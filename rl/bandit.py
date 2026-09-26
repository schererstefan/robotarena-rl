"""Per-opponent strategy bandit (Pluto lesson #4).

Pluto keeps strategy selection *outside* the neural net: a per-opponent bandit
picks the build order, which is fed to the policy as a conditioning vector.
Here the "strategies" are a small set of opening labels
(rush/kite/greedy/control). The label is a pure conditioning signal — the net
discovers the behavior itself (Pluto: "labels may not match behavior"); the
bandit learns which label wins against which opponent.

Persistence mirrors Pluto's bandit file: human-readable JSON with per-arm
win/loss counts and a ``seq`` line — bump ``seq`` to make hand-edits win.

Worker processes read the shared JSON fresh on every episode (the same
live-read pattern as the PFSP file: atomic os.replace on write, no locks,
no IPC). A main-process callback aggregates per-episode outcomes from infos
and rewrites the file periodically.
"""
from __future__ import annotations

import json
import math
import os
import time

STRATEGIES = ["rush", "kite", "greedy", "control"]
N_STRATEGIES = len(STRATEGIES)
BANDIT_FILENAME = "strategy_bandit.json"


def _blank_opponent() -> dict:
    return {s: {"wins": 0, "losses": 0} for s in STRATEGIES}


def _blank_doc(ucb_c: float = 1.0) -> dict:
    return {
        "seq": 0,
        "strategies": list(STRATEGIES),
        "ucb_c": float(ucb_c),
        "opponents": {},
        "updated_ts": time.time(),
    }


class StrategyBandit:
    """UCB1 bandit over strategy labels, one table per opponent."""

    def __init__(self, path: str, ucb_c: float = 1.0):
        self.path = path
        self.ucb_c = float(ucb_c)
        self._doc = _blank_doc(ucb_c)
        self._seq = -1
        self.refresh()

    # -- persistence ------------------------------------------------------
    def refresh(self) -> None:
        """Re-read the JSON file if it is newer than our in-memory copy."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                doc = json.load(f)
        except (OSError, ValueError):
            return
        if not isinstance(doc, dict):
            return
        if int(doc.get("seq", 0)) <= self._seq:
            return
        if doc.get("strategies") != STRATEGIES:
            return  # foreign schema; ignore
        self._doc = doc
        self._seq = int(doc.get("seq", 0))
        self.ucb_c = float(doc.get("ucb_c", self.ucb_c))

    def save(self) -> None:
        self._seq += 1
        self._doc["seq"] = self._seq
        self._doc["updated_ts"] = time.time()
        tmp = self.path + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(self._doc, f, indent=2)
        os.replace(tmp, self.path)

    # -- bandit logic ------------------------------------------------------
    def _table(self, opponent: str) -> dict:
        opps = self._doc.setdefault("opponents", {})
        return opps.setdefault(opponent, _blank_opponent())

    @staticmethod
    def _pulls(cell: dict) -> int:
        return int(cell.get("wins", 0)) + int(cell.get("losses", 0))

    def select(self, opponent: str, exclude: tuple = ()) -> int:
        """UCB1 arm index. Arms with no recorded pulls are tried first,
        skipping `exclude` (arms this worker already tried since the last
        file refresh) so one worker doesn't hammer a single untried arm."""
        self.refresh()
        table = self._table(str(opponent))
        pulls = [self._pulls(table[s]) for s in STRATEGIES]
        excl = {int(e) % N_STRATEGIES for e in exclude}
        for i, p in enumerate(pulls):
            if p == 0 and i not in excl:
                return i
        cands = [i for i in range(N_STRATEGIES) if i not in excl]
        if not cands:
            cands = list(range(N_STRATEGIES))
        total = sum(pulls[i] for i in cands)
        if total <= 0:
            return cands[0]
        log_total = math.log(total)
        best_i, best_score = cands[0], -1e18
        for i in cands:
            wr = table[STRATEGIES[i]]["wins"] / pulls[i]
            score = wr + self.ucb_c * math.sqrt(log_total / pulls[i])
            if score > best_score:
                best_score, best_i = score, i
        return best_i

    @property
    def seq(self) -> int:
        """File generation last loaded (for local exploration tracking)."""
        return self._seq

    def best(self, opponent: str) -> int:
        """Argmax win-rate arm (for eval / export). Ties -> lowest index."""
        self.refresh()
        table = self._table(str(opponent))
        best_i, best_wr, best_pulls = 0, -1.0, -1
        for i, s in enumerate(STRATEGIES):
            cell = table[s]
            p = self._pulls(cell)
            wr = (cell["wins"] / p) if p else 0.0
            if (wr, p) > (best_wr, best_pulls):
                best_wr, best_pulls, best_i = wr, p, i
        return best_i

    def update(self, opponent: str, arm: int, won: bool) -> None:
        table = self._table(str(opponent))
        cell = table[STRATEGIES[int(arm) % N_STRATEGIES]]
        cell["wins" if won else "losses"] = int(cell.get("wins" if won else "losses", 0)) + 1

    def overall_best(self) -> int:
        """Argmax arm by total wins across opponents (export default)."""
        self.refresh()
        wins = [0] * N_STRATEGIES
        pulls = [0] * N_STRATEGIES
        for table in self._doc.get("opponents", {}).values():
            for i, s in enumerate(STRATEGIES):
                cell = table.get(s, {})
                w = int(cell.get("wins", 0))
                p = w + int(cell.get("losses", 0))
                wins[i] += w
                pulls[i] += p
        best_i, best_key = 0, (-1, -1)
        for i in range(N_STRATEGIES):
            key = (wins[i], pulls[i])
            if key > best_key:
                best_key, best_i = key, i
        return best_i

    def arm_counts(self) -> dict[str, int]:
        counts = {s: 0 for s in STRATEGIES}
        for table in self._doc.get("opponents", {}).values():
            for i, s in enumerate(STRATEGIES):
                counts[s] += self._pulls(table.get(s, {}))
        return counts
