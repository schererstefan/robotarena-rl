"""League self-play: opponent pool of scripted bots + policy snapshots.

Each env worker owns a League and samples per episode. Snapshot discovery
reads the snapshot dir live, so newly exported snapshots are picked up
without any IPC. Snapshots are exported policy weights JSON files
(see rl/export_ts.py) consumed by the in-bridge TS policy controller.

Sampling (fixed Sep 2026): the old code compared the roll against
``p_scripted + p_latest`` as absolute thresholds, so with
p_scripted=0.7 / p_latest=0.5 *every* self-play roll drew the newest
snapshot and the historical pool was never touched. The draw is now
two-stage:

1. with prob ``p_scripted`` -> scripted bot (PFSP-weighted by difficulty,
   see below); otherwise a self-play snapshot.
2. within self-play: with prob ``p_latest`` (a fraction *of the self-play
   share*) -> the newest snapshot, else uniform over the historical pool.

PFSP (prioritized fictitious self-play): WinRateEvalCallback writes the
live policy's latest deterministic per-bot win rates to
<run_dir>/league/pfsp.json; the League reads it live and weights scripted
bots by difficulty::

    w_i = (1 - u) * (1 - wr_i)^p / Z + u / K

with ``p = pfsp_exponent`` and ``u = uniform_mix``. The policy spends most
episodes against its hardest scripted opponents while never starving the
easy ones. A missing or unreadable pfsp.json falls back to uniform
sampling over the scripted list.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any

import numpy as np

SCRIPTED_ROSTER = ["wanderer", "rusher", "hunter", "orbiter", "sniper", "brawler"]


class League:
    def __init__(self, snapshot_dir: str,
                 scripted: tuple[str, ...] = tuple(SCRIPTED_ROSTER),
                 p_scripted: float = 0.3,
                 p_latest: float = 0.4,
                 max_snapshots: int = 8,
                 deterministic_snapshots: bool = False,
                 pfsp_exponent: float = 1.5,
                 uniform_mix: float = 0.2):
        self.snapshot_dir = snapshot_dir
        self.scripted = list(scripted)
        self.p_scripted = p_scripted
        self.p_latest = p_latest
        self.max_snapshots = max_snapshots
        self.deterministic_snapshots = deterministic_snapshots
        self.pfsp_exponent = pfsp_exponent
        self.uniform_mix = uniform_mix
        self.pfsp_path = os.path.join(snapshot_dir, "pfsp.json")
        os.makedirs(snapshot_dir, exist_ok=True)

    def snapshots(self) -> list[str]:
        paths = sorted(glob.glob(os.path.join(self.snapshot_dir, "*.json")),
                       key=os.path.getmtime)
        # pfsp.json shares the dir but is not a snapshot.
        paths = [p for p in paths
                 if os.path.basename(p) != "pfsp.json"]
        return paths[-self.max_snapshots:]

    def _pfsp_win_rates(self) -> dict[str, float] | None:
        try:
            with open(self.pfsp_path) as f:
                d = json.load(f)
            wr = d.get("win_rate")
            if isinstance(wr, dict) and wr:
                return {str(k): float(v) for k, v in wr.items()}
        except (OSError, ValueError):
            pass
        return None

    def _sample_scripted(self, rng: np.random.Generator) -> str:
        wr = self._pfsp_win_rates()
        if wr is None:
            return self.scripted[int(rng.integers(0, len(self.scripted)))]
        uniq = list(dict.fromkeys(self.scripted))
        k = len(uniq)
        raw = np.array([(1.0 - wr.get(b, 0.5)) ** self.pfsp_exponent
                        for b in uniq], dtype=np.float64)
        z = raw.sum()
        if z <= 0:
            probs_uniq = np.full(k, 1.0 / k)
        else:
            probs_uniq = ((1.0 - self.uniform_mix) * raw / z
                          + self.uniform_mix / k)
        w_of = dict(zip(uniq, probs_uniq))
        # Duplicate entries keep their extra mass (e.g. brawler listed twice
        # still gets 2x the per-bot weight), preserving the mix's intent.
        weights = np.array([w_of[b] for b in self.scripted])
        weights /= weights.sum()
        return self.scripted[int(rng.choice(len(self.scripted), p=weights))]

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        snaps = self.snapshots()
        if snaps and rng.random() >= self.p_scripted:
            # Self-play share: p_latest of it draws the newest snapshot,
            # the remainder is uniform over the historical pool.
            if rng.random() < self.p_latest or len(snaps) == 1:
                chosen = snaps[-1]
            else:
                chosen = snaps[int(rng.integers(0, len(snaps) - 1))]
            # Pluto upgrade: snapshots also play a random strategy arm so the
            # learner meets label-conditioned opponents (arm is per-episode).
            return {"policy": {"weights": chosen,
                               "deterministic": self.deterministic_snapshots,
                               "strategy": int(rng.integers(0, 4))}}
        return {"bot": self._sample_scripted(rng)}

    def describe_last(self) -> dict[str, Any]:
        snaps = self.snapshots()
        return {"n_snapshots": len(snaps),
                "latest": os.path.basename(snaps[-1]) if snaps else None}
