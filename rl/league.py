"""League self-play: opponent pool of scripted bots + policy snapshots.

Each env worker owns a League and samples per episode. Snapshot discovery
reads the snapshot dir live, so newly exported snapshots are picked up
without any IPC. Snapshots are exported policy weights JSON files
(see rl/export_ts.py) consumed by the in-bridge TS policy controller.
"""
from __future__ import annotations

import glob
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
                 deterministic_snapshots: bool = False):
        self.snapshot_dir = snapshot_dir
        self.scripted = list(scripted)
        self.p_scripted = p_scripted
        self.p_latest = p_latest
        self.max_snapshots = max_snapshots
        self.deterministic_snapshots = deterministic_snapshots
        os.makedirs(snapshot_dir, exist_ok=True)

    def snapshots(self) -> list[str]:
        paths = sorted(glob.glob(os.path.join(self.snapshot_dir, "*.json")),
                       key=os.path.getmtime)
        return paths[-self.max_snapshots:]

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        snaps = self.snapshots()
        roll = rng.random()
        if snaps and roll >= self.p_scripted:
            if roll < self.p_scripted + self.p_latest or len(snaps) == 1:
                chosen = snaps[-1]
            else:
                chosen = snaps[int(rng.integers(0, len(snaps) - 1))]
            return {"policy": {"weights": chosen,
                               "deterministic": self.deterministic_snapshots}}
        bot = self.scripted[int(rng.integers(0, len(self.scripted)))]
        return {"bot": bot}

    def describe_last(self) -> dict[str, Any]:
        snaps = self.snapshots()
        return {"n_snapshots": len(snaps),
                "latest": os.path.basename(snaps[-1]) if snaps else None}
