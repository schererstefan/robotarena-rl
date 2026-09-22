"""Reward shaping for RobotArena 1v1.

Dense shaping on damage deltas + sparse bonuses for kills, pickups, turret
captures, and the terminal outcome. Aim-assist shaping (`aim`, `fireAim`)
bootstraps the hard exploration problem of aiming the sensor tower and
landing hits; the win bonus keeps the agent optimizing for victory, not
just aiming.
"""
from __future__ import annotations

import math
from typing import Any

DEFAULT_WEIGHTS: dict[str, float] = {
    "dealt": 1.0,        # per 100 HP of damage dealt
    "taken": -1.0,       # per 100 HP of damage taken
    "kill": 3.0,         # landing the killing blow
    "death": -3.0,       # dying
    "pickup": 0.3,       # collecting a powerup pad
    "turret": 1.0,       # capturing a map turret
    "aim": 0.01,         # per tick, scaled by (1 - aimErr/pi) when foe visible
    "fireAim": 0.05,     # per tick when firing with aimErr < 0.15 rad
    "tick": -0.002,      # per-tick urgency (avg match ~850 ticks -> ~-1.7)
    "win": 15.0,
    "loss": -15.0,
    "draw": 0.0,
}

_AIM_TOL = 0.15  # radians — "aimed" threshold for the fireAim bonus


def compute_reward(components: dict[str, Any], weights: dict[str, float],
                   *, terminated: bool, won: bool, lost: bool,
                   firing: bool = False):
    """Return (reward, info). Components come from the bridge step reply."""
    w = weights
    dealt = float(components.get("dealt", 0.0))
    taken = float(components.get("taken", 0.0))
    aim_err = float(components.get("aimErr", -1.0))
    reward = (
        w["dealt"] * dealt / 100.0
        + w["taken"] * taken / 100.0
        + w["kill"] * float(bool(components.get("killed", False)))
        + w["death"] * float(bool(components.get("died", False)))
        + w["pickup"] * float(bool(components.get("pickup", False)))
        + w["turret"] * float(bool(components.get("turret", False)))
        + w["tick"]
    )
    if aim_err >= 0:
        reward += w["aim"] * (1.0 - aim_err / math.pi)
        if firing and aim_err < _AIM_TOL:
            reward += w["fireAim"]
    outcome = "none"
    if terminated:
        if won:
            reward += w["win"]
            outcome = "win"
        elif lost:
            reward += w["loss"]
            outcome = "loss"
        else:
            reward += w["draw"]
            outcome = "draw"
    info = {
        "reward/dealt": dealt,
        "reward/taken": taken,
        "reward/aimErr": aim_err,
        "reward/outcome": outcome,
    }
    return float(reward), info
