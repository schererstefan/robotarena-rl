"""Gymnasium env wrapping the RobotArena Node bridge (stdio JSONL)."""
from __future__ import annotations

import json
import os
import subprocess
import threading
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .reward import DEFAULT_WEIGHTS, compute_reward

import shutil

ACTION_DIMS = [5, 5, 5, 2, 2, 2, 2]
OBS_DIM = 124
NODE_BIN = (shutil.which("node") or "/opt/homebrew/bin/node"
            or "/usr/local/bin/node")
_BRIDGE_REL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "bridge", "dist", "bridge.mjs")

_INTENT_KEYS = ["throttle", "turn", "towerTurn", "fire", "charge", "dash", "emp"]


def action_to_intent(action: np.ndarray) -> dict[str, Any]:
    a = [int(x) for x in action]
    return {
        "throttle": a[0] / 2 - 1,
        "turn": a[1] / 2 - 1,
        "towerTurn": a[2] / 2 - 1,
        "fire": a[3] == 1,
        "charge": a[4] == 1,
        "dash": a[5] == 1,
        "emp": a[6] == 1,
    }


class BridgeError(RuntimeError):
    pass


class RobotArenaEnv(gym.Env):
    """1v1 RobotArena match vs a scripted bot or an exported policy snapshot.

    Each env instance owns one Node bridge subprocess. Not thread-safe across
    threads (one lock guards the stdio pair); use one instance per worker.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_sampler: Callable[[np.random.Generator], dict] | None = None,
        arena: str = "open",
        loadout: dict | None = None,
        reward_weights: dict | None = None,
        bridge_path: str | None = None,
        max_episode_ticks: int = 11400,
    ):
        super().__init__()
        self.action_space = spaces.MultiDiscrete(ACTION_DIMS)
        self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(OBS_DIM,),
                                            dtype=np.float32)
        self._opponent_sampler = opponent_sampler
        self._arena = arena
        self._loadout = loadout or {}
        self._weights = dict(DEFAULT_WEIGHTS)
        if reward_weights:
            self._weights.update(reward_weights)
        self._max_episode_ticks = max_episode_ticks
        self._lock = threading.Lock()

        bridge = bridge_path or _BRIDGE_REL
        if not os.path.exists(bridge):
            raise FileNotFoundError(
                f"bridge not found at {bridge} — run bridge/build.sh first")
        self._proc = subprocess.Popen(
            [NODE_BIN, bridge],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        # Handshake: fail fast if the bridge is broken.
        pong = self._rpc({"cmd": "ping"})
        if pong.get("type") != "pong" or pong.get("obsDim") != OBS_DIM:
            raise BridgeError(f"bridge handshake failed: {pong}")
        self._rng = np.random.default_rng()
        self._last_info: dict[str, Any] = {}

    # -- stdio RPC ---------------------------------------------------------
    def _rpc(self, msg: dict) -> dict:
        with self._lock:
            assert self._proc.stdin is not None and self._proc.stdout is not None
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        if not line:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise BridgeError(f"bridge closed stdout; stderr: {err[-2000:]}")
        resp = json.loads(line)
        if resp.get("type") == "error":
            raise BridgeError(resp.get("message", "unknown bridge error"))
        return resp

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        opp_seed = int(self._rng.integers(0, 2 ** 31 - 1))
        opponent = (self._opponent_sampler(self._rng)
                    if self._opponent_sampler else {"bot": "wanderer"})
        resp = self._rpc({"cmd": "reset", "seed": opp_seed,
                          "opponent": opponent, "arena": self._arena,
                          "loadout": self._loadout})
        obs = np.asarray(resp["obs"], dtype=np.float32)
        self._last_info = {"opponent": opponent, "seed": opp_seed,
                           **resp.get("info", {})}
        return obs, self._last_info

    def step(self, action: np.ndarray):
        action = np.asarray(action).reshape(-1)
        intent = action_to_intent(action)
        firing = bool(action[3] == 1)
        resp = self._rpc({"cmd": "step", "intent": intent})
        obs = np.asarray(resp["obs"], dtype=np.float32)
        components = resp.get("components", {})
        terminated = bool(resp.get("terminated", False))
        truncated = bool(resp.get("truncated", False))
        info = resp.get("info", {})
        winner = info.get("winner", -1)
        reward, rinfo = compute_reward(components, self._weights,
                                       terminated=terminated,
                                       won=(winner == 0), lost=(winner == 1),
                                       firing=firing)
        # Dodge: reward moving perpendicular to incoming bullets.
        # A bullet on collision course is dodged by sidestepping, not by
        # outrunning it. Computed from raw obs (pre-VecNormalize).
        import math as _math
        dodge_w = self._weights.get("dodge", 0.0)
        if dodge_w != 0.0:
            speed = obs[5] * 375.0
            if speed > 20.0:
                heading = _math.atan2(obs[2], obs[3])
                vx = speed * _math.cos(heading)
                vy = speed * _math.sin(heading)
                best = 0.0
                for i in range(6):
                    b = 38 + i * 6
                    if obs[b] < 0.5:
                        continue
                    closing = obs[b + 4] * 430.0
                    dist = obs[b + 3] * 540.0
                    if closing < 50.0 or dist > 350.0 or dist < 1e-6:
                        continue
                    # bullet relative pos; perpendicular direction
                    bx = obs[b + 1] * 960.0
                    by = obs[b + 2] * 640.0
                    px, py = -by / dist, bx / dist
                    perp = abs(vx * px + vy * py) / 375.0
                    if perp > best:
                        best = perp
                if best > 0:
                    reward += dodge_w * best
                    rinfo["reward/dodge"] = dodge_w * best

        info.update(rinfo)
        info["opponent"] = self._last_info.get("opponent")
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self._rpc({"cmd": "shutdown"})
        except Exception:
            pass
        if self._proc.poll() is None:
            self._proc.terminate()
