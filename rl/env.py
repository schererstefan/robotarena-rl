"""Gymnasium env wrapping the RobotArena Node bridge (stdio JSONL).

Pluto-upgrade (Sep 2026): memory lives in the net via frame stacking --
the last FRAME_STACK raw bridge frames (128-dim each) are concatenated and
fed to the policy together with a strategy-label one-hot, so the net can
infer foe velocity from position deltas instead of relying solely on the
hand-computed leadFeat crutch. The bridge protocol itself is unchanged
(still 128-dim frames, tick-for-tick).
"""
from __future__ import annotations

import json
import math as _math
import os
import queue
import subprocess
import threading
from collections import deque
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .bandit import N_STRATEGIES, StrategyBandit
from .reward import DEFAULT_WEIGHTS, compute_reward

import shutil

ACTION_DIMS = [5, 5, 5, 2, 2, 2, 2]
FRAME_DIM = 128          # raw per-tick obs emitted by the Node bridge
FRAME_STACK = 4          # Pluto lesson #1: temporal memory in the net
N_STRATS = N_STRATEGIES  # strategy-label conditioning (Pluto lesson #4)
OBS_DIM = FRAME_STACK * FRAME_DIM + N_STRATS  # 516
OPENING_TICKS = 600      # opening-alignment shaping applies below this tick
# Named obs indices mirroring bridge/OBS_SPEC.md:7-18.
SELF_SPEED = 6
FOE_PRESENT = 17
BULLET_BASE = 38
TICK = 121
NODE_BIN = (shutil.which("node") or "/opt/homebrew/bin/node"
            or "/usr/local/bin/node")
_BRIDGE_REL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "bridge", "dist", "bridge.mjs")


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except ValueError:
        return float(default)
    return value if value > 0 else float(default)


# Bridge RPC hardening: a chatty or hung bridge must never wedge a worker.
# Every RPC reply wait is bounded by BRIDGE_RPC_TIMEOUT; close() gets its
# own shorter budgets. All three are env-overridable at import time.
BRIDGE_RPC_TIMEOUT = _env_float("ROBOTARENA_BRIDGE_RPC_TIMEOUT", 30.0)
BRIDGE_SHUTDOWN_TIMEOUT = _env_float("ROBOTARENA_BRIDGE_SHUTDOWN_TIMEOUT", 2.0)
BRIDGE_TERMINATE_TIMEOUT = _env_float("ROBOTARENA_BRIDGE_TERMINATE_TIMEOUT", 3.0)
_STDERR_TAIL_MAXLEN = 200  # bridge stderr lines retained for diagnostics

_INTENT_KEYS = ["throttle", "turn", "towerTurn", "fire", "charge", "dash", "emp"]
# Boolean ability action indices (for the diversity bonus).
_ABILITY_IDX = {4: "charge", 5: "dash", 6: "emp"}


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


def _pad_dist(raw: np.ndarray, base: int, n: int) -> float | None:
    """Nearest active pad/turret distance (arena-fraction), or None."""
    best = None
    for i in range(n):
        b = base + i * 6
        if raw[b] < 0.5:
            continue
        dx = raw[b + 1] * 960.0
        dy = raw[b + 2] * 640.0
        d = _math.hypot(dx, dy) / 960.0
        if best is None or d < best:
            best = d
    return best


def dodge_perp(raw) -> float:
    """Best perpendicular-motion fraction (0..~1) vs incoming bullets. Pure, no bridge."""
    speed = float(raw[SELF_SPEED]) * 375.0
    if speed <= 20.0:
        return 0.0
    heading = _math.atan2(raw[2], raw[3])
    vx = speed * _math.cos(heading)
    vy = speed * _math.sin(heading)
    best = 0.0
    for i in range(6):
        b = BULLET_BASE + i * 6
        if raw[b] < 0.5:
            continue
        closing = raw[b + 4] * 430.0
        dist = raw[b + 3] * 540.0
        if closing < 50.0 or dist > 350.0 or dist < 1e-6:
            continue
        bx = raw[b + 1] * 960.0
        by = raw[b + 2] * 640.0
        norm = _math.hypot(bx, by)
        if norm < 1e-6:
            continue
        px, py = -by / norm, bx / norm
        perp = abs(vx * px + vy * py) / 375.0
        if perp > best:
            best = perp
    return best


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
        bandit_path: str | None = None,
        bandit_mode: str = "ucb",
        fixed_arm: int = 0,
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
        # Strategy conditioning (Pluto lesson #4). bandit_mode: "ucb" (train
        # workers), "best" (eval: argmax win-rate arm), "fixed" (fixed_arm).
        self._bandit_path = bandit_path
        self._bandit_mode = bandit_mode
        self._fixed_arm = int(fixed_arm) % N_STRATS
        self._reader = StrategyBandit(bandit_path) if bandit_path else None
        self._local_tried: dict[str, set[int]] = {}
        self._local_seq: dict[str, int] = {}
        self._frames: deque[np.ndarray] = deque(maxlen=FRAME_STACK)
        self._prev_raw: np.ndarray | None = None
        self._arm = self._fixed_arm
        self._opp_name = "none"
        self._abilities_used: set[str] = set()

        bridge = bridge_path or _BRIDGE_REL
        if not os.path.exists(bridge):
            raise FileNotFoundError(
                f"bridge not found at {bridge} — run bridge/build.sh first")
        self._rpc_timeout = BRIDGE_RPC_TIMEOUT
        self._closed = False
        self._stdout_eof = False
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_MAXLEN)
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._proc = subprocess.Popen(
            [NODE_BIN, bridge],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        # Drain stderr continuously: an undrained PIPE deadlocks once the
        # bridge writes more than the OS pipe buffer (stack traces etc.).
        # Stdout lines go through a queue so _rpc can wait with a timeout.
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name="bridge-stderr")
        self._stderr_thread.start()
        self._stdout_thread = threading.Thread(
            target=self._pump_stdout, daemon=True, name="bridge-stdout")
        self._stdout_thread.start()
        # Handshake: fail fast if the bridge is broken. The bridge still
        # speaks 128-dim frames; stacking happens here.
        try:
            pong = self._rpc({"cmd": "ping"})
            if pong.get("type") != "pong" or pong.get("obsDim") != FRAME_DIM:
                raise BridgeError(f"bridge handshake failed: {pong}")
        except Exception:
            self._closed = True
            self._force_kill()
            raise
        self._rng = np.random.default_rng()
        self._last_info: dict[str, Any] = {}

    # -- bridge process plumbing -------------------------------------------
    def _drain_stderr(self) -> None:
        stream = self._proc.stderr
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                self._stderr_tail.append(line)
        except (OSError, ValueError):
            pass

    def _pump_stdout(self) -> None:
        stream = self._proc.stdout
        if stream is None:
            self._stdout_queue.put(None)
            return
        try:
            for line in iter(stream.readline, ""):
                self._stdout_queue.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._stdout_queue.put(None)

    def _stderr_text(self) -> str:
        return "".join(self._stderr_tail)[-2000:]

    def _force_kill(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=BRIDGE_TERMINATE_TIMEOUT)
        except Exception:
            pass

    # -- stdio RPC ---------------------------------------------------------
    def _rpc(self, msg: dict, timeout: float | None = None) -> dict:
        effective = self._rpc_timeout if timeout is None else timeout
        with self._lock:
            if self._closed:
                raise BridgeError("bridge RPC attempted on a closed env")
            if self._stdout_eof:
                raise BridgeError(
                    f"bridge closed stdout; stderr: {self._stderr_text()}")
            assert self._proc.stdin is not None
            try:
                self._proc.stdin.write(json.dumps(msg) + "\n")
                self._proc.stdin.flush()
            except OSError as exc:
                raise BridgeError(
                    f"bridge stdin closed ({exc}); "
                    f"stderr: {self._stderr_text()}") from exc
            try:
                line = self._stdout_queue.get(timeout=effective)
            except queue.Empty:
                # A hung bridge would otherwise wedge every worker using
                # this env forever. Kill it so the failure is recoverable,
                # and surface what it said on stderr.
                self._force_kill()
                raise BridgeError(
                    f"bridge RPC timed out after {effective:.1f}s waiting "
                    f"for a reply to {msg.get('cmd', '?')!r}; killed the "
                    f"bridge; stderr tail: {self._stderr_text()}")
        if not line:
            self._stdout_eof = True
            raise BridgeError(
                f"bridge closed stdout; stderr: {self._stderr_text()}")
        resp = json.loads(line)
        if resp.get("type") == "error":
            raise BridgeError(resp.get("message", "unknown bridge error"))
        return resp

    # -- strategy ----------------------------------------------------------
    @staticmethod
    def _opponent_name(opponent: dict) -> str:
        if "bot" in opponent:
            return str(opponent["bot"])
        pol = opponent.get("policy", {})
        w = pol.get("weights", "snapshot")
        return "selfplay:" + os.path.basename(str(w))

    def _select_arm(self, opp_name: str) -> int:
        if self._reader is None or self._bandit_mode == "fixed":
            return self._fixed_arm
        if self._bandit_mode == "best":
            return self._reader.best(opp_name)
        # UCB with local exploration: each worker tries every arm once per
        # file generation before repeating, so arms get tried promptly even
        # though the shared file is only rewritten periodically.
        seq = self._reader.seq  # property; refresh() runs inside select()
        tried = self._local_tried.get(opp_name)
        if tried is None or self._local_seq.get(opp_name) != seq:
            tried = set()
            self._local_tried[opp_name] = tried
            self._local_seq[opp_name] = seq
        arm = self._reader.select(opp_name, exclude=tuple(tried))
        self._local_seq[opp_name] = self._reader.seq
        tried.add(arm)
        return arm

    def _stacked_obs(self) -> np.ndarray:
        oh = np.zeros((N_STRATS,), dtype=np.float32)
        oh[self._arm] = 1.0
        return np.concatenate(
            [np.asarray(f, dtype=np.float32) for f in self._frames] + [oh]
        ).astype(np.float32)

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        opp_seed = int(self._rng.integers(0, 2 ** 31 - 1))
        opponent = (self._opponent_sampler(self._rng)
                    if self._opponent_sampler else {"bot": "wanderer"})
        self._opp_name = self._opponent_name(opponent)
        self._arm = self._select_arm(self._opp_name)
        resp = self._rpc({"cmd": "reset", "seed": opp_seed,
                          "opponent": opponent, "arena": self._arena,
                          "loadout": self._loadout})
        raw = np.asarray(resp["obs"], dtype=np.float32)
        self._frames = deque([raw.copy() for _ in range(FRAME_STACK)],
                             maxlen=FRAME_STACK)
        self._prev_raw = raw.copy()
        self._abilities_used = set()
        obs = self._stacked_obs()
        self._last_info = {"opponent": opponent, "seed": opp_seed,
                           "opponent_name": self._opp_name,
                           "strategy_arm": self._arm,
                           **resp.get("info", {})}
        return obs, self._last_info

    def step(self, action: np.ndarray):
        action = np.asarray(action).reshape(-1)
        intent = action_to_intent(action)
        firing = bool(action[3] == 1)
        resp = self._rpc({"cmd": "step", "intent": intent})
        raw = np.asarray(resp["obs"], dtype=np.float32)
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
        # outrunning it. Computed from raw obs (pre-VecNormalize);
        # see dodge_perp helper above.
        dodge_w = self._weights.get("dodge", 0.0)
        if dodge_w != 0.0:
            best = dodge_perp(raw)
            if best > 0:
                reward += dodge_w * best
                rinfo["reward/dodge"] = dodge_w * best
        # Diversity (Pluto lesson #3): one-time per-episode bonus for the
        # first use of each ability. Small anti-mode-collapse nudge.
        div_w = self._weights.get("diversity", 0.0)
        if div_w != 0.0:
            for idx, name in _ABILITY_IDX.items():
                if int(action[idx]) == 1 and name not in self._abilities_used:
                    self._abilities_used.add(name)
                    reward += div_w
            if self._abilities_used:
                rinfo["reward/diversity"] = div_w * len(self._abilities_used)
        # Opening alignment (Pluto lesson #3): for the first OPENING_TICKS,
        # nudge the policy toward its strategy label's reference behavior.
        # Soft nudge, not enforced; computed from raw obs, no bridge changes.
        open_w = self._weights.get("opening", 0.0)
        prev = self._prev_raw
        if open_w != 0.0 and prev is not None:
            tick = float(raw[TICK]) * 11400.0
            if tick < OPENING_TICKS:
                bonus = self._opening_bonus(raw, prev)
                if bonus > 0.0:
                    reward += open_w * bonus
                    rinfo["reward/opening"] = open_w * bonus
        self._prev_raw = raw.copy()
        self._frames.append(raw)

        info.update(rinfo)
        info["opponent"] = self._last_info.get("opponent")
        info["opponent_name"] = self._opp_name
        info["strategy_arm"] = self._arm
        if terminated or truncated:
            # Reported to the main-process bandit callback via infos.
            won = (winner == 0)
            info["bandit"] = {"opponent": self._opp_name,
                              "arm": self._arm, "won": bool(won)}
        return self._stacked_obs(), reward, terminated, truncated, info

    def _opening_bonus(self, raw: np.ndarray, prev: np.ndarray) -> float:
        """Reference-behavior alignment for the current strategy arm."""
        arm = self._arm
        if arm == 0 or arm == 1:
            # rush: close the foe / kite: open distance (foe slot 0).
            if raw[FOE_PRESENT] < 0.5 or prev[FOE_PRESENT] < 0.5:
                return 0.0
            delta = float(prev[20] - raw[20])  # + means closing
            return max(0.0, delta) if arm == 0 else max(0.0, -delta)
        if arm == 2:
            # greedy: approach nearest active pickup pad (dims 94-117).
            d0 = _pad_dist(prev, 94, 4)
            d1 = _pad_dist(raw, 94, 4)
            if d0 is None or d1 is None:
                return 0.0
            return max(0.0, d0 - d1)
        # control: approach nearest active turret (dims 82-93), else center.
        d0 = _pad_dist(prev, 82, 2)
        d1 = _pad_dist(raw, 82, 2)
        if d0 is None or d1 is None:
            cx, cy = float(raw[0]) - 0.5, float(raw[1]) - 0.5
            px, py = float(prev[0]) - 0.5, float(prev[1]) - 0.5
            d0 = _math.hypot(px, py)
            d1 = _math.hypot(cx, cy)
        return max(0.0, d0 - d1)

    def close(self):
        if getattr(self, "_closed", True):
            return
        # Best-effort graceful shutdown, bounded so a hung bridge cannot
        # wedge close() for the full RPC timeout.
        try:
            self._rpc({"cmd": "shutdown"}, timeout=BRIDGE_SHUTDOWN_TIMEOUT)
        except Exception:
            pass
        self._closed = True
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=BRIDGE_TERMINATE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=BRIDGE_TERMINATE_TIMEOUT)
        except OSError:
            pass
