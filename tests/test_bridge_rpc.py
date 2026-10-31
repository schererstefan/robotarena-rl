"""Fake-bridge tests for the bridge RPC hardening in rl/env.py.

A tiny Python script stands in for the Node bridge (spawned via a
monkeypatched NODE_BIN -> sys.executable), so these run anywhere the
pure suite runs: no Node, no torch, no built bridge. They cover the
three failure modes that used to hang or leak silently:

- normal request/response round-trip (incl. a full env.reset()),
- a bridge that floods stderr past the OS pipe buffer before replying
  (deadlocked before stderr was drained continuously),
- a bridge that never replies (RPC must time out, raise BridgeError
  carrying the stderr tail, and kill the bridge),
- close() reaping a process that ignores both shutdown and SIGTERM,
- short / over-long / NaN / inf / missing obs frames and malformed
  JSON replies, which must surface as BridgeError from reset()/step()
  instead of silently shifting the stacked-obs shape or leaking raw
  KeyError/JSONDecodeError into training.
"""

import sys

import pytest

import rl.env as env_mod
from rl.env import OBS_DIM, BridgeError, RobotArenaEnv

FAKE_BRIDGE = r"""
import json
import os
import signal
import sys
import time

mode = os.environ.get("FAKE_BRIDGE_MODE", "normal")
if mode == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


while True:
    raw = sys.stdin.readline()
    if not raw:
        break
    try:
        msg = json.loads(raw)
    except Exception:
        continue
    cmd = msg.get("cmd")
    if cmd == "ping":
        if mode == "flood":
            chunk = "bridge-noise " + "x" * 100 + "\n"
            for _ in range(3000):  # ~340 KB, far past the pipe buffer
                sys.stderr.write(chunk)
            sys.stderr.flush()
        send({"type": "pong", "obsDim": 128})
    elif cmd == "hang":
        sys.stderr.write("fake bridge hanging on purpose\n")
        sys.stderr.flush()
        time.sleep(60)
    elif cmd == "reset":
        if mode == "short":
            send({"type": "state", "obs": [0.0] * 127, "info": {}})
        elif mode == "long":
            send({"type": "state", "obs": [0.0] * 129, "info": {}})
        elif mode == "nan":
            send({"type": "state", "obs": [0.0] * 127 + [float("nan")],
                  "info": {}})
        elif mode == "inf":
            send({"type": "state", "obs": [float("inf")] + [0.0] * 127,
                  "info": {}})
        elif mode == "missing_obs":
            send({"type": "state", "info": {}})
        elif mode == "badjson":
            sys.stdout.write("not json{\n")
            sys.stdout.flush()
        elif mode == "nonobject":
            sys.stdout.write("[1, 2, 3]\n")
            sys.stdout.flush()
        else:
            send({"type": "state", "obs": [0.0] * 128, "info": {}})
    elif cmd == "step":
        if mode == "badstep":
            send({"type": "step", "obs": [0.0] * 3, "components": {},
                  "terminated": False, "truncated": False, "info": {}})
        elif mode == "badjson":
            sys.stdout.write("not json{\n")
            sys.stdout.flush()
        else:
            send({"type": "step", "obs": [0.0] * 128, "components": {},
                  "terminated": False, "truncated": False, "info": {}})
    elif cmd == "shutdown":
        send({"type": "bye"})
        if mode != "stubborn":
            sys.exit(0)
    else:
        send({"type": "ok"})
"""


@pytest.fixture()
def fake_bridge(tmp_path):
    script = tmp_path / "fake_bridge.py"
    script.write_text(FAKE_BRIDGE)
    return str(script)


def make_env(monkeypatch, bridge_path, mode):
    monkeypatch.setenv("FAKE_BRIDGE_MODE", mode)
    monkeypatch.setattr(env_mod, "NODE_BIN", sys.executable)
    return RobotArenaEnv(bridge_path=bridge_path)


def test_rpc_round_trip_and_reset(monkeypatch, fake_bridge):
    env = make_env(monkeypatch, fake_bridge, "normal")
    try:
        assert env._rpc({"cmd": "anything"}) == {"type": "ok"}
        obs, _info = env.reset(seed=0)
        assert obs.shape == (OBS_DIM,)
    finally:
        env.close()
    assert env._proc.poll() is not None
    env.close()  # idempotent: second close is a no-op


def test_stderr_flood_does_not_deadlock(monkeypatch, fake_bridge):
    # Construction performs the ping handshake; the fake floods stderr
    # before replying, so this only completes if stderr is drained.
    env = make_env(monkeypatch, fake_bridge, "flood")
    try:
        assert "bridge-noise" in env._stderr_text()
        # The tail is bounded (deque of lines), not an unbounded buffer.
        assert sum(len(s) for s in env._stderr_tail) <= 200 * 200
    finally:
        env.close()


def test_rpc_timeout_raises_with_stderr_tail(monkeypatch, fake_bridge):
    env = make_env(monkeypatch, fake_bridge, "normal")
    env._rpc_timeout = 0.5
    with pytest.raises(BridgeError, match="timed out") as excinfo:
        env._rpc({"cmd": "hang"})
    assert "hanging on purpose" in str(excinfo.value)
    # The timeout path kills the bridge instead of leaving it wedged.
    assert env._proc.poll() is not None
    env.close()


def test_close_reaps_stubborn_process(monkeypatch, fake_bridge):
    monkeypatch.setattr(env_mod, "BRIDGE_TERMINATE_TIMEOUT", 0.3)
    env = make_env(monkeypatch, fake_bridge, "stubborn")
    env.close()  # shutdown is acked but ignored; SIGTERM is ignored too
    assert env._proc.poll() is not None
    env.close()  # idempotent


@pytest.mark.parametrize("mode", ["short", "long", "nan", "inf", "missing_obs"])
def test_reset_rejects_invalid_obs(monkeypatch, fake_bridge, mode):
    env = make_env(monkeypatch, fake_bridge, mode)
    try:
        with pytest.raises(BridgeError, match="obs"):
            env.reset(seed=0)
    finally:
        env.close()


def test_step_rejects_invalid_obs(monkeypatch, fake_bridge):
    env = make_env(monkeypatch, fake_bridge, "badstep")
    try:
        obs, _info = env.reset(seed=0)
        assert obs.shape == (OBS_DIM,)
        with pytest.raises(BridgeError, match="obs"):
            env.step([0, 0, 0, 0, 0, 0, 0])
    finally:
        env.close()


def test_step_round_trip(monkeypatch, fake_bridge):
    env = make_env(monkeypatch, fake_bridge, "normal")
    try:
        env.reset(seed=0)
        obs, _reward, terminated, truncated, _info = env.step(
            [0, 0, 0, 0, 0, 0, 0])
        assert obs.shape == (OBS_DIM,)
        assert not terminated and not truncated
    finally:
        env.close()


def test_malformed_json_raises_bridge_error(monkeypatch, fake_bridge):
    env = make_env(monkeypatch, fake_bridge, "badjson")
    try:
        with pytest.raises(BridgeError, match="malformed"):
            env.reset(seed=0)
    finally:
        env.close()


def test_nonobject_json_raises_bridge_error(monkeypatch, fake_bridge):
    env = make_env(monkeypatch, fake_bridge, "nonobject")
    try:
        with pytest.raises(BridgeError, match="malformed"):
            env.reset(seed=0)
    finally:
        env.close()


def test_rpc_rejects_oversized_line(monkeypatch, fake_bridge):
    env = make_env(monkeypatch, fake_bridge, "normal")
    monkeypatch.setattr(env_mod, "_MAX_BRIDGE_LINE_CHARS", 5, raising=False)
    try:
        with pytest.raises(BridgeError, match="too large"):
            env._rpc({"cmd": "anything"})
    finally:
        env.close()


def test_validate_raw_obs_accepts_valid_frame():
    from rl.env import FRAME_DIM, _validate_raw_obs

    arr = _validate_raw_obs({"obs": [0.0] * FRAME_DIM})
    assert arr.shape == (FRAME_DIM,)
    assert str(arr.dtype) == "float32"
    ints = _validate_raw_obs({"obs": [0] * FRAME_DIM})
    assert ints.shape == (FRAME_DIM,)
    assert RobotArenaEnv._validate_raw_obs(
        {"obs": [0.0] * FRAME_DIM}).shape == (FRAME_DIM,)


@pytest.mark.parametrize(
    "resp",
    [
        {},
        {"obs": None},
        {"obs": "not-a-list"},
        {"obs": [0.0] * 127},
        {"obs": [0.0] * 129},
        {"obs": [float("nan")] + [0.0] * 127},
        {"obs": [float("inf")] + [0.0] * 127},
        {"obs": [float("-inf")] + [0.0] * 127},
        {"obs": ["0.0"] + [0.0] * 127},
        {"obs": [None] + [0.0] * 127},
        {"obs": [True] + [0.0] * 127},
        {"obs": [[0.0]] + [0.0] * 127},
        {"obs": [1e100] + [0.0] * 127},
        {"obs": [10**400] + [0.0] * 127},
        "not-a-dict",
    ],
)
def test_validate_raw_obs_rejects_invalid(resp):
    from rl.env import _validate_raw_obs

    with pytest.raises(BridgeError):
        _validate_raw_obs(resp)
