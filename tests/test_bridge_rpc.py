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
- close() reaping a process that ignores both shutdown and SIGTERM.
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
        send({"type": "state", "obs": [0.0] * 128, "info": {}})
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
