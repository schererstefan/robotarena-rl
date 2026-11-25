"""PPO / AnchorKLPPO smoke test: actually run ``learn()`` in CI.

No other test instantiates PPO/AnchorKLPPO or calls ``.learn()``, and
``rl/train.py`` ``main()`` never runs in CI either -- the only SB3/torch
coverage elsewhere is import-level. That leaves the hand-copied PPO
update loop in ``rl/anchor.py`` (``train()``, copied verbatim from
stable_baselines3 2.9.0 with the anchor-KL term added) free to break
silently on the next SB3/torch/gymnasium drift.

This test drives that training path end to end for 64 timesteps against
a fake bridge (no Node, no sibling robotarena checkout, no bridge
build): a tiny Python script stands in for the bridge, spawned via a
monkeypatched ``NODE_BIN`` -> ``sys.executable`` exactly as in
``tests/test_bridge_rpc.py``, whose normal-mode stub is extracted here.

The anchor must be *enabled* for this to mean anything:
``AnchorKLPPO.train()`` short-circuits to ``PPO.train()`` when there is
no anchor policy or ``kl_anchor_coef <= 0`` (rl/anchor.py), so a smoke
test with the anchor disabled would never execute the verbatim copy and
never log ``train/anchor_kl``. The test therefore first saves a tiny
PPO as the frozen anchor checkpoint (the role the iteration seed plays
in ``rl/train.py``), then trains ``AnchorKLPPO`` against it with a
positive, non-annealed coefficient.
"""

import math
import sys

import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.logger import KVWriter, Logger
from stable_baselines3.common.vec_env import DummyVecEnv

import rl.env as env_mod
from rl.anchor import AnchorKLPPO
from rl.env import ACTION_DIMS, OBS_DIM, RobotArenaEnv

# Extracted from tests/test_bridge_rpc.py (its FAKE_BRIDGE, normal mode):
# ping/pong handshake, reset/step answering all-zero 128-dim frames,
# shutdown/bye. Bridge frames are 128-dim; RobotArenaEnv stacks them
# into the OBS_DIM observation the policy sees.
FAKE_BRIDGE = r"""
import json
import sys


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
        send({"type": "pong", "obsDim": 128})
    elif cmd == "reset":
        send({"type": "state", "obs": [0.0] * 128, "info": {}})
    elif cmd == "step":
        send({"type": "step", "obs": [0.0] * 128, "components": {},
              "terminated": False, "truncated": False, "info": {}})
    elif cmd == "shutdown":
        send({"type": "bye"})
        sys.exit(0)
    else:
        send({"type": "ok"})
"""

# Tiny and CPU-only so the whole learn() below is seconds in CI.
_TINY_PPO_KWARGS = dict(
    n_steps=32,
    batch_size=32,
    n_epochs=2,
    policy_kwargs=dict(net_arch=[32, 32], activation_fn=th.nn.Tanh),
    verbose=0,
    device="cpu",
)


class _CapturingWriter(KVWriter):
    """SB3 logger writer that keeps every dumped key/value dict."""

    def __init__(self):
        self.dumps: list[dict] = []

    def write(self, key_values, key_excluded, step=0):
        self.dumps.append(dict(key_values))

    def close(self):
        pass


def test_anchor_kl_ppo_learn_smoke(monkeypatch, tmp_path):
    script = tmp_path / "fake_bridge.py"
    script.write_text(FAKE_BRIDGE)
    monkeypatch.setattr(env_mod, "NODE_BIN", sys.executable)

    def make_vec_env():
        return DummyVecEnv(
            [lambda: RobotArenaEnv(bridge_path=str(script))])

    # Frozen anchor checkpoint: a tiny, untrained PPO with the same
    # obs/action spaces and net shape as the live model (a different
    # seed, so live vs anchor KL is a real, nonzero-ish quantity).
    anchor_vec_env = make_vec_env()
    try:
        anchor_model = PPO("MlpPolicy", anchor_vec_env, seed=1,
                           **_TINY_PPO_KWARGS)
        anchor_model.save(str(tmp_path / "anchor_model"))
    finally:
        anchor_vec_env.close()
    anchor_path = tmp_path / "anchor_model.zip"
    assert anchor_path.is_file()

    vec_env = make_vec_env()
    try:
        writer = _CapturingWriter()
        model = AnchorKLPPO(
            "MlpPolicy", vec_env, seed=0,
            anchor_model_path=str(anchor_path),
            kl_anchor_coef=0.05,
            # No anneal: over a 64-step run annealing would take the
            # coefficient to 0, muting the very term under test.
            kl_anchor_anneal=False,
            **_TINY_PPO_KWARGS)
        # A custom logger survives learn()'s logger setup and lets the
        # test assert on what train() actually logged.
        model.set_logger(Logger(None, [writer]))

        model.learn(total_timesteps=64)

        assert model.num_timesteps == 64

        # train/anchor_kl is recorded only by the verbatim train() copy
        # in rl/anchor.py, so its presence in a logger dump proves that
        # copy (not PPO.train) executed during learn().
        anchor_kls = [dump["train/anchor_kl"] for dump in writer.dumps
                      if "train/anchor_kl" in dump]
        assert anchor_kls, (
            "train/anchor_kl never logged; dumps: "
            f"{[sorted(dump) for dump in writer.dumps]}")
        assert all(math.isfinite(float(kl)) and float(kl) >= -1e-8
                   for kl in anchor_kls)
        anchor_coefs = [dump["train/anchor_coef"] for dump in writer.dumps
                        if "train/anchor_coef" in dump]
        assert anchor_coefs == [pytest.approx(0.05)] * len(anchor_coefs)
        assert anchor_coefs  # the anchor term was actually in the loss

        model.save(str(tmp_path / "model"))
        saved = tmp_path / "model.zip"
        assert saved.is_file()

        loaded = PPO.load(str(saved))
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        action, _state = loaded.predict(obs, deterministic=True)
        action = np.asarray(action).reshape(-1)
        assert action.shape == (len(ACTION_DIMS),)
        assert np.all(action >= 0)
        assert np.all(action < np.asarray(ACTION_DIMS))
    finally:
        vec_env.close()
