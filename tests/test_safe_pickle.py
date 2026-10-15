"""Restricted-unpickler tests for rl.safe_pickle.

These need SB3/gymnasium (to build a genuine VecNormalize object), which
CI installs via requirements.txt, same as for any SB3-importing module.
No network, no Node bridge.

Benign case: a real VecNormalize (built on a DummyVecEnv, with a seeded
action space so the pickle carries the numpy RNG globals a real
``runs/*/vecnormalize.pkl`` carries) round-trips through the restricted
loader with identical stats. Malicious cases: a classic ``__reduce__``
payload and a non-allowlisted class must raise before anything executes.
"""

import os
import pickle

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from rl.safe_pickle import load_vecnormalize_stats

PWNED = "/tmp/robotarena-rl-safe-pickle-pwned"


class _TinyEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([5, 5, 5, 2])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


def _write_benign_vecnormalize(path):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([lambda: _TinyEnv()])
    vn = VecNormalize(venv, training=True, norm_obs=True,
                      norm_reward=True, clip_obs=10.0)
    vn.obs_rms.mean = np.array([0.1, 0.2, 0.3, 0.4])
    vn.obs_rms.var = np.array([1.0, 2.0, 3.0, 4.0])
    vn.obs_rms.count = 123.0
    # Seeding embeds a numpy Generator in the pickle, exactly as in a
    # real saved vecnormalize.pkl -- the loader's RNG allowlist entries
    # exist for this, so exercise them.
    vn.action_space.seed(123)
    with open(path, "wb") as f:
        pickle.dump(vn, f)
    return vn


def test_benign_vecnormalize_round_trip(tmp_path):
    path = tmp_path / "vecnormalize.pkl"
    vn = _write_benign_vecnormalize(path)
    stats = load_vecnormalize_stats(path)
    np.testing.assert_allclose(stats["mean"], vn.obs_rms.mean)
    np.testing.assert_allclose(stats["var"], vn.obs_rms.var)
    assert stats["epsilon"] == pytest.approx(vn.epsilon)
    assert stats["clip_obs"] == pytest.approx(vn.clip_obs)


def test_malicious_reduce_payload_rejected(tmp_path):
    if os.path.exists(PWNED):
        os.unlink(PWNED)

    class Exploit:
        def __reduce__(self):
            return (os.system, (f"touch {PWNED}",))

    path = tmp_path / "evil.pkl"
    with open(path, "wb") as f:
        pickle.dump(Exploit(), f)
    with pytest.raises(pickle.UnpicklingError):
        load_vecnormalize_stats(path)
    assert not os.path.exists(PWNED)


def test_malicious_global_opcode_rejected(tmp_path):
    # Hand-built pickle invoking os.system via a GLOBAL opcode, the
    # canonical pickle-code-execution shape.
    if os.path.exists(PWNED):
        os.unlink(PWNED)
    payload = (
        b"cos\nsystem\n"
        b"(S'touch " + PWNED.encode() + b"'\n"
        b"tR."
    )
    path = tmp_path / "evil-global.pkl"
    path.write_bytes(payload)
    with pytest.raises(pickle.UnpicklingError):
        load_vecnormalize_stats(path)
    assert not os.path.exists(PWNED)


def test_non_allowlisted_class_rejected(tmp_path):
    # A perfectly ordinary pickle of a non-allowlisted class: rejected
    # by find_class, not by post-load validation.
    path = tmp_path / "dict-subclass.pkl"
    with open(path, "wb") as f:
        pickle.dump({"mean": [0.0]}, f)
    # dict itself is a pickle builtin (no GLOBAL), so this loads and is
    # then rejected by validation -- assert the loader refuses it.
    with pytest.raises(pickle.UnpicklingError):
        load_vecnormalize_stats(path)

    import collections
    path2 = tmp_path / "ordered-dict.pkl"
    with open(path2, "wb") as f:
        pickle.dump(collections.OrderedDict([("a", 1)]), f)
    with pytest.raises(pickle.UnpicklingError):
        load_vecnormalize_stats(path2)
