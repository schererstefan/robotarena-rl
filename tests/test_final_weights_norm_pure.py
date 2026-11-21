"""Pure final-weights norm tests (no training run).

Scope: end-of-run ``final_weights.json`` contract. Training always
wraps the env in VecNormalize (``norm_obs=True``), and the module
contract (``rl/export_ts.py`` docstring) says consumers MUST normalize
raw obs with the exported ``norm`` dict before the forward pass.
The league-snapshot export already passes the live stats; these tests
pin that the final-weights export path — ``export_weights`` /
``extract_weights`` fed by ``norm_from_vecnormalize`` — carries a
``norm`` dict whose ``mean``/``var`` lists are length ``OBS_DIM``
(516, ``rl/env.py``).

Hermetic by construction: a hand-built dummy model (real
``torch.nn.Linear`` layers, no PPO, no environment) and a mocked
VecNormalize stats object. No training, no Node bridge, no file I/O
beyond ``tmp_path``.
"""

import json
from types import SimpleNamespace

import numpy as np
import torch as th

from rl.env import ACTION_DIMS, OBS_DIM
from rl.export_ts import extract_weights, export_weights, norm_from_vecnormalize


class _FakeMLPExtractor:
    def __init__(self, policy_net):
        self.policy_net = policy_net


class _FakePolicy:
    def __init__(self, policy_net, action_net):
        self.mlp_extractor = _FakeMLPExtractor(policy_net)
        self.action_net = action_net


class _FakeModel:
    def __init__(self, policy):
        self.policy = policy


def _dummy_model(hidden=8):
    """Tiny stand-in for a PPO model exposing only what extract_weights reads."""
    policy_net = [
        th.nn.Linear(OBS_DIM, hidden), th.nn.Tanh(),
        th.nn.Linear(hidden, hidden), th.nn.Tanh(),
    ]
    action_net = th.nn.Linear(hidden, sum(ACTION_DIMS))
    return _FakeModel(_FakePolicy(policy_net, action_net))


def _fake_vec_normalize(mean=None, var=None, epsilon=1e-8, clip_obs=10.0,
                        shape=None):
    """Minimal VecNormalize stand-in: obs_rms mean/var plus epsilon/clip."""
    if mean is None:
        mean = np.linspace(-1.0, 1.0, OBS_DIM, dtype=np.float32)
    if var is None:
        var = np.full(OBS_DIM, 4.0, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    var = np.asarray(var, dtype=np.float32)
    if shape is not None:
        mean = mean.reshape(shape)
        var = var.reshape(shape)
    return SimpleNamespace(
        obs_rms=SimpleNamespace(mean=mean, var=var),
        epsilon=epsilon,
        clip_obs=clip_obs,
    )


def test_obs_dim_is_pinned_to_516():
    assert OBS_DIM == 516


def test_norm_from_vecnormalize_mean_var_length_is_obs_dim():
    norm = norm_from_vecnormalize(_fake_vec_normalize())
    assert norm is not None
    assert set(norm) == {"mean", "var", "epsilon", "clip_obs"}
    assert len(norm["mean"]) == OBS_DIM == 516
    assert len(norm["var"]) == OBS_DIM == 516
    assert norm["epsilon"] == 1e-8
    assert norm["clip_obs"] == 10.0


def test_norm_from_vecnormalize_flattens_batched_stats():
    # Real RunningMeanStd stores shapes like (1, OBS_DIM); the export
    # must still flatten to length OBS_DIM.
    norm = norm_from_vecnormalize(
        _fake_vec_normalize(shape=(1, OBS_DIM)))
    assert len(norm["mean"]) == OBS_DIM
    assert len(norm["var"]) == OBS_DIM


def test_norm_from_vecnormalize_unwraps_nested_venv():
    inner = _fake_vec_normalize()
    outer = SimpleNamespace(venv=inner)
    norm = norm_from_vecnormalize(outer)
    assert norm is not None
    assert len(norm["mean"]) == OBS_DIM
    assert len(norm["var"]) == OBS_DIM


def test_norm_from_vecnormalize_none_without_obs_rms():
    assert norm_from_vecnormalize(object()) is None


def test_extract_weights_with_norm_carries_obs_dim_mean_var():
    norm = norm_from_vecnormalize(_fake_vec_normalize())
    weights = extract_weights(_dummy_model(), norm=norm)
    assert weights["obs_dim"] == OBS_DIM
    assert "norm" in weights
    assert len(weights["norm"]["mean"]) == OBS_DIM
    assert len(weights["norm"]["var"]) == OBS_DIM
    # Layers: [OBS_DIM -> hidden, hidden -> hidden, hidden -> actions].
    assert len(weights["layers"][0]["W"][0]) == OBS_DIM
    assert len(weights["layers"][-1]["W"]) == sum(ACTION_DIMS)


def test_extract_weights_without_norm_omits_key():
    # Pins why callers must pass norm: without it the JSON has no "norm"
    # key and consumers normalize with nothing.
    weights = extract_weights(_dummy_model())
    assert "norm" not in weights


def test_final_weights_export_file_carries_norm(tmp_path):
    path = tmp_path / "final_weights.json"
    norm = norm_from_vecnormalize(_fake_vec_normalize())
    returned = export_weights(_dummy_model(), str(path), norm=norm)
    assert len(returned["norm"]["mean"]) == OBS_DIM
    with open(path) as f:
        saved = json.load(f)
    assert saved["obs_dim"] == OBS_DIM == 516
    assert "norm" in saved
    assert len(saved["norm"]["mean"]) == OBS_DIM
    assert len(saved["norm"]["var"]) == OBS_DIM
    assert saved["norm"]["epsilon"] == 1e-8
    assert saved["norm"]["clip_obs"] == 10.0
