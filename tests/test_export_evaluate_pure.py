"""Characterisation tests for export/evaluate pure helpers.

These tests pin the *current* behaviour (as of ``origin/pluto-upgrade``)
of three previously untested pure helpers:

- ``rl.export_ts.normalize_obs``  — VecNormalize-style obs normalisation
- ``rl.export_ts.numpy_forward``  — NumPy MLP forward pass over an
  exported weights dict
- ``rl.evaluate.elo_update``      — two-player Elo rating update

They are characterisation tests, not a specification of ideal
behaviour: every expected value below is hand-computed independently
in the test body (``math.tanh`` / explicit formulas) from the current
function bodies, so a future change to those bodies shows up here as a
deliberate, reviewable diff.

Scope: hand-built tiny weights dicts and rating tuples only — no Node
bridge, no trained model, no file I/O, no network. Importing
``rl.export_ts`` / ``rl.evaluate`` pulls in torch/stable-baselines3 at
module level (expected; CI installs ``requirements.txt``), but nothing
here instantiates PPO or any environment.
"""

import math

import numpy as np
import pytest

from rl.evaluate import BASE_ELO, K, elo_update
from rl.export_ts import normalize_obs, numpy_forward


def _weights(layers, activation="tanh", norm=None):
    """Build a minimal exported-weights dict around hand-made layers."""
    weights = {
        "obs_dim": len(layers[0]["W"][0]),
        "action_dims": [len(layers[-1]["b"])],
        "activation": activation,
        "layers": layers,
    }
    if norm is not None:
        weights["norm"] = norm
    return weights


# -- normalize_obs -----------------------------------------------------------


def test_normalize_obs_none_returns_float64_values_unchanged():
    obs = np.array([1.5, -2.25, 0.0], dtype=np.float32)
    out = normalize_obs(obs, None)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [1.5, -2.25, 0.0], rtol=1e-12)


def test_normalize_obs_none_does_not_clip():
    obs = np.array([1000.0, -1000.0], dtype=np.float32)
    out = normalize_obs(obs, None)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [1000.0, -1000.0], rtol=1e-12)


def test_normalize_obs_exact_math():
    # Denominators: sqrt(3 + 1) = 2 and sqrt(15 + 1) = 4.
    norm = {"mean": [0.0, 10.0], "var": [3.0, 15.0],
            "epsilon": 1.0, "clip_obs": 10.0}
    out = normalize_obs(np.array([4.0, 18.0]), norm)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [2.0, 2.0], rtol=1e-12)
    # Obs exactly at the mean normalise to zero.
    out = normalize_obs(np.array([0.0, 10.0]), norm)
    np.testing.assert_allclose(out, [0.0, 0.0], rtol=1e-12, atol=1e-12)


def test_normalize_obs_clips_both_signs():
    norm = {"mean": [0.0, 0.0, 0.0], "var": [1.0, 1.0, 1.0],
            "epsilon": 0.0, "clip_obs": 2.0}
    out = normalize_obs(np.array([5.0, -5.0, 1.0]), norm)
    np.testing.assert_allclose(out, [2.0, -2.0, 1.0], rtol=1e-12)
    # Values exactly at the clip boundary pass through unchanged.
    out = normalize_obs(np.array([2.0, -2.0, 0.5]), norm)
    np.testing.assert_allclose(out, [2.0, -2.0, 0.5], rtol=1e-12)


def test_normalize_obs_epsilon_zero_var_finite_unclipped():
    # var=0: denominator is sqrt(epsilon) = sqrt(1e-8) = 1e-4.
    norm = {"mean": [0.0, 0.0], "var": [0.0, 0.0],
            "epsilon": 1e-8, "clip_obs": 10.0}
    out = normalize_obs(np.array([0.0002, -0.0003]), norm)
    expected = [0.0002 / math.sqrt(1e-8), -0.0003 / math.sqrt(1e-8)]
    np.testing.assert_allclose(out, expected, rtol=1e-12)
    np.testing.assert_allclose(out, [2.0, -3.0], rtol=1e-9)
    assert np.all(np.isfinite(out))


def test_normalize_obs_epsilon_zero_var_clips_when_huge():
    # (1 - 0) / sqrt(1e-8) = 10000, far beyond clip_obs.
    norm = {"mean": [0.0, 0.0], "var": [0.0, 0.0],
            "epsilon": 1e-8, "clip_obs": 10.0}
    out = normalize_obs(np.array([1.0, -1.0]), norm)
    np.testing.assert_allclose(out, [10.0, -10.0], rtol=1e-12)


# -- numpy_forward -----------------------------------------------------------


def test_numpy_forward_two_layer_tanh_zero_input():
    weights = _weights([
        {"W": [[1.0, 0.0], [0.0, 1.0]], "b": [0.0, 0.0]},
        {"W": [[1.0, 1.0]], "b": [0.5]},
    ])
    out = numpy_forward(weights, np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [0.5], rtol=1e-12, atol=1e-12)


def test_numpy_forward_two_layer_tanh_opposing_inputs_cancel():
    weights = _weights([
        {"W": [[1.0, 0.0], [0.0, 1.0]], "b": [0.0, 0.0]},
        {"W": [[1.0, 1.0]], "b": [0.5]},
    ])
    out = numpy_forward(weights, np.array([1.0, -1.0]))
    expected = math.tanh(1.0) + math.tanh(-1.0) + 0.5
    np.testing.assert_allclose(out, [expected], rtol=1e-12)
    np.testing.assert_allclose(out, [0.5], rtol=1e-12, atol=1e-12)


def test_numpy_forward_two_layer_tanh_general_weights():
    weights = _weights([
        {"W": [[1.0, 2.0], [0.0, 1.0]], "b": [0.5, -0.5]},
        {"W": [[2.0, -1.0]], "b": [0.25]},
    ])
    out = numpy_forward(weights, np.array([0.5, -1.0]))
    # Hidden pre-activation: [1*0.5 + 2*(-1) + 0.5, 0*0.5 + 1*(-1) - 0.5]
    #                      = [-1.0, -1.5], then tanh, then the linear head.
    expected = 2.0 * math.tanh(-1.0) - 1.0 * math.tanh(-1.5) + 0.25
    np.testing.assert_allclose(out, [expected], rtol=1e-12)


def test_numpy_forward_last_layer_has_no_activation():
    weights = _weights([
        {"W": [[1.0, 0.0], [0.0, 1.0]], "b": [0.0, 0.0]},
        {"W": [[1.0, 1.0]], "b": [-3.0]},
    ])
    # Final pre-activation is -3: returned as-is (relu would give 0,
    # tanh would give tanh(-3) ~= -0.995).
    out = numpy_forward(weights, np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [-3.0], rtol=1e-12, atol=1e-12)
    out = numpy_forward(weights, np.array([1.0, 0.0]))
    np.testing.assert_allclose(out, [math.tanh(1.0) - 3.0], rtol=1e-12)
    assert out[0] < 0.0


def test_numpy_forward_hidden_activation_tanh_vs_relu():
    layers = [
        {"W": [[1.0, 0.0], [0.0, 1.0]], "b": [0.0, 0.0]},
        {"W": [[1.0, 1.0]], "b": [0.0]},
    ]
    obs = np.array([-1.0, 2.0])  # hidden pre-activation [-1, 2]
    out_tanh = numpy_forward(_weights(layers, activation="tanh"), obs)
    np.testing.assert_allclose(
        out_tanh, [math.tanh(-1.0) + math.tanh(2.0)], rtol=1e-12)
    # Any activation other than "tanh" takes the relu branch: the
    # negative hidden unit is zeroed, the positive one passes through.
    out_relu = numpy_forward(_weights(layers, activation="relu"), obs)
    np.testing.assert_allclose(out_relu, [0.0 + 2.0], rtol=1e-12)
    out_other = numpy_forward(_weights(layers, activation="anything"), obs)
    np.testing.assert_allclose(out_other, out_relu, rtol=1e-12)
    assert out_tanh[0] != pytest.approx(out_relu[0])


def test_numpy_forward_single_layer_no_activation():
    weights = _weights(
        [{"W": [[1.0, -2.0], [0.5, 1.0]], "b": [-1.0, 2.0]}],
        activation="tanh",
    )
    out = numpy_forward(weights, np.array([1.0, 1.0]))
    # Exactly W @ x + b, negatives included: [1-2-1, 0.5+1+2].
    np.testing.assert_allclose(out, [-2.0, 3.5], rtol=1e-12)


def test_numpy_forward_with_norm_normalises_first():
    norm = {"mean": [1.0, 2.0], "var": [4.0, 9.0],
            "epsilon": 0.0, "clip_obs": 10.0}
    weights = _weights(
        [{"W": [[1.0, 0.0], [0.0, 1.0]], "b": [0.0, 0.0]}], norm=norm)
    out = numpy_forward(weights, np.array([3.0, 8.0]))
    hand_normalised = [(3.0 - 1.0) / math.sqrt(4.0),
                       (8.0 - 2.0) / math.sqrt(9.0)]
    np.testing.assert_allclose(out, hand_normalised, rtol=1e-12)
    np.testing.assert_allclose(out, [1.0, 2.0], rtol=1e-12)
    # Normalisation clips before the linear layer: (101-1)/2 = 50 -> 10.
    out = numpy_forward(weights, np.array([101.0, 2.0]))
    np.testing.assert_allclose(out, [10.0, 0.0], rtol=1e-12, atol=1e-12)


def test_numpy_forward_norm_applied_before_hidden_layers():
    norm = {"mean": [1.0, 2.0], "var": [4.0, 9.0],
            "epsilon": 0.0, "clip_obs": 10.0}
    weights = _weights([
        {"W": [[1.0, 0.0], [0.0, 1.0]], "b": [0.0, 0.0]},
        {"W": [[1.0, 1.0]], "b": [0.5]},
    ], norm=norm)
    # Raw obs equals the norm mean, so the net sees [0, 0] and the
    # logit is just the head bias (unnormalised it would be
    # tanh(1) + tanh(2) + 0.5).
    out = numpy_forward(weights, np.array([1.0, 2.0]))
    np.testing.assert_allclose(out, [0.5], rtol=1e-12, atol=1e-12)


def test_numpy_forward_without_norm_uses_obs_as_is():
    weights = _weights(
        [{"W": [[1.0, 0.0], [0.0, 1.0]], "b": [0.0, 0.0]}])
    assert "norm" not in weights
    out = numpy_forward(weights, np.array([3.0, 8.0]))
    np.testing.assert_allclose(out, [3.0, 8.0], rtol=1e-12)
    # No norm also means no clipping of large raw obs.
    out = numpy_forward(weights, np.array([101.0, 2.0]))
    np.testing.assert_allclose(out, [101.0, 2.0], rtol=1e-12)


# -- elo_update --------------------------------------------------------------


def test_elo_constants_pinned():
    assert BASE_ELO == 1200.0
    assert K == 32.0


def test_elo_equal_ratings_win():
    assert elo_update(1200.0, 1200.0, 1.0) == (1216.0, 1184.0)


def test_elo_equal_ratings_loss():
    assert elo_update(1200.0, 1200.0, 0.0) == (1184.0, 1216.0)


def test_elo_equal_ratings_draw():
    assert elo_update(1200.0, 1200.0, 0.5) == (1200.0, 1200.0)


@pytest.mark.parametrize("ra,rb,score_a", [
    (1200.0, 1200.0, 1.0),
    (1200.0, 1200.0, 0.0),
    (1200.0, 1200.0, 0.5),
    (1400.0, 1000.0, 1.0),
    (1000.0, 1400.0, 0.0),
    (1350.0, 1250.0, 0.5),
    (2000.0, 800.0, 0.5),
])
def test_elo_zero_sum(ra, rb, score_a):
    ra2, rb2 = elo_update(ra, rb, score_a)
    assert ra2 + rb2 == pytest.approx(ra + rb)


def test_elo_underdog_win_gains_more_than_16():
    ra, rb = 1000.0, 1400.0
    ra2, rb2 = elo_update(ra, rb, 1.0)
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))  # = 1/11
    assert ra2 == pytest.approx(ra + K * (1.0 - ea))
    assert rb2 == pytest.approx(rb + K * (0.0 - (1.0 - ea)))
    assert ra2 - ra > 16.0


def test_elo_favourite_win_gains_fewer_than_16():
    ra, rb = 1400.0, 1000.0
    ra2, rb2 = elo_update(ra, rb, 1.0)
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))  # = 10/11
    assert ra2 == pytest.approx(ra + K * (1.0 - ea))
    assert rb2 == pytest.approx(rb + K * (0.0 - (1.0 - ea)))
    assert 0.0 < ra2 - ra < 16.0


@pytest.mark.parametrize("ra,rb,score_a", [
    (1200.0, 1400.0, 1.0),
    (1000.0, 1600.0, 0.0),
    (1350.0, 1250.0, 0.5),
    (1200.0, 1200.0, 1.0),
])
def test_elo_symmetry(ra, rb, score_a):
    ra2, rb2 = elo_update(ra, rb, score_a)
    rb2_swapped, ra2_swapped = elo_update(rb, ra, 1.0 - score_a)
    assert ra2 == pytest.approx(ra2_swapped, abs=1e-9)
    assert rb2 == pytest.approx(rb2_swapped, abs=1e-9)


def test_elo_draw_unequal_ratings_move_toward_each_other():
    ra, rb = 1000.0, 1400.0  # ra is the lower-rated player
    ra2, rb2 = elo_update(ra, rb, 0.5)
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))  # = 1/11
    assert ra2 == pytest.approx(ra + K * (0.5 - ea))
    assert rb2 == pytest.approx(rb + K * (0.5 - (1.0 - ea)))
    assert ra2 > ra  # lower-rated gains from a draw
    assert rb2 < rb  # higher-rated loses from a draw
    assert rb2 - ra2 < rb - ra  # ratings move toward each other
