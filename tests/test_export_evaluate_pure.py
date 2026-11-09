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

import json
import math

import numpy as np
import pytest

from rl.env import (
    EVENT_BASE, FOE_BASE, FOE_DIST, FRAME_DIM, FRAME_STACK, MATCH_BASE,
    N_STRATS, OBS_DIM, PAD_BASE, TRACK_BASE, TURRET_BASE, WALL_BASE,
    ZONE_BASE,
)
from rl.evaluate import BASE_ELO, K, elo_update
from rl.export_ts import (
    build_robot_source, normalize_obs, numpy_forward, sample_realistic_obs,
)


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


# -- build_robot_source: CLI strings cannot break out of the TS literals ----
#
# rl/export_ts main() writes <out>.ts for the operator to copy into
# src/robots/ and register, i.e. generated TypeScript that gets compiled
# and run. robot_id / robot_name / loadout used to be substituted raw,
# so a name like "O'Brien" terminated the '...' literal early and the
# rest landed in executable position. These tests pin the serialized
# substitution: whatever the operator passes, the meta values stay a
# single string literal and round-trip to the exact input.


def _robot_source(robot_id="rl-ppo", robot_name="RL PPO", loadout=None):
    weights = {
        "layers": [{"W": [[1.0]], "b": [0.0]}],
        "activation": "tanh",
        "norm": {"mean": [0.0], "var": [1.0],
                 "epsilon": 1e-8, "clip_obs": 10.0},
    }
    if loadout is None:
        loadout = {"overdrive": 2, "trigger": 2, "plating": 2}
    return build_robot_source(weights, robot_id, robot_name, loadout, 0,
                              "function buildObs() { return []; }")


def _ts_string_value(src, key):
    """Decode the single-quoted TS literal assigned to `key` in `src`.

    Scans escape-aware to the closing quote and asserts a comma follows
    immediately, i.e. the value is exactly one string literal with no
    tokens leaked after it.
    """
    marker = key + ": '"
    i = src.index(marker) + len(marker)
    out = []
    while src[i] != "'":
        if src[i] == "\\":
            nxt = src[i + 1]
            if nxt == "u":
                out.append(chr(int(src[i + 2:i + 6], 16)))
                i += 6
                continue
            out.append({"n": "\n", "r": "\r", "t": "\t",
                        "b": "\b", "f": "\f"}.get(nxt, nxt))
            i += 2
        else:
            out.append(src[i])
            i += 1
    assert src[i + 1] == ",", "literal must end the meta value: " + src
    return "".join(out)


def test_robot_source_apostrophe_and_backslash_name_stays_one_literal():
    name = "O'Brien \\ test"
    src = _robot_source(robot_name=name)
    assert _ts_string_value(src, "name") == name
    assert _ts_string_value(src, "id") == "rl-ppo"


def test_robot_source_hostile_id_cannot_inject_code():
    robot_id = "x'; globalThis.pwned = true; //"
    src = _robot_source(robot_id=robot_id)
    assert _ts_string_value(src, "id") == robot_id


@pytest.mark.parametrize("name", [
    "O'Brien \\ test",
    'say "hi"',
    "back\\slash",
    "line\nbreak",
    "café",
    "both ' and \" and \\",
])
def test_robot_source_name_round_trips(name):
    assert _ts_string_value(_robot_source(robot_name=name), "name") == name


def test_robot_source_loadout_is_the_parsed_object():
    line = next(l for l in _robot_source().splitlines()
                if l.startswith("export const loadout"))
    assert json.loads(line.split("=", 1)[1].removesuffix(";")) == {
        "overdrive": 2, "trigger": 2, "plating": 2}


def test_robot_source_defaults_keep_their_generated_shape():
    lines = _robot_source().splitlines()
    assert "    id: 'rl-ppo'," in lines
    assert "    name: 'RL PPO'," in lines
    assert ('export const loadout = '
            '{"overdrive": 2, "trigger": 2, "plating": 2};') in lines


# -- sample_realistic_obs ---------------------------------------------------
# These tests pin the export verifier sampler to the live 128-dim
# layout in bridge/OBS_SPEC.md:7-18 (the layout buildObs emits), so
# the next spec change breaks a test instead of hollowing out
# verify(). All dim numbers below are literal, taken straight from
# the spec, so they cross-check the shared rl.env bases rather than
# following them.

_LEAD_SIN = EVENT_BASE + 1  # 123
_LEAD_COS = EVENT_BASE + 3  # 125


def _frame(seed):
    return sample_realistic_obs(np.random.default_rng(seed))[:FRAME_DIM]


def _frames(n=200, seed0=10_000):
    return [_frame(seed0 + i) for i in range(n)]


def test_env_index_constants_match_obs_spec_bases():
    assert FOE_BASE == 17
    assert FOE_DIST == 20
    assert TRACK_BASE == 33
    assert WALL_BASE == 74
    assert ZONE_BASE == 78
    assert TURRET_BASE == 82
    assert PAD_BASE == 94
    assert MATCH_BASE == 118
    assert EVENT_BASE == 122


def test_sample_realistic_obs_shape_dtype_stack_and_onehot():
    out = sample_realistic_obs(np.random.default_rng(0))
    assert out.shape == (OBS_DIM,) == (516,)
    assert out.dtype == np.float64
    first = out[:FRAME_DIM]
    for k in range(1, FRAME_STACK):
        np.testing.assert_array_equal(
            out[k * FRAME_DIM:(k + 1) * FRAME_DIM], first)
    assert N_STRATS == 4
    np.testing.assert_array_equal(out[-4:], [1.0, 0.0, 0.0, 0.0])


def test_sample_realistic_obs_deterministic_for_seed():
    a = sample_realistic_obs(np.random.default_rng(7))
    b = sample_realistic_obs(np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    c = sample_realistic_obs(np.random.default_rng(8))
    assert not np.array_equal(a, c)


def test_sample_realistic_obs_ranges_over_many_samples():
    flag_dims = {10, 13, 33, 79, 81, 122, 124, 126, 127}
    flag_dims.update(FOE_BASE + 8 * s for s in range(2))
    flag_dims.update(38 + 6 * s for s in range(6))
    flag_dims.update(TURRET_BASE + 6 * s + k for s in range(2)
                     for k in (0, 3, 4))
    flag_dims.update(PAD_BASE + 6 * s for s in range(4))
    flag_dims.update(PAD_BASE + 6 * s + k for s in range(4)
                     for k in (3, 4, 5))
    sincos_dims = [2, 3, 4, 5, 15, 16, _LEAD_SIN, _LEAD_COS]
    sincos_dims += [FOE_BASE + 8 * s + k for s in range(2) for k in (4, 5)]
    unit_dims = [0, 1, 6, 7, 8, 9, 11, 12, 14]  # self [0,1] dims
    unit_dims += [FOE_BASE + 8 * s + k for s in range(2) for k in (3, 6, 7)]
    unit_dims += [TRACK_BASE + 3, TRACK_BASE + 4]
    unit_dims += [38 + 6 * s + k for s in range(6) for k in (3, 4, 5)]
    unit_dims += [WALL_BASE + k for k in range(4)]
    unit_dims += [ZONE_BASE, ZONE_BASE + 2]
    unit_dims += [MATCH_BASE, MATCH_BASE + 1, 121]
    signed_dims = [FOE_BASE + 8 * s + k for s in range(2) for k in (1, 2)]
    signed_dims += [TRACK_BASE + 1, TRACK_BASE + 2]
    signed_dims += [38 + 6 * s + k for s in range(6) for k in (1, 2)]
    signed_dims += [TURRET_BASE + 6 * s + k for s in range(2)
                    for k in (1, 2, 5)]
    signed_dims += [PAD_BASE + 6 * s + k for s in range(4) for k in (1, 2)]

    for o in _frames():
        assert np.all(np.isfinite(o))
        for d in flag_dims:
            assert o[d] == 0.0 or o[d] == 1.0, (d, o[d])
        for d in sincos_dims:
            assert -1.0 <= o[d] <= 1.0, (d, o[d])
        for d in unit_dims:
            assert 0.0 <= o[d] <= 1.0, (d, o[d])
        for d in signed_dims:
            assert -1.0 <= o[d] <= 1.0, (d, o[d])
        # Self sin/cos pairs: heading, tower, lastDamage bearing.
        for a, b in ((2, 3), (4, 5), (15, 16)):
            assert o[a] ** 2 + o[b] ** 2 == pytest.approx(1.0, abs=1e-9)
        # Walls derived from the sampled self position.
        assert o[WALL_BASE] == o[0]
        assert o[WALL_BASE + 1] == 1.0 - o[0]
        assert o[WALL_BASE + 2] == o[1]
        assert o[WALL_BASE + 3] == 1.0 - o[1]
        # Foes: bearing unit-circle when present, slot zero when absent.
        for s in range(2):
            b = FOE_BASE + 8 * s
            if o[b] == 1.0:
                assert o[b + 4] ** 2 + o[b + 5] ** 2 == pytest.approx(
                    1.0, abs=1e-9)
            else:
                np.testing.assert_array_equal(o[b:b + 8], np.zeros(8))
        # Track: the rest of the slot stays 0 when seenNow = 0.
        if o[TRACK_BASE] == 0.0:
            np.testing.assert_array_equal(
                o[TRACK_BASE:TRACK_BASE + 5], np.zeros(5))
        # Bullets: the slot stays 0 when present = 0.
        for s in range(6):
            b = 38 + 6 * s
            if o[b] == 0.0:
                np.testing.assert_array_equal(o[b:b + 6], np.zeros(6))
        # Turrets: owner flags never both 1; slot zero when inactive.
        for s in range(2):
            b = TURRET_BASE + 6 * s
            assert o[b + 3] + o[b + 4] <= 1.0
            if o[b] == 0.0:
                np.testing.assert_array_equal(o[b:b + 6], np.zeros(6))
        # Pads: kind one-hot iff active; slot zero when inactive.
        for s in range(4):
            b = PAD_BASE + 6 * s
            kind_sum = o[b + 3] + o[b + 4] + o[b + 5]
            if o[b] == 1.0:
                assert kind_sum == 1.0
            else:
                assert kind_sum == 0.0
                np.testing.assert_array_equal(o[b:b + 6], np.zeros(6))
        # Match: kills quantized to k/5, aliveFoes to k/3.
        for d in (MATCH_BASE, MATCH_BASE + 1):
            assert o[d] * 5 == pytest.approx(round(float(o[d] * 5)),
                                              abs=1e-9)
        assert any(o[MATCH_BASE + 2] == pytest.approx(v, abs=1e-12)
                   for v in (0.0, 1 / 3, 2 / 3, 1.0))
        # leadFeat: unit circle with a foe, exactly (0, 1) without one.
        if o[FOE_BASE] == 1.0 or o[FOE_BASE + 8] == 1.0:
            assert o[_LEAD_SIN] ** 2 + o[_LEAD_COS] ** 2 == pytest.approx(
                1.0, abs=1e-9)
        else:
            assert o[_LEAD_SIN] == 0.0 and o[_LEAD_COS] == 1.0


def test_sample_realistic_obs_events_actually_exercised():
    frames = _frames()
    assert any(o[EVENT_BASE] == 1.0 for o in frames)        # wasHit
    assert any(o[EVENT_BASE + 2] == 1.0 for o in frames)    # gotKill
    assert any(o[EVENT_BASE + 4] == 1.0 for o in frames)    # pickup
    assert any(o[EVENT_BASE + 5] == 1.0 for o in frames)    # turret-captured
    assert len({float(o[_LEAD_COS]) for o in frames}) > 1
    # Both leadFeat branches are hit across the draws.
    assert any(o[FOE_BASE] == 1.0 or o[FOE_BASE + 8] == 1.0
               for o in frames)
    assert any(o[FOE_BASE] == 0.0 and o[FOE_BASE + 8] == 0.0
               for o in frames)
