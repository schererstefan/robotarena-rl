"""Pure, bridge-free logic tests (no Node bridge, no torch/SB3).

Scope: exercises only pure Python logic that runs on a stock hosted
runner — rl.reward.compute_reward and rl.env.action_to_intent plus the
OBS_DIM shape invariant. Nothing here instantiates RobotArenaEnv (that
spawns a Node bridge subprocess), and nothing imports torch or
stable_baselines3 (pulled in only by rl.train / rl.evaluate / rl.anchor,
which are deliberately not imported).

Bridge-dependent checks (scripts/smoke.sh, intent_check, aim_check) are
deliberately excluded: they need the sibling robotarena checkout and a
built Node bridge, unavailable on stock hosted runners.
"""

import math

import numpy as np
import pytest

from rl.env import FRAME_DIM, FRAME_STACK, N_STRATS, OBS_DIM, action_to_intent
from rl.reward import DEFAULT_WEIGHTS, compute_reward


def weights():
    return dict(DEFAULT_WEIGHTS)


def test_empty_components_baseline():
    reward, info = compute_reward({}, weights(), terminated=False, won=False, lost=False)
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"])
    assert info["reward/outcome"] == "none"


def test_damage_dealt():
    reward, _ = compute_reward({"dealt": 100}, weights(), terminated=False, won=False, lost=False)
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"] + 1.0)


def test_damage_taken():
    reward, _ = compute_reward({"taken": 100}, weights(), terminated=False, won=False, lost=False)
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"] - 1.0)


def test_damage_combined():
    reward, _ = compute_reward(
        {"dealt": 50, "taken": 25}, weights(), terminated=False, won=False, lost=False
    )
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"] + 0.5 - 0.25)


def test_kill_bonus():
    reward, _ = compute_reward({"killed": True}, weights(), terminated=False, won=False, lost=False)
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"] + 3.0)


def test_death_penalty():
    reward, _ = compute_reward({"died": True}, weights(), terminated=False, won=False, lost=False)
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"] - 3.0)


def test_pickup_bonus():
    reward, _ = compute_reward({"pickup": True}, weights(), terminated=False, won=False, lost=False)
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"] + 0.3)


def test_turret_bonus():
    reward, _ = compute_reward({"turret": True}, weights(), terminated=False, won=False, lost=False)
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"] + 1.0)


def test_terminal_win():
    reward, info = compute_reward({}, weights(), terminated=True, won=True, lost=False)
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"] + 15.0)
    assert info["reward/outcome"] == "win"


def test_terminal_loss():
    reward, info = compute_reward({}, weights(), terminated=True, won=False, lost=True)
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"] - 15.0)
    assert info["reward/outcome"] == "loss"


def test_terminal_draw():
    reward, info = compute_reward({}, weights(), terminated=True, won=False, lost=False)
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"] + 0.0)
    assert info["reward/outcome"] == "draw"


def test_nonterminal_won_flag_no_bonus():
    reward, info = compute_reward({}, weights(), terminated=False, won=True, lost=False)
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"])
    assert info["reward/outcome"] == "none"


def test_aim_full_bonus():
    reward, _ = compute_reward(
        {"leadErr": 0.0}, weights(), terminated=False, won=False, lost=False
    )
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"] + 0.01)


def test_aim_zero_at_pi():
    reward, _ = compute_reward(
        {"leadErr": math.pi}, weights(), terminated=False, won=False, lost=False
    )
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"])


def test_aim_fallback_to_aim_err():
    reward, _ = compute_reward(
        {"aimErr": 0.0}, weights(), terminated=False, won=False, lost=False
    )
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"] + 0.01)


def test_no_aim_bonus_by_default():
    reward, _ = compute_reward({}, weights(), terminated=False, won=False, lost=False)
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"])


def test_fire_lead_bonus():
    reward, _ = compute_reward(
        {"leadErr": 0.0}, weights(), terminated=False, won=False, lost=False, firing=True
    )
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"] + 0.01 + 0.05)


def test_fire_lead_above_tolerance():
    lead_err = 0.10
    reward, _ = compute_reward(
        {"leadErr": lead_err}, weights(), terminated=False, won=False, lost=False, firing=True
    )
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"] + 0.01 * (1 - lead_err / math.pi))


def test_no_fire_lead_when_not_firing():
    reward, _ = compute_reward(
        {"leadErr": 0.0}, weights(), terminated=False, won=False, lost=False, firing=False
    )
    assert reward == pytest.approx(DEFAULT_WEIGHTS["tick"] + 0.01)


def test_action_all_min():
    intent = action_to_intent(np.array([0, 0, 0, 0, 0, 0, 0]))
    assert intent == {
        "throttle": -1.0,
        "turn": -1.0,
        "towerTurn": -1.0,
        "fire": False,
        "charge": False,
        "dash": False,
        "emp": False,
    }


def test_action_all_mid():
    intent = action_to_intent(np.array([2, 2, 2, 0, 0, 0, 0]))
    assert intent == {
        "throttle": 0.0,
        "turn": 0.0,
        "towerTurn": 0.0,
        "fire": False,
        "charge": False,
        "dash": False,
        "emp": False,
    }


def test_action_all_max():
    intent = action_to_intent(np.array([4, 4, 4, 1, 1, 1, 1]))
    assert intent == {
        "throttle": 1.0,
        "turn": 1.0,
        "towerTurn": 1.0,
        "fire": True,
        "charge": True,
        "dash": True,
        "emp": True,
    }


def test_action_mixed():
    intent = action_to_intent(np.array([0, 4, 2, 1, 0, 1, 0]))
    assert intent == {
        "throttle": -1.0,
        "turn": 1.0,
        "towerTurn": 0.0,
        "fire": True,
        "charge": False,
        "dash": True,
        "emp": False,
    }


def test_action_half_steps():
    intent = action_to_intent(np.array([1, 3, 1, 0, 0, 0, 0]))
    assert intent["throttle"] == pytest.approx(-0.5)
    assert intent["turn"] == pytest.approx(0.5)
    assert intent["towerTurn"] == pytest.approx(-0.5)


def test_obs_dim_invariant():
    assert FRAME_DIM == 128
    assert FRAME_STACK == 4
    assert N_STRATS == 4
    assert OBS_DIM == FRAME_STACK * FRAME_DIM + N_STRATS == 516
