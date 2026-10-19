"""Pure League + StrategyBandit tests (no Node bridge, no torch/SB3).

Scope: the two bridge-free seams that shape who the learner plays and
which strategy label it is conditioned on — ``rl.league.League``
(snapshot discovery, PFSP-weighted scripted sampling, two-stage
self-play draw) and ``rl.bandit.StrategyBandit`` (UCB1 selection,
in-memory updates, ``seq``-guarded JSON persistence).

Everything here is deterministic: bandit ``select`` is a pure function
of recorded counts, and every stochastic league check draws from a
fresh ``np.random.default_rng(<fixed int>)`` (Generator API only — no
``np.random.seed``, no wall-clock assertions, no network, no
subprocess). Nothing imports ``rl.train``.
"""

import json
import os
from collections import Counter

import numpy as np

from rl.bandit import N_STRATEGIES, STRATEGIES, StrategyBandit
from rl.league import League


def _bandit(tmp_path) -> StrategyBandit:
    path = tmp_path / "bandit.json"
    assert not path.exists()
    return StrategyBandit(str(path))


def _fill(bandit: StrategyBandit, opponent: str, arm: int, wins: int, losses: int) -> None:
    for _ in range(wins):
        bandit.update(opponent, arm, True)
    for _ in range(losses):
        bandit.update(opponent, arm, False)


def _make_snapshots(snap_dir, names) -> list[str]:
    paths = []
    for i, name in enumerate(names):
        p = snap_dir / name
        p.write_text("{}")
        t = 1_700_000_000 + i * 10
        os.utime(p, (t, t))
        paths.append(str(p))
    return paths


# -- StrategyBandit --------------------------------------------------------


def test_cold_start_selects_lowest_index(tmp_path):
    assert STRATEGIES == ["rush", "kite", "greedy", "control"]
    assert N_STRATEGIES == 4
    bandit = _bandit(tmp_path)
    assert bandit.select("oppX") == 0


def test_untried_arms_tried_first_with_exclude(tmp_path):
    bandit = _bandit(tmp_path)
    assert bandit.select("o", exclude=(0,)) == 1
    assert bandit.select("o", exclude=(0, 1, 2)) == 3


def test_update_increments_counts(tmp_path):
    bandit = _bandit(tmp_path)
    bandit.update("o", 2, True)
    bandit.update("o", 2, True)
    bandit.update("o", 2, False)
    assert bandit.arm_counts() == {"rush": 0, "kite": 0, "greedy": 3, "control": 0}
    assert bandit.best("o") == 2
    assert sum(bandit.arm_counts().values()) == 3


def test_ucb_prefers_underpulled_arm(tmp_path):
    bandit = _bandit(tmp_path)
    _fill(bandit, "o", 0, wins=90, losses=10)
    for arm in (1, 2, 3):
        _fill(bandit, "o", arm, wins=0, losses=1)
    # arm0: 0.9 + sqrt(ln(103)/100) ~= 1.12; arms 1..3: sqrt(ln(103)) ~= 2.15.
    assert bandit.select("o") == 1


def test_ucb_prefers_higher_winrate_when_pulls_equal(tmp_path):
    bandit = _bandit(tmp_path)
    for arm, wins in enumerate([2, 18, 5, 5]):
        _fill(bandit, "o", arm, wins=wins, losses=20 - wins)
    assert bandit.select("o") == 1


def test_overall_best_aggregates_opponents(tmp_path):
    bandit = _bandit(tmp_path)
    _fill(bandit, "a", 1, wins=3, losses=0)
    _fill(bandit, "b", 2, wins=5, losses=0)
    assert bandit.overall_best() == 2


def test_persistence_roundtrip_seq_bump(tmp_path):
    path = str(tmp_path / "bandit.json")
    bandit = StrategyBandit(path)
    _fill(bandit, "o", 2, wins=4, losses=1)
    _fill(bandit, "o", 0, wins=1, losses=1)
    bandit.save()
    assert os.path.exists(path)
    reloaded = StrategyBandit(path)
    assert reloaded.arm_counts() == bandit.arm_counts()
    assert reloaded.select("o") == bandit.select("o")
    assert reloaded.seq >= 0


# -- League ------------------------------------------------------------------


def test_snapshot_eviction_oldest_first(tmp_path):
    snap_dir = tmp_path / "snaps"
    league = League(str(snap_dir), max_snapshots=3)
    paths = _make_snapshots(snap_dir, [f"s{i}.json" for i in range(5)])
    pfsp = snap_dir / "pfsp.json"
    pfsp.write_text("{}")
    newest = 1_700_000_000 + 100
    os.utime(pfsp, (newest, newest))
    assert league.snapshots() == paths[2:]
    assert str(pfsp) not in league.snapshots()


def test_no_pfsp_falls_back_to_uniform(tmp_path):
    snap_dir = tmp_path / "snaps"
    league = League(str(snap_dir), scripted=("a", "b"), p_scripted=1.0)
    assert not (snap_dir / "pfsp.json").exists()
    rng = np.random.default_rng(7)
    bots = [league.sample(rng) for _ in range(200)]
    assert all(set(s) == {"bot"} and s["bot"] in {"a", "b"} for s in bots)
    assert {s["bot"] for s in bots} == {"a", "b"}


def test_pfsp_weights_hardest_over_extremes(tmp_path):
    snap_dir = tmp_path / "snaps"
    league = League(str(snap_dir), scripted=("easy", "mid", "hard"),
                    pfsp_exponent=1.5, uniform_mix=0.2, p_scripted=1.0)
    (snap_dir / "pfsp.json").write_text(
        json.dumps({"win_rate": {"easy": 1.0, "mid": 0.5, "hard": 0.0}}))
    rng = np.random.default_rng(123)
    counts = Counter(league.sample(rng)["bot"] for _ in range(3000))
    assert counts["hard"] > counts["mid"] > counts["easy"] > 0
    # Missing keys default to wr=0.5 and must never raise.
    (snap_dir / "pfsp.json").write_text(json.dumps({"win_rate": {"easy": 0.9}}))
    rng = np.random.default_rng(9)
    bots = [league.sample(rng)["bot"] for _ in range(200)]
    assert all(b in {"easy", "mid", "hard"} for b in bots)


def test_sample_deterministic_under_seed(tmp_path):
    snap_dir = tmp_path / "snaps"
    league = League(str(snap_dir), p_scripted=0.3, p_latest=0.4)
    _make_snapshots(snap_dir, ["s0.json", "s1.json", "s2.json"])
    rng_a = np.random.default_rng(2024)
    seq_a = [league.sample(rng_a) for _ in range(50)]
    rng_b = np.random.default_rng(2024)
    seq_b = [league.sample(rng_b) for _ in range(50)]
    assert seq_a == seq_b
    rng_c = np.random.default_rng(2025)
    [league.sample(rng_c) for _ in range(50)]  # allowed to differ; just run it
    for s in seq_a:
        if "policy" in s:
            assert set(s) == {"policy"}
            pol = s["policy"]
            assert pol["strategy"] in range(N_STRATEGIES)
            assert os.path.dirname(pol["weights"]) == str(snap_dir)
        else:
            assert set(s) == {"bot"}
            assert isinstance(s["bot"], str)


def test_single_snapshot_always_latest(tmp_path):
    snap_dir = tmp_path / "snaps"
    league = League(str(snap_dir), p_scripted=0.0, p_latest=0.0)
    (only,) = _make_snapshots(snap_dir, ["only.json"])
    rng = np.random.default_rng(3)
    for _ in range(20):
        s = league.sample(rng)
        assert set(s) == {"policy"}
        assert s["policy"]["weights"] == only
        assert s["policy"]["deterministic"] is False
        assert isinstance(s["policy"]["strategy"], int)
        assert s["policy"]["strategy"] in range(N_STRATEGIES)


def test_describe_last_and_empty(tmp_path):
    snap_dir = tmp_path / "snaps"
    league = League(str(snap_dir))
    assert league.describe_last() == {"n_snapshots": 0, "latest": None}
    _make_snapshots(snap_dir, ["snap_a.json"])
    desc = league.describe_last()
    assert desc["n_snapshots"] == 1
    assert desc["latest"] == "snap_a.json"
