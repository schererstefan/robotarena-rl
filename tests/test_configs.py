"""Guard tests for every training config in configs/.

WHY: ``rl/train.py`` dereferences a small set of config keys
unconditionally, so a config missing one of them (or holding the
wrong shape) crashes with ``KeyError``/``TypeError`` at launch --
typically discovered only when someone starts that training run.
These tests load every shipped config and assert the contract
``rl/train.py`` actually relies on, so a broken config fails in CI
instead of at launch time.

The hard-required keys are exactly the unconditional dereferences in
``rl/train.py`` (verified against ``main()`` and ``make_env_fn()``):

- ``cfg["ppo"]`` in ``main()`` (then read via ``ppo_cfg.get(...)``),
- ``cfg["env"]`` in ``main()`` and ``make_env_fn()``
  (``cfg["env"].get(...)``),
- ``cfg["training"]["total_timesteps"]`` in ``main()`` -- the only
  hard-required subkey -- plus ``cfg["training"].get(...)`` for
  ``eval_bots`` / ``eval_episodes`` / ``eval_freq`` /
  ``checkpoint_freq`` / ``snapshot_freq``,
- ``cfg["league"]`` via ``cfg["league"].get(...)`` in
  ``make_env_fn()`` (``scripted``, ``p_scripted``, ``p_latest``,
  ``max_snapshots``, ``pfsp_exponent``, ``uniform_mix``) and in
  ``main()`` (``max_snapshots``, ``min_roster_rate``).

Every other key ``rl/train.py`` reads goes through ``.get(...)`` with
a default, so those keys are OPTIONAL: this module checks them only
when present (sane ranges/shapes), and their absence never fails.

Bot names in ``league.scripted`` / ``training.eval_bots`` are checked
for shape only (non-empty strings), never against a fixed roster:
configs legitimately name bots such as ``ghost`` and ``hunter-hc2``
that are not members of any single hard-coded roster.

Scope: pure and bridge-free, mirroring tests/test_pure.py -- imports
only stdlib + yaml + pytest. Nothing imports ``rl.train`` (which
pulls torch/stable_baselines3), ``rl.env``, or spawns anything, so
this runs on a stock hosted runner. Configs are discovered by
globbing ``configs/*.yaml`` (plus ``*.yml``) at import time, so new
configs are covered automatically.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATHS = sorted(
    p
    for p in (*CONFIG_DIR.glob("*.yaml"), *CONFIG_DIR.glob("*.yml"))
    if p.is_file()
)
CONFIG_IDS = [p.name for p in CONFIG_PATHS]

# Top-level sections rl/train.py dereferences unconditionally.
REQUIRED_SECTIONS = ("ppo", "env", "training", "league")

# Optional top-level sections (read via cfg.get(..., {}) or {}).
OPTIONAL_SECTIONS = ("anchor", "rollback")

# Optional keys that must lie in [0, 1] when present.
PROBABILITY_KEYS = (
    ("league", "p_scripted"),
    ("league", "p_latest"),
    ("league", "uniform_mix"),
    ("league", "min_roster_rate"),
    ("ppo", "gamma"),
    ("ppo", "gae_lambda"),
    ("ppo", "clip_range"),
    ("rollback", "max_drop"),
)

# Optional count keys that must be positive ints when present.
POSITIVE_INT_KEYS = (
    ("env", "n_envs"),
    ("ppo", "n_steps"),
    ("ppo", "batch_size"),
    ("ppo", "n_epochs"),
    ("league", "max_snapshots"),
    ("training", "checkpoint_freq"),
    ("training", "snapshot_freq"),
    ("training", "eval_freq"),
    ("training", "eval_episodes"),
)

# Optional keys that must be positive real numbers when present.
POSITIVE_NUMBER_KEYS = (
    ("league", "pfsp_exponent"),
    ("ppo", "learning_rate"),
)

# Optional keys that must be non-negative real numbers when present.
NONNEGATIVE_NUMBER_KEYS = (("ppo", "ent_coef"),)

# Optional bot-name lists: non-empty list of non-empty strings.
# Duplicates are meaningful (uniform sampling over entries weights
# the mix), so uniqueness is deliberately not asserted.
BOT_LIST_KEYS = (
    ("league", "scripted"),
    ("training", "eval_bots"),
)

_MISSING = object()


def _is_number(value) -> bool:
    """True for a finite int/float; bool is not a number here."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _is_positive_int(value) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), (
        f"{path.name}: top level must be a mapping, "
        f"got {type(cfg).__name__}"
    )
    return cfg


def _lookup(cfg: dict, section: str, key: str):
    """Return cfg[section][key], or _MISSING if section/key is absent.

    A section that is present but not a mapping is a config error,
    not absence, so it asserts here (required sections are asserted
    separately in test_required_sections_present_and_mappings).
    """
    section_value = cfg.get(section, _MISSING)
    if section_value is _MISSING or section_value is None:
        return _MISSING
    assert isinstance(section_value, dict), (
        f"{section} must be a mapping, got {type(section_value).__name__}"
    )
    return section_value.get(key, _MISSING)


def test_config_discovery_finds_all_configs():
    # Guards against accidental config deletion or a broken glob
    # silently parametrizing the tests below to zero cases.
    assert CONFIG_PATHS, f"no configs discovered in {CONFIG_DIR}"
    assert len(CONFIG_PATHS) >= 17, (
        f"expected at least 17 configs in {CONFIG_DIR}, "
        f"found {len(CONFIG_PATHS)}: {CONFIG_IDS}"
    )


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=CONFIG_IDS)
def test_config_loads_to_mapping(config_path: Path):
    cfg = _load_config(config_path)
    assert cfg, f"{config_path.name}: config mapping is empty"


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=CONFIG_IDS)
def test_required_sections_present_and_mappings(config_path: Path):
    cfg = _load_config(config_path)
    for name in REQUIRED_SECTIONS:
        assert name in cfg, (
            f"{config_path.name}: missing required top-level key "
            f"{name!r} (dereferenced unconditionally in rl/train.py)"
        )
        assert isinstance(cfg[name], dict), (
            f"{config_path.name}: {name!r} must be a mapping, "
            f"got {type(cfg[name]).__name__}"
        )


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=CONFIG_IDS)
def test_training_total_timesteps(config_path: Path):
    cfg = _load_config(config_path)
    training = cfg.get("training")
    assert isinstance(training, dict), (
        f"{config_path.name}: 'training' must be a mapping, "
        f"got {type(training).__name__}"
    )
    assert "total_timesteps" in training, (
        f"{config_path.name}: missing training.total_timesteps "
        f"(read as cfg['training']['total_timesteps'] in rl/train.py)"
    )
    value = training["total_timesteps"]
    assert _is_positive_int(value), (
        f"{config_path.name}: training.total_timesteps must be an "
        f"int > 0, got {value!r}"
    )


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=CONFIG_IDS)
def test_probabilities_in_unit_interval(config_path: Path):
    cfg = _load_config(config_path)
    for section, key in PROBABILITY_KEYS:
        value = _lookup(cfg, section, key)
        if value is _MISSING:
            continue
        assert _is_number(value) and 0 <= value <= 1, (
            f"{config_path.name}: {section}.{key} must be a real "
            f"number in [0, 1], got {value!r}"
        )


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=CONFIG_IDS)
def test_positive_counts_when_present(config_path: Path):
    cfg = _load_config(config_path)
    for section, key in POSITIVE_INT_KEYS:
        value = _lookup(cfg, section, key)
        if value is _MISSING:
            continue
        assert _is_positive_int(value), (
            f"{config_path.name}: {section}.{key} must be an "
            f"int > 0, got {value!r}"
        )


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=CONFIG_IDS)
def test_positive_numbers_when_present(config_path: Path):
    cfg = _load_config(config_path)
    for section, key in POSITIVE_NUMBER_KEYS:
        value = _lookup(cfg, section, key)
        if value is _MISSING:
            continue
        assert _is_number(value) and value > 0, (
            f"{config_path.name}: {section}.{key} must be a real "
            f"number > 0, got {value!r}"
        )
    for section, key in NONNEGATIVE_NUMBER_KEYS:
        value = _lookup(cfg, section, key)
        if value is _MISSING:
            continue
        assert _is_number(value) and value >= 0, (
            f"{config_path.name}: {section}.{key} must be a real "
            f"number >= 0, got {value!r}"
        )


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=CONFIG_IDS)
def test_ppo_net_arch_when_present(config_path: Path):
    cfg = _load_config(config_path)
    value = _lookup(cfg, "ppo", "net_arch")
    if value is _MISSING:
        return
    assert isinstance(value, (list, tuple)) and len(value) > 0, (
        f"{config_path.name}: ppo.net_arch must be a non-empty "
        f"list of positive ints, got {value!r}"
    )
    for width in value:
        assert _is_positive_int(width), (
            f"{config_path.name}: ppo.net_arch entries must be "
            f"positive ints, got {value!r}"
        )


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=CONFIG_IDS)
def test_bot_lists_when_present(config_path: Path):
    cfg = _load_config(config_path)
    for section, key in BOT_LIST_KEYS:
        value = _lookup(cfg, section, key)
        if value is _MISSING:
            continue
        assert isinstance(value, (list, tuple)) and len(value) > 0, (
            f"{config_path.name}: {section}.{key} must be a "
            f"non-empty list of bot names, got {value!r}"
        )
        for bot in value:
            assert isinstance(bot, str) and bot.strip() != "", (
                f"{config_path.name}: {section}.{key} entries must "
                f"be non-empty strings, got {value!r}"
            )


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=CONFIG_IDS)
def test_env_loadout_when_present(config_path: Path):
    cfg = _load_config(config_path)
    value = _lookup(cfg, "env", "loadout")
    if value is _MISSING:
        return
    assert isinstance(value, dict), (
        f"{config_path.name}: env.loadout must be a mapping, "
        f"got {type(value).__name__}"
    )


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=CONFIG_IDS)
def test_optional_sections_are_mappings(config_path: Path):
    cfg = _load_config(config_path)
    for name in OPTIONAL_SECTIONS:
        # rl/train.py reads these as cfg.get(name, {}) or {}, so an
        # absent or null section is equivalent to an empty one.
        if name not in cfg or cfg[name] is None:
            continue
        assert isinstance(cfg[name], dict), (
            f"{config_path.name}: optional section {name!r} must "
            f"be a mapping, got {type(cfg[name]).__name__}"
        )

