"""Restricted unpickling for VecNormalize checkpoint stats.

Why this exists: a ``vecnormalize.pkl`` is a Python pickle, and a plain
``pickle.load`` of one is arbitrary code execution for whoever wrote the
file. Run folders are shared/swapped between campaigns in this repo's
workflow, so the two *direct* pickle sites that only need observation
normalization stats -- ``rl/anchor.py`` (anchor stats) and
``rl/export_ts.py:load_norm`` -- must not unpickle unrestricted.

``RestrictedUnpickler`` allowlists exactly the globals found by a
``pickletools`` opcode scan (no execution) of real
``runs/*/vecnormalize.pkl`` / ``anchor_vecnormalize.pkl`` files at
``pluto-upgrade@b651eb8`` (SB3 2.9.0 / gymnasium 1.3.0 / numpy 2.5.3):

* ``stable_baselines3...VecNormalize`` + ``RunningMeanStd``
* ``gymnasium.spaces`` ``Box`` + ``MultiDiscrete`` (embedded spaces)
* ``numpy.dtype``, ``numpy._core.multiarray.scalar``,
  ``numpy._core.numeric._frombuffer`` (ndarray reconstruction)
* numpy RNG constructors embedded in a seeded space:
  ``numpy.random._pickle.__generator_ctor`` / ``__bit_generator_ctor``,
  ``numpy.random._pcg64.PCG64``,
  ``numpy.random.bit_generator.__pyx_unpickle_SeedSequence`` /
  ``SeedSequence``

Everything else raises ``pickle.UnpicklingError`` from ``find_class``
before any of it can execute. After loading, ``load_vecnormalize_stats``
additionally validates the object graph (exact types, finite 1-D float
mean/var of equal length, finite epsilon/clip_obs) and returns only the
stats dict -- callers never receive the raw unpickled object.

NOT covered (trusted-local-only): SB3-mediated loads elsewhere in this
repo -- ``PPO.load`` and ``VecNormalize.load`` in ``rl/train.py``,
``rl/evaluate.py``, ``rl/export_ts.py`` and ``scripts/*``. In SB3 2.9.0
those unpickle via ``pickle``/``cloudpickle`` internally and expose no
``weights_only``/unpickler parameter, so they cannot be fixed from this
repo without forking SB3. Never point them at a checkpoint or run folder
that did not originate on a machine you trust.
"""
from __future__ import annotations

import math
import os
import pickle
from typing import BinaryIO

import numpy as np

# (module, name) pairs permitted in a VecNormalize pickle. See the module
# docstring for how this list was derived; do not widen it without
# re-scanning a real file with pickletools.
_ALLOWED_GLOBALS = frozenset({
    ("stable_baselines3.common.vec_env.vec_normalize", "VecNormalize"),
    ("stable_baselines3.common.running_mean_std", "RunningMeanStd"),
    ("gymnasium.spaces.box", "Box"),
    ("gymnasium.spaces.multi_discrete", "MultiDiscrete"),
    ("numpy", "dtype"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy.random._pickle", "__generator_ctor"),
    ("numpy.random._pickle", "__bit_generator_ctor"),
    ("numpy.random._pcg64", "PCG64"),
    ("numpy.random.bit_generator", "__pyx_unpickle_SeedSequence"),
    ("numpy.random.bit_generator", "SeedSequence"),
})


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that resolves only the VecNormalize allowlist above."""

    def find_class(self, module: str, name: str):  # noqa: D102
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"forbidden global in VecNormalize pickle: {module}.{name}")


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise pickle.UnpicklingError(
            f"VecNormalize.{field} is not a real number: {type(value)!r}")
    result = float(value)
    if not math.isfinite(result):
        raise pickle.UnpicklingError(
            f"VecNormalize.{field} is not finite: {result!r}")
    return result


def _validate_stats(obj: object) -> dict:
    # Imported lazily so importing this module (e.g. in pure tests) does
    # not pull in SB3/torch; a real load needs them anyway, because the
    # allowlisted classes live there.
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    from stable_baselines3.common.vec_env import VecNormalize

    if not isinstance(obj, VecNormalize):
        raise pickle.UnpicklingError(
            f"expected a VecNormalize object, got {type(obj)!r}")
    obs_rms = getattr(obj, "obs_rms", None)
    if not isinstance(obs_rms, RunningMeanStd):
        raise pickle.UnpicklingError(
            f"VecNormalize.obs_rms is not a RunningMeanStd: "
            f"{type(obs_rms)!r}")
    mean = np.asarray(obs_rms.mean)
    var = np.asarray(obs_rms.var)
    for label, arr in (("mean", mean), ("var", var)):
        if arr.ndim != 1 or arr.size == 0:
            raise pickle.UnpicklingError(
                f"obs_rms.{label} must be a non-empty 1-D array, "
                f"got shape {arr.shape}")
        if arr.dtype.kind != "f":
            raise pickle.UnpicklingError(
                f"obs_rms.{label} must be a float array, got {arr.dtype}")
        if not np.all(np.isfinite(arr)):
            raise pickle.UnpicklingError(
                f"obs_rms.{label} contains non-finite values")
    if mean.shape != var.shape:
        raise pickle.UnpicklingError(
            f"obs_rms mean/var shape mismatch: {mean.shape} vs {var.shape}")
    epsilon = _finite_float(getattr(obj, "epsilon", None), "epsilon")
    clip_obs = _finite_float(getattr(obj, "clip_obs", None), "clip_obs")
    if epsilon < 0:
        raise pickle.UnpicklingError(
            f"VecNormalize.epsilon must be >= 0, got {epsilon}")
    if clip_obs <= 0:
        raise pickle.UnpicklingError(
            f"VecNormalize.clip_obs must be > 0, got {clip_obs}")
    return {
        "mean": np.array(mean, dtype=np.float64, copy=True),
        "var": np.array(var, dtype=np.float64, copy=True),
        "epsilon": epsilon,
        "clip_obs": clip_obs,
    }


def load_vecnormalize_stats(path: str | os.PathLike) -> dict:
    """Load a ``vecnormalize.pkl`` under the restricted unpickler.

    Returns ``{"mean": float64 ndarray, "var": float64 ndarray,
    "epsilon": float, "clip_obs": float}`` -- the same values the former
    direct-``pickle.load`` call sites read off the raw object. Any file
    that references a non-allowlisted global, or whose object graph does
    not validate as VecNormalize stats, raises
    ``pickle.UnpicklingError`` and nothing in it executes.
    """
    with open(path, "rb") as f:
        return load_vecnormalize_stats_from_file(f)


def load_vecnormalize_stats_from_file(file_obj: BinaryIO) -> dict:
    """File-object variant of :func:`load_vecnormalize_stats`."""
    obj = RestrictedUnpickler(file_obj).load()
    return _validate_stats(obj)
