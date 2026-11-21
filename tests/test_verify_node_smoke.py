"""Smoke test that bridge/verify_node.mjs actually runs and agrees with NumPy.

bridge/verify_node.mjs is otherwise never executed by CI: the only
Python-to-Node agreement check is the manual rl/export_ts.py verify()
3-way torch-vs-numpy-vs-Node check (rl/export_ts.py:153, tol 1e-4),
which no test calls, and the pure-tests CI job
(.github/workflows/ci.yml:17-43) has no Node step. Meanwhile
tests/test_export_evaluate_pure.py exercises rl/export_ts.numpy_forward
(rl/export_ts.py:112) in isolation only, so a drift in the
verify_node.mjs normalize/clip/MLP code would pass CI silently.

This test closes that gap with a tiny synthetic net (8->4->2, tanh):
it writes the weights dict to tmp_path, feeds a raw obs JSON array to
`node bridge/verify_node.mjs <weights.json>` over stdin, and asserts the
printed logits match rl/export_ts.numpy_forward on the same obs to
<1e-6. It runs inside the existing pure-tests job (no CI change):
ubuntu-latest runners preinstall Node, and this test skips only when
the node binary is absent.

Scope: hand-built tiny weights in the style of
tests/test_export_evaluate_pure.py; no trained model, no bridge build,
no network. Importing rl.export_ts pulls in torch/stable-baselines3
at module level (CI installs requirements.txt), but nothing here
instantiates PPO or any environment.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from rl.export_ts import numpy_forward

_VERIFY_NODE = Path(__file__).resolve().parent.parent / "bridge" / "verify_node.mjs"

needs_node = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="node binary not found",
)


def _tiny_weights():
    """8->4->2 tanh net with a VecNormalize-style norm block.

    Norm keys mirror rl/export_ts.normalize_obs (rl/export_ts.py:55-62):
    mean, var, epsilon, clip_obs.
    """
    return {
        "obs_dim": 8,
        "action_dims": [2],
        "activation": "tanh",
        "norm": {
            "mean": [0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            "var": [1.0, 4.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "epsilon": 1e-8,
            "clip_obs": 10.0,
        },
        "layers": [
            {
                "W": [
                    [0.5, -0.25, 0.1, 0.0, 0.2, -0.1, 0.05, 0.0],
                    [-0.3, 0.4, 0.0, 0.2, -0.1, 0.3, 0.0, 0.1],
                    [0.0, 0.1, -0.5, 0.3, 0.0, -0.2, 0.4, -0.1],
                    [0.2, 0.0, -0.1, 0.0, 0.5, 0.0, -0.3, 0.2],
                ],
                "b": [0.1, -0.2, 0.0, 0.05],
            },
            {
                "W": [
                    [0.7, -0.6, 0.2, 0.1],
                    [-0.4, 0.5, -0.3, 0.8],
                ],
                "b": [0.05, -0.1],
            },
        ],
    }


def _run_verify_node(weights_path, obs_list):
    """Run verify_node.mjs on a raw obs JSON array fed over stdin."""
    return subprocess.run(
        [shutil.which("node"), str(_VERIFY_NODE), str(weights_path)],
        input=json.dumps(obs_list),
        capture_output=True,
        text=True,
        timeout=30,
    )


@needs_node
def test_verify_node_logits_match_numpy_forward(tmp_path):
    weights = _tiny_weights()
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps(weights))

    # Raw obs; the trailing 100.0 exercises the norm clip branch.
    obs = np.array([0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 0.25, 100.0],
                   dtype=np.float64)
    expected = numpy_forward(weights, obs)

    proc = _run_verify_node(weights_path, obs.astype(np.float64).tolist())
    assert proc.returncode == 0, f"verify_node.mjs failed: {proc.stderr}"
    logits = np.asarray(json.loads(proc.stdout)["logits"], dtype=np.float64)
    assert logits.shape == expected.shape
    diff = float(np.max(np.abs(logits - expected)))
    assert diff < 1e-6, f"node/numpy logit mismatch: {diff} >= 1e-6"


@needs_node
def test_verify_node_rejects_malformed_weights(tmp_path):
    weights_path = tmp_path / "weights.json"
    weights_path.write_text("{ not valid json")

    proc = _run_verify_node(weights_path, [0.0] * 8)
    assert proc.returncode != 0
