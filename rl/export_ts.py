"""Export a trained PPO policy to portable MLP weights JSON + a RobotArena robot.

The weights JSON feeds two consumers:
  1. the Node bridge policy opponent (league self-play), and
  2. `robot_<name>.ts` — a drop-in RobotArena robot file (see below).

Layout: {"obs_dim": 124, "action_dims": [5,5,5,2,2,2,2],
          "activation": "tanh",
          "norm": {"mean": [...], "var": [...], "epsilon": 1e-8,
                   "clip_obs": 10.0},   # VecNormalize stats (optional)
          "layers": [{"W": [[out][in]...], "b": [...]}, ...]}
The final layer is the action head (no activation); all earlier layers use
the configured activation. Consumers MUST normalize raw obs with `norm`
before the forward pass (see bridge normalizeObs / robot template).

Usage:
    python -m rl.export_ts --model runs/<run>/model.zip --out export/rl_bot
    # writes export/rl_bot_weights.json + export/rl_bot.ts, then verifies
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from .env import ACTION_DIMS, OBS_DIM


def load_norm(vecnormalize_path: str | None) -> dict | None:
    """Read VecNormalize obs stats from a .pkl into a JSON-able dict."""
    if not vecnormalize_path or not os.path.exists(vecnormalize_path):
        return None
    with open(vecnormalize_path, "rb") as f:
        vn = pickle.load(f)
    return {
        "mean": np.asarray(vn.obs_rms.mean).reshape(-1).tolist(),
        "var": np.asarray(vn.obs_rms.var).reshape(-1).tolist(),
        "epsilon": float(vn.epsilon),
        "clip_obs": float(vn.clip_obs),
    }


def normalize_obs(obs: np.ndarray, norm: dict | None) -> np.ndarray:
    x = obs.astype(np.float64)
    if norm is None:
        return x
    mean = np.asarray(norm["mean"], dtype=np.float64)
    var = np.asarray(norm["var"], dtype=np.float64)
    x = (x - mean) / np.sqrt(var + norm["epsilon"])
    return np.clip(x, -norm["clip_obs"], norm["clip_obs"])


def extract_weights(model: PPO, activation: str = "tanh",
                    norm: dict | None = None) -> dict:
    policy = model.policy
    layers = []
    for mod in policy.mlp_extractor.policy_net:
        if isinstance(mod, th.nn.Linear):
            layers.append({
                "W": mod.weight.detach().cpu().numpy().tolist(),
                "b": mod.bias.detach().cpu().numpy().tolist(),
            })
    head = policy.action_net
    layers.append({
        "W": head.weight.detach().cpu().numpy().tolist(),
        "b": head.bias.detach().cpu().numpy().tolist(),
    })
    weights = {"obs_dim": OBS_DIM, "action_dims": ACTION_DIMS,
               "activation": activation, "layers": layers}
    if norm is not None:
        weights["norm"] = norm
    return weights


def export_weights(model: PPO, path: str, activation: str = "tanh",
                   norm: dict | None = None) -> dict:
    weights = extract_weights(model, activation, norm)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(weights, f)
    return weights


def norm_from_vecnormalize(venv) -> dict | None:
    """Build the JSON-able norm dict from a live VecNormalize wrapper."""
    vn = venv
    # Unwrap in case of nesting (e.g. VecNormalize(VecNormalize(...))).
    while not hasattr(vn, "obs_rms") and hasattr(vn, "venv"):
        vn = vn.venv
    if not hasattr(vn, "obs_rms"):
        return None
    return {
        "mean": np.asarray(vn.obs_rms.mean).reshape(-1).tolist(),
        "var": np.asarray(vn.obs_rms.var).reshape(-1).tolist(),
        "epsilon": float(vn.epsilon),
        "clip_obs": float(vn.clip_obs),
    }


def numpy_forward(weights: dict, obs: np.ndarray) -> np.ndarray:
    x = normalize_obs(obs, weights.get("norm"))
    layers = weights["layers"]
    act = weights["activation"]
    for i, layer in enumerate(layers):
        W = np.asarray(layer["W"], dtype=np.float64)
        b = np.asarray(layer["b"], dtype=np.float64)
        x = W @ x + b
        if i < len(layers) - 1:
            x = np.tanh(x) if act == "tanh" else np.maximum(x, 0.0)
    return x


def torch_logits(model: PPO, obs_norm: np.ndarray) -> np.ndarray:
    policy = model.policy
    with th.no_grad():
        obs_t = th.as_tensor(obs_norm, dtype=th.float32).unsqueeze(0)
        features = policy.extract_features(obs_t)
        latent_pi, _ = policy.mlp_extractor(features)
        logits = policy.action_net(latent_pi)
    return logits.squeeze(0).cpu().numpy()


def node_logits(weights_path: str, obs: np.ndarray,
                node_bin: str = "node") -> np.ndarray:
    """Run bridge/verify_node.mjs on a raw obs; returns the JS logits."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "bridge", "verify_node.mjs")
    proc = subprocess.run(
        [node_bin, script, weights_path],
        input=json.dumps(obs.astype(np.float64).tolist()),
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"verify_node.mjs failed: {proc.stderr.strip()}")
    return np.asarray(json.loads(proc.stdout)["logits"], dtype=np.float64)


def verify(weights: dict, model: PPO, weights_path: str,
           n: int = 5, tol: float = 1e-4, node_bin: str = "node") -> None:
    """3-way check: torch vs numpy vs Node.js, on raw obs (norm in the loop).

    Raw obs are sampled from realistic ranges (positions, angles, health…)
    rather than uniform(-1, 1), so the VecNormalize path is exercised.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(n):
        obs = sample_realistic_obs(rng).astype(np.float32)
        obs_n = normalize_obs(obs, weights.get("norm")).astype(np.float32)
        ref = torch_logits(model, obs_n).astype(np.float64)
        got_np = numpy_forward(weights, obs)
        got_js = node_logits(weights_path, obs, node_bin)
        for name, got in (("numpy", got_np), ("node", got_js)):
            diff = float(np.max(np.abs(ref - got)))
            worst = max(worst, diff)
            assert diff < tol, f"{name} logit mismatch: {diff} >= {tol}"
        # argmax actions must agree too
        off = 0
        for d in ACTION_DIMS:
            a_ref = int(np.argmax(ref[off:off + d]))
            assert int(np.argmax(got_np[off:off + d])) == a_ref, \
                "numpy action mismatch after export"
            assert int(np.argmax(got_js[off:off + d])) == a_ref, \
                "node action mismatch after export"
            off += d
    print(f"verify OK: max |logit diff| torch/numpy/node over {n} obs "
          f"= {worst:.2e} (tol {tol})")


def sample_realistic_obs(rng: np.random.Generator) -> np.ndarray:
    """Sample a raw obs with plausible per-feature ranges (pre-normalization)."""
    o = np.zeros((OBS_DIM,), dtype=np.float64)
    # 0-1: position, 2-3: speed, 4-7: tower/chassis sin/cos, 8-9: health/heat
    o[0:2] = rng.uniform(0, 1, 2)
    o[2] = rng.uniform(0, 1)
    o[3] = rng.uniform(0, 1)
    a = rng.uniform(-np.pi, np.pi, 2)
    o[4:6] = np.sin(a[0]), np.cos(a[0])
    o[6:8] = np.sin(a[1]), np.cos(a[1])
    o[8] = rng.uniform(0, 1)
    o[9] = rng.uniform(0, 1)
    # 10-12: cooldowns, 13: overdrive, 14-17: enemy dir/dash/emp, 18-19: alive
    o[10:13] = rng.uniform(0, 1, 3)
    o[13] = rng.uniform(0, 1)
    o[14:16] = np.sin(a[0]), np.cos(a[0])
    o[16] = rng.uniform(0, 1)
    o[17] = rng.uniform(0, 1)
    o[18] = rng.choice([0.0, 1.0])
    o[19] = rng.choice([0.0, 1.0])
    # 20-43: two foe slots of 12
    for s in range(2):
        b = s * 12
        o[20 + b] = rng.choice([0.0, 1.0])
        o[21 + b:23 + b] = np.sin(a[0]), np.cos(a[0])
        o[23 + b] = rng.uniform(0, 1)
        o[24 + b] = rng.uniform(0, 1)
        o[25 + b:27 + b] = np.sin(a[1]), np.cos(a[1])
        o[27 + b] = rng.uniform(0, 1)
        o[28 + b] = rng.uniform(0, 1)
        o[29 + b] = rng.choice([0.0, 1.0])
        o[30 + b] = rng.choice([0.0, 1.0])
        o[31 + b] = rng.uniform(0, 1)
    # 44-67: two track slots of 12 (same layout)
    for s in range(2):
        b = 44 + s * 12
        o[b] = rng.choice([0.0, 1.0])
        o[b + 1:b + 3] = np.sin(a[0]), np.cos(a[0])
        o[b + 3] = rng.uniform(0, 1)
        o[b + 4] = rng.uniform(0, 1)
        o[b + 5:b + 7] = np.sin(a[1]), np.cos(a[1])
        o[b + 7] = rng.uniform(0, 1)
        o[b + 8] = rng.uniform(0, 1)
        o[b + 9] = rng.choice([0.0, 1.0])
        o[b + 10] = rng.choice([0.0, 1.0])
        o[b + 11] = rng.uniform(0, 1)
    # 68-79: 4 powerup slots of 3; 80-91: 4 turret slots of 3
    for i in range(68, 92):
        o[i] = rng.uniform(0, 1) if (i - 68) % 3 else rng.choice([0.0, 1.0])
    # 92-99: one-hot timer; 100-103: match one-hots; 104-111: my skills;
    # 112-115: foe skills; 116-119: scoreboard; 120-123: timer/pos/vel/score
    o[92 + rng.integers(0, 8)] = 1.0
    o[100 + rng.integers(0, 4)] = 1.0
    o[104:112] = rng.integers(0, 4, 8) / 3.0
    o[112:116] = rng.integers(0, 4, 4) / 3.0
    o[116] = rng.uniform(-3, 3)
    o[117] = rng.uniform(-3, 3)
    o[118] = rng.uniform(0, 1)
    o[119] = rng.uniform(0, 1)
    o[120] = rng.uniform(0, 1)
    o[121:123] = rng.uniform(0, 1, 2)
    o[123] = rng.uniform(-3, 3)
    return o


ROBOT_TEMPLATE = """// Exported RobotArena RL policy — drop into src/robots/ and register it.
// Generated by robotarena-rl (rl/export_ts.py). Obs layout: bridge/OBS_SPEC.md.
import type {{ Intent, RobotController, RobotMeta, SenseState }} from '../sim/types';

export const meta: RobotMeta = {{
    id: '__ROBOT_ID__',
    name: '__ROBOT_NAME__',
    author: 'robotarena-rl',
    version: '1.0.0',
    description: 'PPO policy trained with league self-play.',
}};

export const loadout = __LOADOUT__;

const W_DIMS = __ACTION_DIMS__;
const OBS_DIM = __OBS_DIM__;
// weights: layers of {{W: [out][in], b: [out]}}; activation between layers, linear head
const ACT = '__ACTIVATION__' as 'tanh' | 'relu';
const LAYERS: Array<{{ W: number[][]; b: number[] }}> = __WEIGHTS__;
// VecNormalize stats from training — raw obs MUST be normalized first.
const NORM_MEAN: number[] = __NORM_MEAN__;
const NORM_VAR: number[] = __NORM_VAR__;
const NORM_EPS = __NORM_EPS__;
const NORM_CLIP = __NORM_CLIP__;

function normalizeObs(obs: number[]): number[] {{
    const out = new Array<number>(obs.length);
    for (let i = 0; i < obs.length; i += 1) {{
        const v = (obs[i] - NORM_MEAN[i]) / Math.sqrt(NORM_VAR[i] + NORM_EPS);
        out[i] = Math.min(NORM_CLIP, Math.max(-NORM_CLIP, v));
    }}
    return out;
}}

function mlp(obs: number[]): number[] {{
    const act = ACT === 'tanh' ? Math.tanh : (v: number): number => Math.max(0, v);
    let x = obs;
    for (let li = 0; li < LAYERS.length; li += 1) {{
        const {{ W, b }} = LAYERS[li];
        const y = new Array<number>(b.length);
        for (let j = 0; j < b.length; j += 1) {{
            let acc = b[j];
            const row = W[j];
            for (let k = 0; k < x.length; k += 1) acc += row[k] * x[k];
            y[j] = acc;
        }}
        x = li === LAYERS.length - 1 ? y : y.map(act);
    }}
    return x;
}}

function argmax(xs: number[]): number {{
    let bi = 0;
    for (let i = 1; i < xs.length; i += 1) if (xs[i] > xs[bi]) bi = i;
    return bi;
}}

{OBS_BUILDER}

export function create(): RobotController {{
    function update(sense: SenseState): Intent {{
        const obs = normalizeObs(buildObs(sense));
        const logits = mlp(obs);
        const a: number[] = [];
        let off = 0;
        for (const d of W_DIMS) {{
            a.push(argmax(logits.slice(off, off + d)));
            off += d;
        }}
        const bin5 = (i: number): number => i / 2 - 1;
        return {{
            throttle: bin5(a[0] ?? 2),
            turn: bin5(a[1] ?? 2),
            towerTurn: bin5(a[2] ?? 2),
            fire: (a[3] ?? 0) === 1,
            charge: (a[4] ?? 0) === 1,
            dash: (a[5] ?? 0) === 1,
            emp: (a[6] ?? 0) === 1,
        }};
    }}
    return {{ meta, loadout, update }};
}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True,
                    help="output prefix, e.g. export/rl_bot")
    ap.add_argument("--robot-id", default="rl-ppo")
    ap.add_argument("--robot-name", default="RL PPO")
    ap.add_argument("--loadout", default='{"overdrive": 2, "trigger": 2, "plating": 2}')
    ap.add_argument("--vecnormalize", default=None,
                    help="path to vecnormalize.pkl; defaults to <model_dir>/vecnormalize.pkl")
    ap.add_argument("--node", default="node",
                    help="node binary for the Node.js numeric check")
    args = ap.parse_args()

    model = PPO.load(args.model)
    vn_path = args.vecnormalize
    if vn_path is None:
        cand = os.path.join(os.path.dirname(os.path.abspath(args.model)),
                            "vecnormalize.pkl")
        vn_path = cand if os.path.exists(cand) else None
    norm = load_norm(vn_path)
    if norm is None:
        print("WARNING: no VecNormalize stats found — exporting without "
              "obs normalization (policy will misbehave if trained normalized)")
    else:
        print(f"loaded VecNormalize stats from {vn_path}")
    weights = extract_weights(model, norm=norm)

    weights_path = args.out + "_weights.json"
    with open(weights_path, "w") as f:
        json.dump(weights, f)
    print(f"wrote {weights_path}")

    verify(weights, model, weights_path, node_bin=args.node)

    # Inline the obs builder from the bridge template for a self-contained file.
    # Brace-balanced extraction (the function contains nested blocks).
    tpl_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "bridge", "bridge.template.ts")
    with open(tpl_path) as f:
        src = f.read()
    sig = "function buildObs(sense: SenseState, ctx: ObsCtx): number[] {"
    start = src.index(sig)
    depth = 0
    end = None
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    assert end is not None, "could not extract buildObs from bridge template"
    builder = src[start:end]
    # Adapt: robot file has no ObsCtx; maxHealth from the loadout
    # (START_HEALTH 100 + 15 per plating point, mirroring skills.ts).
    loadout_obj = json.loads(args.loadout)
    maxh = 100 + 15 * int(loadout_obj.get("plating", 0))
    builder = builder.replace(
        "function buildObs(sense: SenseState, ctx: ObsCtx): number[] {",
        f"const MAXH = {maxh};\n"
        "function buildObs(sense: SenseState): number[] {\n"
        "    const ctx = { maxHealth: MAXH, myTeam: sense.self.team as 0 | 1 };")
    # Helpers + constants the builder needs.
    header = (
        "const clamp = (v: number, lo: number, hi: number): number =>\n"
        "    Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : 0;\n"
        "const normAngle = (a: number): number => {\n"
        "    let x = a % (Math.PI * 2);\n"
        "    if (x > Math.PI) x -= Math.PI * 2;\n"
        "    if (x < -Math.PI) x += Math.PI * 2;\n"
        "    return x;\n"
        "};\n"
        "const ARENA_WIDTH = 960;\nconst ARENA_HEIGHT = 640;\n"
        "const SENSOR_RANGE = 540;\nconst MAX_TICKS_TOTAL = 11400;\n"
        "const MAX_SPEED = 150;\nconst GUN_COOLDOWN_TICKS = 24;\n"
        "const DASH_COOLDOWN_TICKS = 480;\nconst EMP_COOLDOWN_TICKS = 720;\n"
    )
    builder = header + builder

    norm = weights.get("norm") or {"mean": [0.0] * OBS_DIM,
                                       "var": [1.0] * OBS_DIM,
                                       "epsilon": 1e-8, "clip_obs": 10.0}
    robot_src = ROBOT_TEMPLATE \
        .replace("__ROBOT_ID__", args.robot_id) \
        .replace("__ROBOT_NAME__", args.robot_name) \
        .replace("__LOADOUT__", args.loadout) \
        .replace("__ACTION_DIMS__", json.dumps(ACTION_DIMS)) \
        .replace("__OBS_DIM__", str(OBS_DIM)) \
        .replace("__WEIGHTS__", json.dumps([l for l in weights["layers"]])) \
        .replace("__ACTIVATION__", weights["activation"]) \
        .replace("__NORM_MEAN__", json.dumps(norm["mean"])) \
        .replace("__NORM_VAR__", json.dumps(norm["var"])) \
        .replace("__NORM_EPS__", repr(float(norm["epsilon"]))) \
        .replace("__NORM_CLIP__", repr(float(norm["clip_obs"]))) \
        .replace("{OBS_BUILDER}", builder)
    # The template above is written with doubled braces; collapse to real TS.
    robot_src = robot_src.replace("{{", "{").replace("}}", "}")

    robot_path = args.out + ".ts"
    with open(robot_path, "w") as f:
        f.write(robot_src)
    print(f"wrote {robot_path} "
          f"({os.path.getsize(robot_path) / 1024:.0f} KB)")
    print("To use in RobotArena: copy into src/robots/, add to registry.ts "
          "and sources.ts (see docs/ROBOT_API.md).")


if __name__ == "__main__":
    main()
