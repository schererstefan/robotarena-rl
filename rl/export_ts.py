"""Export a trained PPO policy to portable MLP weights JSON + a RobotArena robot.

The weights JSON feeds two consumers:
  1. the Node bridge policy opponent (league self-play), and
  2. `robot_<name>.ts` — a drop-in RobotArena robot file (see below).

Layout: {"obs_dim": 516, "action_dims": [5,5,5,2,2,2,2],
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
import subprocess

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from .env import (
    ACTION_DIMS, BULLET_BASE, EVENT_BASE, FOE_BASE, FRAME_DIM,
    FRAME_STACK, MATCH_BASE, N_STRATS, OBS_DIM, PAD_BASE, TICK,
    TRACK_BASE, TURRET_BASE, WALL_BASE, ZONE_BASE,
)
from .safe_pickle import load_vecnormalize_stats


def load_norm(vecnormalize_path: str | None) -> dict | None:
    """Read VecNormalize obs stats from a .pkl into a JSON-able dict."""
    if not vecnormalize_path or not os.path.exists(vecnormalize_path):
        return None
    # Restricted unpickler: a vecnormalize.pkl is executable pickle;
    # only its obs stats are needed here. See rl/safe_pickle.py.
    stats = load_vecnormalize_stats(vecnormalize_path)
    return {
        "mean": np.asarray(stats["mean"]).reshape(-1).tolist(),
        "var": np.asarray(stats["var"]).reshape(-1).tolist(),
        "epsilon": float(stats["epsilon"]),
        "clip_obs": float(stats["clip_obs"]),
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

    Raw obs are drawn from sample_realistic_obs, which follows the
    live bridge layout (see bridge/OBS_SPEC.md), rather than
    uniform(-1, 1), so the VecNormalize path is exercised.
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
    """Sample a raw obs with plausible per-feature ranges (pre-normalization).

    Drawn section by section against the live 128-dim layout in
    bridge/OBS_SPEC.md:7-18 (which bridge.template.ts ``buildObs``
    emits), so verify() exercises the VecNormalize path on inputs
    shaped like real bridge frames. Section bases come from rl.env,
    so this sampler shares one index source of truth with the env.
    Pin: tests/test_export_evaluate_pure.py.
    """
    o = np.zeros((FRAME_DIM,), dtype=np.float64)

    def flag() -> float:
        return float(rng.integers(0, 2))

    def angle() -> float:
        return float(rng.uniform(-np.pi, np.pi))

    # -- self 0-16 --
    o[0] = rng.uniform(0, 1)   # x/960
    o[1] = rng.uniform(0, 1)   # y/640
    a = angle()
    o[2], o[3] = np.sin(a), np.cos(a)   # sin/cos(heading)
    a = angle()
    o[4], o[5] = np.sin(a), np.cos(a)   # sin/cos(tower)
    o[6] = rng.uniform(0, 1)   # speed
    o[7] = rng.uniform(0, 1)   # health
    o[8] = rng.uniform(0, 1)   # cooldown/24
    o[9] = rng.uniform(0, 1)   # charge
    o[10] = flag()             # charged
    o[11] = rng.uniform(0, 1)  # dashCd/480
    o[12] = rng.uniform(0, 1)  # empCd/720
    o[13] = flag()             # slowed
    o[14] = rng.uniform(0, 1)  # lastDamage/100
    a = angle()
    o[15], o[16] = np.sin(a), np.cos(a)  # lastDamage bearing - heading

    # -- foes 17-32 (2 slots x 8) --
    any_foe = False
    for s in range(2):
        b = FOE_BASE + 8 * s
        if rng.integers(0, 2) == 0:
            continue  # absent: the whole slot stays 0
        any_foe = True
        o[b] = 1.0                      # present
        o[b + 1] = rng.uniform(-1, 1)   # dx/960
        o[b + 2] = rng.uniform(-1, 1)   # dy/640
        o[b + 3] = rng.uniform(0, 1)    # dist/540
        a = angle()
        o[b + 4], o[b + 5] = np.sin(a), np.cos(a)  # bearing - heading
        o[b + 6] = rng.uniform(0, 1)    # speed
        o[b + 7] = rng.uniform(0, 1)    # health

    # -- stale track 33-37 --
    if rng.integers(0, 2) == 1:  # seenNow; else the slot stays 0
        o[TRACK_BASE] = 1.0
        o[TRACK_BASE + 1] = rng.uniform(-1, 1)  # dx/960
        o[TRACK_BASE + 2] = rng.uniform(-1, 1)  # dy/640
        o[TRACK_BASE + 3] = rng.uniform(0, 1)   # age/600
        o[TRACK_BASE + 4] = rng.uniform(0, 1)   # dist/960

    # -- bullets 38-73 (6 slots x 6) --
    for s in range(6):
        b = BULLET_BASE + 6 * s
        if rng.integers(0, 2) == 0:
            continue  # absent: the whole slot stays 0
        o[b] = 1.0                      # present
        o[b + 1] = rng.uniform(-1, 1)   # dx/960
        o[b + 2] = rng.uniform(-1, 1)   # dy/640
        o[b + 3] = rng.uniform(0, 1)    # dist/540
        o[b + 4] = rng.uniform(0, 1)    # closing/430
        o[b + 5] = rng.uniform(0, 1)    # damage/25

    # -- walls 74-77: derived from the sampled self position --
    o[WALL_BASE] = o[0]             # left/960
    o[WALL_BASE + 1] = 1.0 - o[0]   # right/960
    o[WALL_BASE + 2] = o[1]         # top/640
    o[WALL_BASE + 3] = 1.0 - o[1]   # bottom/640

    # -- zone 78-81 --
    o[ZONE_BASE] = rng.uniform(0, 1)      # distToSafety/960
    o[ZONE_BASE + 1] = flag()             # inside
    o[ZONE_BASE + 2] = rng.uniform(0, 1)  # suddenDeathIn/11400
    o[ZONE_BASE + 3] = flag()             # shrinking

    # -- turrets 82-93 (2 slots x 6) --
    for s in range(2):
        b = TURRET_BASE + 6 * s
        if rng.integers(0, 2) == 0:
            continue  # inactive: the whole slot stays 0
        o[b] = 1.0                      # active
        o[b + 1] = rng.uniform(-1, 1)   # dx/960
        o[b + 2] = rng.uniform(-1, 1)   # dy/640
        owner = int(rng.integers(0, 3))  # 0 mine, 1 foe's, 2 unowned
        o[b + 3] = 1.0 if owner == 0 else 0.0   # owned-by-me
        o[b + 4] = 1.0 if owner == 1 else 0.0   # owned-by-foe
        o[b + 5] = rng.uniform(-1, 1)   # progress

    # -- pickup pads 94-117 (4 slots x 6) --
    for s in range(4):
        b = PAD_BASE + 6 * s
        if rng.integers(0, 2) == 0:
            continue  # inactive: the whole slot stays 0
        o[b] = 1.0                      # active
        o[b + 1] = rng.uniform(-1, 1)   # dx/960
        o[b + 2] = rng.uniform(-1, 1)   # dy/640
        o[b + 3 + int(rng.integers(0, 3))] = 1.0  # kind one-hot

    # -- match 118-121 --
    o[MATCH_BASE] = float(rng.integers(0, 6)) / 5.0      # killsYou/5
    o[MATCH_BASE + 1] = float(rng.integers(0, 6)) / 5.0  # killsTeam/5
    o[MATCH_BASE + 2] = float(rng.integers(0, 4)) / 3.0  # aliveFoes/3
    o[TICK] = rng.uniform(0, 1)                          # tick/11400

    # -- last-tick events 122-127 --
    o[EVENT_BASE] = flag()      # wasHit
    # leadFeat sin/cos (EVENT_BASE+1/+3): a unit-circle pair from one
    # lead-error angle, or exactly (0, 1) when no foe is visible.
    if any_foe:
        a = angle()
        o[EVENT_BASE + 1] = np.sin(a)
        o[EVENT_BASE + 3] = np.cos(a)
    else:
        o[EVENT_BASE + 1] = 0.0
        o[EVENT_BASE + 3] = 1.0
    o[EVENT_BASE + 2] = flag()  # gotKill
    o[EVENT_BASE + 4] = flag()  # pickup
    o[EVENT_BASE + 5] = flag()  # turret-captured

    # Pluto upgrade: the policy consumes FRAME_STACK stacked frames
    # + a strategy one-hot (OBS_DIM = 516). Tile one realistic frame
    # x FRAME_STACK + arm-0 one-hot.
    oh = np.zeros((N_STRATS,), dtype=np.float64)
    oh[0] = 1.0
    out = np.concatenate([o] * FRAME_STACK + [oh])
    assert out.shape == (OBS_DIM,)
    return out


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
const OBS_DIM = __OBS_DIM__; // 516 = 4 stacked 128-dim frames + 4-dim strategy one-hot
const FRAME_DIM = 128;
const FRAMES = 4;
const N_STRAT = 4;
// Strategy conditioning (Pluto lesson #4). Default arm is baked at export
// from the bandit file (--bandit); override per match via setStrategyArm().
let STRATEGY_ARM = __DEFAULT_STRAT__;
export function setStrategyArm(i: number): void {
    STRATEGY_ARM = Math.max(0, Math.min(N_STRAT - 1, Math.floor(i)));
}
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
    // Frame-stack buffer: memory lives in the net (Pluto lesson #1).
    // Pre-filled with the first obs, mirroring rl/env.py reset().
    let frames: number[][] = [];
    function update(sense: SenseState): Intent {{
        const raw = buildObs(sense);
        frames.push(raw);
        if (frames.length > FRAMES) frames.shift();
        while (frames.length < FRAMES) frames.unshift(frames[0]);
        const oh = [0, 0, 0, 0];
        oh[STRATEGY_ARM] = 1;
        const obs = normalizeObs(([] as number[]).concat(...frames, oh));
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
    ap.add_argument("--bandit", default=None,
                    help="path to strategy_bandit.json; the overall-best arm "
                         "is baked in as the default strategy (default: 0)")
    args = ap.parse_args()

    default_strat = 0
    if args.bandit and os.path.exists(args.bandit):
        from .bandit import StrategyBandit
        default_strat = StrategyBandit(args.bandit).overall_best()
        print(f"default strategy arm {default_strat} from {args.bandit}")

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
    # The obs builder calls leadError(), which is defined in the bridge
    # template and needs BULLET_SPEED from the game constants. Extract both
    # so the generated robot file is self-contained (previously the output
    # referenced leadError/BULLET_SPEED without defining them).
    # NOTE: match the full signature (incl. the braced return type) so brace
    # counting starts at the function *body*, not the return type annotation.
    lead_sig = ("function leadError(sense: SenseState): "
                "{ sin: number; cos: number; err: number } {")
    ls = src.index(lead_sig)
    ldepth = 1
    lend = None
    for i in range(ls + len(lead_sig), len(src)):
        if src[i] == "{":
            ldepth += 1
        elif src[i] == "}":
            ldepth -= 1
            if ldepth == 0:
                lend = i + 1
                break
    assert lend is not None, "could not extract leadError from bridge template"
    lead_fn = src[ls:lend]
    import re
    bullet_speed = None
    ra_path = os.environ.get("ROBOTARENA_PATH", "/Users/stefan/dev/robotarena")
    const_path = os.path.join(ra_path, "src", "sim", "constants.ts")
    try:
        with open(const_path) as f:
            m = re.search(r"BULLET_SPEED\s*=\s*([0-9.]+)", f.read())
        if m:
            bullet_speed = m.group(1)
    except OSError:
        pass
    if bullet_speed is None:
        bullet_speed = "430"
        print("WARNING: could not read BULLET_SPEED from game constants; "
              "defaulting to 430")
    else:
        print(f"inlined BULLET_SPEED={bullet_speed} from {const_path}")
    builder = header + f"const BULLET_SPEED = {bullet_speed};\n" + lead_fn + "\n\n" + builder

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
        .replace("__DEFAULT_STRAT__", str(default_strat)) \
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
