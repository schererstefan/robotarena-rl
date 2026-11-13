# robotarena-rl

Deep reinforcement learning for RobotArena
(2D pixel-art robot battles): PPO with league self-play, trained against the
real game sim running headless.

## Architecture

```
┌─────────────┐  JSONL over stdio   ┌──────────────────────────┐
│ Python      │◄──────────────────►│ Node bridge (bridge.mjs) │
│ gymnasium   │  reset/step        │  real Match engine,      │
│ env + SB3   │  obs[128] raw /    │  tick-for-tick           │
│ PPO         │  516 stacked       │                          │
└─────────────┘                    └──────────────────────────┘
```

- **No sim port, no engine changes.** The bridge drives the actual TypeScript
  `Match` engine tick-for-tick. A bridge-controlled robot returns the pending
  intent queued by Python; the opponent is a scripted registry bot or an
  exported policy snapshot running its MLP in plain TS.
- **Obs:** 128-dim float vector raw per tick (self state, 2 nearest foes, stale
  tracks, 6 nearest bullets, walls, safe zone, turrets, pickup pads, match
  state, last-tick events), 4× frame-stacked plus a 4-dim strategy one-hot =
  516 policy obs (see `rl/env.py` `FRAME_DIM`/`FRAME_STACK`/`OBS_DIM` and
  `bridge/OBS_SPEC.md`).
- **Actions:** `MultiDiscrete([5,5,5,2,2,2,2])` → throttle/turn/towerTurn in
  5 bins (−1…1) + fire/charge/dash/emp booleans.
- **Reward:** dense damage deltas (±1 per 100 HP) + kill/death (±3) +
  pickup (+0.3) + turret capture (+1) + per-tick urgency (−0.002) +
  terminal win/loss (±15). Configurable in `rl/reward.py`.

## Branches

Default branch `main` is the 124-dim legacy line; live work is on
`pluto-upgrade` (128 raw / 516 stacked, per `rl/env.py`). This README describes
the `pluto-upgrade` line. Start with `docs/UPGRADE_SPEC.md` for the design and
the train/test checklist, and verify with `pytest tests/ -q`.

## RL best practices baked in

- **PPO first** (on-policy, stable, forgiving), SAC on the roadmap for sample
  efficiency. Defaults: LR 3e-4 with linear decay, γ=0.995, λ=0.95,
  clip 0.2, entropy bonus 0.01, gradient norm clip 0.5, orthogonal init.
- **League self-play** — 30% scripted roster, 40% latest snapshot, 30% past
  snapshots — so the policy can't overfit to one opponent.
- **VecNormalize** on observations and rewards (reward clip 10).
- **Truncation vs termination:** draws at the tick cap are truncations, so PPO
  bootstraps the value instead of learning "draw = death".
- **Win-rate evals** against fixed scripted bots every N rollouts, plus Elo.

## Prerequisites

The bridge bundles the TypeScript game engine, so you need the sibling
`robotarena` checkout with `esbuild` installed:

```sh
git clone https://github.com/schererstefan/robotarena ~/dev/robotarena  # skip if you already have it
cd ~/dev/robotarena && npm install     # provides node_modules/.bin/esbuild
export ROBOTARENA_PATH=~/dev/robotarena   # default is /Users/stefan/dev/robotarena
```

Then create the Python venv (CI runs Python 3.12; 3.14 works locally but is
not the CI-tested version):

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Quickstart (Mac mini)

```sh
./bridge/build.sh            # bundles the TS bridge (needs ROBOTARENA_PATH)
./scripts/smoke.sh           # random-policy sanity check
pytest tests/ -q             # pure tests (no bridge needed)
./scripts/train.sh --timesteps 300000   # short run
.venv/bin/tensorboard --logdir runs/<run>/tb
```

Full run: `./scripts/train.sh` (10M steps per `configs/ppo_1v1.yaml`).
Resume: `./scripts/train.sh --resume runs/<run> --timesteps 20000000`.

Evaluate: `.venv/bin/python -m rl.evaluate --model runs/<run>/model.zip`
Export a browser-ready robot: `.venv/bin/python -m rl.export_ts --model runs/<run>/model.zip --out export/rl_bot`
(copy `export/rl_bot.ts` into the game's `src/robots/` and register it).

## Environment variables

| Variable | Default | Consumed by |
|---|---|---|
| `ROBOTARENA_PATH` | `/Users/stefan/dev/robotarena` | `bridge/build.sh` (esbuild path), `rl/export_ts.py` |
| `ROBOTARENA_BRIDGE_RPC_TIMEOUT` | `30.0` s | `rl/env.py` `BRIDGE_RPC_TIMEOUT` |
| `ROBOTARENA_BRIDGE_SHUTDOWN_TIMEOUT` | `2.0` s | `rl/env.py` `BRIDGE_SHUTDOWN_TIMEOUT` |
| `ROBOTARENA_BRIDGE_TERMINATE_TIMEOUT` | `3.0` s | `rl/env.py` `BRIDGE_TERMINATE_TIMEOUT` |
| `CAMP_DIR` | `runs/campaign` | `scripts/campaign.sh` (`CAMP`) |
| `CONFIG` | `configs/ppo_continue.yaml` | `scripts/campaign.sh` (train config) |

`ROBOTARENA_PATH` points at the game checkout; rebuild the bridge if it moves.

## Layout

- `bridge/` — TS stdio bridge (`bridge.template.ts` → `dist/bridge.mjs`), obs spec
- `rl/env.py` — `RobotArenaEnv` gymnasium env (one Node subprocess each)
- `rl/train.py` — PPO + league + checkpoints + win-rate evals
- `rl/league.py` — opponent pool (scripted roster + policy snapshots)
- `rl/reward.py` — reward shaping weights
- `rl/evaluate.py` — roster eval + Elo
- `rl/export_ts.py` — policy → weights JSON + drop-in TS robot (numerically verified)
- `configs/ppo_1v1.yaml` — all hyperparameters
- `scripts/` — smoke / train entry points

## Roadmap

- SAC (off-policy, more sample-efficient) alongside PPO
- 2v2/3v3 team play (shared policy + radio channel)
- Imitation pretraining from match replays
- Full league with Nash-weighted opponent sampling
- Hyperparameter sweeps (Optuna)
