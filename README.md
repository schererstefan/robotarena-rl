# robotarena-rl

Deep reinforcement learning for RobotArena
(2D pixel-art robot battles): PPO with league self-play, trained against the
real game sim running headless.

## Architecture

```
┌─────────────┐  JSONL over stdio   ┌──────────────────────────┐
│ Python      │◄──────────────────►│ Node bridge (bridge.mjs) │
│ gymnasium   │  reset/step        │  real Match engine,      │
│ env + SB3   │  obs[124]          │  tick-for-tick           │
│ PPO         │  reward            │                          │
└─────────────┘                    └──────────────────────────┘
```

- **No sim port, no engine changes.** The bridge drives the actual TypeScript
  `Match` engine tick-for-tick. A bridge-controlled robot returns the pending
  intent queued by Python; the opponent is a scripted registry bot or an
  exported policy snapshot running its MLP in plain TS.
- **Obs:** 124-dim float vector — self state, 2 nearest foes, stale tracks,
  6 nearest bullets, walls, safe zone, turrets, pickup pads, match state,
  last-tick events. See `bridge/OBS_SPEC.md`.
- **Actions:** `MultiDiscrete([5,5,5,2,2,2,2])` → throttle/turn/towerTurn in
  5 bins (−1…1) + fire/charge/dash/emp booleans.
- **Reward:** dense damage deltas (±1 per 100 HP) + kill/death (±3) +
  pickup (+0.3) + turret capture (+1) + per-tick urgency (−0.002) +
  terminal win/loss (±15). Configurable in `rl/reward.py`.

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

## Quickstart (Mac mini)

```sh
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
./bridge/build.sh            # bundles the TS bridge (needs ROBOTARENA_PATH)
./scripts/smoke.sh           # random-policy sanity check
./scripts/train.sh --timesteps 300000   # short run
tensorboard --logdir runs/<run>/tb
```

Full run: `./scripts/train.sh` (10M steps per `configs/ppo_1v1.yaml`).
Resume: `./scripts/train.sh --resume runs/<run> --timesteps 20000000`.

Evaluate: `.venv/bin/python -m rl.evaluate --model runs/<run>/model.zip`
Export a browser-ready robot: `.venv/bin/python -m rl.export_ts --model runs/<run>/model.zip --out export/rl_bot`
(copy `export/rl_bot.ts` into the game's `src/robots/` and register it).

`ROBOTARENA_PATH` env var points at the game checkout (default
`/Users/stefan/dev/robotarena`); rebuild the bridge if it moves.

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
