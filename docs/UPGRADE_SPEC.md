# Pluto-Inspired RobotArena Upgrade — Spec (v1)

Date: 2026-09-26. Branch: `pluto-upgrade` (robotarena-rl, from local main).
Status: approved by Stefan ("run the full Pluto-inspired upgrade").

## Background

Pluto (tscmoo, StarCraft: Brood War AI, CoG 2026 winner) is a single 315M-param
net trained by pure self-play PPO: memory lives **in the net** (GRU + slow
transformer cache, no hand-coded trackers), dense staged rewards + small
diversity bonus, and strategy selection kept **outside the net** (per-opponent
bandit over build orders, fed back as a conditioning vector).

RobotArena's diagnosed gaps: 105K-param feedforward MLP, single-frame obs (the
net cannot infer foe velocity — the `leadFeat` bridge crutch exists because of
this), ghost/brawler blockers (fast strafers), pure self-play collapsed
0.82 → 0.37 without the 70% scripted floor.

## Design decisions

### 1. Memory: frame stacking (K=4), not LSTM — documented tradeoff

Pluto's headline lesson is recurrence in the net. A full LSTM port
(sb3-contrib RecurrentPPO) would touch 6+ files with subtle correctness risk:
custom AnchorKLRecurrentPPO, hidden-state handling in eval/rollout code, LSTM
forward + state-reset semantics in the TS export template, the bridge policy
opponent, and the 3-way numeric verification. Frame stacking captures the
diagnosed gap (foe-velocity inference from position deltas — exactly what
ghost/brawler exploit) while keeping every hard-won invariant: the verified
MLP export pipeline, the KL anchor, the snapshot admission gate, and
tick-for-tick determinism. LSTM remains the documented follow-up once the
stacked model plateaus.

- `FRAME_DIM = 128` (raw bridge obs, unchanged; handshake untouched).
- `FRAME_STACK = 4`, `N_STRATEGIES = 4`.
- `OBS_DIM = 4*128 + 4 = 516` (RL-facing dim; stacked frames + unstacked
  strategy one-hot).
- Env pre-fills the deque with the first obs (standard); TS consumers
  (export template, bridge policy controller) keep a 4-frame buffer in the
  controller closure with the same pre-fill rule.

### 2. Scale: ~2.1M params (~20x)

MLP `[1024, 1024, 512]` tanh, orthogonal init (unchanged init scheme).
516→1024→1024→512→(23 logits + value). Browser/Node inference cost is
negligible at this size (Pluto runs 315M int8 at 20–60 ms/step; 2.1M float32
MLP is <1 ms).

### 3. Strategy conditioning + per-opponent bandit (Pluto lesson #4, faithful)

- 4 strategy labels: `rush`, `kite`, `greedy`, `control`. The label is a pure
  conditioning signal — the net discovers the behavior (Pluto: "labels may not
  match behavior"). Soft grounding only (see §4).
- `rl/bandit.py`: per-opponent UCB1 over arms, persisted as human-readable
  JSON (`runs/<run>/bandit/strategy_bandit.json`, Pluto-style: per-arm
  wins/losses, `seq` line to let hand-edits win).
- Worker envs select arms via UCB1, reading the shared JSON fresh each episode
  (same live-read pattern as the existing PFSP file — no IPC, no locks;
  atomic `os.replace` on write).
- A main-process `BanditCallback` aggregates per-episode (opponent, arm, won)
  reports from infos and rewrites the JSON.
- Eval (`WinRateEvalCallback`, `eval_roster.py --bandit`) uses best-arm
  (argmax win rate) per opponent — this measures the deployed system.
- Export (`export_ts.py --bandit`) bakes the overall-best arm as default;
  the generated robot exposes `setStrategyArm(i)` for override.

### 4. Reward: diversity bonus + opening nudge (Pluto lesson #3)

- Keep all existing shaping weights; terminal win/loss stays dominant late.
- `diversity` (new, default 0.0; 0.05 in new configs): one-time per-episode
  bonus on first use of each of dash/emp/charge. Small anti-mode-collapse
  nudge, computed env-side from actions (no bridge changes).
- `opening` (new, default 0.0; 0.002 in new configs): for the first 600 ticks
  only, tiny bonus for moving the arm's reference distance in the intended
  direction (rush→close foe, kite→open distance, greedy→approach nearest
  active pad, control→approach nearest active turret/center). Computed
  env-side from raw obs, guarded by present/active flags. Pluto-faithful
  "soft nudge, not enforced".

### 5. What stays (deliberately NOT copied from Pluto)

- 70% permanent scripted floor + PFSP: RobotArena's own experiment showed
  pure self-play collapses (0.82→0.37) at this scale. The floor is load-bearing
  bootstrap; it fades only when capacity proves it can.
- Snapshot admission gate, KL anchor, rollback gate, min-of-6 champion
  scoring, tick-for-tick bridge (latency matched by construction — Pluto's
  principle, already satisfied).
- `leadFeat` crutch and stale-track dims stay for now (bootstrap); ablation
  is a follow-up once the stacked model is healthy.

## File changes (robotarena-rl, branch pluto-upgrade)

| File | Change |
|---|---|
| `rl/bandit.py` | NEW: StrategyBandit (UCB1), BanditReader, best_arm, atomic JSON |
| `rl/env.py` | Frame stack K=4, strategy arm (ucb/best/fixed), diversity + opening rewards, bandit info reporting; OBS_DIM=516 |
| `rl/reward.py` | `diversity`, `opening` weights (default 0.0) |
| `rl/train.py` | BanditCallback; wire bandit into make_env_fn; eval uses best-arm |
| `rl/league.py` | Policy-opponent spec gains uniform-random `strategy` arm |
| `rl/export_ts.py` | Template: frame buffer + strategy one-hot + setStrategyArm; --bandit flag; 516-dim realistic obs |
| `bridge/bridge.template.ts` | makePolicyController: frame buffer + strategy arm (uniform random per match unless specified) |
| `scripts/eval_roster.py` | --bandit flag (best-arm per opponent) |
| `scripts/smoke.sh` | obs shape (516,) |
| `scripts/campaign.sh` | carry bandit JSON across iterations; strict_eval passes --bandit |
| `configs/ppo_pluto_smoke.yaml` | NEW: 1.5M steps, [1024,1024,512], wanderer+rusher, no anchor |
| `configs/ppo_pluto_campaign.yaml` | NEW: 8M steps, league-v2 + pluto upgrades, anchor on |
| `scripts/pluto_pipeline.sh` | NEW: smoke → (on success) campaign_pluto |
| `docs/UPGRADE_SPEC.md` | this file (repo copy) |

## Training plan

1. Smoke: 1.5M steps on `ppo_pluto_smoke.yaml` in tmux (`pluto-smoke`).
   Validate: loss decreases, eval win rates move, export 3-way verify passes.
2. Pipeline auto-continues: `campaign.sh` seeded from smoke run's
   `best_roster.zip`, `CAMP_DIR=runs/campaign_pluto`,
   `CONFIG=configs/ppo_pluto_campaign.yaml`, 50 iterations max, patience 10.
3. Ship rule unchanged: export to game repo + merge only on Stefan's approval.

## Follow-ups (not in v1)

- LSTM/GRU policy (sb3-contrib) if frame-stacked model plateaus vs
  ghost/brawler.
- Ablate `leadFeat` + stale-track once recurrence/stack proves sufficient.
- Richer strategy set; opening-alignment shaping with bridge-side components.
- Nash-weighted league (research notes roadmap).
