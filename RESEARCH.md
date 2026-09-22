# RL Research Notes — why this harness looks the way it does

Surveyed the canonical actor-critic / deep-RL starter repos before building.
Decision for v1: **Stable-Baselines3 + Gymnasium, PPO first**.

## Repos surveyed

- **Stable-Baselines3** — https://github.com/DLR-RM/stable-baselines3
  Reliable, well-tested PyTorch implementations (PPO, SAC, TD3, DQN…) with a
  uniform API, VecEnv/VecNormalize utilities, and TensorBoard logging. JMLR
  paper: https://jmlr.org/papers/volume22/20-1364/20-1364.pdf. The reference
  choice when you want an algorithm that "just works" rather than a research
  prototype.
- **CleanRL** — https://github.com/vwxyzjn/cleanrl
  Single-file, readable implementations (one file per algorithm) with
  experiment tracking. Excellent for learning how PPO actually works and for
  hacking custom variants, but you own every bug — no test suite like SB3's.
- **Gymnasium** — https://github.com/Farama-Foundation/Gymnasium
  The standard RL env API (successor to OpenAI Gym). Our env subclasses
  `gymnasium.Env`; SB3 consumes it directly.
- **PettingZoo** — https://github.com/Farama-Foundation/PettingZoo
  Multi-agent API (AEC / parallel). Relevant if we ever train both sides
  simultaneously with a shared league; overkill for v1 where the opponent is
  part of the environment.
- **RLlib** — https://github.com/ray-project/ray (ray[rllib])
  Scales to clusters and has strong multi-agent + self-play support
  (AlphaStar-style leagues). Much heavier operationally than SB3; justified
  when we outgrow a single Mac mini, not before.
- **Unity ML-Agents** — https://github.com/Unity-Technologies/ml-agents
  Ties training to the Unity editor. Wrong fit: our sim is a TypeScript web
  game, not Unity.
- **OpenAI Spinning Up** — https://github.com/openai/spinningup (archived)
  Educational implementations with clear explanations. Superseded by CleanRL
  as a learning resource; not a training framework.

## Why SB3 + Gymnasium for v1

1. Correctness first: SB3's PPO is benchmarked against reference
   implementations; we debug our env/reward, not the algorithm.
2. `VecNormalize` + `SubprocVecEnv` solve the two most common beginner
   failures (unscaled obs, slow sampling) out of the box.
3. The Gymnasium API keeps the door open to swap in CleanRL or RLlib later
   without rewriting the bridge.

## Why PPO before SAC

- Our action space is **hybrid** (3 × 5-bin discrete + 4 × binary). SB3's SAC
  needs a continuous (or flattened discrete) space; PPO handles
  `MultiDiscrete` natively.
- Episodes are long (~1k ticks) with delayed outcomes; PPO's clipped
  on-policy updates with GAE are more stable than off-policy Q-learning here.
- SAC becomes interesting later for sample efficiency once the policy can
  already aim and hit (e.g. fine-tuning a PPO warm start).

## Key practices applied

- Orthogonal init + tanh MLP, linear LR decay, gradient clipping (SB3 PPO
  defaults, cf. CleanRL's `ppo.py` and the SB3 Zoo tuned hyperparams).
- Observation/reward normalization via `VecNormalize` — and the stats are
  **exported with the policy**, since a normalized-input network is garbage
  without them (easy to forget; we forgot it once).
- Reward shaping only as a bootstrap (aim assist), keeping the terminal
  win/loss bonus dominant so the agent optimizes victory, not the shaping.
- Deterministic eval vs a scripted roster + Elo, separate from training.
- Export verification in three directions: PyTorch ↔ NumPy ↔ Node.js
  (< 1e-4), because the shipped artifact runs in TypeScript.
