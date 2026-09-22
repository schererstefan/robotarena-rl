#!/usr/bin/env bash
# Smoke test: build bridge, run a random policy through the env.
set -euo pipefail
cd "$(dirname "$0")/.."

./bridge/build.sh

.venv/bin/python - <<'EOF'
import numpy as np
from rl.env import RobotArenaEnv

env = RobotArenaEnv(opponent_sampler=lambda rng: {"bot": "wanderer"})
obs, info = env.reset(seed=0)
assert obs.shape == (124,), obs.shape
assert obs.dtype == np.float32
rewards, terms, truncs = [], 0, 0
ep = 0
obs, _ = env.reset(seed=1)
for t in range(2000):
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    assert np.isfinite(obs).all() and np.isfinite(r), f"NaN at step {t}"
    rewards.append(r)
    if term or trunc:
        terms += term; truncs += trunc; ep += 1
        obs, _ = env.reset()
print(f"steps=2000 episodes={ep} terminated={terms} truncated={truncs} "
      f"reward_mean={np.mean(rewards):.3f} min={np.min(rewards):.3f} max={np.max(rewards):.3f}")
assert ep >= 1, "no episode finished in 2000 steps"
assert not all(r == 0 for r in rewards), "rewards are all zero"
env.close()
print("SMOKE OK")
EOF
