"""Probe a trained policy's behavior: action frequencies, movement, damage."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl.env import RobotArenaEnv

model_path = sys.argv[1]
vecnorm_path = sys.argv[2] if len(sys.argv) > 2 else None

model = PPO.load(model_path)
env = RobotArenaEnv(opponent_sampler=lambda rng: {"bot": "wanderer"},
                     arena="open", loadout={})
venv = DummyVecEnv([lambda: env])
if vecnorm_path:
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False

names = ["throttle", "turn", "towerTurn", "fire", "charge", "dash", "emp"]
counts = {n: {} for n in names}
obs = venv.reset()
done = False
steps = 0
while not done and steps < 12000:
    action, _ = model.predict(obs, deterministic=True)
    a = action[0]
    for n, v in zip(names, a):
        counts[n][int(v)] = counts[n].get(int(v), 0) + 1
    obs, reward, done, infos = venv.step(action)
    steps += 1

print(f"steps={steps} outcome={infos[0].get('reward/outcome')}")
for n in names:
    total = sum(counts[n].values())
    dist = {k: round(v / total, 3) for k, v in sorted(counts[n].items())}
    print(f"{n:>10}: {dist}")
venv.close()
