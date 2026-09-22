"""Check if a policy ever deals damage (exploration diagnosis)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl.env import RobotArenaEnv

model_path = sys.argv[1]
vecnorm_path = sys.argv[2]

model = PPO.load(model_path)
dealt_any = 0
dealt_total = 0.0
for ep in range(10):
    env = RobotArenaEnv(opponent_sampler=lambda rng: {"bot": "wanderer"},
                        arena="open", loadout={})
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False
    obs = venv.reset()
    done = False
    ep_dealt = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = venv.step(action)
        ep_dealt += infos[0].get("reward/dealt", 0.0)
    dealt_total += ep_dealt
    dealt_any += 1 if ep_dealt > 0.5 else 0
    print(f"ep {ep}: dealt={ep_dealt:.1f} outcome={infos[0].get('reward/outcome')}")
    venv.close()
print(f"episodes with damage>0.5: {dealt_any}/10, total dealt={dealt_total:.1f}")
