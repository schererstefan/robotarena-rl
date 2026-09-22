"""Probe shaped policy: actions, distance to foe, aim quality."""
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
env = RobotArenaEnv(opponent_sampler=lambda rng: {"bot": "wanderer"},
                    arena="open", loadout={"overdrive": 2, "trigger": 2, "plating": 2})
venv = DummyVecEnv([lambda: env])
venv = VecNormalize.load(vecnorm_path, venv)
venv.training = False
venv.norm_reward = False

names = ["throttle", "turn", "towerTurn", "fire", "charge", "dash", "emp"]
counts = {n: {} for n in names}
obs = venv.reset()
done = False
steps = 0
dists = []
aims = []
while not done and steps < 12000:
    action, _ = model.predict(obs, deterministic=True)
    a = action[0]
    for n, v in zip(names, a):
        counts[n][int(v)] = counts[n].get(int(v), 0) + 1
    obs, reward, done, infos = venv.step(action)
    steps += 1
    # raw obs foe distance (dim 23) and aimErr from info
    raw = venv.venv.envs[0].bridge_obs if hasattr(venv.venv.envs[0], 'bridge_obs') else None
    aims.append(infos[0].get("reward/aimErr", -1))

print(f"steps={steps} outcome={infos[0].get('reward/outcome')}")
for n in names:
    total = sum(counts[n].values())
    dist = {k: round(v / total, 3) for k, v in sorted(counts[n].items())}
    print(f"{n:>10}: {dist}")
aims = np.array([a for a in aims if a >= 0])
print(f"aimErr: frac_visible={(np.array(aims) >= 0).mean():.3f}" if len(aims) else "never visible")
if len(aims):
    print(f"  mean={aims.mean():.3f} p50={np.median(aims):.3f}")
venv.close()
