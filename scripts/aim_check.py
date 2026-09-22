"""Sanity-check aimErr from the bridge (foe visibility + aiming error)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rl.env import RobotArenaEnv

env = RobotArenaEnv(opponent_sampler=lambda rng: {"bot": "wanderer"})
env.reset(seed=0)
errs = []
for t in range(300):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    errs.append(info["reward/aimErr"])
    if term or trunc:
        env.reset()
env.close()
errs = np.array(errs)
vis = errs[errs >= 0]
print("frac visible: %.3f" % (errs >= 0).mean())
if len(vis):
    print("aimErr visible: mean=%.3f min=%.3f max=%.3f" % (vis.mean(), vis.min(), vis.max()))
else:
    print("WARNING: foe never visible in 300 random steps")
