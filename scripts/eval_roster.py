#!/usr/bin/env python3
"""Deterministic roster eval: N episodes per scripted bot, raw JSON out.

Mirrors the in-training WinRateEvalCallback (deterministic policy, same
VecNormalize handling) but as a standalone script so any model.zip can be
evaluated after the fact, with results persisted as JSON.

Usage:
    .venv/bin/python scripts/eval_roster.py runs/<run>/model.zip runs/<run>/vecnormalize.pkl \
        --bots wanderer rusher hunter --episodes 20 --out runs/<run>/eval_roster.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.env import RobotArenaEnv


def eval_bot(model, vecnorm_path, bot, episodes, arena, loadout):
    wins = losses = draws = 0
    for _ in range(episodes):
        env = RobotArenaEnv(
            opponent_sampler=lambda rng, b=bot: {"bot": b},
            arena=arena, loadout=loadout)
        venv = DummyVecEnv([lambda: env])
        if vecnorm_path:
            venv = VecNormalize.load(vecnorm_path, venv)
            venv.training = False
            venv.norm_reward = False
        obs = venv.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = venv.step(action)
            if done:
                outcome = infos[0].get("reward/outcome", "none")
                wins += outcome == "win"
                losses += outcome == "loss"
                draws += outcome == "draw"
        venv.close()
    return {"wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / episodes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to model.zip")
    ap.add_argument("vecnormalize", nargs="?", default=None,
                    help="path to vecnormalize.pkl (omit only for debugging)")
    ap.add_argument("--bots", nargs="+",
                    default=["wanderer", "rusher", "hunter"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--arena", default="open")
    ap.add_argument("--out", default=None, help="write raw JSON results here")
    args = ap.parse_args()

    model = PPO.load(args.model)
    results = {
        "model": os.path.abspath(args.model),
        "vecnormalize": (os.path.abspath(args.vecnormalize)
                         if args.vecnormalize else None),
        "bots": args.bots,
        "episodes_per_bot": args.episodes,
        "arena": args.arena,
        "deterministic": True,
        "ts": time.time(),
        "results": {},
    }
    for bot in args.bots:
        r = eval_bot(model, args.vecnormalize, bot, args.episodes,
                     args.arena, {})
        results["results"][bot] = r
        print(f"{bot}: W{r['wins']} L{r['losses']} D{r['draws']} "
              f"win_rate={r['win_rate']:.2f}", flush=True)
    agg = float(np.mean([results["results"][b]["win_rate"]
                         for b in args.bots]))
    total_w = sum(r["wins"] for r in results["results"].values())
    total_n = args.episodes * len(args.bots)
    results["aggregate_win_rate"] = agg
    results["total"] = {"wins": total_w, "episodes": total_n}
    print(f"aggregate: {agg:.3f} ({total_w}/{total_n})", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
