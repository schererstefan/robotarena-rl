"""Evaluate a trained model vs the scripted roster; report win rates + Elo.

Usage:
    python -m rl.evaluate --model runs/<run>/model.zip --bots wanderer rusher hunter
    python -m rl.evaluate --model runs/<run>/model.zip   # full roster
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .env import RobotArenaEnv
from .league import SCRIPTED_ROSTER

BASE_ELO = 1200.0
K = 32.0


def elo_update(ra: float, rb: float, score_a: float) -> tuple[float, float]:
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    ra2 = ra + K * (score_a - ea)
    rb2 = rb + K * ((1 - score_a) - (1 - ea))
    return ra2, rb2


DEFAULT_EVAL_LOADOUT = {"overdrive": 2, "trigger": 2, "plating": 2}


def evaluate(model_path: str, vecnorm_path: str | None, bots: list[str],
             episodes: int, arena: str, loadout: dict,
             deterministic: bool = True) -> dict:
    model = PPO.load(model_path)
    results: dict = {"model": model_path, "bots": {}}
    agent_elo = BASE_ELO
    for bot in bots:
        wins = losses = draws = 0
        ep_rewards: list[float] = []
        ep_ticks: list[int] = []
        bot_elo = BASE_ELO
        for ep in range(episodes):
            env = RobotArenaEnv(
                opponent_sampler=lambda rng, b=bot: {"bot": b},
                arena=arena, loadout=loadout)
            venv = DummyVecEnv([lambda: env])
            if vecnorm_path and os.path.exists(vecnorm_path):
                venv = VecNormalize.load(vecnorm_path, venv)
                venv.training = False
                venv.norm_reward = False
            obs = venv.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, infos = venv.step(action)
                ep_reward += float(np.asarray(reward).flat[0])
                if done:
                    # A time-cap ("none") is a draw, not a KeyError.
                    outcome = infos[0].get("reward/outcome", "none")
                    if outcome == "none":
                        outcome = "draw"
                    score = {"win": 1.0, "draw": 0.5, "loss": 0.0}[outcome]
                    agent_elo, bot_elo = elo_update(agent_elo, bot_elo, score)
                    wins += outcome == "win"
                    losses += outcome == "loss"
                    draws += outcome == "draw"
                    ep_rewards.append(ep_reward)
                    ep_ticks.append(int(infos[0].get("tick", 0)))
            venv.close()
        n = episodes
        results["bots"][bot] = {
            "win_rate": wins / n, "loss_rate": losses / n, "draw_rate": draws / n,
            "mean_ep_reward": float(np.mean(ep_rewards)),
            "mean_ep_ticks": float(np.mean(ep_ticks)),
            "elo": round(bot_elo, 1),
        }
        print(f"{bot:>12}: W{wins:>3} L{losses:>3} D{draws:>3} "
              f"win={wins / n:5.2f} elo={bot_elo:7.1f}", flush=True)
    results["agent_elo"] = round(agent_elo, 1)
    print(f"agent elo: {agent_elo:.1f} (pool baseline {BASE_ELO})", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnormalize", default=None)
    ap.add_argument("--bots", nargs="*", default=list(SCRIPTED_ROSTER))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--arena", default="open")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    vnp = args.vecnormalize
    if vnp is None:
        cand = os.path.join(os.path.dirname(args.model), "vecnormalize.pkl")
        vnp = cand if os.path.exists(cand) else None
    results = evaluate(args.model, vnp, args.bots, args.episodes,
                       args.arena, DEFAULT_EVAL_LOADOUT)
    out = args.out or os.path.join(os.path.dirname(args.model), "eval_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
