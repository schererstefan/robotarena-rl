"""RobotArena RL harness: PPO + league self-play on the RobotArena sim."""
from .env import RobotArenaEnv, ACTION_DIMS, OBS_DIM

__all__ = ["RobotArenaEnv", "ACTION_DIMS", "OBS_DIM"]
