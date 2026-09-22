"""Print first/last values of key TB scalars for a run."""
import glob
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo_smoke_20260922-101402"
files = glob.glob(f"{run}/tb/ppo_1/*")
print("event files:", files)
ea = EventAccumulator(files[0])
ea.Reload()
for tag in ["rollout/ep_rew_mean", "rollout/ep_len_mean",
            "train/entropy_loss", "train/explained_variance",
            "eval/win_rate_wanderer"]:
    try:
        ev = ea.Scalars(tag)
        vals = [round(e.value, 3) for e in ev]
        print(f"{tag} n={len(vals)}")
        print(f"  first5: {vals[:5]}")
        print(f"  last5 : {vals[-5:]}")
    except KeyError:
        print(f"{tag}: MISSING")
