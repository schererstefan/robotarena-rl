"""Regression guard for dodge obs-index fix (pure Python, no bridge)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rl.env import dodge_perp, SELF_SPEED, BULLET_BASE, TICK, FOE_PRESENT


def make_frame():
    frame = np.zeros(128, dtype=np.float32)
    frame[BULLET_BASE] = 1.0
    frame[BULLET_BASE + 1] = 100.0 / 960.0
    frame[BULLET_BASE + 2] = 0.0
    frame[BULLET_BASE + 3] = 100.0 / 540.0
    frame[BULLET_BASE + 4] = 200.0 / 430.0
    frame[BULLET_BASE + 5] = 0.0
    return frame


assert SELF_SPEED == 6, f"SELF_SPEED={SELF_SPEED}"
assert FOE_PRESENT == 17, f"FOE_PRESENT={FOE_PRESENT}"
assert BULLET_BASE == 38, f"BULLET_BASE={BULLET_BASE}"
assert TICK == 121, f"TICK={TICK}"

# Case A: stationary + tower +x, heading +y. Must be ~0.
frame_a = make_frame()
frame_a[2] = 1.0
frame_a[3] = 0.0
frame_a[4] = 0.0
frame_a[5] = 1.0
frame_a[SELF_SPEED] = 0.0
val_a = dodge_perp(frame_a)
print(f"case A (stationary): {val_a}")
assert val_a < 1e-6, f"case A should be ~0, got {val_a}"

# Case B: genuine perpendicular motion. Expected 150/375 = 0.4.
frame_b = make_frame()
frame_b[2] = 1.0
frame_b[3] = 0.0
frame_b[5] = 0.0
frame_b[SELF_SPEED] = 150.0 / 375.0
val_b = dodge_perp(frame_b)
print(f"case B (perp motion): {val_b}")
assert val_b > 0.1, f"case B should be >0.1, got {val_b}"
assert abs(val_b - 0.4) < 1e-3, f"case B should be ~0.4, got {val_b}"

print("DODGE CHECK OK")
