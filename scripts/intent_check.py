"""Verify learner intents actually move the robot (intent plumbing check)."""
import json
import subprocess

BRIDGE = "/Users/stefan/dev/robotarena-rl/bridge/dist/bridge.mjs"
NODE = "/opt/homebrew/bin/node"


def run_match(intent_fn, steps=600, seed=42):
    p = subprocess.Popen([NODE, BRIDGE], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, bufsize=1)

    def rpc(m):
        p.stdin.write(json.dumps(m) + "\n")
        p.stdin.flush()
        return json.loads(p.stdout.readline())

    r = rpc({"cmd": "reset", "seed": seed, "opponent": {"bot": "wanderer"},
             "arena": "open", "loadout": {}})
    o0 = r["obs"]
    last = None
    for t in range(steps):
        last = rpc({"cmd": "step", "intent": intent_fn(t)})
        if last["terminated"] or last["truncated"]:
            break
    x0, y0 = o0[0] * 960, o0[1] * 640
    x1, y1 = last["obs"][0] * 960, last["obs"][1] * 640
    done = last["terminated"] or last["truncated"]
    rpc({"cmd": "shutdown"})
    p.wait()
    return (x0, y0, x1, y1, done)


def fwd(t):
    return {"throttle": 1, "turn": 0, "towerTurn": 0, "fire": False,
            "charge": False, "dash": False, "emp": False}


def rev(t):
    return {"throttle": -1, "turn": 0, "towerTurn": 0, "fire": False,
            "charge": False, "dash": False, "emp": False}


def idle(t):
    return {"throttle": 0, "turn": 0, "towerTurn": 0, "fire": False,
            "charge": False, "dash": False, "emp": False}


for name, fn in [("fwd", fwd), ("rev", rev), ("idle", idle)]:
    x0, y0, x1, y1, done = run_match(fn)
    moved = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    print(f"{name}: start=({x0:.0f},{y0:.0f}) end=({x1:.0f},{y1:.0f}) "
          f"moved={moved:.0f} done={done}")
