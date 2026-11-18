"""Check that our C environment matches Gymnasium's InvertedDoublePendulum-v5.

OUR CODE (whole file). It runs Gymnasium itself (Farama Foundation, MIT License,
https://github.com/Farama-Foundation/Gymnasium) as the reference: gym.make,
env.reset(seed=...), env.step(...) and env.unwrapped.data are Gymnasium's
standard API (https://gymnasium.farama.org/api/env/).

Runs the same start states and random actions through both, and reports the
largest difference in reward and observation. From the project folder:

    python3 scripts/check_env.py      (or: make check)
"""
import subprocess
import sys

import gymnasium as gym
import numpy as np

EPISODES = 20
STEPS = 300


def run_c(qpos, qvel, actions):
    text = f"{len(qpos)} {len(qvel)}\n"
    text += " ".join(repr(float(v)) for v in qpos) + "\n"
    text += " ".join(repr(float(v)) for v in qvel) + "\n"
    text += f"{len(actions)}\n" + " ".join(repr(float(a)) for a in actions) + "\n"
    out = subprocess.run(["./bin/env_test", "models/inverted_double_pendulum.xml"],
                         input=text, capture_output=True, text=True, check=True).stdout
    return np.array([list(map(float, line.split())) for line in out.strip().splitlines()])


def main():
    env = gym.make("InvertedDoublePendulum-v5")
    rng = np.random.default_rng(0)
    worst_r = worst_o = 0.0
    total_steps = 0
    for ep in range(EPISODES):
        env.reset(seed=ep)
        base = env.unwrapped
        qpos, qvel = base.data.qpos.copy(), base.data.qvel.copy()
        # Random actions, large on even episodes and small on odd ones, so episodes vary in length.
        actions = rng.uniform(-1, 1, STEPS) * (0.3 if ep % 2 else 1.0)

        g_rows = []
        for a in actions:
            obs, r, term, trunc, _ = env.step(np.array([a]))
            g_rows.append([r, float(term), *obs])
            if term:
                break
        g = np.array(g_rows)
        c = run_c(qpos, qvel, actions)
        if c.shape != g.shape:
            print(f"episode {ep}: length mismatch, C {c.shape[0]} steps vs Gymnasium {g.shape[0]}")
            sys.exit(1)
        worst_r = max(worst_r, np.abs(c[:, 0] - g[:, 0]).max())
        worst_o = max(worst_o, np.abs(c[:, 2:] - g[:, 2:]).max())
        total_steps += len(g)

    print(f"Compared {EPISODES} episodes, {total_steps} steps")
    print(f"Largest reward difference:      {worst_r:.3e}")
    print(f"Largest observation difference: {worst_o:.3e}")
    ok = worst_r < 1e-9 and worst_o < 1e-9
    print("MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
