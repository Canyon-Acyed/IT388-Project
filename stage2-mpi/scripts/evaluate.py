"""Success rate of a trained controller over many random start states.

OUR CODE (whole file). It reuses replay.py's controller loader and
observation function, which follow Gymnasium's _get_obs() and
reset_model() [GYM] (MIT License, https://github.com/Farama-Foundation/Gymnasium).
Start states use NumPy's generator, so they differ from the 8 fixed
evaluation states used during training: this is a test on unseen starts.

    python3 scripts/evaluate.py models/trained/double_mlp32.theta
    python3 scripts/evaluate.py my.theta --episodes 500
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay import get_obs, load_controller  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("controller")
    ap.add_argument("--model", default="models/inverted_double_pendulum.xml")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--noise", type=float, default=0.1, help="start-state noise ([GYM] reset_noise_scale)")
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(a.model)
    d = mujoco.MjData(m)
    act, _ = load_controller(a.controller)
    tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
    lengths, returns = [], []
    for ep in range(a.episodes):
        rng = np.random.default_rng(ep)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = m.qpos0 + rng.uniform(-a.noise, a.noise, m.nq)       # [GYM] reset_model
        d.qvel[:] = a.noise * rng.standard_normal(m.nv)
        mujoco.mj_forward(m, d)
        t, total = 0, 0.0
        while t < 1000:
            d.ctrl[0] = np.clip(act(get_obs(d)), -1, 1)
            for _ in range(5):
                mujoco.mj_step(m, d)
            t += 1
            x, _, y = d.site_xpos[tip]                                    # [GYM] step()
            fell = y <= 1
            v = d.qvel[1:]
            w = np.array([1e-3] + [5e-3] * (len(v) - 1))                  # [GYM] _get_rew()
            total += (0 if fell else 10) - (0.01 * x ** 2 + (y - 2) ** 2) - float(w @ v ** 2)
            if fell:
                break
        lengths.append(t)
        returns.append(total)
    ok = sum(t == 1000 for t in lengths)
    print(f"{a.controller}: balanced {ok}/{a.episodes} episodes for all 1000 steps "
          f"({100 * ok / a.episodes:.0f}%), mean return {np.mean(returns):.1f}")
    failed = sorted(t for t in lengths if t < 1000)
    if failed:
        print("steps before falling, failed episodes:", failed)


if __name__ == "__main__":
    main()
