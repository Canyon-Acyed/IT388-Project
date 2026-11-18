"""Classical-control reference: can an LQR controller balance the model?

LQR (linear-quadratic regulator) is the optimal linear controller for the
pendulum linearized around upright. Unlike ES it needs the physics equations.
We use it to check that a model is balanceable at all, and how the start-state
noise affects that. This is how we chose 0.05 noise for the triple pendulum.

SOURCES
  [MJLQR] MuJoCo's official LQR tutorial, python/LQR.ipynb (Google DeepMind,
          Apache 2.0): https://github.com/google-deepmind/mujoco/blob/main/python/LQR.ipynb
          Copied pattern:
              P = scipy.linalg.solve_discrete_are(A, B, Q, R)
              K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
          The tutorial gets A and B from mujoco.mjd_transitionFD(...). That
          function does not support the RK4 integrator our models use
          ("mjd_transitionFD: RK4 integrator is not supported"), so OUR CODE
          computes A and B the same way by hand: forward finite differences over
          one control step (5 physics steps).
  [GYM]   Gymnasium inverted_double_pendulum_v5.py (MIT): start-state noise,
          frame_skip = 5, termination when the tip height <= 1.
  SciPy (BSD): scipy.linalg.solve_discrete_are.
OUR CODE: everything else, including the Q and R weights.

    python3 scripts/lqr_baseline.py models/inverted_double_pendulum.xml
    python3 scripts/lqr_baseline.py models/inverted_triple_pendulum.xml --noise 0.1
"""
import argparse

import mujoco
import numpy as np
import scipy.linalg as sl

FRAME_SKIP = 5  # [GYM]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--noise", type=float, help="start-state noise (default: the model's own setting)")
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(a.model)
    d = mujoco.MjData(m)
    nx = 2 * m.nv
    noise = a.noise
    if noise is None:  # same rule as src/env.c: the model may set reset_noise_scale
        nid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "reset_noise_scale")
        noise = m.numeric_data[m.numeric_adr[nid]] if nid >= 0 else 0.1

    # OUR CODE: linearize one control step around upright by finite differences
    # (replaces mjd_transitionFD from [MJLQR], see above).
    def step_from(x, u):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[0] = x[:m.nq], x[m.nq:], u
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(m, d)
        return np.concatenate([d.qpos, d.qvel])

    x0, eps = np.zeros(nx), 1e-6
    f0 = step_from(x0, 0.0)
    A = np.stack([(step_from(x0 + eps * np.eye(nx)[i], 0.0) - f0) / eps for i in range(nx)], axis=1)
    B = ((step_from(x0, eps) - f0) / eps)[:, None]

    # OUR CODE: cost weights (cart 1, hinge angles 10, velocities 0.1, push 0.1).
    Q = np.diag([1.0] + [10.0] * (m.nv - 1) + [0.1] * m.nv)
    R = np.eye(1) * 0.1
    # SOURCE [MJLQR]: discrete Riccati equation and feedback gain.
    P = sl.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    # OUR CODE: run the controller u = -K x from random starts ([GYM] reset rule).
    tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
    lengths = []
    for ep in range(a.episodes):
        rng = np.random.default_rng(ep)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = rng.uniform(-noise, noise, m.nq)
        d.qvel[:] = noise * rng.standard_normal(m.nv)
        mujoco.mj_forward(m, d)
        t = 0
        while t < 1000:
            d.ctrl[0] = np.clip(-(K @ np.concatenate([d.qpos, d.qvel]))[0], -1, 1)
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(m, d)
            t += 1
            if d.site_xpos[tip][2] <= 1.0:  # [GYM] termination
                break
        lengths.append(t)
    ok = sum(t == 1000 for t in lengths)
    print(f"{a.model}, start noise {noise}: LQR balanced {ok}/{a.episodes} episodes for all 1000 steps")
    print("episode lengths:", lengths)


if __name__ == "__main__":
    main()
