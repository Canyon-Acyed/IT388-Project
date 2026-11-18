"""Watch a trained controller balance the pendulum, or save it as a GIF.

SOURCES
  [MJPY]  MuJoCo Python bindings documentation (Apache 2.0):
          https://mujoco.readthedocs.io/en/stable/python.html
          - Passive viewer: "with mujoco.viewer.launch_passive(m, d) as viewer:
            ... mujoco.mj_step(m, d) ... viewer.sync()"
          - Rendering: mujoco.Renderer(model, height, width),
            renderer.update_scene(data, camera), renderer.render()
  [GYM]   Gymnasium inverted_double_pendulum_v5.py (MIT License): _get_obs()
          and reset_model(), re-written here in Python exactly as in src/env.c.
  [OAI]   OpenAI evolution-strategies-starter policies.py (MIT License):
          input normalization clip((ob - mean) / std, -5, 5).
  imageio (BSD License): imageio.mimsave writes the GIF.
OUR CODE: everything else (loading our saved controller format, the loop,
the --push test, camera settings).

    python3 scripts/replay.py models/trained/double_mlp32.theta
    MUJOCO_GL=egl python3 scripts/replay.py models/trained/double_mlp32.theta --gif double.gif

Without --gif it opens MuJoCo's viewer (needs a display). With --gif it renders
off-screen; with no display, set MUJOCO_GL=egl (or osmesa) first.
Needs: pip install mujoco numpy imageio
"""
import argparse
import time

import mujoco
import numpy as np


def load_controller(path):
    """OUR CODE: reads the file written by es_serial --save."""
    with open(path) as f:
        od, hidden, n, _count = f.readline().split()
        od, hidden, n = int(od), int(hidden), int(n)
        mu = np.array(f.readline().split(), float)
        sd = np.array(f.readline().split(), float)
        theta = np.array([float(f.readline()) for _ in range(n)])

    def act(obs):
        x = np.clip((obs - mu) / sd, -5, 5)              # [OAI] normalization
        if hidden == 0:
            return np.tanh(theta[:od] @ x + theta[od])
        W1 = theta[:hidden * od].reshape(hidden, od)     # same layout as src/policy.c
        b1 = theta[hidden * od:hidden * od + hidden]
        w2 = theta[hidden * od + hidden:hidden * od + 2 * hidden]
        b2 = theta[-1]
        return np.tanh(w2 @ np.tanh(W1 @ x + b1) + b2)
    return act, od


def get_obs(d):
    """SOURCE [GYM] _get_obs()."""
    return np.concatenate([d.qpos[:1], np.sin(d.qpos[1:]), np.cos(d.qpos[1:]),
                           np.clip(d.qvel, -10, 10), np.clip(d.qfrc_constraint, -10, 10)[:1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("controller")
    ap.add_argument("--model", default="models/inverted_double_pendulum.xml")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push", type=float, default=0.0,
                    help="kick the top pole every 200 steps (rad/s) to test recovery")
    ap.add_argument("--gif", help="save a GIF here instead of opening the viewer")
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(a.model)
    d = mujoco.MjData(m)
    act, od = load_controller(a.controller)
    if od != 3 * (m.nq - 1) + 3:
        raise SystemExit("This controller was trained on a different model; pass the matching --model.")
    # SOURCE [GYM] reset_model() with reset_noise_scale = 0.1.
    noise = 0.1
    rng = np.random.default_rng(a.seed)
    d.qpos[:] = m.qpos0 + rng.uniform(-noise, noise, m.nq)
    d.qvel[:] = noise * rng.standard_normal(m.nv)
    mujoco.mj_forward(m, d)
    tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")

    def step(t):
        if a.push and t > 0 and t % 200 == 0:
            d.qvel[-1] += a.push
        d.ctrl[0] = np.clip(act(get_obs(d)), -1, 1)
        for _ in range(5):                    # [GYM] frame_skip = 5
            mujoco.mj_step(m, d)
        return d.site_xpos[tip][2] > 1.0      # [GYM] terminated when y <= 1

    if a.gif:
        import imageio
        r = mujoco.Renderer(m, 360, 480)      # [MJPY] Renderer
        cam = mujoco.MjvCamera()
        cam.distance, cam.lookat[:], cam.elevation, cam.azimuth = 4.1, (0, 0, 0.3), -5, 90
        frames, t = [], 0
        for t in range(a.steps):
            ok = step(t)
            if t % 2 == 0:
                r.update_scene(d, cam)        # [MJPY]
                frames.append(r.render())     # [MJPY]
            if not ok:
                break
        imageio.mimsave(a.gif, frames, duration=0.1, loop=0)   # 0.1 s per frame = real time
        print(f"Saved {a.gif}: {len(frames)} frames, balanced for {t + 1} steps ({(t + 1) * 0.05:.0f} s)")
    else:
        from mujoco import viewer
        with viewer.launch_passive(m, d) as v:  # [MJPY] passive viewer
            for t in range(a.steps):
                if not step(t) or not v.is_running():
                    break
                v.sync()
                time.sleep(0.05)


if __name__ == "__main__":
    main()
