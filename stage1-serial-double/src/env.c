/*
 * env.c - Inverted pendulum on a cart: a C version of Gymnasium's
 *         InvertedDoublePendulum-v5.
 *
 * SOURCES
 *   [GYM]  Farama Foundation, Gymnasium 1.4.0 (MIT License),
 *          gymnasium/envs/mujoco/inverted_double_pendulum_v5.py and
 *          gymnasium/envs/mujoco/mujoco_env.py.
 *          https://github.com/Farama-Foundation/Gymnasium
 *          Each Python line we translated is quoted next to its C version.
 *   [MJB]  MuJoCo 3.3.7, sample/basic.cc (Apache 2.0): the pattern
 *          mj_loadXML -> mj_makeData -> mj_step -> mj_deleteData/mj_deleteModel.
 *          https://github.com/google-deepmind/mujoco/blob/main/sample/basic.cc
 *   [MJ]   MuJoCo API reference: mj_resetData, mj_forward, mj_name2id,
 *          site_xpos, qfrc_constraint.
 *          https://mujoco.readthedocs.io/en/stable/APIreference/index.html
 *
 * OUR CODE: everything not marked, including support for any number of links
 * (Gymnasium's file handles exactly two).
 *
 * Checked against Gymnasium by scripts/check_env.py (make check): same
 * observations and rewards to within 1e-15.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "env.h"

/* [GYM] np.clip */
static double clip(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

int env_spec_load(EnvSpec *s, const char *xml_path, char *err, int err_sz) {
    memset(s, 0, sizeof(*s));
    /* SOURCE [MJB]: m = mj_loadXML(argv[1], 0, error, 1000); */
    s->m = mj_loadXML(xml_path, NULL, err, err_sz);
    if (!s->m) return -1;

    /* OUR CODE: read the shape of the model instead of assuming two links. */
    int nq = (int)s->m->nq;
    s->nlinks = nq - 1;
    if (s->nlinks < 1 || s->nlinks > ENV_MAX_LINKS || (int)s->m->nu != 1) {
        snprintf(err, err_sz, "model must have a cart plus 1..%d hinges and one actuator", ENV_MAX_LINKS);
        return -1;
    }
    /* [GYM] uses self.data.site_xpos[0]; we look the site up by name [MJ]. */
    s->tip_site = mj_name2id(s->m, mjOBJ_SITE, "tip");
    if (s->tip_site < 0) { snprintf(err, err_sz, "model has no site named \"tip\""); return -1; }

    /* SOURCE [GYM] __init__ defaults: frame_skip=5, healthy_reward=10.0,
       reset_noise_scale=0.1; step(): terminated = bool(y <= 1);
       _get_rew(): (y - 2) ** 2. The 1000-step limit is Gymnasium's
       max_episode_steps for this environment. */
    s->obs_dim = 3 * s->nlinks + 3;
    s->frame_skip = 5;
    s->max_steps = 1000;
    s->healthy_reward = 10.0;
    s->term_height = 1.0;
    s->target_height = 2.0;
    s->reset_noise = 0.1;
    /* SOURCE [GYM] _get_rew(): vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2 */
    for (int k = 0; k < s->nlinks; k++) s->vel_w[k] = (k == 0) ? 1e-3 : 5e-3;
    return 0;
}

/* SOURCE [MJB]: mj_deleteModel(m); */
void env_spec_free(EnvSpec *s) { if (s->m) mj_deleteModel(s->m); s->m = NULL; }

/* SOURCE [MJB]: d = mj_makeData(m); ... mj_deleteData(d); */
void env_init(Env *e, const EnvSpec *s) { e->s = s; e->d = mj_makeData(s->m); e->t = 0; }
void env_free(Env *e) { if (e->d) mj_deleteData(e->d); e->d = NULL; }

/* SOURCE [GYM] _get_obs():
       np.concatenate([qpos[:1], np.sin(qpos[1:]), np.cos(qpos[1:]),
                       np.clip(qvel, -10, 10), np.clip(qfrc_constraint, -10, 10)[:1]]) */
static void get_obs(const Env *e, double *obs) {
    const mjData *d = e->d;
    int L = e->s->nlinks, k = 0;
    obs[k++] = d->qpos[0];                                              /* qpos[:1]       */
    for (int i = 1; i <= L; i++) obs[k++] = sin(d->qpos[i]);            /* sin(qpos[1:])  */
    for (int i = 1; i <= L; i++) obs[k++] = cos(d->qpos[i]);            /* cos(qpos[1:])  */
    for (int i = 0; i <= L; i++) obs[k++] = clip(d->qvel[i], -10, 10);  /* clip(qvel)     */
    obs[k++] = clip(d->qfrc_constraint[0], -10.0, 10.0);                /* clip(qfrc)[:1] */
}

/* OUR CODE: set an exact state (used by env_reset and by the Gymnasium check).
   mj_resetData / mj_forward per the MuJoCo API [MJ]. */
void env_set_state(Env *e, const double *qpos, const double *qvel, double *obs) {
    const mjModel *m = e->s->m;
    mj_resetData(m, e->d);
    memcpy(e->d->qpos, qpos, sizeof(double) * m->nq);
    memcpy(e->d->qvel, qvel, sizeof(double) * m->nv);
    mj_forward(m, e->d);
    e->t = 0;
    get_obs(e, obs);
}

/* SOURCE [GYM] reset_model():
       qpos = init_qpos + uniform(low=-noise, high=noise, size=nq)
       qvel = init_qvel + standard_normal(nv) * noise
   OUR CHANGE: our own seeded random numbers (rng.h) instead of NumPy's.
   (init_qvel is zero, so it is left out.) */
void env_reset(Env *e, uint64_t seed, double *obs) {
    const mjModel *m = e->s->m;
    double qpos[ENV_MAX_LINKS + 1], qvel[ENV_MAX_LINKS + 1];
    double a = e->s->reset_noise;
    Rng r = { seed };
    for (int i = 0; i < m->nq; i++) qpos[i] = m->qpos0[i] + rng_uniform(&r, -a, a);
    for (int i = 0; i < m->nv; i++) qvel[i] = a * rng_normal(&r);
    env_set_state(e, qpos, qvel, obs);
}

double env_step(Env *e, double action, double *obs, int *done, int *fell) {
    const EnvSpec *s = e->s;
    mjData *d = e->d;

    /* SOURCE [GYM] mujoco_env.py _step_mujoco_simulation():
           self.data.ctrl[:] = ctrl
           mujoco.mj_step(self.model, self.data, nstep=n_frames)
       The C mj_step has no nstep argument, so we loop. Gymnasium then calls
       mj_rnePostConstraint, which only fills quantities this environment never
       reads, so we skip it (check_env.py confirms identical results).
       OUR CHANGE: clip the action to the actuator's ctrlrange [-1, 1] ourselves. */
    d->ctrl[0] = clip(action, -1.0, 1.0);
    for (int i = 0; i < s->frame_skip; i++) mj_step(s->m, d);
    e->t++;

    /* SOURCE [GYM] step():  x, _, y = self.data.site_xpos[0]
                             terminated = bool(y <= 1)                    */
    double x = d->site_xpos[3 * s->tip_site + 0];
    double y = d->site_xpos[3 * s->tip_site + 2];
    int terminated = (y <= s->term_height);

    /* SOURCE [GYM] _get_rew():
           dist_penalty = 0.01 * x**2 + (y - 2) ** 2
           vel_penalty  = 1e-3 * v1**2 + 5e-3 * v2**2
           alive_bonus  = self._healthy_reward * int(not terminated)
           reward       = alive_bonus - dist_penalty - vel_penalty
       OUR CHANGE: the velocity penalty loops over however many hinges exist. */
    double dist = 0.01 * x * x + (y - s->target_height) * (y - s->target_height);
    double vel = 0.0;
    for (int k = 0; k < s->nlinks; k++) vel += s->vel_w[k] * d->qvel[k + 1] * d->qvel[k + 1];
    double reward = (terminated ? 0.0 : s->healthy_reward) - dist - vel;

    get_obs(e, obs);
    *fell = terminated;
    /* OUR CODE: Gymnasium's TimeLimit wrapper ends the episode at 1000 steps. */
    *done = terminated || (e->t >= s->max_steps);
    return reward;
}
