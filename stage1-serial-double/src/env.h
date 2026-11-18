/*
 * env.h - Inverted pendulum on a cart, simulated with MuJoCo.
 *
 * SOURCES
 *   [GYM]  Farama Foundation, Gymnasium 1.4.0,
 *          gymnasium/envs/mujoco/inverted_double_pendulum_v5.py (MIT License).
 *          https://github.com/Farama-Foundation/Gymnasium
 *          The settings below (frame skip 5, healthy reward 10, end when the
 *          tip height <= 1, distance target 2, reset noise 0.1) are its defaults.
 *   [MJ]   MuJoCo C API (Apache 2.0): mjModel, mjData.
 *          https://mujoco.readthedocs.io/en/stable/APIreference/index.html
 *
 * OUR CODE: the C structs and function names (a C version of Gymnasium's
 * Python class), and the generalization to any number of links.
 */
#ifndef ENV_H
#define ENV_H

#include <mujoco/mujoco.h>
#include "rng.h"

#define ENV_MAX_LINKS 8

/* Settings shared by every copy of the environment (read-only). */
typedef struct {
    mjModel *m;            /* [MJ] the model, loaded once                  */
    int nlinks;            /* number of poles (nq - 1)                     */
    int obs_dim;           /* 3 * nlinks + 3, as in [GYM] _get_obs         */
    int frame_skip;        /* [GYM] frame_skip = 5                         */
    int max_steps;         /* [GYM] episode limit (1000 steps)             */
    int tip_site;          /* [GYM] uses site_xpos[0], the "tip" site      */
    double healthy_reward; /* [GYM] healthy_reward = 10                    */
    double term_height;    /* [GYM] terminated when y <= 1                 */
    double target_height;  /* [GYM] distance penalty (y - 2)^2             */
    double reset_noise;    /* [GYM] reset_noise_scale = 0.1                */
    double vel_w[ENV_MAX_LINKS]; /* [GYM] 1e-3 * v1^2 + 5e-3 * v2^2        */
} EnvSpec;

/* One running copy of the environment. */
typedef struct {
    const EnvSpec *s;
    mjData *d;             /* [MJ] simulation state */
    int t;                 /* steps taken in this episode */
} Env;

int    env_spec_load(EnvSpec *s, const char *xml_path, char *err, int err_sz);
void   env_spec_free(EnvSpec *s);
void   env_init(Env *e, const EnvSpec *s);
void   env_free(Env *e);
void   env_reset(Env *e, uint64_t seed, double *obs);
void   env_set_state(Env *e, const double *qpos, const double *qvel, double *obs);
/* Returns the reward. *done = episode over; *fell = pendulum fell. */
double env_step(Env *e, double action, double *obs, int *done, int *fell);

#endif
