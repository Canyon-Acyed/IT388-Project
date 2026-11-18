/*
 * policy.c - Small feed-forward neural network controller.
 *
 * SOURCES
 *   [OAI] OpenAI, evolution-strategies-starter (MIT License),
 *         https://github.com/openai/evolution-strategies-starter
 *         - policies.py MujocoPolicy._make_net: hidden layers
 *             x = self.nonlin(U.dense(x, hd, ..., U.normc_initializer(1.0)))
 *           with nonlin = tanh, and the output layer
 *             a = U.dense(x, adim, 'out', U.normc_initializer(0.01))
 *         - tf_util.py normc_initializer(std):
 *             out = np.random.randn(*shape)
 *             out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
 *           i.e. each neuron's incoming weights are random, scaled to length std.
 *           U.dense starts biases at zero.
 *
 * OUR CODE: the C implementation, the flat parameter layout, and tanh on the
 * output so the action is always within the motor's range [-1, 1].
 */
#include <math.h>
#include "policy.h"
#include "rng.h"

/* OUR CODE */
void policy_shape(PolicyShape *p, int obs_dim, int hidden) {
    p->obs_dim = obs_dim;
    p->hidden = hidden;
    if (hidden > 0) p->n_params = hidden * obs_dim + hidden + hidden + 1;   /* W1, b1, w2, b2 */
    else            p->n_params = obs_dim + 1;                              /* w, b           */
}

/* SOURCE [OAI] normc_initializer: random normal weights, each neuron's incoming
   weight vector scaled to length `std`. OUR CODE: the C loop, using our
   seeded normals (rng.h). */
static void normc(double *w, int n, double std, uint64_t key, uint64_t offset) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) { w[i] = normal_at(key, offset + i); ss += w[i] * w[i]; }
    double scale = std / sqrt(ss);
    for (int i = 0; i < n; i++) w[i] *= scale;
}

void policy_init(const PolicyShape *p, double *theta, uint64_t seed) {
    uint64_t key = rng_key(seed, 0, 0, STREAM_INIT);
    for (int j = 0; j < p->n_params; j++) theta[j] = 0.0;      /* biases start at 0 [OAI] */
    int D = p->obs_dim, H = p->hidden;
    if (H == 0) {
        normc(theta, D, 0.01, key, 0);                         /* output layer: std 0.01 [OAI] */
        return;
    }
    for (int h = 0; h < H; h++)
        normc(theta + h * D, D, 1.0, key, (uint64_t)h * D);    /* hidden layer: std 1.0 [OAI]  */
    normc(theta + H * D + H, H, 0.01, key, (uint64_t)H * D);   /* output layer: std 0.01 [OAI] */
}

/* OUR CODE (structure from [OAI] _make_net: tanh(dense) per hidden layer, then
   a dense output layer; we add tanh on the output). */
double policy_act(const PolicyShape *p, const double *theta, const double *x, double *work) {
    int D = p->obs_dim, H = p->hidden;
    if (H == 0) {
        double z = theta[D];
        for (int i = 0; i < D; i++) z += theta[i] * x[i];
        return tanh(z);
    }
    const double *W1 = theta, *b1 = theta + H * D, *w2 = b1 + H, *b2 = w2 + H;
    double z = b2[0];
    for (int h = 0; h < H; h++) {
        const double *row = W1 + h * D;
        double a = b1[h];
        for (int i = 0; i < D; i++) a += row[i] * x[i];
        work[h] = tanh(a);
        z += w2[h] * work[h];
    }
    return tanh(z);
}
