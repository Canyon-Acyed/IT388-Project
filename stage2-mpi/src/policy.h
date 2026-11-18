/*
 * policy.h - The controller: a small feed-forward neural network.
 *
 *   x      = clip((obs - mean) / std, -5, 5)   (done by the caller, es.c)
 *   hidden = tanh(W1 * x + b1)                 (H neurons)
 *   action = tanh(w2 . hidden + b2)            (one number in [-1, 1])
 *
 * theta (the parameters) is one flat array: W1 (row by row), b1, w2, b2.
 * With hidden = 0 the network is linear: action = tanh(w . x + b).
 *
 * SOURCE: layer structure (tanh hidden layers) and the "normc" starting
 * weights follow OpenAI's evolution-strategies-starter, es_distributed/
 * policies.py (MujocoPolicy) and tf_util.py (normc_initializer), MIT License.
 * https://github.com/openai/evolution-strategies-starter
 * OUR CODE: the C implementation, the flat theta layout, and the tanh on the
 * output (OpenAI's output layer is linear).
 */
#ifndef POLICY_H
#define POLICY_H

#include <stdint.h>

typedef struct {
    int obs_dim;
    int hidden;     /* number of hidden neurons (0 = linear policy) */
    int n_params;   /* size of theta */
} PolicyShape;

void   policy_shape(PolicyShape *p, int obs_dim, int hidden);
void   policy_init(const PolicyShape *p, double *theta, uint64_t seed);
/* work must hold at least p->hidden doubles. */
double policy_act(const PolicyShape *p, const double *theta, const double *x, double *work);

#endif
