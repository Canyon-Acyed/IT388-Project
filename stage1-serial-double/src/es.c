/*
 * es.c - Stage 1: serial Evolution Strategies (ES) for the inverted double pendulum.
 *
 * SOURCES
 *   [ES]   T. Salimans, J. Ho, X. Chen, S. Sidor, I. Sutskever, "Evolution
 *          Strategies as a Scalable Alternative to Reinforcement Learning",
 *          arXiv:1703.03864, 2017. Algorithm 1 (the ES update), mirrored
 *          sampling, fitness shaping, weight decay, observation normalization.
 *   [OAI]  OpenAI, evolution-strategies-starter, es_distributed/es.py and
 *          optimizers.py (MIT License). The reference code for [ES].
 *          https://github.com/openai/evolution-strategies-starter
 *          Each translated Python line is quoted next to its C version.
 *   [ADAM] D. Kingma, J. Ba, "Adam: A Method for Stochastic Optimization",
 *          ICLR 2015, Algorithm 1.
 *   [FNV]  G. Fowler, L. Noll, K.-P. Vo, FNV-1a hash (public domain).
 *          http://www.isthe.com/chongo/tech/comp/fnv/
 *
 * OUR CODE: the C program around these pieces: options, memory, the
 * generation loop, evaluation of the unperturbed controller, the stop-at-target
 * rule, printing, the CSV log and saving the controller.
 *
 * One generation:
 *   1. Make P random perturbations eps_p of the parameters theta.
 *   2. Test theta + sigma*eps_p and theta - sigma*eps_p (mirrored sampling),
 *      both from the same starting state.
 *   3. Rank all 2P returns and move theta toward perturbations that scored
 *      well and away from those that scored badly (Adam step).
 *   4. Every few generations, test the unperturbed theta to track progress.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "env.h"
#include "policy.h"
#include "rng.h"

/* ------------------------------------------------------------------------ */
/* OUR CODE: settings and command-line options                              */
/* ------------------------------------------------------------------------ */
typedef struct {
    const char *model;
    int pairs, hidden, gens, eval_every, eval_eps;
    double sigma, lr, l2, target;
    uint64_t seed;
    const char *log_path, *save_path;
} Config;

static void usage(void) {
    fprintf(stderr,
        "Options:\n"
        "  --model FILE       MuJoCo model (default models/inverted_double_pendulum.xml)\n"
        "  --pairs N          mirrored perturbation pairs per generation (default 128)\n"
        "  --hidden N         hidden neurons, 0 = linear policy (default 32)\n"
        "  --sigma X          noise scale (default 0.1)\n"
        "  --lr X             Adam learning rate (default 0.05)\n"
        "  --l2 X             L2 weight decay (default 0.005)\n"
        "  --gens N           maximum generations (default 300)\n"
        "  --seed N           random seed (default 1)\n"
        "  --eval-every N     evaluate every N generations (default 5)\n"
        "  --eval-eps N       episodes per evaluation (default 8)\n"
        "  --target X         stop at this evaluation return, 0 = run all gens (default 9000)\n"
        "  --log FILE         per-generation CSV log\n"
        "  --save FILE        save the final controller\n");
}

static int parse(Config *c, int argc, char **argv) {
    /* Defaults. l2 = 0.005 is [OAI]'s "l2coeff" in configurations/humanoid.json.
       sigma, lr, pairs and hidden were tuned by us for this task. */
    *c = (Config){ "models/inverted_double_pendulum.xml", 128, 32, 300, 5, 8,
                   0.1, 0.05, 0.005, 9000.0, 1, NULL, NULL };
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = (i + 1 < argc) ? argv[i + 1] : NULL;
#define TAKE(name) (strcmp(a, name) == 0 && v && (++i, 1))
        if      (TAKE("--model"))      c->model = v;
        else if (TAKE("--pairs"))      c->pairs = atoi(v);
        else if (TAKE("--hidden"))     c->hidden = atoi(v);
        else if (TAKE("--sigma"))      c->sigma = atof(v);
        else if (TAKE("--lr"))         c->lr = atof(v);
        else if (TAKE("--l2"))         c->l2 = atof(v);
        else if (TAKE("--gens"))       c->gens = atoi(v);
        else if (TAKE("--seed"))       c->seed = strtoull(v, NULL, 10);
        else if (TAKE("--eval-every")) c->eval_every = atoi(v);
        else if (TAKE("--eval-eps"))   c->eval_eps = atoi(v);
        else if (TAKE("--target"))     c->target = atof(v);
        else if (TAKE("--log"))        c->log_path = v;
        else if (TAKE("--save"))       c->save_path = v;
        else { fprintf(stderr, "Unknown or incomplete option: %s\n", a); return -1; }
#undef TAKE
    }
    if (c->pairs < 1 || c->gens < 1 || c->eval_every < 1 || c->eval_eps < 1) return -1;
    return 0;
}

/* OUR CODE: wall-clock timer. */
static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* SOURCE [FNV]: 64-bit FNV-1a over the bytes of an array:
       hash = offset_basis; for each byte: hash ^= byte; hash *= FNV_prime
   Used to print a fingerprint of the trained controller. */
static uint64_t hash_doubles(const double *x, int n) {
    uint64_t h = 14695981039346656037ULL;                  /* FNV-1a 64-bit offset basis */
    const unsigned char *b = (const unsigned char *)x;
    for (size_t i = 0; i < (size_t)n * sizeof(double); i++) {
        h ^= b[i];
        h *= 1099511628211ULL;                             /* FNV 64-bit prime */
    }
    return h;
}

/* ------------------------------------------------------------------------ */
/* Observation normalization                                                */
/* SOURCE [OAI] es.py class RunningStat:                                     */
/*     sum = zeros; sumsq = full(eps); count = eps        (eps = 1e-2)      */
/*     increment(s, ssq, c): sum += s; sumsq += ssq; count += c             */
/*     mean = sum / count                                                    */
/*     std  = sqrt(maximum(sumsq / count - square(mean), 1e-2))             */
/* and policies.py: the network input is clip((ob - ob_mean) / ob_std, -5, 5) */
/* OUR CHANGE: [OAI] samples statistics from 1% of episodes                  */
/* (calc_obstat_prob = 0.01); we use every episode.                         */
/* ------------------------------------------------------------------------ */
typedef struct { int n; double *sum, *sumsq, count, *mean, *std; } RunningStat;

static void stat_init(RunningStat *r, int n) {
    r->n = n;
    r->sum = calloc(n, sizeof(double));
    r->sumsq = malloc(sizeof(double) * n);
    r->mean = malloc(sizeof(double) * n);
    r->std = malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) r->sumsq[i] = 1e-2;   /* [OAI] sumsq = full(eps) */
    r->count = 1e-2;                                  /* [OAI] count = eps       */
}
static void stat_update(RunningStat *r) {             /* [OAI] mean, std properties */
    for (int i = 0; i < r->n; i++) {
        r->mean[i] = r->sum[i] / r->count;
        double var = r->sumsq[i] / r->count - r->mean[i] * r->mean[i];
        r->std[i] = sqrt(var > 1e-2 ? var : 1e-2);
    }
}
static void stat_free(RunningStat *r) { free(r->sum); free(r->sumsq); free(r->mean); free(r->std); }

/* OUR CODE: run one episode and return its total reward (the "return").
   Inputs are normalized as in [OAI] policies.py. If stats != NULL, the raw
   observations are added to stats = [sums (od values), sums of squares (od)],
   like [OAI] rollout_and_update_ob_stat. */
static double run_episode(Env *env, const PolicyShape *ps, const double *theta,
                          const RunningStat *rs, double *stats,
                          double *obs, double *x, double *work,
                          uint64_t reset_seed, int *steps) {
    int done = 0, fell = 0, od = ps->obs_dim;
    double total = 0.0;
    env_reset(env, reset_seed, obs);
    while (!done) {
        for (int i = 0; i < od; i++) {
            if (stats) { stats[i] += obs[i]; stats[od + i] += obs[i] * obs[i]; }
            double z = (obs[i] - rs->mean[i]) / rs->std[i];
            x[i] = z < -5.0 ? -5.0 : (z > 5.0 ? 5.0 : z);   /* [OAI] clip(..., -5, 5) */
        }
        double a = policy_act(ps, theta, x, work);
        total += env_step(env, a, obs, &done, &fell);
    }
    *steps = env->t;
    return total;
}

/* ------------------------------------------------------------------------ */
/* SOURCE [OAI] es.py:                                                       */
/*   def compute_ranks(x):  ranks[x.argsort()] = np.arange(len(x))          */
/*   def compute_centered_ranks(x):  y = compute_ranks(x); y /= (x.size - 1); */
/*                                   y -= .5                                */
/* OUR CODE: the C sort; ties are broken by index so results never depend   */
/* on the sort routine.                                                      */
/* ------------------------------------------------------------------------ */
typedef struct { double v; int i; } RankItem;
static int cmp_rank(const void *a, const void *b) {
    const RankItem *x = a, *y = b;
    if (x->v < y->v) return -1;
    if (x->v > y->v) return 1;
    return x->i - y->i;
}
static void centered_ranks(const double *v, int n, double *out, RankItem *tmp) {
    for (int i = 0; i < n; i++) { tmp[i].v = v[i]; tmp[i].i = i; }
    qsort(tmp, n, sizeof(RankItem), cmp_rank);                  /* argsort        */
    for (int r = 0; r < n; r++)
        out[tmp[r].i] = (n > 1) ? (double)r / (n - 1) - 0.5 : 0.0;   /* /(size-1) - .5 */
}

/* ------------------------------------------------------------------------ */
int main(int argc, char **argv) {
    Config cfg;
    if (parse(&cfg, argc, argv)) { usage(); return 1; }

    /* OUR CODE: set up the environment, controller and buffers. */
    char err[1024];
    EnvSpec spec;
    if (env_spec_load(&spec, cfg.model, err, sizeof err)) {
        fprintf(stderr, "Could not load %s: %s\n", cfg.model, err);
        return 1;
    }
    Env env;
    env_init(&env, &spec);
    PolicyShape ps;
    policy_shape(&ps, spec.obs_dim, cfg.hidden);
    const int D = ps.n_params, P = cfg.pairs, N = 2 * cfg.pairs, OD = spec.obs_dim;

    double *theta = malloc(sizeof(double) * D), *grad = malloc(sizeof(double) * D);
    double *th_plus = malloc(sizeof(double) * D), *th_minus = malloc(sizeof(double) * D);
    double *adam_m = calloc(D, sizeof(double)), *adam_v = calloc(D, sizeof(double));
    double *obs = malloc(sizeof(double) * OD), *x = malloc(sizeof(double) * OD);
    double *work = malloc(sizeof(double) * (cfg.hidden > 0 ? cfg.hidden : 1));
    double *ret = malloc(sizeof(double) * N), *len = malloc(sizeof(double) * N);
    double *shaped = malloc(sizeof(double) * N), *wpair = malloc(sizeof(double) * P);
    double *stats = malloc(sizeof(double) * 2 * OD);
    uint64_t *keys = malloc(sizeof(uint64_t) * P);
    RankItem *tmp = malloc(sizeof(RankItem) * N);
    RunningStat ob;
    stat_init(&ob, OD);
    stat_update(&ob);                       /* mean 0, std 1 before any data */
    policy_init(&ps, theta, cfg.seed);

    FILE *logf = cfg.log_path ? fopen(cfg.log_path, "w") : NULL;
    if (logf) fprintf(logf, "gen,wall,pop_mean,pop_max,eval_mean,mean_len,steps\n");
    printf("Model %s: %d links, %d observations\n", cfg.model, spec.nlinks, OD);
    printf("Controller: %d hidden neurons, %d parameters\n", cfg.hidden, D);
    printf("Population %d (%d mirrored pairs), serial\n", N, P);
    printf("%5s %9s %10s %10s %10s %8s\n", "gen", "time(s)", "pop mean", "pop max", "eval", "avg len");

    double t_start = now(), eval_mean = NAN, time_to_target = -1.0;
    long long total_steps = 0;
    int gen;
    for (gen = 0; gen < cfg.gens; gen++) {
        /* ---- Steps 1-2: test every mirrored pair ----
           SOURCE [OAI] es.py run_worker:
               v = config.noise_stdev * noise.get(noise_idx, policy.num_params)
               policy.set_trainable_flat(task_data.params + v)  -> rollout
               policy.set_trainable_flat(task_data.params - v)  -> rollout
               returns.append([rews_pos.sum(), rews_neg.sum()])
           OUR CHANGE: [OAI] reads v from a big shared table of noise; we compute
           it from a seed (rng.h). Both halves start from the same state. */
        long long steps = 0;
        for (int p = 0; p < P; p++) {
            uint64_t key = rng_key(cfg.seed, gen, p, STREAM_NOISE);
            for (int j = 0; j < D; j++) {
                double e = cfg.sigma * normal_at(key, j);
                th_plus[j] = theta[j] + e;
                th_minus[j] = theta[j] - e;
            }
            uint64_t rs = rng_key(cfg.seed, gen, p, STREAM_RESET);
            int sp, sm;
            memset(stats, 0, sizeof(double) * 2 * OD);
            ret[2 * p]     = run_episode(&env, &ps, th_plus,  &ob, stats, obs, x, work, rs, &sp);
            ret[2 * p + 1] = run_episode(&env, &ps, th_minus, &ob, stats, obs, x, work, rs, &sm);
            len[2 * p] = sp; len[2 * p + 1] = sm;
            steps += sp + sm;
            /* [OAI] ob_stat.increment(result.ob_sum, result.ob_sumsq, result.ob_count) */
            for (int i = 0; i < OD; i++) { ob.sum[i] += stats[i]; ob.sumsq[i] += stats[OD + i]; }
            ob.count += sp + sm;
        }

        /* ---- Step 3: rank the returns and take one Adam step ----
           SOURCE [OAI] es.py:
               proc_returns_n2 = compute_centered_ranks(returns_n2)
               g, count = batched_weighted_sum(proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
                                               (noise.get(idx, ...) for idx in noise_inds_n), ...)
               g /= returns_n2.size
               update_ratio = optimizer.update(-g + config.l2coeff * theta)
           OUR CHANGE: we also divide by sigma, as in [ES] Algorithm 1
           (theta += alpha / (n sigma) * sum F_i eps_i), and step uphill
           directly instead of minimizing -g. */
        centered_ranks(ret, N, shaped, tmp);
        for (int p = 0; p < P; p++) {
            wpair[p] = shaped[2 * p] - shaped[2 * p + 1];       /* pos rank - neg rank */
            keys[p] = rng_key(cfg.seed, gen, p, STREAM_NOISE);  /* rebuild eps_p       */
        }
        for (int j = 0; j < D; j++) {
            double g = 0.0;
            for (int p = 0; p < P; p++) g += wpair[p] * normal_at(keys[p], j);
            grad[j] = g / (N * cfg.sigma) - cfg.l2 * theta[j];
        }
        /* SOURCE [ADAM] Algorithm 1 (beta1 = 0.9, beta2 = 0.999, eps = 1e-8,
           the same defaults as [OAI] optimizers.py class Adam):
               m = b1 m + (1 - b1) g;  v = b2 v + (1 - b2) g^2
               theta += lr * (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps) */
        const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
        double c1 = 1.0 - pow(b1, gen + 1), c2 = 1.0 - pow(b2, gen + 1);
        for (int j = 0; j < D; j++) {
            adam_m[j] = b1 * adam_m[j] + (1 - b1) * grad[j];
            adam_v[j] = b2 * adam_v[j] + (1 - b2) * grad[j] * grad[j];
            theta[j] += cfg.lr * (adam_m[j] / c1) / (sqrt(adam_v[j] / c2) + eps);
        }
        /* [OAI] policy.set_ob_stat(ob_stat.mean, ob_stat.std) after each update. */
        stat_update(&ob);

        /* ---- Step 4 (OUR CODE): test the unperturbed controller on the same
           8 starting states every time, so results are comparable. ---- */
        int do_eval = ((gen + 1) % cfg.eval_every == 0) || (gen + 1 == cfg.gens);
        if (do_eval) {
            eval_mean = 0;
            for (int k = 0; k < cfg.eval_eps; k++) {
                int s;
                eval_mean += run_episode(&env, &ps, theta, &ob, NULL, obs, x, work,
                                         rng_key(cfg.seed, 0, k, STREAM_EVAL), &s);
            }
            eval_mean /= cfg.eval_eps;
        }
        total_steps += steps;

        /* OUR CODE: progress output and log. */
        double pmean = 0, pmax = -1e300, mlen = 0, t = now() - t_start;
        for (int i = 0; i < N; i++) { pmean += ret[i]; mlen += len[i]; if (ret[i] > pmax) pmax = ret[i]; }
        pmean /= N; mlen /= N;
        if (logf) fprintf(logf, "%d,%.6f,%.3f,%.3f,%.3f,%.2f,%lld\n",
                          gen, t, pmean, pmax, do_eval ? eval_mean : NAN, mlen, steps);
        if (do_eval) printf("%5d %9.2f %10.1f %10.1f %10.1f %8.1f\n", gen + 1, t, pmean, pmax, eval_mean, mlen);
        fflush(stdout);
        if (do_eval && cfg.target > 0 && eval_mean >= cfg.target) { time_to_target = t; gen++; break; }
    }

    /* OUR CODE: summary, fingerprint and saving. */
    double t_total = now() - t_start;
    uint64_t h = hash_doubles(theta, D) ^ (hash_doubles(ob.mean, OD) * 31) ^ (hash_doubles(ob.std, OD) * 131);
    printf("\nDone: %d generations in %.2f s (%s)\n", gen, t_total,
           time_to_target >= 0 ? "target reached" : (cfg.target > 0 ? "target NOT reached" : "fixed length"));
    printf("  final evaluation return  %.1f\n", eval_mean);
    printf("  simulation steps         %lld  (%.0f steps/s)\n", total_steps, total_steps / t_total);
    printf("  theta hash               %016llx\n", (unsigned long long)h);

    if (cfg.save_path) {
        FILE *f = fopen(cfg.save_path, "w");
        if (f) {
            /* Line 1: obs_dim hidden n_params observations_counted.
               Line 2: input means. Line 3: input standard deviations.
               Then one parameter per line (W1 row by row, b1, w2, b2). */
            fprintf(f, "%d %d %d %.17g\n", OD, cfg.hidden, D, ob.count);
            for (int i = 0; i < OD; i++) fprintf(f, "%.17g%c", ob.mean[i], i + 1 < OD ? ' ' : '\n');
            for (int i = 0; i < OD; i++) fprintf(f, "%.17g%c", ob.std[i], i + 1 < OD ? ' ' : '\n');
            for (int j = 0; j < D; j++) fprintf(f, "%.17g\n", theta[j]);
            fclose(f);
        }
    }
    if (logf) fclose(logf);

    free(theta); free(grad); free(th_plus); free(th_minus); free(adam_m); free(adam_v);
    free(obs); free(x); free(work); free(ret); free(len); free(shaped); free(wpair);
    free(stats); free(keys); free(tmp);
    stat_free(&ob);
    env_free(&env);
    env_spec_free(&spec);
    return 0;
}
