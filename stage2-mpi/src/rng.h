/*
 * rng.h - Seeded random numbers that can be recomputed anywhere.
 *
 * SOURCES
 *   [SM]  S. Vigna, "splitmix64.c", public domain.
 *         https://prng.di.unimi.it/splitmix64.c
 *         (mix64 below is SplitMix64's next() step, constants unchanged.)
 *   [VD]  S. Vigna, "xoshiro / xoroshiro generators and the PRNG shootout",
 *         section "Generating uniform doubles in the unit interval".
 *         https://prng.di.unimi.it/   (the (x >> 11) * 2^-53 conversion)
 *   [CB]  J. Salmon, M. Moraes, R. Dror, D. Shaw, "Parallel Random Numbers:
 *         As Easy as 1, 2, 3", SC '11, 2011. Idea of a counter-based generator:
 *         random number = hash(key, counter), so any number can be computed
 *         directly without generating the ones before it.
 *   [BM]  G. Box, M. Muller, "A Note on the Generation of Random Normal
 *         Deviates", Annals of Mathematical Statistics 29(2), 1958.
 *         (Box-Muller transform: uniform -> normal.)
 *   [ES]  T. Salimans et al., "Evolution Strategies as a Scalable Alternative
 *         to Reinforcement Learning", arXiv:1703.03864, 2017. Algorithm 2 gives
 *         every worker the same random seeds so it can rebuild any perturbation.
 *
 * OUR CODE: rng_key (how seed, generation, pair and stream are combined), the
 * stream names, normal_at, the +0.5 in u01, and the Rng wrapper.
 */
#ifndef RNG_H
#define RNG_H

#include <stdint.h>
#include <math.h>

/* OUR CODE: separate streams so different uses of randomness never overlap. */
enum { STREAM_NOISE = 1, STREAM_RESET = 2, STREAM_EVAL = 3, STREAM_INIT = 4 };

/* SOURCE [SM]: SplitMix64 next(), state passed by value. */
static inline uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* OUR CODE, following the counter-based idea of [CB] and the shared seeds of
   [ES]: one key per (seed, generation, index, stream). The odd constants only
   spread the inputs apart before mixing. */
static inline uint64_t rng_key(uint64_t seed, uint64_t gen, uint64_t idx, uint64_t stream) {
    uint64_t k = mix64(seed);
    k = mix64(k ^ (gen + 0x1000193ULL));
    k = mix64(k ^ (idx * 0xD6E8FEB86659FD93ULL + 0x632BE59BD9B4E019ULL));
    k = mix64(k ^ (stream * 0xA0761D6478BD642FULL));
    return k;
}

/* SOURCE [VD]: top 53 bits -> double. OUR CHANGE: + 0.5 so the result is never
   exactly 0, because Box-Muller takes log(u). */
static inline double u01(uint64_t x) {
    return ((double)(x >> 11) + 0.5) * (1.0 / 9007199254740992.0);   /* 2^-53 */
}

/* Normal number j of stream "key". Counter-based per [CB]: element j is
   computed directly. Uniform -> normal by the Box-Muller transform [BM]. */
static inline double normal_at(uint64_t key, uint64_t j) {
    uint64_t a = mix64(key ^ (2 * j * 0x9E3779B97F4A7C15ULL));
    uint64_t b = mix64(a ^ 0xC2B2AE3D27D4EB4FULL);
    double u1 = u01(a), u2 = u01(b);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);      /* [BM] */
}

/* OUR CODE: small sequential generator (SplitMix64 [SM] with a counter state),
   used for episode start states. */
typedef struct { uint64_t s; } Rng;
static inline uint64_t rng_next(Rng *r) { r->s += 0x9E3779B97F4A7C15ULL; return mix64(r->s); }
static inline double rng_uniform(Rng *r, double lo, double hi) { return lo + (hi - lo) * u01(rng_next(r)); }
static inline double rng_normal(Rng *r) {                            /* [BM] */
    double u1 = u01(rng_next(r)), u2 = u01(rng_next(r));
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

#endif
