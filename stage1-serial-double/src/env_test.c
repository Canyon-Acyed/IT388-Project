/*
 * env_test.c - Helper for scripts/check_env.py: runs given start states and
 * actions through our C environment so they can be compared with Gymnasium.
 *
 * OUR CODE (whole file). Uses only our env.c.
 *
 * Reads from stdin: "nq nv", then qpos, qvel, then a count of actions
 * followed by the actions. Prints one line per step:
 *   reward fell obs[0] ... obs[obs_dim-1]
 */
#include <stdio.h>
#include <stdlib.h>
#include "env.h"

int main(int argc, char **argv) {
    const char *xml = argc > 1 ? argv[1] : "models/inverted_double_pendulum.xml";
    char err[1024];
    EnvSpec s;
    if (env_spec_load(&s, xml, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    Env e; env_init(&e, &s);

    int nq, nv, n;
    if (scanf("%d %d", &nq, &nv) != 2) return 1;
    double qpos[ENV_MAX_LINKS + 1], qvel[ENV_MAX_LINKS + 1], obs[64];
    for (int i = 0; i < nq; i++) if (scanf("%lf", &qpos[i]) != 1) return 1;
    for (int i = 0; i < nv; i++) if (scanf("%lf", &qvel[i]) != 1) return 1;
    if (scanf("%d", &n) != 1) return 1;
    env_set_state(&e, qpos, qvel, obs);
    for (int t = 0; t < n; t++) {
        double a; int done, fell;
        if (scanf("%lf", &a) != 1) return 1;
        double r = env_step(&e, a, obs, &done, &fell);
        printf("%.17g %d", r, fell);
        for (int k = 0; k < s.obs_dim; k++) printf(" %.17g", obs[k]);
        printf("\n");
        if (fell) break;
    }
    env_free(&e); env_spec_free(&s);
    return 0;
}
