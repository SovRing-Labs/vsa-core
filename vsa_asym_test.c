/* ===========================================================================
 * vsa_asym_test.c — RSPK step 6: asymmetry control -> ACF -> IFS A/B
 * on the phase kernel.  (VSAW2-S6-ASYM)
 *
 * Build/run (locked by the wave yaml, do not change):
 *   cd vsa-core && gcc -O2 -mavx2 -o /tmp/vsa_asym_test \
 *     vsa_asym_test.c vsa_resonator_min.c vsa_phase.c -lm && /tmp/vsa_asym_test
 *
 * Design:
 *   F = 3 roles, M = 22 codewords/role, D = 5120, K = 16, 1,000 synthetic
 *   problems, one resonator per query, fixed seeds. Every arm sees the SAME
 *   1,000 problems.
 *
 *   arm (a)   baseline            no perturbation
 *   arm (b1)  one copy perturbed  frozen two-sided mask on the unbind use
 *   arm (b2)  BOTH perturbed      frozen two-sided mask, identical on both
 *   arm (c)   ACF                 frozen ONE-SIDED mask (the asymmetry)
 *   arm (d)   NOISE               live two-sided RANDOM perturbation, per iteration.
 *                                 NOT an IFS -- kept as its honest control.
 *   arm (e)   IFS (real)          VSA_RESONO_IFS_MAPS fixed DETERMINISTIC contraction
 *                                 maps, cycled per iteration. Added 2026-09-28 after
 *                                 re-verification found (d) was noise wearing IFS's name.
 *
 *   r is swept over the PUBLISHED ACF range (0.001-0.04, RSPK-PHYS §36). No
 *   threshold is fitted anywhere; the primary column is r = 0.04, the largest
 *   published amplitude, so that a kill is conservative.
 *
 *   Kill criterion (fixed by the contract, not changed here):
 *     limit-cycle rate changes < 2x  =>  the noise thread (incl. IFS) is dead.
 * ========================================================================= */

#include "vsa_resonator_min.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef N_PROBLEMS
#define N_PROBLEMS 1000
#endif
#define SEED_CODEBOOK 0xA5A5C0DEULL
#define SEED_PROBLEMS 0x5EEDBEEFULL

#define N_ARMS   6
#define N_SPARSE 3
#define N_SUBSAMPLE 200   /* the r-sweep sensitivity subsample; the primary
                          * contract arms always run the full 1,000 */

static const char *ARM_NAME[N_ARMS] = {
    "(a) baseline            ",
    "(b1) ONE copy (unbind)  ",
    "(b2) BOTH copies, same  ",
    "(c) ACF frozen 1-sided  ",
    "(d) NOISE live 2-sided  ",
    "(e) IFS live MAPS       "
};

static const vsa_resonator_pert_mode_t ARM_MODE[N_ARMS] = {
    VSA_RESONO_PERT_NONE,
    VSA_RESONO_PERT_FROZEN_TWO_SIDED,
    VSA_RESONO_PERT_FROZEN_TWO_SIDED,
    VSA_RESONO_PERT_FROZEN_ONE_SIDED,
    VSA_RESONO_PERT_LIVE_TWO_SIDED,
    VSA_RESONO_PERT_LIVE_MAPS
};

static const vsa_resonator_mask_scope_t ARM_SCOPE[N_ARMS] = {
    VSA_RESONO_MASK_BOTH_USES,
    VSA_RESONO_MASK_UNBIND_ONLY,
    VSA_RESONO_MASK_BOTH_USES,
    VSA_RESONO_MASK_BOTH_USES,
    VSA_RESONO_MASK_BOTH_USES,
    VSA_RESONO_MASK_BOTH_USES
};

static const float ARM_R[N_SPARSE] = { 0.001f, 0.01f, 0.04f };

typedef struct {
    int n_cycle;        /* any repeat within 50 iters  (brief's primary)     */
    int n_period_ge2;   /* genuine non-trivial cycle                            */
    int n_fixed;        /* period-1 fixed point                                 */
    int n_role_correct; /* role-slots correct, out of 3N                       */
    int n_tuple_correct;/* exact factor tuple correct                          */
    int n_bary;         /* converged BUT wrong tuple (barycentre failure)       */
    int n_soa_mismatch; /* recall-path conformance disagreements                */
    int iters[N_PROBLEMS];
} agg_t;

static vsa_phase_hv_t gA[VSA_RESONO_ROLES * VSA_RESONO_M];
static vsa_phase_hv_t gB[VSA_RESONO_ROLES * VSA_RESONO_M];
static vsa_resonator_ctx_t gCtx;
static int g_truth[3 * N_PROBLEMS];
static vsa_phase_table_t gTable;

static void make_pert(int arm, float r, vsa_resonator_pert_t *p)
{
    p->mode     = ARM_MODE[arm];
    p->scope    = ARM_SCOPE[arm];
    p->sparsity = r;
    p->mask_seed = 0x1D0B1ED0ULL + (uint64_t)arm * 7919ULL + (uint64_t)(r * 100000.0f);
    p->live_seed = 0x1F5A1F5AULL + (uint64_t)arm * 104729ULL + (uint64_t)(r * 100000.0f);
}

/* ---- Wilson 95% interval on a rate, so a "2x" claim can be judged against
 *      its own sampling noise instead of being read as exact --------------- */
static void wilson(int k, int n, double *lo, double *hi)
{
    const double z = 1.959963984540054;
    double p, den, c, h;
    if (n <= 0) { *lo = *hi = 0.0; return; }
    p = (double)k / (double)n;
    den = 1.0 + z * z / (double)n;
    c  = (p + z * z / (2.0 * (double)n)) / den;
    h  = z * sqrt(p * (1.0 - p) / (double)n + z * z / (4.0 * (double)n * (double)n)) / den;
    *lo = c - h;  *hi = c + h;
}

static int cmp_int(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static void run_arm(int arm, float r, int nprob, agg_t *out)
{
    vsa_resonator_pert_t pert;
    vsa_resonator_result_t res;
    int p;

    memset(out, 0, sizeof(*out));
    make_pert(arm, r, &pert);

    vsa_resonator_prepare(&gCtx, gA, gB, &pert);

    for (p = 0; p < nprob; p++) {
        vsa_phase_hv_t q;
        int f, ok_roles = 1;   /* AND over all three roles */

        vsa_resonator_make_query(gA, g_truth[3 * p + 0], g_truth[3 * p + 1],
                                 g_truth[3 * p + 2], &gTable, &q);
        vsa_resonator_run(&gCtx, &q, &gTable, &pert, &res);

        out->iters[p] = res.iters;
        if (res.cycle_found) {
            out->n_cycle++;
            if (res.cycle_period >= 2) out->n_period_ge2++;
        }
        if (res.converged) out->n_fixed++;

        for (f = 0; f < VSA_RESONO_ROLES; f++) {
            if (res.role_top1[f] == g_truth[3 * p + f]) out->n_role_correct++;
            else                                        ok_roles = 0;
        }
        if (ok_roles) out->n_tuple_correct++;

        /* barycentre failure: reached a fixed point but the tuple is wrong */
        if (res.converged && !ok_roles) out->n_bary++;

        out->n_soa_mismatch += res.soa_mismatch;
    }

    vsa_resonator_release(&gCtx);
}

int main(int argc, char **argv)
{
    /* COMPILED CONFIG — printed from the actual macros the binary was built with.
       The old banner said "K = 16" in a COMMENT, which is unfalsifiable: a -D that
       silently loses to a header #define cannot be caught by a reader. If EXPECT_K
       was passed at compile time and does not match what we compiled, abort. */
    printf("== COMPILED CONFIG ==\n");
    printf("  K (phases/dim)     = %d   %s\n", (int)VSA_RESONO_K,
           VSA_PHASE_K_MAX >= VSA_RESONO_K ? "" : "*** EXCEEDS K_MAX ***");
    printf("  D (dims)           = %d\n", (int)VSA_PHASE_DIM);
    printf("  roles F            = %d\n", (int)VSA_RESONO_ROLES);
    printf("  codewords/role M   = %d\n", (int)VSA_RESONO_M);
    printf("  problems           = %d\n", (int)N_PROBLEMS);
    printf("  arms               = %d\n", (int)N_ARMS);
    printf("  max iter / window  = %d / %d\n", (int)VSA_RESONO_MAX_ITER, (int)VSA_RESONO_CYCLE_WIN);
    {
        const int want = VSA_RESONO_K;
#ifdef EXPECT_K
        if (want != EXPECT_K) {
            printf("\nABORT: compiled K=%d but EXPECT_K=%d. A -D lost to a header #define;\n"
                   "       the sweep would silently have measured the same thing twice.\n",
                   want, (int)EXPECT_K);
            return 3;
        }
        printf("  EXPECT_K check     = %d OK\n", (int)EXPECT_K);
#else
        (void)want;
        printf("  (no -DEXPECT_K given: config not cross-checked)\n");
#endif
    }

    /* A self-cancelling IFS is a BUG, not a finding. Round 1 of arm (e) was
       exactly that and its inert 1.01x was reported as physics. Refuse to run. */
    {
        int net = vsa_ifs_selfcheck(VSA_RESONO_K);
        if (net == 0) {
            printf("\nABORT: the IFS map cycle is the identity (net displacement 0 mod K).\n"
                   "       Arm (e) would measure my own bug. Fix the maps before reading\n"
                   "       anything from this run.\n");
            return 2;
        }
        printf("IFS self-check: net displacement %d mod %d (non-zero, maps compound) OK\n", net, (int)VSA_RESONO_K);
    }
    static agg_t agg[N_ARMS][N_SPARSE];
    static agg_t repeat_check;
    static int iters_sorted[N_PROBLEMS];

    int nprob = N_PROBLEMS;
    int nsub = N_SUBSAMPLE;
    int bundle_bad;
    int a, s, p, f;
    double base_rate, base_p2;
    double best_ratio = 0.0, best_p2_ratio = 0.0;
    int best_arm = 0;
    int verdict_pass;
    const char *verdict_reason;
    int total_soa_mismatch = 0;

    if (argc > 1) {          /* timing probe only; the canonical run uses
                               the default 1,000 problems */
        nprob = atoi(argv[1]);
        if (nprob < 1 || nprob > N_PROBLEMS) nprob = N_PROBLEMS;
    }

    vsa_phase_table_init(&gTable, VSA_RESONO_K);
    vsa_resonator_init(&gTable);
    vsa_resonator_build_codebooks(gA, gB, SEED_CODEBOOK);

    /* fixed problem set, shared by every arm */
    for (p = 0; p < N_PROBLEMS; p++) {
        uint64_t st = SEED_PROBLEMS + (uint64_t)p * 0x9E3779B97F4A7C15ULL;
        for (f = 0; f < VSA_RESONO_ROLES; f++) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            g_truth[3 * p + f] = (int)((st >> 33) % VSA_RESONO_M);
        }
    }

    printf("== RSPK step 6: asymmetry control -> ACF -> IFS A/B ==\n");
    printf("F=%d  M=%d  D=%d  K=%d  problems=%d  max_iter=%d  cycle_window=%d\n",
           VSA_RESONO_ROLES, VSA_RESONO_M, VSA_PHASE_DIM, VSA_RESONO_K,
           nprob, VSA_RESONO_MAX_ITER, VSA_RESONO_CYCLE_WIN);
    printf("codebook_seed=0x%llX  problem_seed=0x%llX\n",
           (unsigned long long)SEED_CODEBOOK, (unsigned long long)SEED_PROBLEMS);
    printf("chance: role top-1 = 1/%d = %.3f%%   |   exact tuple = 1/%d = %.5f%%\n",
           VSA_RESONO_M, 100.0 / VSA_RESONO_M,
           VSA_RESONO_M * VSA_RESONO_M * VSA_RESONO_M,
           100.0 / ((double)VSA_RESONO_M * VSA_RESONO_M * VSA_RESONO_M));
    printf("primary: all %d arms at r=0.04 (largest published) on %d problems\n",
           N_ARMS, nprob);
    printf("sweep:   all %d arms at r=0.001 and r=0.01 on the first %d problems "
           "(r=0.04 is the primary table above)\n", N_ARMS, nsub);

    /* Fast path (VSA_FAST_BASELINE=1): answer CONFIG questions, not physics ones.
       Baseline arm only, no r-sweep, no conformance, no determinism block -- seconds
       instead of minutes. The full factorial stays the default for real S6 runs. */
    if (getenv("VSA_FAST_BASELINE")) {
        agg_t base;
        run_arm(0, ARM_R[N_SPARSE - 1], N_PROBLEMS, &base);
        {   /* Split the headline number. n_cycle counts ANY repeat, which INCLUDES
               period-1 (frozen) states -- so "limit cycle" here is really
               "stuck", and a 100% arm may be total convergence, not oscillation. */
            double n = (double)N_PROBLEMS;
            printf("== FAST BASELINE ==\n  K=%d  n=%d\n", (int)VSA_RESONO_K, (int)N_PROBLEMS);
            printf("  any-repeat (headline) = %6.2f%%\n", 100.0*base.n_cycle/n);
            printf("  frozen (period-1)     = %6.2f%%\n", 100.0*base.n_fixed/n);
            printf("  TRUE period>=2        = %6.2f%%\n", 100.0*base.n_period_ge2/n);
            printf("  neither               = %6.2f%%\n", 100.0*(n-base.n_cycle)/n);
            printf("  role_correct (3N)     = %6.2f%%\n", 100.0*base.n_role_correct/(3.0*n));
            printf("  tuple_correct         = %6.2f%%\n", 100.0*base.n_tuple_correct/n);
        }
        return 0;
    }

    bundle_bad = vsa_resonator_bundle_conformance(&gTable, 20, SEED_CODEBOOK ^ 0xBEEFULL);
    printf("bundle conformance: memoised-bundle vs vsa_phase_bundle disagreements = %d of %d dims\n\n",
           bundle_bad, 20 * VSA_PHASE_DIM);

    for (a = 0; a < N_ARMS; a++)
        run_arm(a, ARM_R[N_SPARSE - 1], nprob, &agg[a][N_SPARSE - 1]);

    for (s = 0; s < N_SPARSE - 1; s++)
        for (a = 0; a < N_ARMS; a++)
            run_arm(a, ARM_R[s], nsub, &agg[a][s]);

    /* Determinism self-check. The baseline arm ignores r (mode NONE), so its
     * r=0.001 and r=0.01 subsample runs must also agree with each other --
     * compare the re-run against agg[0][0], which is n=nsub, not n=nprob. */
    run_arm(0, ARM_R[N_SPARSE - 1], nsub, &repeat_check);

    for (a = 0; a < N_ARMS; a++)
        for (s = 0; s < N_SPARSE; s++)
            total_soa_mismatch += agg[a][s].n_soa_mismatch;
    total_soa_mismatch += repeat_check.n_soa_mismatch;

    printf("determinism: baseline re-run (n=%d) n_cycle=%d, expected %d -> %s\n",
           nsub, repeat_check.n_cycle, agg[0][0].n_cycle,
           repeat_check.n_cycle == agg[0][0].n_cycle ? "STABLE" : "MISMATCH");
    printf("baseline r-invariance: r=0.001 n_cycle=%d vs r=0.01 n_cycle=%d -> %s\n",
           agg[0][0].n_cycle, agg[0][1].n_cycle,
           agg[0][0].n_cycle == agg[0][1].n_cycle ? "STABLE" : "MISMATCH");
    printf("recall-path conformance: SoA-fast-path vs AoS-similarity argmax disagreements = %d of %d role-readouts\n",
           total_soa_mismatch,
           (N_ARMS * nprob + N_ARMS * (N_SPARSE - 1) * nsub + nsub) * VSA_RESONO_ROLES);

    /* ---------------- primary table: r = largest published (0.04) --------- */
    printf("\n== Measurements (r = 0.04, largest published) ==\n");
    printf("arm                      limcycle  95%% CI            period>=2  fixed  med_it  p95  nonconv  role_acc (chance 4.545%%)  tuple_acc (chance 0.00939%%)  barycentre\n");

    for (a = 0; a < N_ARMS; a++) {
        agg_t *g = &agg[a][N_SPARSE - 1];
        double lo, hi, med, p95;
        memcpy(iters_sorted, g->iters, sizeof(int) * (size_t)nprob);
        qsort(iters_sorted, (size_t)nprob, sizeof(int), cmp_int);
        med = (double)iters_sorted[nprob / 2];
        p95 = (double)iters_sorted[(int)((double)(nprob - 1) * 0.95)];
        wilson(g->n_cycle, nprob, &lo, &hi);
        printf("%s  %6.2f%%  [%.1f,%.1f]  %8.2f%%  %5.1f%%  %5.0f %4.0f  %7d  %6.2f%%              %6.2f%%              %6.2f%%\n",
               ARM_NAME[a],
               100.0 * g->n_cycle / nprob, 100.0 * lo, 100.0 * hi,
               100.0 * g->n_period_ge2 / nprob,
               100.0 * g->n_fixed / nprob,
               med, p95, nprob - g->n_fixed,
               100.0 * g->n_role_correct / (3.0 * nprob),
               100.0 * g->n_tuple_correct / nprob,
               100.0 * g->n_bary / nprob);
    }

    /* ---------------- r sweep (sensitivity; note the denominators) -------- */
    printf("\n== limit-cycle rate across the published r sweep ==\n");
    printf("  NOTE: r=0.001 and r=0.01 are measured on the first %d problems;\n"
           "        r=0.04 is the full n=%d run from the primary table above.\n", nsub, nprob);
    printf("arm                         r=0.001    r=0.01     r=0.04\n");
    for (a = 0; a < N_ARMS; a++) {
        printf("%s", ARM_NAME[a]);
        for (s = 0; s < N_SPARSE - 1; s++)
            printf("   %6.2f%% ", 100.0 * agg[a][s].n_cycle / nsub);
        printf("   %6.2f%%\n", 100.0 * agg[a][N_SPARSE - 1].n_cycle / nprob);
    }
    printf("\n== period>=2 rate across the published r sweep ==\n");
    printf("arm                         r=0.001    r=0.01     r=0.04\n");
    for (a = 0; a < N_ARMS; a++) {
        printf("%s", ARM_NAME[a]);
        for (s = 0; s < N_SPARSE - 1; s++)
            printf("   %6.2f%% ", 100.0 * agg[a][s].n_period_ge2 / nsub);
        printf("   %6.2f%%\n", 100.0 * agg[a][N_SPARSE - 1].n_period_ge2 / nprob);
    }

    /* ---------------- kill criterion -------------------------------------- */
    base_rate = (double)agg[0][N_SPARSE - 1].n_cycle / nprob;
    base_p2   = (double)agg[0][N_SPARSE - 1].n_period_ge2 / nprob;

    printf("\n== Kill criterion ==\n");
    printf("stated: \"Limit-cycle rate changes < 2x => the noise thread (incl. IFS) is dead.\"\n");
    printf("primary metric = brief's definition (any repeat within %d iterations)\n", VSA_RESONO_CYCLE_WIN);
    printf("baseline (a) limit-cycle rate = %.2f%%\n", 100.0 * base_rate);

    for (a = 1; a < N_ARMS; a++) {
        double r_arm = (double)agg[a][N_SPARSE - 1].n_cycle / nprob;
        double p2    = (double)agg[a][N_SPARSE - 1].n_period_ge2 / nprob;
        double ratio, ratio2;

        if (base_rate > 0.0) ratio = (r_arm >= base_rate) ? (r_arm / base_rate) : (base_rate / r_arm);
        else                  ratio = (r_arm > 0.0) ? 1e9 : 1.0;
        if (base_p2 > 0.0)   ratio2 = (p2 >= base_p2) ? (p2 / base_p2) : (base_p2 / p2);
        else                 ratio2 = (p2 > 0.0) ? 1e9 : 1.0;

        printf("  %s  rate=%6.2f%%  ratio_vs_baseline=%.2fx   (period>=2: %.2f%% -> %.2fx)\n",
               ARM_NAME[a], 100.0 * r_arm, ratio, 100.0 * p2, ratio2);
        if (ratio > best_ratio)     { best_ratio = ratio; best_arm = a; }
        if (ratio2 > best_p2_ratio) { best_p2_ratio = ratio2; }
    }

    printf("largest limit-cycle change: %.2fx  (arm %s)\n", best_ratio, ARM_NAME[best_arm]);
    printf("largest period>=2 change:   %.2fx\n", best_p2_ratio);

    if (best_ratio >= 2.0) {
        verdict_pass = 1;
        verdict_reason = "limit-cycle rate changed by >= 2x";
    } else {
        verdict_pass = 0;
        verdict_reason = "no arm changed the limit-cycle rate by >= 2x";
    }

    if (verdict_pass) printf("\nPASS\n");
    else               printf("\nKILL: %s\n", verdict_reason);

    return 0;
}
