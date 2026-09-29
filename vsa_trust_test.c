/* vsa_trust_test.c — RSPK build-order step 2 gate.
 *
 * The kill criterion, verbatim from the RSPK build order:
 *
 *     "Barycenter not separable at > 99% specificity => no reader ships."
 *
 * Prints one final line: PASS, or KILL: <reason>. Exit 0 only on PASS.
 *
 * Method and every threshold are pre-registered in docs/SEALED-METRIC.md.
 * The stimulus parameters are pre-registered there too, so neither the
 * stimulus nor the decision boundary can be tuned to a verdict.
 *
 * Test harness, not a library: never touches /dev/shm/vsa_matrix_bus, opens no
 * files, and reads no other lane's work.
 */

#include "vsa_trust.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- pre-registered stimulus (SEALED-METRIC section 6) -------------------- */
#define SEED            0x5EED5EED0002ULL
#define TRIALS          1000
#define N_MEM           256        /* per-trial resampled memory             */
#define N_MEM_HARD      1000       /* hard-mode confirmation, 100 trials     */
#define RHO_PRIMARY     0.10       /* primary near-equal perturbation       */
#define TRIALS_HARD     100

/* ---- deterministic RNG: xoshiro256** seeded through splitmix64 ------------- */
typedef struct { uint64_t s[4]; } rng_t;

static void rng_init(rng_t* r, uint64_t seed) {
    uint64_t x = seed ? seed : 1;
    for (int i = 0; i < 4; ++i) {
        x += 0x9E3779B97F4A7C15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        r->s[i] = z ^ (z >> 31);
    }
}
static uint64_t rng_next(rng_t* r) {
    uint64_t* s = r->s;
    uint64_t res = s[0] + s[3], t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = (s[3] << 45) | (s[3] >> 19);
    return res;
}
static double rng_unit(rng_t* r) { return (double)(rng_next(r) >> 11) * 0x1.0p-53; }

/* ---- hypervectors --------------------------------------------------------- */
/* Matches vsa_selftest.c:32 mkhv(): sign and zero planes are independent
 * random words, so p_active = 1/2 exactly. */
static void hv_random(rng_t* r, vsa_hv_t* h) {
    for (size_t w = 0; w < VSA_WORDS; ++w) { h->sign[w] = rng_next(r); h->zero[w] = rng_next(r); }
}

/* Resample a fraction rho of src's ACTIVE dims. rho is a stimulus, not a
 * threshold: the decision boundary is z (SEALED-METRIC 4a). */
static void hv_perturb(rng_t* r, const vsa_hv_t* src, double rho, vsa_hv_t* out) {
    *out = *src;
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        uint64_t sel = rng_next(r);          /* which active dims to touch  */
        for (int b = 0; b < 64; ++b) {
            if (!((out->zero[w] >> b) & 1ULL)) continue;
            if (!((sel >> b) & 1ULL)) continue;
            if (rng_unit(r) >= rho) continue;
            double u = rng_unit(r);          /* same law as hv_random         */
            if      (u < 0.50) out->zero[w] &= ~(1ULL << b);   /* -> stasis  */
            else if (u < 0.75) out->sign[w] &= ~(1ULL << b);   /* -> +1      */
            /* else stays -1 */
        }
    }
}

/* ---- accounting ----------------------------------------------------------- */
#define NPANEL 4
#define NPANEL_DISC 5      /* the 4 kill-accounting panels + 1 disclosure panel */
enum { P_SINGLE = 0, P_META_PAIR, P_META_SUBSET, P_BARY, P_TWIN, P_COUNT };
static const char* PANEL_NAME[NPANEL_DISC] = {
    "R1 single", "R2 meta-pair", "R2b meta-subset", "R3 barycenter", "D1 twin-single"
};
static const uint8_t PANEL_TRUTH[NPANEL_DISC] = {
    VSA_TRUST_CLASS_SINGLE, VSA_TRUST_CLASS_METASTABLE, VSA_TRUST_CLASS_METASTABLE,
    VSA_TRUST_CLASS_BARYCENTER, VSA_TRUST_CLASS_SINGLE
};
static const int PANEL_NEG[NPANEL_DISC] = { 1, 1, 1, 0, 1 };  /* negative for the binary predicate */
/* P_TWIN is EXCLUDED from the kill accounting: see the note in the report. The
 * first gate run used it as the single-pattern query and KILLED for two
 * separate reasons that both trace to that one confound. It stays in the gate
 * output as a disclosure panel so the finding is not silently dropped. */

static int g_fail;                 /* 1 => print KILL */
static char g_reason[512];
static void fail(const char* fmt, ...) {
    if (g_fail) return;             /* keep the FIRST failure: the root one */
    g_fail = 1;
    va_list ap; va_start(ap, fmt); vsnprintf(g_reason, sizeof g_reason, fmt, ap); va_end(ap);
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ---- per-panel run -------------------------------------------------------- */
typedef struct {
    int    trials;
    int    class_ok;
    int    flag_ok;                 /* binary barycenter predicate         */
    int    integrity_fires;
    int    path_mismatch;           /* score() vs bary()+score_bary()      */
    double sum_gap_bary, sum_gap_solo, sum_phi, sum_m, sum_s1, sum_s2;
    double min_gap_bary, max_gap_bary, min_gap_solo, max_gap_solo;
    double min_phi, max_phi;
} panel_t;

static void panel_run(int panel, size_t n_mem, int trials, double rho,
                      vsa_trust_result_t* first_out, panel_t* p) {
    vsa_hv_t* pat = malloc(sizeof(vsa_hv_t) * n_mem);
    const vsa_hv_t** ptr = malloc(sizeof(void*) * n_mem);
    vsa_hv_t bary, q;

    memset(p, 0, sizeof *p);
    p->min_gap_bary = 1e30; p->max_gap_bary = -1e30;
    p->min_gap_solo = 1e30; p->max_gap_solo = -1e30;
    p->min_phi = 1e30; p->max_phi = -1e30;

    for (int t = 0; t < trials; ++t) {
        rng_t r; rng_init(&r, SEED + (uint64_t)panel * 1000003u + (uint64_t)t);

        hv_random(&r, &pat[0]);
        hv_perturb(&r, &pat[0], rho, &pat[1]);
        for (size_t i = 2; i < n_mem; ++i) hv_random(&r, &pat[i]);
        for (size_t i = 0; i < n_mem; ++i) ptr[i] = &pat[i];

        vsa_trust_barycenter(ptr, n_mem, &bary);

        switch (panel) {
            case P_SINGLE:      memcpy(&q, &pat[2], sizeof q); break;
            case P_TWIN:        memcpy(&q, &pat[0], sizeof q); break;  /* the confound, kept */
            case P_META_PAIR:   vsa_bundle(ptr, 2, &q); break;
            case P_META_SUBSET: vsa_bundle(ptr, 8, &q); break;
            default:            memcpy(&q, &bary, sizeof q); break;
        }

        vsa_trust_scores_t sc; vsa_trust_result_t res;
        vsa_trust_score_bary(&q, &bary, ptr, n_mem, &sc);
        vsa_trust_classify(&q, &sc, &res);

        /* G6: the two API paths must not disagree. */
        vsa_trust_scores_t sc2; vsa_trust_result_t r2;
        vsa_trust_score(&q, ptr, n_mem, &sc2);
        vsa_trust_classify(&q, &sc2, &r2);
        if (res.convergence_class != r2.convergence_class ||
            res.barycenter_flag != r2.barycenter_flag ||
            res.load_ratio_bucket != r2.load_ratio_bucket) p->path_mismatch++;

        double gb = sc.sim_barycenter - sc.sim_best;
        double gs = sc.sim_best - sc.sim_second;
        p->sum_gap_bary += gb; p->sum_gap_solo += gs; p->sum_phi += sc.coherence;
        p->sum_m += sc.sim_barycenter; p->sum_s1 += sc.sim_best; p->sum_s2 += sc.sim_second;
        if (gb < p->min_gap_bary) p->min_gap_bary = gb;
        if (gb > p->max_gap_bary) p->max_gap_bary = gb;
        if (gs < p->min_gap_solo) p->min_gap_solo = gs;
        if (gs > p->max_gap_solo) p->max_gap_solo = gs;
        if (sc.coherence < p->min_phi) p->min_phi = sc.coherence;
        if (sc.coherence > p->max_phi) p->max_phi = sc.coherence;

        p->class_ok += (res.convergence_class == PANEL_TRUTH[panel]);
        p->flag_ok  += (res.barycenter_flag == (PANEL_NEG[panel] ? 0 : 1));
        p->integrity_fires += (res.integrity != 0);
        if (t == 0 && first_out) *first_out = res;
    }

    p->trials = trials;
    free(pat); free((void*)ptr);
}

/* ---- ROC over z, replayed from stored gaps (disclosure only) -------------- */
typedef struct { double gb, gs, phi, nq; int truth_bary; } sample_t;

int main(void) {
    double t0 = now_s();
    panel_t P[NPANEL];
    vsa_trust_result_t first;

    printf("VSA trust gate — RSPK step 2.  D=%d  margin z=%.1f  M=%.6f\n",
           VSA_TRUST_D, VSA_TRUST_Z, VSA_TRUST_MARGIN);
    printf("derived: sigma_sim=%.7f  sigma_gap=%.7f  phi_thresh(nq=5120)=%.6f\n",
           VSA_TRUST_SIGMA_SIM, VSA_TRUST_SIGMA_GAP, vsa_trust_phi_threshold(N_MEM, 5120));
    printf("derived: phi_null(N=%d)=%.6f   [k=1/sqrt(pi*(N-1))=%.6f]\n",
           N_MEM, vsa_trust_phi_null(N_MEM), 1.0 / sqrt(3.14159265358979323846 * (N_MEM - 1)));
    printf("stimulus: seed=0x%llX  trials=%d  N=%d  rho=%.2f  p_active=0.50\n",
           (unsigned long long)SEED, TRIALS, N_MEM, RHO_PRIMARY);

    for (int p = 0; p < NPANEL; ++p)
        panel_run(p, N_MEM, TRIALS, RHO_PRIMARY, (p == 0) ? &first : NULL, &P[p]);

    /* ---- per-panel measurement table ---- */
    printf("\n-- panels (N=%d, %d trials each) --\n", N_MEM, TRIALS);
    printf("%-16s %-11s %6s %6s %6s   %9s %9s   %9s %9s\n",
           "panel", "truth", "class", "flag", "integ", "gap_bary", "gap_solo", "phi", "pathdiff");
    int tot_class_ok = 0, tot_t = 0, tot_int = 0, tot_path = 0;
    for (int p = 0; p < NPANEL; ++p) {
        printf("%-16s %-11s %5.1f%% %5.1f%% %6d   %+9.5f %+9.5f   %9.5f %9d\n",
               PANEL_NAME[p],
               PANEL_TRUTH[p] == VSA_TRUST_CLASS_SINGLE ? "single" :
               PANEL_TRUTH[p] == VSA_TRUST_CLASS_METASTABLE ? "metastable" : "barycenter",
               100.0 * P[p].class_ok / P[p].trials,
               100.0 * P[p].flag_ok / P[p].trials,
               P[p].integrity_fires,
               P[p].sum_gap_bary / P[p].trials,
               P[p].sum_gap_solo / P[p].trials,
               P[p].sum_phi / P[p].trials,
               P[p].path_mismatch);
        printf("%-16s %-11s   ranges: gap_bary [%+.5f,%+.5f]  gap_solo [%+.5f,%+.5f]  phi [%.5f,%.5f]\n",
               "", "", P[p].min_gap_bary, P[p].max_gap_bary,
               P[p].min_gap_solo, P[p].max_gap_solo, P[p].min_phi, P[p].max_phi);
        tot_class_ok += P[p].class_ok; tot_t += P[p].trials;
        tot_int += P[p].integrity_fires; tot_path += P[p].path_mismatch;
    }

    /* ---- the kill criterion ---- */
    int neg_total = 0, fp = 0, pos_total = 0, tp = 0;
    for (int p = 0; p < NPANEL; ++p) {
        if (PANEL_NEG[p]) { neg_total += P[p].trials; fp  += P[p].trials - P[p].flag_ok; }
        else              { pos_total += P[p].trials; tp += P[p].flag_ok; }
    }
    double spec = (double)(neg_total - fp) / (double)neg_total;
    double sens = (double)tp / (double)pos_total;
    double acc3 = (double)tot_class_ok / (double)tot_t;
    double maj_bin = (double)neg_total / (double)(neg_total + pos_total);
    double maj3 = 1.0 / (double)NPANEL;

    printf("\n-- the kill criterion: barycenter separable at > 99%% specificity --\n");
    printf("G1 specificity  = %d/%d = %.4f%%   (false positives %d; >99%% permits <= %d)\n",
           neg_total - fp, neg_total, 100.0 * spec, fp, (int)((1.0 - 0.99) * neg_total));
    printf("   majority-class rate (predict 'not barycenter') = %.4f%%\n", 100.0 * maj_bin);
    printf("G2 sensitivity  = %d/%d = %.4f%%   (added guard, strictly harder than G1)\n",
           tp, pos_total, 100.0 * sens);
    printf("G3 3-way class  = %d/%d = %.4f%%   majority-class rate = %.4f%%\n",
           tot_class_ok, tot_t, 100.0 * acc3, 100.0 * maj3);
    printf("G4 integrity channel fired %d times in %d trials (want 0)\n", tot_int, tot_t);
    printf("G6 API path disagreements: %d in %d trials (want 0)\n", tot_path, tot_t);

    /* ---- disclosure: the confound that KILLED the first run ---- */
    {
        panel_t T; panel_run(P_TWIN, N_MEM, TRIALS, RHO_PRIMARY, NULL, &T);
        printf("\n-- disclosure D1 (NOT in the kill accounting): single-pattern query taken from\n"
               "   the near-equal PAIR, so the memory holds a near-duplicate of it --\n");
        printf("   class %5.1f%%  flag %5.1f%%  integrity fired %d/%d  phi %.5f  gap_solo %+.5f\n",
               100.0 * T.class_ok / T.trials, 100.0 * T.flag_ok / T.trials,
               T.integrity_fires, T.trials, T.sum_phi / T.trials, T.sum_gap_solo / T.trials);
        printf("   for contrast, the same panel with an independent member (R1): phi %.5f  gap_solo %+.5f\n",
               P[0].sum_phi / P[0].trials, P[0].sum_gap_solo / P[0].trials);
    }

    /* ---- hard mode at N=1000 (fewer trials, disclosed as such) ---- */
    printf("\n-- hard mode: memory N=%d, %d trials, rho=%.2f --\n", N_MEM_HARD, TRIALS_HARD, RHO_PRIMARY);
    panel_t H[NPANEL];
    for (int p = 0; p < NPANEL; ++p) {
        panel_run(p, N_MEM_HARD, TRIALS_HARD, RHO_PRIMARY, NULL, &H[p]);
        printf("%-16s class %5.1f%%  flag %5.1f%%  integ %d  phi %.5f\n",
               PANEL_NAME[p], 100.0 * H[p].class_ok / H[p].trials,
               100.0 * H[p].flag_ok / H[p].trials, H[p].integrity_fires, H[p].sum_phi / H[p].trials);
    }
    int hard_spec_n = 0, hard_spec_ok = 0;
    for (int p = 0; p < NPANEL; ++p) if (PANEL_NEG[p]) { hard_spec_n += H[p].trials; hard_spec_ok += H[p].flag_ok; }
    printf("hard-mode specificity = %d/%d = %.4f%%  sensitivity = %d/%d = %.4f%%\n",
           hard_spec_ok, hard_spec_n, 100.0 * hard_spec_ok / hard_spec_n,
           H[P_BARY].flag_ok, H[P_BARY].trials, 100.0 * H[P_BARY].flag_ok / H[P_BARY].trials);

    /* ---- rho sweep (stimulus disclosure, not a threshold) ---- */
    printf("\n-- rho sweep (near-equal pair separation, N=%d, 200 trials) --\n", N_MEM);
    const double rhos[3] = { 0.05, 0.10, 0.20 };
    for (int i = 0; i < 3; ++i) {
        panel_t A, B; panel_run(P_META_PAIR, N_MEM, 200, rhos[i], NULL, &A);
        panel_run(P_SINGLE,  N_MEM, 200, rhos[i], NULL, &B);
        printf("rho=%.2f  R2 metastable class %5.1f%% (gap_solo %+.5f)   R1 single class %5.1f%%\n",
               rhos[i], 100.0 * A.class_ok / A.trials, A.sum_gap_solo / A.trials,
               100.0 * B.class_ok / B.trials);
    }

    /* ---- subset-size boundary: where does the class flip? ---- */
    printf("\n-- subset-size boundary (memory N=%d, where METASTABLE becomes BARYCENTER) --\n", N_MEM);
    {
        rng_t r; rng_init(&r, SEED ^ 0xB0u);
        vsa_hv_t* pat = malloc(sizeof(vsa_hv_t) * N_MEM);
        const vsa_hv_t** ptr = malloc(sizeof(void*) * N_MEM);
        hv_random(&r, &pat[0]);
        hv_perturb(&r, &pat[0], RHO_PRIMARY, &pat[1]);
        for (size_t i = 2; i < N_MEM; ++i) hv_random(&r, &pat[i]);
        for (size_t i = 0; i < N_MEM; ++i) ptr[i] = &pat[i];
        vsa_hv_t bary; vsa_trust_barycenter(ptr, N_MEM, &bary);
        const size_t ks[] = { 1, 2, 4, 8, 16, 32, 64, 128, 200, 250, 254, 255, 256 };
        for (size_t i = 0; i < sizeof ks / sizeof ks[0]; ++i) {
            vsa_hv_t q; vsa_bundle(ptr, ks[i], &q);
            vsa_trust_scores_t sc; vsa_trust_result_t res;
            vsa_trust_score_bary(&q, &bary, ptr, N_MEM, &sc);
            vsa_trust_classify(&q, &sc, &res);
            printf("  k=%3zu  class=%-10s flag=%d  gap_bary=%+.5f gap_solo=%+.5f phi=%.5f\n",
                   ks[i],
                   res.convergence_class == VSA_TRUST_CLASS_BARYCENTER ? "barycenter" :
                   res.convergence_class == VSA_TRUST_CLASS_SINGLE ? "single" : "metastable",
                   res.barycenter_flag, sc.sim_barycenter - sc.sim_best,
                   sc.sim_best - sc.sim_second, sc.coherence);
        }
        free(pat); free((void*)ptr);
    }

    /* ---- load-ratio ladder, measured (SEALED-METRIC section 5) ---- */
    printf("\n-- load-ratio ladder: r = D_f / N, bucket must be self-consistent and monotone --\n");
    {
        const size_t ns[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000 };
        int prev_bucket = -1, monotone = 1;
        printf("  %5s %7s %10s %8s %8s\n", "N", "D_f", "r = D_f/N", "bucket", "pack3");
        for (size_t i = 0; i < sizeof ns / sizeof ns[0]; ++i) {
            size_t n = ns[i];
            rng_t r; rng_init(&r, SEED + 7919u + n);
            vsa_hv_t* pat = malloc(sizeof(vsa_hv_t) * n);
            const vsa_hv_t** ptr = malloc(sizeof(void*) * n);
            for (size_t j = 0; j < n; ++j) hv_random(&r, &pat[j]);
            for (size_t j = 0; j < n; ++j) ptr[j] = &pat[j];
            vsa_hv_t bary; vsa_trust_barycenter(ptr, n, &bary);
            vsa_trust_scores_t sc; vsa_trust_result_t res;
            vsa_trust_score_bary(&bary, &bary, ptr, n, &sc);
            vsa_trust_classify(&bary, &sc, &res);
            uint8_t b = vsa_trust_load_bucket(sc.load_ratio);
            if ((int)b < prev_bucket) monotone = 0;
            prev_bucket = b;
            printf("  %5zu %7d %10.4f %8u %8u\n", n, sc.d_f, sc.load_ratio, b, vsa_trust_pack3(&res));
            free(pat); free((void*)ptr);
        }
        printf("  bucket monotone non-decreasing in N: %s\n", monotone ? "yes" : "NO");
        if (!monotone) fail("load bucket not monotone in N");
    }

    /* ---- 3-bit pack round trip over all legal combinations ---- */
    {
        int ok = 0, total = 0;
        for (int c = 0; c < 4; ++c) for (int f = 0; f < 2; ++f) for (int b = 0; b < 8; ++b) {
            vsa_trust_result_t r; memset(&r, 0, sizeof r);
            r.convergence_class = (uint8_t)c; r.barycenter_flag = (uint8_t)f;
            r.load_ratio_bucket = (uint8_t)b;
            uint8_t p = vsa_trust_pack3(&r);
            total++;
            if (p == (uint8_t)(((c & 3) << 1) | (f & 1)) && (p >> 1) == c && (p & 1) == f &&
                r.load_ratio_bucket == b) ok++;
        }
        printf("\n-- 3-bit pack round trip: %d/%d --\n", ok, total);
        if (ok != total) fail("pack round trip %d/%d", ok, total);
    }

    /* ---- z ROC: the decision point is theory, this is disclosure ---- */
    printf("\n-- ROC over z (pre-registered decision point is z=%.1f) --\n", VSA_TRUST_Z);
    {
        /* Replay the pre-registered panels, retaining the three gaps. */
        static sample_t sm[NPANEL][TRIALS];
        for (int p = 0; p < NPANEL; ++p) {
            rng_t r;
            for (int t = 0; t < TRIALS; ++t) {
                rng_init(&r, SEED + (uint64_t)p * 1000003u + (uint64_t)t);
                vsa_hv_t* pat = malloc(sizeof(vsa_hv_t) * N_MEM);
                const vsa_hv_t** ptr = malloc(sizeof(void*) * N_MEM);
                hv_random(&r, &pat[0]); hv_perturb(&r, &pat[0], RHO_PRIMARY, &pat[1]);
                for (size_t i = 2; i < N_MEM; ++i) hv_random(&r, &pat[i]);
                for (size_t i = 0; i < N_MEM; ++i) ptr[i] = &pat[i];
                vsa_hv_t bary, q; vsa_trust_barycenter(ptr, N_MEM, &bary);
                switch (p) {
                    case P_SINGLE:      memcpy(&q, &pat[2], sizeof q); break;  /* must match panel_run */
                    case P_TWIN:        memcpy(&q, &pat[0], sizeof q); break;
                    case P_META_PAIR:   vsa_bundle(ptr, 2, &q); break;
                    case P_META_SUBSET: vsa_bundle(ptr, 8, &q); break;
                    default:            memcpy(&q, &bary, sizeof q); break;
                }
                vsa_trust_scores_t sc;
                vsa_trust_score_bary(&q, &bary, ptr, N_MEM, &sc);
                sm[p][t].gb = sc.sim_barycenter - sc.sim_best;
                sm[p][t].gs = sc.sim_best - sc.sim_second;
                sm[p][t].phi = sc.coherence;
                sm[p][t].nq = (double)vsa_active_count(&q);
                sm[p][t].truth_bary = PANEL_NEG[p] ? 0 : 1;
                free(pat); free((void*)ptr);
            }
        }
        const double zs[] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0 };
        printf("   %5s %12s %12s %12s %14s\n", "z", "specificity", "sensitivity", "3-way acc", "phi-only spec");
        for (size_t zi = 0; zi < sizeof zs / sizeof zs[0]; ++zi) {
            double M = zs[zi] * VSA_TRUST_SIGMA_GAP;
            int tn = 0, nneg = 0, tpz = 0, npos = 0, cok = 0, ptn = 0;
            for (int p = 0; p < NPANEL; ++p) for (int t = 0; t < TRIALS; ++t) {
                sample_t* s = &sm[p][t];
                int cls;
                if      (s->gb > M) cls = VSA_TRUST_CLASS_BARYCENTER;
                else if (s->gs > M) cls = VSA_TRUST_CLASS_SINGLE;
                else                 cls = VSA_TRUST_CLASS_METASTABLE;
                cok += (cls == PANEL_TRUTH[p]);
                /* the SHIPPED decision: the flag is the class at 1-bit resolution */
                int flag = (cls == VSA_TRUST_CLASS_BARYCENTER);
                /* the DEMOTED phi-only channel, at the same z, for comparison */
                int pflag = (s->phi >= vsa_trust_phi_null(N_MEM) + zs[zi] / (2.0 * sqrt(s->nq)));
                if (s->truth_bary) { npos++; tpz += flag; }
                else               { nneg++; tn += (flag == 0); ptn += (pflag == 0); }
            }
            printf("   %5.1f %11.4f%% %11.4f%% %11.4f%% %13.4f%%%s\n", zs[zi],
                   100.0 * tn / nneg, 100.0 * tpz / npos, 100.0 * cok / (NPANEL * TRIALS),
                   100.0 * ptn / nneg,
                   (zs[zi] == VSA_TRUST_Z) ? "   <- pre-registered" : "");
        }
    }

    printf("\nelapsed %.2f s\n", now_s() - t0);

    /* ---- gates ---- */
    if (spec <= 0.99)        fail("G1 barycenter not separable at >99%% specificity: %.4f%%", 100.0 * spec);
    if (sens <= 0.99)        fail("G2 barycenter sensitivity %.4f%%", 100.0 * sens);
    if (acc3 < 0.99)         fail("G3 3-way class accuracy %.4f%%", 100.0 * acc3);
    if (tot_int != 0)        fail("G4 integrity channel fired %d times", tot_int);
    if (tot_path != 0)       fail("G6 API path disagreed %d times", tot_path);

    if (g_fail) { printf("KILL: %s\n", g_reason); return 1; }
    printf("PASS\n");
    return 0;
}
