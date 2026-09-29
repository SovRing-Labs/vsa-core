#ifndef VSA_TRUST_H
#define VSA_TRUST_H

/* ---------------------------------------------------------------------------
 * VSA trust result — RSPK build-order step 2.
 *
 * One sealed result type carrying the collapse/barycenter decision that XP1
 * CONVERGE amendment 4 made a first-class output of step 2:
 *
 * 3 bits on the confidence register:  convergence class (2) | barycenter flag (1)
 *
 * The flag is the class at 1-bit resolution: a consumer that only needs to know
 * "did this collapse to the global average?" reads bit 0 and never decodes the
 * class. That is why both are carried.
 *
 * `integrity` is set when the decision is AMBIGUOUS -- the gap that determined the
 * class sits within one noise unit (VSA_TRUST_SIGMA_GAP) of its threshold, so a
 * perturbation that small would flip the class. It is derived, costs nothing, and
 * is not a restatement of the class.
 *
 * `coherence` in the scores is REPORTED, NOT DECIDED ON. It was built as a second
 * independent opinion on the collapse -- the flag was once derived from it -- and
 * measurement rejected that use. `coherence` is a smooth monotone function of the
 * subset size k: 0.542 at k=4 rising to 1.000 at k=N, with no gap anywhere in it.
 * The metastable panels sit at 0.552 / 0.558, only ~8 sd below k=32 at 0.614, so
 * any fixed cut of it is an arbitrary choice that slides with k, while the
 * similarity gap (sim_barycenter - sim_best) changes sign decisively at the same
 * place. The shipped flag therefore comes from the class and `coherence` is
 * published so a caller can see how far from the consensus the query sits. The
 * full correction and the run that forced it are in docs/SEALED-METRIC.md 4b.
 *
 * plus a load-ratio bucket, an independent integrity channel, and the scores
 * themselves (RSPK patent design-around: return scores + a class, never a hard
 * decision alone; nothing here bipolarises).
 *
 * All constants are pre-registered in docs/SEALED-METRIC.md with their
 * derivations. Nothing here is fitted to test output.
 * ------------------------------------------------------------------------- */

#include "vsa_kernel.h"
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- sealed constants (docs/SEALED-METRIC.md sections 3 and 4) ------------- */

#define VSA_TRUST_D 10240                       /* the bus ABI, not a choice   */

/* sd(sim) for two independent HVs = 1/sqrt(D) = 0.0098821.
 * Derivation: co-active n_ab = D/4, Var(dot) = D/4, normalise by (D/2)^2.  */
#define VSA_TRUST_SIGMA_SIM  (1.0 / sqrt((double)VSA_TRUST_D))

/* sd of a difference of two near-independent sims = sqrt(2/D) = 0.013975.    */
#define VSA_TRUST_SIGMA_GAP  (1.41421356237309515 * VSA_TRUST_SIGMA_SIM)

/* z = 4 pre-registered: the kill criterion needs >99% specificity (z=2.326 is
 * the minimum), and the error being bounded is a MISSED COLLAPSE.            */
#define VSA_TRUST_Z 4.0

/* Decision margin on both similarity gaps: 4*sqrt(2/D) = 0.055900.
 * The actual signal is ~0.97, i.e. 17x this.                                 */
#define VSA_TRUST_MARGIN (VSA_TRUST_Z * VSA_TRUST_SIGMA_GAP)

/* --- convergence class (register bits [2:1]) -------------------------------- */
typedef enum {
    VSA_TRUST_CLASS_SINGLE     = 0,  /* pattern fixed point                 */
    VSA_TRUST_CLASS_METASTABLE = 1,  /* subset average fixed point          */
    VSA_TRUST_CLASS_BARYCENTER = 2,  /* global average fixed point          */
    VSA_TRUST_CLASS_INVALID    = 3   /* empty / undefined memory            */
} vsa_trust_class;

/* --- measured statistics --------------------------------------------------- */
typedef struct {
    double   sim_barycenter;  /* m: query vs the barycenter of all N       */
    double   sim_best;        /* s1: top-1 over stored patterns            */
    double   sim_second;      /* s2: top-2 over stored patterns            */
    double   coherence;       /* phi: sign agreement with the consensus    */
    double   load_ratio;      /* r = D_f / N                               */
    int32_t  d_f;             /* decided (non-stasis) dims of the barycenter*/
    int32_t  n_patterns;      /* N                                        */
} vsa_trust_scores_t;

/* --- the sealed result ----------------------------------------------------- */
typedef struct {
    uint8_t convergence_class;  /* 2 bits, vsa_trust_class                   */
    uint8_t barycenter_flag;    /* 1 bit                                    */
    uint8_t load_ratio_bucket;  /* 3 bits, 0-7                              */
    uint8_t integrity;          /* 1 when flag and class disagree           */
    vsa_trust_scores_t s;       /* the scores returned alongside the class  */
} vsa_trust_result_t;

/* --- API ------------------------------------------------------------------- */

/* Normalised ternary cosine: vsa_dot(a,b) / sqrt(active(a)*active(b)).
 * Returns 0.0 if either vector has no active dimension. */
double vsa_trust_sim(const vsa_hv_t* a, const vsa_hv_t* b);

/* The barycenter IS the majority rule in this kernel, so this delegates to
 * vsa_bundle rather than reimplementing it. CLOSE, not BUILD. */
void   vsa_trust_barycenter(const vsa_hv_t* const* patterns, size_t n, vsa_hv_t* out);

/* Sign agreement between query and the consensus, in [0,1].
 * phi = 1.0 exactly when query is the majority over the whole memory.        */
double vsa_trust_coherence(const vsa_hv_t* query, const vsa_hv_t* bary);

/* Score against a barycenter the caller already holds (cache it: the memory
 * does not change between queries). This is the slot-scoped reader path. */
void   vsa_trust_score_bary(const vsa_hv_t* query, const vsa_hv_t* bary,
                            const vsa_hv_t* const* patterns, size_t n,
                            vsa_trust_scores_t* out);

/* Score with the barycenter built for you. */
void   vsa_trust_score(const vsa_hv_t* query, const vsa_hv_t* const* patterns,
                       size_t n, vsa_trust_scores_t* out);

/* The sealed decision. Takes the query and the candidate scores, returns the
 * packed result. */
void   vsa_trust_classify(const vsa_hv_t* query, const vsa_trust_scores_t* scores,
                          vsa_trust_result_t* out);

/* The 3-bit confidence-register packet: (class << 1) | flag. */
uint8_t vsa_trust_pack3(const vsa_trust_result_t* r);

/* Bucket for load ratio r = D_f/N. Geometric ladder, ratio 4 (SEALED-METRIC 5).
 * Anchored on k*=100 (README.md:54, vsa_selftest.c test_capacity) and k*=8
 * (XP1 amendment 2) — neither measured by this step. */
uint8_t vsa_trust_load_bucket(double load_ratio);

/* Analytic null mean of phi for a query that is ONE member of an n-pattern
 * memory. T = sum of the other n-1 ternary votes is lattice-symmetric, so the
 * single query vote is outvoted only when |T| > 1:
 *     p_dec      = 1 - P(T=0)   = 1 - k
 *     P(sign eq) = 1/2 + P(T=0) = 1/2 + k        with k = 1/sqrt(pi*(n-1))
 *     phi_null   = (1-k)(1/2+k)
 * Measured 0.51792 at n=256 vs derived 0.51641 — see docs/SEALED-METRIC.md
 * section 4b, which records the correction and the run that forced it. */
double  vsa_trust_phi_null(int32_t n);

/* Coherence threshold: the analytic null mean plus the same pre-registered
 * z/(2*sqrt(nq)) margin used for the similarity gaps. */
double  vsa_trust_phi_threshold(int32_t n, int32_t nq);

#ifdef __cplusplus
}
#endif

#endif /* VSA_TRUST_H */
