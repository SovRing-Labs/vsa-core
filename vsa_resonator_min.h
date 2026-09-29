/* ===========================================================================
 * vsa_resonator_min.h — RSPK Step 6: minimal single-resonator iteration loop
 * on the phase kernel (VSAW2-S6-ASYM).
 *
 * Scope (locked by _Pads/VSA/vsa-steps-w2.wave.yaml):
 *   synthetic only; F = 3 factors, M = 22 codewords per factor,
 *   D = 5,120, K = 16; one resonator per query; scores returned, never a class.
 *
 * Design note — the two codebook uses:
 *   A  is the codebook used for cleanup (recall) and rebind (write-back).
 *   B  is a byte-identical copy of A used for unbind (read-out).
 *   Both are the SAME logical codebook. A perturbation can therefore be
 *   applied to (i) the unbind use only  -> copy-asymmetric, or
 *   (ii) both uses identically         -> copy-symmetric, and self-cancelling
 *   in the round trip (s - B') + A'  ==  s - B + A. That cancellation is
 *   exactly the control the step-6 asymmetry test needs.
 *
 * Codebook preparation (perturbation + SoA transpose) is hoisted OUT of the
 * per-query loop: it is a property of the arm, not of the query. The recall
 * path uses vsa_phase_cleanup_avx2_pretransposed, which computes the SAME
 * metric as vsa_phase_cleanup (mean cos of the phase difference) over a
 * transposed codebook. The harness cross-checks the two paths agree on the
 * argmax and reports the agreement, so the speedup is verified, not assumed.
 *
 * Clean-room note (RSPKO-REL-STR §8): this module returns SCORE VECTORS. It
 * exposes no class-assignment API and performs no bipolarize step. `role_top1`
 * is the argmax of the returned score vector (the mechanism's own recall
 * readout), not a system-level class.
 *
 * LINK CONSTRAINT: the step-6 test_cmd compiles only
 *   vsa_asym_test.c vsa_resonator_min.c vsa_phase.c
 * so vsa_phase_int8.c is NOT linked. This module may call only symbols
 * provided by vsa_phase.c.
 * ========================================================================= */

#ifndef VSA_RESONATOR_MIN_H
#define VSA_RESONATOR_MIN_H

#include <stdint.h>
#include <stddef.h>
#include "vsa_phase.h"

#define VSA_RESONO_ROLES     3
#define VSA_RESONO_M         22
#ifndef VSA_RESONO_K
#define VSA_RESONO_K         16   /* overridable: -DVSA_RESONO_K=5 to sweep the split */
#endif
#define VSA_RESONO_MAX_ITER  50
#define VSA_RESONO_CYCLE_WIN 50   /* brief: a cycle = revisiting a prior state
                                     within 50 iterations */

/* ---- perturbation modes --------------------------------------------------
 * FROZEN_TWO_SIDED  random subset of dims, +/-1 mod K. Symmetric control.
 * FROZEN_ONE_SIDED  random subset of dims, +1  mod K. This is ACF's
 *                    asymmetry: a codebook perturbation at initialisation
 *                    that is one-sided, digital, and frozen (RSPK-PHYS:
 *                    "asymmetric codebook perturbation at initialisation",
 *                    one-sided, sparsity r, "the finding is asymmetry,
 *                    not noise").
 *   LIVE_TWO_SIDED    re-drawn against the state at every iteration. This is
 *                    stochastic NOISE, not an IFS. Kept as the honest
 *                    control for LIVE_MAPS below. (S6 re-verification,
 *                    2026-09-28: this arm was previously labelled "IFS" and
 *                    the KILL it produced was reported as "incl. IFS". It is
 *                    not an IFS -- it is a re-drawn random mask.)
 *   LIVE_MAPS         a REAL iterated function system: M fixed, DETERMINISTIC
 *                    contraction maps applied in a fixed cycle, one per
 *                    iteration. Determinism is the whole point and the thing
 *                    noise cannot fake: the map applied at iteration t is a
 *                    function of t alone, so the orbit has attractor
 *                    structure and the run is exactly reproducible. Amplitude
 *                    r still selects the fraction of dimensions each map
 *                    touches, matching the published ACF range.
 * ---------------------------------------------------------------------- */
typedef enum {
    VSA_RESONO_PERT_NONE             = 0,
    VSA_RESONO_PERT_FROZEN_TWO_SIDED = 1,
    VSA_RESONO_PERT_FROZEN_ONE_SIDED = 2,
    VSA_RESONO_PERT_LIVE_TWO_SIDED   = 3,
    VSA_RESONO_PERT_LIVE_MAPS        = 4   /* the real IFS */
} vsa_resonator_pert_mode_t;

/* IFS shape. N_MAPS fixed contractive maps, cycled in order. */
#define VSA_RESONO_IFS_MAPS  4

/* Returns net displacement of the IFS cycle mod K. 0 == self-cancelling == BUG. */
int vsa_ifs_selfcheck(uint8_t K);

/* Which codebook uses carry the mask. */
typedef enum {
    VSA_RESONO_MASK_UNBIND_ONLY = 0,  /* one copy   (asymmetric use) */
    VSA_RESONO_MASK_BOTH_USES   = 1   /* both copies, identical mask */
} vsa_resonator_mask_scope_t;

typedef struct {
    vsa_resonator_pert_mode_t  mode;
    vsa_resonator_mask_scope_t scope;
    float    sparsity;    /* r = fraction of dimensions perturbed */
    uint64_t mask_seed;   /* seed for the frozen mask                 */
    uint64_t live_seed;   /* seed for the per-iteration perturbation */
} vsa_resonator_pert_t;

/* ---- prepared arm context (built once per arm, reused for every query) -- */
typedef struct {
    vsa_phase_hv_t       A_l[VSA_RESONO_ROLES * VSA_RESONO_M]; /* perturbed  */
    vsa_phase_hv_t       B_l[VSA_RESONO_ROLES * VSA_RESONO_M]; /* perturbed  */
    vsa_phase_codebook_t A_soa[VSA_RESONO_ROLES];              /* transposed  */
    int                  soa_ready;
} vsa_resonator_ctx_t;

/* Result of one resonator run. Carries scores, plus the loop's own
 * convergence telemetry. */
typedef struct {
    float role_score[VSA_RESONO_ROLES][VSA_RESONO_M];  /* per-role recall scores */
    float tuple_score;   /* score of the joint argmax tuple                  */

    int   iters;         /* iterations actually run                          */
    int   converged;     /* 1 if a period-1 fixed point was reached           */
    int   cycle_found;   /* 1 if any state repeated inside the window
                          * (the brief's primary limit-cycle definition)     */
    int   cycle_period;  /* period of first repeat; 1 == fixed point, 0 none */
    int   cycle_at;      /* iteration index of the first repeat, -1 if none  */

    int   role_top1[VSA_RESONO_ROLES]; /* argmax of role_score[] (mechanism) */

    /* recall-path conformance: does the SoA fast path pick the same codeword
     * as the AoS similarity scan? 0 == agree (expected for every run) */
    int   soa_mismatch;
} vsa_resonator_result_t;

/* ---- API ---------------------------------------------------------------- */

/* Build the memoised 3-input bundle table from the phase table. Call once
 * before any run. Idempotent. */
void vsa_resonator_init(const vsa_phase_table_t *table);

/* Proof that the memoised bundle equals vsa_phase_bundle(): returns the
 * number of disagreeing dimensions over `trials` random 3-input bundles
 * (expected 0). */
int vsa_resonator_bundle_conformance(const vsa_phase_table_t *table, int trials,
                                     uint64_t seed);

/* Build the role codebooks (identical in both uses). Deterministic in seed. */
void vsa_resonator_build_codebooks(vsa_phase_hv_t *A, vsa_phase_hv_t *B, uint64_t seed);

/* Compose a query by binding the three role codewords (generative model, uses
 * the CLEAN codebook). */
void vsa_resonator_make_query(const vsa_phase_hv_t *A, int g0, int g1, int g2,
                              const vsa_phase_table_t *table, vsa_phase_hv_t *q);

/* Prepare one arm: apply the frozen codebook mask, transpose the recall
 * codebooks. Call once per arm; release with vsa_resonator_release(). */
void vsa_resonator_prepare(vsa_resonator_ctx_t *ctx,
                           const vsa_phase_hv_t *A, const vsa_phase_hv_t *B,
                           const vsa_resonator_pert_t *pert);

void vsa_resonator_release(vsa_resonator_ctx_t *ctx);

/* Run one resonator over one query under the prepared arm. */
void vsa_resonator_run(const vsa_resonator_ctx_t *ctx,
                       const vsa_phase_hv_t *query,
                       const vsa_phase_table_t *table,
                       const vsa_resonator_pert_t *pert,
                       vsa_resonator_result_t *out);

#endif /* VSA_RESONATOR_MIN_H */
