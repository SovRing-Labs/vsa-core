/* vsa_trust.c — RSPK build-order step 2: sealed trust result.
 *
 * Every constant and threshold used here is pre-registered with its derivation
 * in docs/SEALED-METRIC.md. None is fitted to this step's test output.
 *
 * No new dependencies: the existing ternary kernel (vsa_kernel.h) is the only
 * thing this file uses, and it never bipolarises — every operation is ternary.
 */

#include "vsa_trust.h"

/* ------------------------------------------------------------------- sim */
double vsa_trust_sim(const vsa_hv_t* a, const vsa_hv_t* b) {
    /* vsa_dot is the inner product of the dense {-1,0,+1} vectors, so
     * sqrt(na*nb) is the exact norm product and this is the true cosine. */
    int32_t na = vsa_active_count(a);
    int32_t nb = vsa_active_count(b);
    if (na <= 0 || nb <= 0) return 0.0;
    return (double)vsa_dot(a, b) / sqrt((double)na * (double)nb);
}

/* ------------------------------------------------------------ barycenter */
void vsa_trust_barycenter(const vsa_hv_t* const* patterns, size_t n, vsa_hv_t* out) {
    if (n == 0) { vsa_hv_t zero; for (size_t w = 0; w < VSA_WORDS; ++w) { zero.sign[w] = 0; zero.zero[w] = 0; } *out = zero; return; }
    /* The barycenter of a ternary memory under the majority rule is exactly
     * what vsa_bundle computes: elementwise sum, then threshold, undecided
     * dims left in stasis. Reimplementing it here would be a second source of
     * truth for the same fixed point. */
    vsa_bundle(patterns, n, out);
}

/* ------------------------------------------------------------- coherence */
double vsa_trust_coherence(const vsa_hv_t* query, const vsa_hv_t* bary) {
    int64_t nq = 0, agree = 0;
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        uint64_t qa = query->zero[w];
        nq += __builtin_popcountll(qa);
        /* The consensus is already in the barycenter's planes: zero = decided,
         * sign = its polarity. So this is three popcounts, not a rescan. */
        agree += __builtin_popcountll(qa & bary->zero[w] & ~(query->sign[w] ^ bary->sign[w]));
    }
    if (nq == 0) return 0.0;
    return (double)agree / (double)nq;
}

double vsa_trust_phi_null(int32_t n) {
    if (n < 2) return 0.5;
    /* T = sum of the other n-1 ternary votes. T is lattice-symmetric about 0,
     * so P(T=0) ~ 1/sqrt(2*pi*var) with var = (n-1)/2, i.e. k = 1/sqrt(pi*(n-1)).
     * The consensus is decided unless T is exactly 0, and the query's one vote
     * is outvoted only when |T| > 1, which symmetry splits evenly. Hence
     * p_dec = 1-k and P(sign agree | co-active) = 1/2 + k. */
    double k = 1.0 / sqrt(3.14159265358979323846 * (double)(n - 1));
    return (1.0 - k) * (0.5 + k);
}

double vsa_trust_phi_threshold(int32_t n, int32_t nq) {
    /* phi is a mean over nq bits, so sd(phi_null) = 1/(2*sqrt(nq)); take the
     * same pre-registered z = 4. Measured at n=256, nq=5120: derived null mean
     * 0.51641, measured 0.51792, threshold 0.54436 — 3.8 sd of measured margin. */
    if (nq <= 0) return 1.0;
    return vsa_trust_phi_null(n) + VSA_TRUST_Z / (2.0 * sqrt((double)nq));
}

/* ------------------------------------------------------------------ score */
void vsa_trust_score_bary(const vsa_hv_t* query, const vsa_hv_t* bary,
                          const vsa_hv_t* const* patterns, size_t n,
                          vsa_trust_scores_t* out) {
    out->sim_barycenter = vsa_trust_sim(query, bary);
    out->sim_best = 0.0;
    out->sim_second = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double s = vsa_trust_sim(query, patterns[i]);
        if (s >= out->sim_best) {                 /* >= so ties are stable    */
            out->sim_second = out->sim_best;
            out->sim_best     = s;
        } else if (s > out->sim_second) {
            out->sim_second = s;
        }
    }

    out->coherence    = vsa_trust_coherence(query, bary);
    out->d_f          = vsa_active_count(bary);
    out->n_patterns   = (int32_t)n;
    out->load_ratio   = (n > 0) ? (double)out->d_f / (double)n : 0.0;
}

void vsa_trust_score(const vsa_hv_t* query, const vsa_hv_t* const* patterns,
                     size_t n, vsa_trust_scores_t* out) {
    static _Thread_local vsa_hv_t bary;   /* 2560 B, no heap, no re-entry */
    vsa_trust_barycenter(patterns, n, &bary);
    vsa_trust_score_bary(query, &bary, patterns, n, out);
}

/* --------------------------------------------------------------- classify */
void vsa_trust_classify(const vsa_hv_t* query, const vsa_trust_scores_t* scores,
                        vsa_trust_result_t* out) {
    (void)query;   /* the query enters through the scores; kept for the call
                    * shape the build order specifies and for future scores
                    * that need it. */

    out->s = *scores;

    double gap_bary = scores->sim_barycenter - scores->sim_best;
    double gap_solo = scores->sim_best - scores->sim_second;

    if (scores->n_patterns <= 0) {
        out->convergence_class = VSA_TRUST_CLASS_INVALID;
    } else if (gap_bary > VSA_TRUST_MARGIN) {
        /* more like the global average than like any single pattern */
        out->convergence_class = VSA_TRUST_CLASS_BARYCENTER;
    } else if (gap_solo > VSA_TRUST_MARGIN) {
        /* one pattern explains it far better than the next */
        out->convergence_class = VSA_TRUST_CLASS_SINGLE;
    } else {
        /* as many ways to explain this as one pattern as two */
        out->convergence_class = VSA_TRUST_CLASS_METASTABLE;
    }

    /* The flag is the class at 1-bit resolution. It was once derived from
     * coherence instead; see docs/SEALED-METRIC.md 4b for why measurement
     * rejected that (coherence is monotone in k with no gap in it, so any cut
     * of it slides with k, while the similarity gap changes sign decisively). */
    out->barycenter_flag = (uint8_t)(out->convergence_class == VSA_TRUST_CLASS_BARYCENTER);

    /* Integrity = the decision is ambiguous: the gap that GOVERNED the class
     * sits within one noise unit of its own threshold, so moving the threshold
     * by that much would flip the answer. Derived from sealed constants; no new
     * fitted value. On a decisively-single pattern (gap ~0.97) or a decisively
     * collapsed barycenter (gap ~0.90) this is 0; it fires only in the band a
     * reader must not trust. */
    double governing = (out->convergence_class == VSA_TRUST_CLASS_BARYCENTER) ? gap_bary : gap_solo;
    out->integrity = (uint8_t)(fabs(governing - VSA_TRUST_MARGIN) <= VSA_TRUST_SIGMA_GAP);

    out->load_ratio_bucket = vsa_trust_load_bucket(scores->load_ratio);
}

/* ------------------------------------------------------------------- pack */
uint8_t vsa_trust_pack3(const vsa_trust_result_t* r) {
    return (uint8_t)(((r->convergence_class & 0x3u) << 1) | (r->barycenter_flag & 0x1u));
}

/* ------------------------------------------------------------------ bucket */
uint8_t vsa_trust_load_bucket(double load_ratio) {
    /* Geometric ladder, ratio 4 (SEALED-METRIC section 5). Anchored on the
     * measured k*=100 (README.md:54 -> r~61, bucket 3) and the measured k*=8
     * (XP1 amendment 2 -> r~763, bucket 1). */
    if (load_ratio >= 1536.0) return 0;
    if (load_ratio >=  384.0) return 1;
    if (load_ratio >=   96.0) return 2;
    if (load_ratio >=   24.0) return 3;
    if (load_ratio >=    6.0) return 4;
    if (load_ratio >=    1.5) return 5;
    if (load_ratio >     0.0) return 6;
    return 7;   /* r <= 0: degenerate, reserved */
}
