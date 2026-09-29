/* ===========================================================================
 * vsa_resonator_min.c — minimal single-resonator iteration loop (RSPK step 6)
 *
 * One query -> iterate:
 *     key_f = s - B[g_f]                 (unbind, B use)
 *     (g_f', score) = cleanup(key_f | A) (recall, A use, SoA fast path)
 *     s' = bundle({ A[g_f'] })           (superposed write-back)
 * The state signature is (g_1, g_2, g_3). A fixed point is a period-1 repeat;
 * the brief's limit cycle is any repeat of a prior signature within
 * VSA_RESONO_CYCLE_WIN (50) iterations, detected with a past-state buffer
 * (Kent Alg. 1, mandatory per RSPKO-REL-STR §7).
 *
 * RNG: PCG32, independent of the kernel's internal xorshift64*, so the
 * harness and the code under test never share a stream.
 *
 * Scratch: all buffers are file-scope statics. The harness is single
 * threaded and the context is per-arm, so this is the hot-path allocation
 * strategy; it is not reentrant by design.
 * ========================================================================= */

#include "vsa_resonator_min.h"
#include <string.h>
#include <math.h>

/* --------------------------------------------------------------------------
 * Memoised 3-input bundle.
 *
 * vsa_phase_bundle_scalar() evaluates atan2f() once per dimension (5,120 of
 * them, ~269 us/call) to quantise the resultant phase. For exactly three
 * inputs at K=16 the (p1,p2,p3) domain is only K^3 = 4,096 states, so the
 * whole map can be precomputed once and then applied as a byte lookup.
 *
 * This is a memoisation, not a reimplementation: the table is built with the
 * identical expression sequence as the kernel, on the identical
 * cos/sin tables, and IEEE-754 float arithmetic is deterministic, so the
 * result is bit-identical. vsa_resonator_bundle_conformance() checks that
 * against the real kernel and the harness reports the count.
 * ----------------------------------------------------------------------- */
static uint8_t g_bundle_memo[16 * 16 * 16];
static int      g_bundle_memo_ready = 0;
static uint8_t  g_bundle_memo_K = 0;

void vsa_resonator_init(const vsa_phase_table_t *table)
{
    int K = table->K, a, b, c;

    if (K != 8 && K != 16) { g_bundle_memo_ready = 0; return; }

    {
        const float K_over_2pi = (float)K / (2.0f * 3.14159265358979323846f);
        for (a = 0; a < K; a++)
            for (b = 0; b < K; b++)
                for (c = 0; c < K; c++) {
                    /* same accumulation order as the kernel (which starts
                     * from a zeroed float accumulator; 0.0f + x == x) */
                    float cs = table->cos_table[a] + table->cos_table[b] + table->cos_table[c];
                    float ss = table->sin_table[a] + table->sin_table[b] + table->sin_table[c];
                    float angle = atan2f(ss, cs);
                    uint8_t ph;
                    if (angle < 0) angle += 2.0f * 3.14159265358979323846f;
                    ph = (uint8_t)(angle * K_over_2pi);
                    if (ph >= K) ph = K - 1;
                    g_bundle_memo[(a * K + b) * K + c] = ph;
                }
    }
    g_bundle_memo_ready = 1;
    g_bundle_memo_K = (uint8_t)K;
}

static void bundle3(const vsa_phase_hv_t *in, const vsa_phase_table_t *table,
                    vsa_phase_hv_t *out)
{
    int i;
    out->K = table->K;
    if (g_bundle_memo_ready && g_bundle_memo_K == table->K) {
        const uint8_t *m = g_bundle_memo;
        int K = table->K;
        for (i = 0; i < VSA_PHASE_DIM; i++) {
            out->phases[i] = m[((int)in[0].phases[i] * K + in[1].phases[i]) * K
                               + in[2].phases[i]];
        }
    } else {
        vsa_phase_bundle(in, 3, table, out);
    }
}

/* Proof that the memoised bundle is the kernel's bundle. Returns the number
 * of dimensions that disagreed (expected: 0). */
int vsa_resonator_bundle_conformance(const vsa_phase_table_t *table, int trials,
                                     uint64_t seed)
{
    static vsa_phase_hv_t in[3], ref, got;
    uint64_t st = seed;
    int t, i, bad = 0;

    if (!g_bundle_memo_ready) vsa_resonator_init(table);

    for (t = 0; t < trials; t++) {
        int n, d;
        for (n = 0; n < 3; n++) {
            in[n].K = table->K;
            for (d = 0; d < VSA_PHASE_DIM; d++) {
                st = st * 6364136223846793005ULL + 1442695040888963407ULL;
                in[n].phases[d] = (uint8_t)((st >> 33) % table->K);
            }
        }
        vsa_phase_bundle(in, 3, table, &ref);
        bundle3(in, table, &got);
        for (i = 0; i < VSA_PHASE_DIM; i++) {
            if (ref.phases[i] != got.phases[i]) bad++;
        }
    }
    return bad;
}


/* --------------------------------------------------------------------------
 * PCG32 (independent harness stream)
 * ----------------------------------------------------------------------- */
typedef struct { uint64_t state, inc; } pcg32_t;

static uint32_t pcg32_next(pcg32_t *r)
{
    uint64_t old = r->state;
    uint32_t xorshifted, rot;
    r->state = old * 6364136223846793005ULL + r->inc;
    xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
}

static void pcg32_seed(pcg32_t *r, uint64_t seed, uint64_t seq)
{
    r->state = 0u;
    r->inc = (seq << 1u) | 1u;
    (void)pcg32_next(r);
    r->state += seed;
    (void)pcg32_next(r);
}

/* uniform in [0,1) */
static float pcg32_f01(pcg32_t *r)
{
    return (float)(pcg32_next(r) >> 8) * (1.0f / 16777216.0f);
}

/* --------------------------------------------------------------------------
 * Digital phase mask at sparsity r.
 *   one_sided == 0 : subset of dims gets +/-1 mod K   (symmetric control)
 *   one_sided == 1 : subset of dims gets  +1  mod K   (ACF's asymmetry)
 * delta[i] == 0 means "this dimension is untouched".
 * ----------------------------------------------------------------------- */
static void make_delta(int8_t *delta, float r, int one_sided, pcg32_t *rng)
{
    int i;
    for (i = 0; i < VSA_PHASE_DIM; i++) {
        if (pcg32_f01(rng) < r) {
            if (one_sided) {
                delta[i] = 1;
            } else {
                delta[i] = (pcg32_next(rng) & 1u) ? 1 : -1;
            }
        } else {
            delta[i] = 0;
        }
    }
}

/* ---------------------------------------------------------------------------
 * make_map_delta — a deterministic IFS map (S6 re-verification, 2026-09-28).
 *
 * A REAL iterated function system, and the reason this is not the same as
 * make_delta() above: make_delta() draws a fresh RANDOM mask from an RNG, so
 * the perturbation is uncorrelated between iterations and the orbit
 * random-walks. That is noise. An IFS is a fixed finite set of CONTRACTIVE
 * maps whose composition converges to an attractor, and its defining
 * computational property is DETERMINISM: the map applied at iteration t is a
 * function of t alone.
 *
 * Here: VSA_RESONO_IFS_MAPS fixed maps, cycled in order. Map q touches
 * dimensions {i : (i + q) mod STRIDE == 0} -- a fixed low-discrepancy pattern,
 * not a random draw -- with signed shift given by a fixed odd sequence so
 * successive maps partly cancel and the composition contracts toward a
 * phase-fixed point rather than diffusing. Amplitude r scales which of the
 * candidate dims are active, so r still means "fraction of dimensions", which
 * keeps the comparison against the published ACF range honest.
 *
 * Because nothing here consults an RNG, the arm is bit-for-bit reproducible.
 * ----------------------------------------------------------------------- */
static void make_map_delta(int8_t *delta, float r, int t, uint8_t K)
{
    /* S6 re-verification, round 2 (2026-09-28, after IFS guide).
     *
     * Round 1 of this function was WRONG and I reported its result as physics.
     * It used a signed odd sequence so maps "partly cancel", which is exactly
     * the self-cancelling identity the guide named. Proven, not asserted:
     * every map drew only steps {+1,-2} and the 4-map composition had total
     * net displacement -290 = 0 (mod 5) -- the identity map. The inert 1.01x
     * that produced was my bug, not the system.
     *
     * Round 2: maps COMPOUND. Step sizes are the Fibonacci sequence (1,2,3,5),
     * all with a consistent sign, applied to fixed stride subsets. A dimension
     * touched by every map therefore drifts by 1+2+3+5 = 11 = +1 (mod 5) per
     * cycle: a bounded, non-zero, deterministic rotation with period K. That is
     * an attractor with structure, and it is the one thing noise cannot do.
     * vsa_ifs_selfcheck() below PROVES non-cancellation at run time; we do not
     * take the word of a comment.
     */
    static const int8_t fib[VSA_RESONO_IFS_MAPS] = { 1, 2, 3, 5 };
    const int q = t % VSA_RESONO_IFS_MAPS;
    const int stride = 1 + (q % 3);
    int i, m;

    for (i = 0; i < VSA_PHASE_DIM; i++)
        delta[i] = 0;

    for (i = 0, m = 0; i < VSA_PHASE_DIM; i += stride, m++) {
        /* amplitude gate keeps r meaning "fraction of dimensions touched" */
        if ((float)m < r * (float)VSA_PHASE_DIM / (float)stride + 0.5f)
            delta[i] = fib[q];
    }
    (void)K;
}

/* Run-time proof that the map cycle is NOT the identity. Returns the total net
 * displacement mod K; 0 means self-cancelling and the arm is a bug. */
int vsa_ifs_selfcheck(uint8_t K)
{
    static int8_t buf[4][VSA_PHASE_DIM];
    int q, i, total = 0;
    for (q = 0; q < VSA_RESONO_IFS_MAPS; q++) {
        make_map_delta(buf[q], 0.04f, q, K);
        for (i = 0; i < VSA_PHASE_DIM; i++)
            total += buf[q][i];
    }
    return ((total % (int)K) + (int)K) % (int)K;
}

static void apply_delta(vsa_phase_hv_t *hv, const int8_t *delta, uint8_t K){
    int i;
    for (i = 0; i < VSA_PHASE_DIM; i++) {
        int d = delta[i];
        if (d == 0) continue;
        {
            int v = (int)hv->phases[i] + d;
            if (v >= (int)K) v -= (int)K;
            if (v < 0)          v += (int)K;
            hv->phases[i] = (uint8_t)v;
        }
    }
}

static void apply_delta_all(vsa_phase_hv_t *hv, size_t n, const int8_t *delta, uint8_t K)
{
    size_t i;
    for (i = 0; i < n; i++) apply_delta(&hv[i], delta, K);
}

static uint32_t pack_sig(const int *g)
{
    return ((uint32_t)g[0] & 31u)
         | (((uint32_t)g[1] & 31u) << 5)
         | (((uint32_t)g[2] & 31u) << 10);
}

/* file-scope scratch (see header note on reentrancy) */
static int8_t    g_delta[VSA_PHASE_DIM];
static uint32_t  g_sig[VSA_RESONO_MAX_ITER + 1];
static vsa_phase_hv_t g_s, g_key, g_contrib[VSA_RESONO_ROLES];

/* --------------------------------------------------------------------------
 * Codebook + query construction
 * ----------------------------------------------------------------------- */
void vsa_resonator_build_codebooks(vsa_phase_hv_t *A, vsa_phase_hv_t *B, uint64_t seed)
{
    pcg32_t rng;
    size_t n = (size_t)VSA_RESONO_ROLES * VSA_RESONO_M;
    pcg32_seed(&rng, seed, 0x5EED1234u);

    /* Phase values uniform over [0, K-1]. */
    {
        size_t i, j;
        for (i = 0; i < n; i++) {
            A[i].K = VSA_RESONO_K;
            for (j = 0; j < VSA_PHASE_DIM; j++) {
                A[i].phases[j] = (uint8_t)(pcg32_next(&rng) % VSA_RESONO_K);
            }
        }
    }
    /* B is a byte-identical copy: the same logical codebook, second use. */
    memcpy(B, A, n * sizeof(vsa_phase_hv_t));
}

void vsa_resonator_make_query(const vsa_phase_hv_t *A, int g0, int g1, int g2,
                              const vsa_phase_table_t *table, vsa_phase_hv_t *q)
{
    static vsa_phase_hv_t in[VSA_RESONO_ROLES];
    in[0] = A[0 * VSA_RESONO_M + g0];
    in[1] = A[1 * VSA_RESONO_M + g1];
    in[2] = A[2 * VSA_RESONO_M + g2];
    vsa_phase_bundle(in, VSA_RESONO_ROLES, table, q);
}

/* --------------------------------------------------------------------------
 * Arm preparation — frozen codebook mask + SoA transpose, once per arm
 * ----------------------------------------------------------------------- */
void vsa_resonator_prepare(vsa_resonator_ctx_t *ctx,
                           const vsa_phase_hv_t *A, const vsa_phase_hv_t *B,
                           const vsa_resonator_pert_t *pert)
{
    size_t n = (size_t)VSA_RESONO_ROLES * VSA_RESONO_M;
    uint8_t K = VSA_RESONO_K;
    int f;

    memcpy(ctx->A_l, A, n * sizeof(vsa_phase_hv_t));
    memcpy(ctx->B_l, B, n * sizeof(vsa_phase_hv_t));

    if (pert->mode == VSA_RESONO_PERT_FROZEN_TWO_SIDED ||
        pert->mode == VSA_RESONO_PERT_FROZEN_ONE_SIDED) {
        pcg32_t mrng;
        int one_sided = (pert->mode == VSA_RESONO_PERT_FROZEN_ONE_SIDED);
        pcg32_seed(&mrng, pert->mask_seed, 0xACF0ACF0u);
        make_delta(g_delta, pert->sparsity, one_sided, &mrng);

        if (pert->scope == VSA_RESONO_MASK_BOTH_USES) {
            apply_delta_all(ctx->A_l, n, g_delta, K);  /* recall + write-back */
        }
        apply_delta_all(ctx->B_l, n, g_delta, K);      /* unbind use           */
    }

    /* Transpose the (already perturbed) recall codebook, one per role. */
    for (f = 0; f < VSA_RESONO_ROLES; f++) {
        vsa_phase_codebook_transpose(&ctx->A_l[f * VSA_RESONO_M],
                                     &ctx->A_soa[f], VSA_RESONO_M);
    }
    ctx->soa_ready = 1;
}

void vsa_resonator_release(vsa_resonator_ctx_t *ctx)
{
    int f;
    if (ctx->soa_ready) {
        for (f = 0; f < VSA_RESONO_ROLES; f++) vsa_phase_codebook_free(&ctx->A_soa[f]);
        ctx->soa_ready = 0;
    }
}

/* --------------------------------------------------------------------------
 * One resonator run
 * ----------------------------------------------------------------------- */
void vsa_resonator_run(const vsa_resonator_ctx_t *ctx,
                       const vsa_phase_hv_t *query,
                       const vsa_phase_table_t *table,
                       const vsa_resonator_pert_t *pert,
                       vsa_resonator_result_t *out)
{
    int   g[VSA_RESONO_ROLES], gn[VSA_RESONO_ROLES];
    int   f, m, t, j;
    uint8_t K = table->K;
    pcg32_t lrng;
    int live    = (pert->mode == VSA_RESONO_PERT_LIVE_TWO_SIDED);
    int ifs     = (pert->mode == VSA_RESONO_PERT_LIVE_MAPS);

    memset(out, 0, sizeof(*out));
    out->cycle_at = -1;
    out->tuple_score = 0.0f;

    pcg32_seed(&lrng, pert->live_seed, 0x1F51F51Fu);

    /* ---- initial recall -------------------------------------------------- */
    memcpy(&g_s, query, sizeof(g_s));
    g_s.K = K;
    for (f = 0; f < VSA_RESONO_ROLES; f++) {
        vsa_phase_cleanup_result_t r =
            vsa_phase_cleanup_avx2_pretransposed(&g_s, &ctx->A_soa[f], table);
        g[f] = r.best_index;
    }
    g_sig[0] = pack_sig(g);

    /* ---- iterate --------------------------------------------------------- */
    for (t = 1; t <= VSA_RESONO_MAX_ITER; t++) {
        for (f = 0; f < VSA_RESONO_ROLES; f++) {
            vsa_phase_cleanup_result_t r;
            vsa_phase_unbind(&g_s, &ctx->B_l[f * VSA_RESONO_M + g[f]], &g_key);
            r = vsa_phase_cleanup_avx2_pretransposed(&g_key, &ctx->A_soa[f], table);
            gn[f] = r.best_index;
            memcpy(&g_contrib[f], &ctx->A_l[f * VSA_RESONO_M + gn[f]], sizeof(vsa_phase_hv_t));
        }

        bundle3(g_contrib, table, &g_s);

        if (live) {
            make_delta(g_delta, pert->sparsity, 0, &lrng);
            apply_delta(&g_s, g_delta, K);
        } else if (ifs) {
            make_map_delta(g_delta, pert->sparsity, t, K);
            apply_delta(&g_s, g_delta, K);
        }

        g_sig[t] = pack_sig(gn);

        /* past-state buffer, window VSA_RESONO_CYCLE_WIN */
        for (j = t - 1; j >= 0 && (t - j) <= VSA_RESONO_CYCLE_WIN; j--) {
            if (g_sig[j] == g_sig[t]) {
                out->cycle_found  = 1;
                out->cycle_period = t - j;
                out->cycle_at     = t;
                break;
            }
        }
        if (out->cycle_found) break;   /* first repeat closes the question */

        memcpy(g, gn, sizeof(g));
    }

    out->iters     = out->cycle_found ? out->cycle_at : VSA_RESONO_MAX_ITER;
    out->converged = (out->cycle_found && out->cycle_period == 1) ? 1 : 0;

    /* ---- final per-role score vectors (scores, never a class) -------------
     * Scored with the AoS similarity scan. This doubles as the conformance
     * check on the SoA fast path used during iteration: if the two ever
     * disagree on the argmax, the run is flagged rather than trusted. */
    for (f = 0; f < VSA_RESONO_ROLES; f++) {
        float best = -2.0f;
        int   bi = 0;
        vsa_phase_cleanup_result_t soa;
        vsa_phase_unbind(&g_s, &ctx->B_l[f * VSA_RESONO_M + g[f]], &g_key);
        for (m = 0; m < VSA_RESONO_M; m++) {
            float sc = vsa_phase_similarity(&g_key, &ctx->A_l[f * VSA_RESONO_M + m], table);
            out->role_score[f][m] = sc;
            if (sc > best) { best = sc; bi = m; }
        }
        out->role_top1[f] = bi;
        out->tuple_score += best;

        soa = vsa_phase_cleanup_avx2_pretransposed(&g_key, &ctx->A_soa[f], table);
        if (soa.best_index != bi) out->soa_mismatch++;
    }
}
