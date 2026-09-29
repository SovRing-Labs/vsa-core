#ifndef VSA_PHASE_H
#define VSA_PHASE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * qFHRR-style Phase Kernel (RSPK Step 5)
 *
 * Phase hypervectors: D dimensions, each element is a phase in {0, ..., K-1}
 * Stored as packed uint8_t (K <= 16 fits in 4 bits, but we use full byte for simplicity)
 *
 * D=5,120 (aligned to 64 for AVX2), K in {8, 16}
 *
 * Operations:
 *   bind      = elementwise (a + b) mod K
 *   unbind    = elementwise (a - b) mod K
 *   bundle    = integer accumulation of unit phasors via K-entry cos/sin table
 *   similarity= mean cos(phase_diff * 2π/K)
 *   cleanup   = argmax similarity over a codebook
 * ------------------------------------------------------------------------- */

#define VSA_PHASE_DIM 5120
#define VSA_PHASE_WORDS (VSA_PHASE_DIM / 64)  /* 80 for AVX2 alignment */
#define VSA_PHASE_K_MAX 16

typedef struct {
    uint8_t phases[VSA_PHASE_DIM];  /* each in [0, K-1] */
    uint8_t K;                      /* 8 or 16 */
} vsa_phase_hv_t;

/* Cos/sin lookup table for bundling (K entries) */
typedef struct {
    float cos_table[VSA_PHASE_K_MAX];
    float sin_table[VSA_PHASE_K_MAX];
    float cos_diff_table[VSA_PHASE_K_MAX];  /* cos(2π * d / K) for d in 0..K-1 */
    uint8_t K;
} vsa_phase_table_t;

/* Result of cleanup operation */
typedef struct {
    int best_index;      /* index in codebook, -1 if empty */
    float best_score;    /* similarity score */
} vsa_phase_cleanup_result_t;

/* ---------------------------------------------------------------------------
 * Initialization
 * ------------------------------------------------------------------------- */

/* Build cos/sin table for given K (8 or 16) */
void vsa_phase_table_init(vsa_phase_table_t *table, uint8_t K);

/* Random phase hypervector (uniform over [0, K-1]) */
void vsa_phase_random(vsa_phase_hv_t *hv, uint8_t K, uint64_t *rng_state);

/* Zero-initialize (all phases = 0) */
void vsa_phase_zero(vsa_phase_hv_t *hv, uint8_t K);

/* ---------------------------------------------------------------------------
 * Core operations (scalar reference)
 * ------------------------------------------------------------------------- */

/* bind: c = (a + b) mod K */
void vsa_phase_bind_scalar(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b,
                           vsa_phase_hv_t *c);

/* unbind: c = (a - b) mod K */
void vsa_phase_unbind_scalar(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b,
                             vsa_phase_hv_t *c);

/* bundle: accumulate unit phasors into integer sums, then quantize to phase */
void vsa_phase_bundle_scalar(const vsa_phase_hv_t *inputs, size_t n_inputs,
                             const vsa_phase_table_t *table,
                             vsa_phase_hv_t *output);

/* similarity: mean cos(phase_diff) using precomputed cos_diff_table */
float vsa_phase_similarity_scalar(const vsa_phase_hv_t *a,
                                  const vsa_phase_hv_t *b,
                                  const vsa_phase_table_t *table);

/* cleanup: find best match in codebook */
vsa_phase_cleanup_result_t vsa_phase_cleanup_scalar(const vsa_phase_hv_t *query,
                                                     const vsa_phase_hv_t *codebook,
                                                     size_t codebook_size,
                                                     const vsa_phase_table_t *table);

/* ---------------------------------------------------------------------------
 * AVX2 optimized paths (when available)
 * ------------------------------------------------------------------------- */

#ifdef __AVX2__
#include <immintrin.h>

/* bind: c = (a + b) mod K */
void vsa_phase_bind_avx2(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b,
                         vsa_phase_hv_t *c);

/* unbind: c = (a - b) mod K */
void vsa_phase_unbind_avx2(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b,
                           vsa_phase_hv_t *c);

/* bundle: AVX2 optimized */
void vsa_phase_bundle_avx2(const vsa_phase_hv_t *inputs, size_t n_inputs,
                           const vsa_phase_table_t *table,
                           vsa_phase_hv_t *output);

/* similarity: AVX2 optimized */
float vsa_phase_similarity_avx2(const vsa_phase_hv_t *a,
                                const vsa_phase_hv_t *b,
                                const vsa_phase_table_t *table);

/* cleanup: AVX2 optimized */
vsa_phase_cleanup_result_t vsa_phase_cleanup_avx2(const vsa_phase_hv_t *query,
                                                   const vsa_phase_hv_t *codebook,
                                                   size_t codebook_size,
                                                   const vsa_phase_table_t *table);

/* Transposed codebook (SoA layout) for vectorized cleanup */
typedef struct {
    uint8_t *data;           /* size: VSA_PHASE_DIM * codebook_size */
    size_t codebook_size;
    uint8_t K;
} vsa_phase_codebook_t;

/* Transpose codebook from AoS to SoA for vectorized cleanup */
void vsa_phase_codebook_transpose(const vsa_phase_hv_t *codebook_aos,
                                   vsa_phase_codebook_t *codebook_soa,
                                   size_t codebook_size);

void vsa_phase_codebook_free(vsa_phase_codebook_t *codebook_soa);

/* AVX2 cleanup using pre-transposed codebook (for repeated queries) */
vsa_phase_cleanup_result_t vsa_phase_cleanup_avx2_pretransposed(const vsa_phase_hv_t *query,
                                                                  const vsa_phase_codebook_t *codebook_soa,
                                                                  const vsa_phase_table_t *table);

/* ---------------------------------------------------------------------------
 * INT8 PSHUFB cleanup path (RSPK Step 5 re-brief, VSAW2-S5-INT8)
 * --------------------------------------------------------------------------- */
vsa_phase_cleanup_result_t vsa_phase_cleanup_int8_scalar_ref(
                                        const vsa_phase_hv_t *query,
                                        const vsa_phase_hv_t *codebook,
                                        size_t codebook_size,
                                        const vsa_phase_table_t *table);

vsa_phase_cleanup_result_t vsa_phase_cleanup_int8_avx2_pshufb(
                                        const vsa_phase_hv_t *query,
                                        const vsa_phase_hv_t *codebook,
                                        size_t codebook_size,
                                        const vsa_phase_table_t *table);

/* Dispatch to best available implementation */
static inline void vsa_phase_bind(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b,
                                  vsa_phase_hv_t *c) {
#ifdef __AVX2__
    vsa_phase_bind_avx2(a, b, c);
#else
    vsa_phase_bind_scalar(a, b, c);
#endif
}

static inline void vsa_phase_unbind(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b,
                                    vsa_phase_hv_t *c) {
#ifdef __AVX2__
    vsa_phase_unbind_avx2(a, b, c);
#else
    vsa_phase_unbind_scalar(a, b, c);
#endif
}

static inline void vsa_phase_bundle(const vsa_phase_hv_t *inputs, size_t n_inputs,
                                    const vsa_phase_table_t *table,
                                    vsa_phase_hv_t *output) {
#ifdef __AVX2__
    vsa_phase_bundle_avx2(inputs, n_inputs, table, output);
#else
    vsa_phase_bundle_scalar(inputs, n_inputs, table, output);
#endif
}

static inline float vsa_phase_similarity(const vsa_phase_hv_t *a,
                                         const vsa_phase_hv_t *b,
                                         const vsa_phase_table_t *table) {
#ifdef __AVX2__
    return vsa_phase_similarity_avx2(a, b, table);
#else
    return vsa_phase_similarity_scalar(a, b, table);
#endif
}

static inline vsa_phase_cleanup_result_t vsa_phase_cleanup(const vsa_phase_hv_t *query,
                                                            const vsa_phase_hv_t *codebook,
                                                            size_t codebook_size,
                                                            const vsa_phase_table_t *table) {
#ifdef __AVX2__
    return vsa_phase_cleanup_avx2(query, codebook, codebook_size, table);
#else
    return vsa_phase_cleanup_scalar(query, codebook, codebook_size, table);
#endif
}

#else  /* !__AVX2__ */

static inline void vsa_phase_bind(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b,
                                  vsa_phase_hv_t *c) {
    vsa_phase_bind_scalar(a, b, c);
}

static inline void vsa_phase_unbind(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b,
                                    vsa_phase_hv_t *c) {
    vsa_phase_unbind_scalar(a, b, c);
}

static inline void vsa_phase_bundle(const vsa_phase_hv_t *inputs, size_t n_inputs,
                                    const vsa_phase_table_t *table,
                                    vsa_phase_hv_t *output) {
    vsa_phase_bundle_scalar(inputs, n_inputs, table, output);
}

static inline float vsa_phase_similarity(const vsa_phase_hv_t *a,
                                         const vsa_phase_hv_t *b,
                                         const vsa_phase_table_t *table) {
    return vsa_phase_similarity_scalar(a, b, table);
}

static inline vsa_phase_cleanup_result_t vsa_phase_cleanup(const vsa_phase_hv_t *query,
                                                            const vsa_phase_hv_t *codebook,
                                                            size_t codebook_size,
                                                            const vsa_phase_table_t *table) {
    return vsa_phase_cleanup_scalar(query, codebook, codebook_size, table);
}

#endif  /* __AVX2__ */

/* ---------------------------------------------------------------------------
 * Conformance test helpers
 * ------------------------------------------------------------------------- */

/* Compare two phase hypervectors for exact equality */
int vsa_phase_equal(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b);

/* Verify bind/unbind round-trip: unbind(bind(a, b), b) == a */
int vsa_phase_roundtrip_test(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b);

#ifdef __cplusplus
}
#endif

#endif /* VSA_PHASE_H */