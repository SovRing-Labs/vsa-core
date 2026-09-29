#include "vsa_phase.h"
#include <math.h>
#include <string.h>
#include <stdint.h>

/* ---------------------------------------------------------------------------
 * Xorshift64* RNG for deterministic testing
 * ------------------------------------------------------------------------- */
static inline uint64_t xorshift64star(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* ---------------------------------------------------------------------------
 * Table initialization
 * ------------------------------------------------------------------------- */
void vsa_phase_table_init(vsa_phase_table_t *table, uint8_t K) {
    table->K = K;
    const float two_pi_over_K = 2.0f * 3.14159265358979323846f / (float)K;
    for (uint8_t i = 0; i < K; i++) {
        float angle = (float)i * two_pi_over_K;
        table->cos_table[i] = cosf(angle);
        table->sin_table[i] = sinf(angle);
        table->cos_diff_table[i] = cosf(angle);  /* cos(2π * i / K) for phase diff i */
    }
}

/* ---------------------------------------------------------------------------
 * Random phase hypervector
 * ------------------------------------------------------------------------- */
void vsa_phase_random(vsa_phase_hv_t *hv, uint8_t K, uint64_t *rng_state) {
    hv->K = K;
    for (size_t i = 0; i < VSA_PHASE_DIM; i++) {
        hv->phases[i] = (uint8_t)(xorshift64star(rng_state) % K);
    }
}

/* ---------------------------------------------------------------------------
 * Zero initialize
 * ------------------------------------------------------------------------- */
void vsa_phase_zero(vsa_phase_hv_t *hv, uint8_t K) {
    hv->K = K;
    memset(hv->phases, 0, VSA_PHASE_DIM);
}

/* ---------------------------------------------------------------------------
 * Scalar implementations
 * ------------------------------------------------------------------------- */

void vsa_phase_bind_scalar(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b,
                           vsa_phase_hv_t *c) {
    uint8_t K = a->K;
    c->K = K;
    for (size_t i = 0; i < VSA_PHASE_DIM; i++) {
        uint8_t sum = a->phases[i] + b->phases[i];
        c->phases[i] = (sum >= K) ? (sum - K) : sum;
    }
}

void vsa_phase_unbind_scalar(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b,
                             vsa_phase_hv_t *c) {
    uint8_t K = a->K;
    c->K = K;
    for (size_t i = 0; i < VSA_PHASE_DIM; i++) {
        int8_t diff = (int8_t)a->phases[i] - (int8_t)b->phases[i];
        c->phases[i] = (diff < 0) ? (diff + K) : diff;
    }
}

void vsa_phase_bundle_scalar(const vsa_phase_hv_t *inputs, size_t n_inputs,
                             const vsa_phase_table_t *table,
                             vsa_phase_hv_t *output) {
    uint8_t K = table->K;
    output->K = K;

    /* Accumulate cos/sin sums for each dimension */
    float cos_sum[VSA_PHASE_DIM];
    float sin_sum[VSA_PHASE_DIM];
    memset(cos_sum, 0, sizeof(cos_sum));
    memset(sin_sum, 0, sizeof(sin_sum));

    for (size_t n = 0; n < n_inputs; n++) {
        const vsa_phase_hv_t *hv = &inputs[n];
        for (size_t i = 0; i < VSA_PHASE_DIM; i++) {
            uint8_t phase = hv->phases[i];
            cos_sum[i] += table->cos_table[phase];
            sin_sum[i] += table->sin_table[phase];
        }
    }

    /* Quantize resultant phase: atan2(sin_sum, cos_sum) -> [0, K-1] */
    const float K_over_2pi = (float)K / (2.0f * 3.14159265358979323846f);
    for (size_t i = 0; i < VSA_PHASE_DIM; i++) {
        float angle = atan2f(sin_sum[i], cos_sum[i]);
        if (angle < 0) angle += 2.0f * 3.14159265358979323846f;
        uint8_t phase = (uint8_t)(angle * K_over_2pi);
        if (phase >= K) phase = K - 1;
        output->phases[i] = phase;
    }
}

float vsa_phase_similarity_scalar(const vsa_phase_hv_t *a,
                                  const vsa_phase_hv_t *b,
                                  const vsa_phase_table_t *table) {
    float sum = 0.0f;

    for (size_t i = 0; i < VSA_PHASE_DIM; i++) {
        int8_t diff = (int8_t)a->phases[i] - (int8_t)b->phases[i];
        if (diff < 0) diff += table->K;
        sum += table->cos_diff_table[diff];
    }

    return sum / (float)VSA_PHASE_DIM;
}

vsa_phase_cleanup_result_t vsa_phase_cleanup_scalar(const vsa_phase_hv_t *query,
                                                     const vsa_phase_hv_t *codebook,
                                                     size_t codebook_size,
                                                     const vsa_phase_table_t *table) {
    vsa_phase_cleanup_result_t result = {-1, -2.0f};  /* similarity in [-1, 1] */

    for (size_t i = 0; i < codebook_size; i++) {
        float sim = vsa_phase_similarity_scalar(query, &codebook[i], table);
        if (sim > result.best_score) {
            result.best_score = sim;
            result.best_index = (int)i;
        }
    }

    return result;
}

/* ---------------------------------------------------------------------------
 * AVX2 implementations
 * ------------------------------------------------------------------------- */

#ifdef __AVX2__
#include <immintrin.h>

/* AVX2 bind: process 32 bytes (32 elements) per iteration */
void vsa_phase_bind_avx2(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b,
                         vsa_phase_hv_t *c) {
    uint8_t K = a->K;
    c->K = K;

    /* For K=8 or 16 (powers of 2), modulo is just bitwise AND with (K-1) */
    const __m256i mask = _mm256_set1_epi8((char)(K - 1));

    size_t i = 0;
    for (; i + 31 < VSA_PHASE_DIM; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a->phases + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b->phases + i));

        __m256i sum = _mm256_add_epi8(va, vb);
        sum = _mm256_and_si256(sum, mask);

        _mm256_storeu_si256((__m256i*)(c->phases + i), sum);
    }

    /* Tail: scalar */
    for (; i < VSA_PHASE_DIM; i++) {
        uint8_t sum = a->phases[i] + b->phases[i];
        c->phases[i] = sum & (K - 1);
    }
}

/* AVX2 unbind: process 32 bytes per iteration */
void vsa_phase_unbind_avx2(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b,
                           vsa_phase_hv_t *c) {
    uint8_t K = a->K;
    c->K = K;

    const __m256i vK = _mm256_set1_epi8((char)K);

    size_t i = 0;
    for (; i + 31 < VSA_PHASE_DIM; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a->phases + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b->phases + i));

        /* Subtract with modulo K */
        __m256i diff = _mm256_sub_epi8(va, vb);

        /* For K=8 or 16, add K where negative */
        __m256i cmp = _mm256_cmpgt_epi8(_mm256_setzero_si256(), diff);  /* diff < 0 */
        __m256i add = _mm256_and_si256(cmp, vK);
        diff = _mm256_add_epi8(diff, add);

        _mm256_storeu_si256((__m256i*)(c->phases + i), diff);
    }

    /* Tail: scalar */
    for (; i < VSA_PHASE_DIM; i++) {
        int8_t diff = (int8_t)a->phases[i] - (int8_t)b->phases[i];
        c->phases[i] = (diff < 0) ? (diff + K) : diff;
    }
}

/* AVX2 bundle: more complex, use scalar for now with cos/sin table lookup */
/* The bundle operation involves floating-point accumulation which doesn't
 * vectorize as cleanly with AVX2 for the quantization step. */
void vsa_phase_bundle_avx2(const vsa_phase_hv_t *inputs, size_t n_inputs,
                           const vsa_phase_table_t *table,
                           vsa_phase_hv_t *output) {
    /* Fall back to scalar for bundle - the floating point accumulation
     * and atan2 quantization don't benefit significantly from AVX2 */
    vsa_phase_bundle_scalar(inputs, n_inputs, table, output);
}

/* AVX2 similarity: compute mean cos(phase_diff) using vectorized lookup */
float vsa_phase_similarity_avx2(const vsa_phase_hv_t *a,
                                const vsa_phase_hv_t *b,
                                const vsa_phase_table_t *table) {
    uint8_t K = a->K;
    float sum = 0.0f;

    /* Load cos_diff_table into AVX2 registers for fast lookup */
    /* K is 8 or 16, table has K entries */
    __m256i cos_lut = _mm256_set1_epi32(0);  /* placeholder, we'll load per-lane */
    
    /* Build lookup vectors: for K=8, we need 8 floats; for K=16, 16 floats.
     * Use _mm256_i32gather_ps but it's slow. Better: process in chunks
     * using scalar for the table lookup since K is small. */
    
    /* Actually, the fastest approach for small K: process 32 elements at a time,
     * compute phase diffs with AVX2, then do scalar lookup for each of the 32 diffs.
     * This avoids the cosf call overhead and is much faster. */
    
    size_t i = 0;
    for (; i + 31 < VSA_PHASE_DIM; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a->phases + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b->phases + i));

        /* Compute diff = (a - b) mod K */
        __m256i diff = _mm256_sub_epi8(va, vb);
        __m256i cmp = _mm256_cmpgt_epi8(_mm256_setzero_si256(), diff);  /* diff < 0 */
        __m256i vK = _mm256_set1_epi8((char)K);
        __m256i add = _mm256_and_si256(cmp, vK);
        diff = _mm256_add_epi8(diff, add);

        /* Store diffs and do scalar lookup */
        uint8_t diff_arr[32];
        _mm256_storeu_si256((__m256i*)diff_arr, diff);

        for (int j = 0; j < 32; j++) {
            sum += table->cos_diff_table[diff_arr[j]];
        }
    }

    /* Tail: scalar */
    for (; i < VSA_PHASE_DIM; i++) {
        int8_t diff = (int8_t)a->phases[i] - (int8_t)b->phases[i];
        if (diff < 0) diff += K;
        sum += table->cos_diff_table[diff];
    }

    return sum / (float)VSA_PHASE_DIM;
}

/* ---------------------------------------------------------------------------
 * Optimized cleanup using transposed codebook (SoA layout) for AVX2
 * ------------------------------------------------------------------------- */

/* Transpose codebook from AoS to SoA for vectorized cleanup */
void vsa_phase_codebook_transpose(const vsa_phase_hv_t *codebook_aos,
                                   vsa_phase_codebook_t *codebook_soa,
                                   size_t codebook_size) {
    codebook_soa->codebook_size = codebook_size;
    codebook_soa->K = codebook_aos[0].K;
    codebook_soa->data = aligned_alloc(32, VSA_PHASE_DIM * codebook_size);
    
    for (size_t dim = 0; dim < VSA_PHASE_DIM; dim++) {
        for (size_t i = 0; i < codebook_size; i++) {
            codebook_soa->data[dim * codebook_size + i] = codebook_aos[i].phases[dim];
        }
    }
}

void vsa_phase_codebook_free(vsa_phase_codebook_t *codebook_soa) {
    free(codebook_soa->data);
    codebook_soa->data = NULL;
    codebook_soa->codebook_size = 0;
}

/* AVX2 cleanup: vectorized similarity against pre-transposed codebook (SoA layout)
 * Best scalar implementation with 8x dimension unrolling and precomputed lookup */
vsa_phase_cleanup_result_t vsa_phase_cleanup_avx2_pretransposed(const vsa_phase_hv_t *query,
                                                                  const vsa_phase_codebook_t *codebook_soa,
                                                                  const vsa_phase_table_t *table) {
    uint8_t K = query->K;
    size_t codebook_size = codebook_soa->codebook_size;
    const float inv_dim = 1.0f / (float)VSA_PHASE_DIM;
    
    /* Precompute cos_diff lookup table for all (q, c) pairs */
    /* cos_diff_table[q][c] = cos(2π * (q - c) mod K / K) */
    float cos_diff_table[16][16];
    for (int q = 0; q < K; q++) {
        for (int c = 0; c < K; c++) {
            int diff = q - c;
            if (diff < 0) diff += K;
            cos_diff_table[q][c] = table->cos_diff_table[diff];
        }
    }
    
    vsa_phase_cleanup_result_t result = {-1, -2.0f};
    
    /* Process codebook in chunks of 8 vectors for register locality */
    for (size_t base = 0; base < codebook_size; base += 8) {
        size_t chunk = (codebook_size - base >= 8) ? 8 : (codebook_size - base);
        
        /* Accumulators for 8 vectors - kept in registers */
        float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        
        /* Process all dimensions with 8x unrolling for maximum ILP */
        size_t dim = 0;
        for (; dim + 7 < VSA_PHASE_DIM; dim += 8) {
            uint8_t q[8];
            #pragma unroll
            for (int u = 0; u < 8; u++) {
                q[u] = query->phases[dim + u];
            }
            
            const uint8_t *rows[8];
            #pragma unroll
            for (int u = 0; u < 8; u++) {
                rows[u] = codebook_soa->data + (dim + u) * codebook_size + base;
            }
            
            /* Unrolled inner loop for 8 codevectors */
            for (int j = 0; j < (int)chunk; j++) {
                #pragma unroll
                for (int u = 0; u < 8; u++) {
                    sums[j] += cos_diff_table[q[u]][rows[u][j]];
                }
            }
        }
        
        /* Tail: scalar */
        for (; dim < VSA_PHASE_DIM; dim++) {
            uint8_t q = query->phases[dim];
            const uint8_t *row = codebook_soa->data + dim * codebook_size + base;
            for (int j = 0; j < (int)chunk; j++) {
                sums[j] += cos_diff_table[q][row[j]];
            }
        }
        
        /* Finalize and check */
        for (int j = 0; j < (int)chunk; j++) {
            float sim = sums[j] * inv_dim;
            if (sim > result.best_score) {
                result.best_score = sim;
                result.best_index = (int)(base + j);
            }
        }
    }
    
    return result;
}


vsa_phase_cleanup_result_t vsa_phase_cleanup_avx2(const vsa_phase_hv_t *query,
                                                   const vsa_phase_hv_t *codebook,
                                                   size_t codebook_size,
                                                   const vsa_phase_table_t *table) {
    /* For small codebooks, fall back to scalar */
    if (codebook_size < 8) {
        vsa_phase_cleanup_result_t result = {-1, -2.0f};
        for (size_t i = 0; i < codebook_size; i++) {
            float sim = vsa_phase_similarity_scalar(query, &codebook[i], table);
            if (sim > result.best_score) {
                result.best_score = sim;
                result.best_index = (int)i;
            }
        }
        return result;
    }

    /* Transpose codebook for vectorized access */
    vsa_phase_codebook_t codebook_soa;
    vsa_phase_codebook_transpose(codebook, &codebook_soa, codebook_size);
    
    vsa_phase_cleanup_result_t result = vsa_phase_cleanup_avx2_pretransposed(query, &codebook_soa, table);
    
    vsa_phase_codebook_free(&codebook_soa);
    return result;
}

#endif  /* __AVX2__ */

/* ---------------------------------------------------------------------------
 * Conformance helpers
 * ------------------------------------------------------------------------- */

int vsa_phase_equal(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b) {
    if (a->K != b->K) return 0;
    return memcmp(a->phases, b->phases, VSA_PHASE_DIM) == 0;
}

int vsa_phase_roundtrip_test(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b) {
    vsa_phase_hv_t bound, unbound;
    vsa_phase_bind(a, b, &bound);
    vsa_phase_unbind(&bound, b, &unbound);
    return vsa_phase_equal(a, &unbound);
}