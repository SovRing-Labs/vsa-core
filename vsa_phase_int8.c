// vsa_phase_int8.c — int8 pshufb cleanup path + int scalar reference
// RSPK Step 5 re-brief (VSAW2-S5-INT8)
// K=16 phase vectors: one byte per dim, PSHUFB lookup on cos_u8 table.
#include "vsa_phase.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

/* Precompute uint8 cos lookup: cos_u8[k] = round(cos(2π*k/K) * 127 + 127) */
static void build_cos_u8(uint8_t cos_u8[16], uint8_t K)
{
    for (int k = 0; k < K; k++) {
        float c = cosf(2.0f * 3.14159265358979323846f * (float)k / (float)K);
        cos_u8[k] = (uint8_t)lrintf(c * 127.0f + 127.0f);
    }
}

/* ---------------------------------------------------------------------------
 * Integer scalar reference — same logic as float cleanup but uses uint8 table
 * --------------------------------------------------------------------------- */
vsa_phase_cleanup_result_t vsa_phase_cleanup_int8_scalar_ref(
                                        const vsa_phase_hv_t *query,
                                        const vsa_phase_hv_t *codebook,
                                        size_t codebook_size,
                                        const vsa_phase_table_t *table)
{
    uint8_t cos_u8[16];
    build_cos_u8(cos_u8, query->K);
    uint8_t K = query->K;
    uint8_t mask = K - 1;
    float inv_dim = 1.0f / (float)VSA_PHASE_DIM;

    vsa_phase_cleanup_result_t result = {-1, -2.0f};

    for (size_t n = 0; n < codebook_size; n++) {
        const vsa_phase_hv_t *cb = &codebook[n];
        int sum = 0;
        for (size_t i = 0; i < VSA_PHASE_DIM; i++) {
            uint8_t diff = (query->phases[i] - cb->phases[i]) & mask;
            sum += cos_u8[diff];
        }
        float score = (float)sum * inv_dim;
        if (score > result.best_score) {
            result.best_score = score;
            result.best_index = (int)n;
        }
    }
    return result;
}

/* ---------------------------------------------------------------------------
 * AVX2 PSHUFB cleanup path — 32 bytes per iteration
 * --------------------------------------------------------------------------- */
vsa_phase_cleanup_result_t vsa_phase_cleanup_int8_avx2_pshufb(
                                        const vsa_phase_hv_t *query,
                                        const vsa_phase_hv_t *codebook,
                                        size_t codebook_size,
                                        const vsa_phase_table_t *table)
{
    uint8_t cos_u8[16];
    build_cos_u8(cos_u8, query->K);
    uint8_t K = query->K;
    uint8_t mask = K - 1;
    float inv_dim = 1.0f / (float)VSA_PHASE_DIM;

    /* Broadcast 16-byte cos table across both 128-bit lanes of 256-bit reg */
    __m128i lut128 = _mm_loadu_si128((const __m128i*)cos_u8);
    __m256i lut     = _mm256_broadcastsi128_si256(lut128);
    __m256i vmask   = _mm256_set1_epi8((char)mask);
    __m256i zero    = _mm256_setzero_si256();

    vsa_phase_cleanup_result_t result = {-1, -2.0f};

    for (size_t n = 0; n < codebook_size; n++) {
        const vsa_phase_hv_t *cb = &codebook[n];
        __m256i acc = zero;

        size_t i = 0;
        for (; i + 31 < VSA_PHASE_DIM; i += 32) {
            __m256i qvec = _mm256_loadu_si256((const __m256i*)(query->phases + i));
            __m256i cvec = _mm256_loadu_si256((const __m256i*)(cb->phases + i));

            /* diff = (q - c) & mask  — unsigned subtraction then mask low bits */
            __m256i diff = _mm256_sub_epi8(qvec, cvec);
            diff = _mm256_and_si256(diff, vmask);

            /* PSHUFB table lookup */
            __m256i looked = _mm256_shuffle_epi8(lut, diff);

            /* SAD against zero = sum of all bytes */
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(looked, zero));
        }

        int64_t total = 0;
        int64_t parts[4];
        _mm256_storeu_si256((__m256i*)parts, acc);
        for (int j = 0; j < 4; j++) total += parts[j];

        /* Tail */
        for (; i < VSA_PHASE_DIM; i++) {
            uint8_t diff = (query->phases[i] - cb->phases[i]) & mask;
            total += cos_u8[diff];
        }

        float score = (float)total * inv_dim;
        if (score > result.best_score) {
            result.best_score = score;
            result.best_index = (int)n;
        }
    }
    return result;
}
