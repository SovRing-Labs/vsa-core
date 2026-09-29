#include "vsa_kernel.h"
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ------------------------------------------------------------------ bind */
void vsa_bind(const vsa_hv_t* a, const vsa_hv_t* b, vsa_hv_t* out) {
#if defined(__AVX2__)
    const __m256i* as = (const __m256i*)a->sign; const __m256i* az = (const __m256i*)a->zero;
    const __m256i* bs = (const __m256i*)b->sign; const __m256i* bz = (const __m256i*)b->zero;
    __m256i* os = (__m256i*)out->sign;           __m256i* oz = (__m256i*)out->zero;
    for (size_t i = 0; i < VSA_DIMENSIONS / 256; ++i) {
        _mm256_storeu_si256(&os[i], _mm256_xor_si256(_mm256_loadu_si256(&as[i]), _mm256_loadu_si256(&bs[i])));   /* product is negative iff exactly one operand is */
        _mm256_storeu_si256(&oz[i], _mm256_and_si256(_mm256_loadu_si256(&az[i]), _mm256_loadu_si256(&bz[i])));   /* product is active iff both operands are        */
    }
#else
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        out->sign[w] = a->sign[w] ^ b->sign[w];
        out->zero[w] = a->zero[w] & b->zero[w];
    }
#endif
}

/* ------------------------------------------------------------------- dot */
#if defined(__AVX2__)
/* AVX2 has no VPOPCNTQ (that is AVX512_VPOPCNTDQ). Use the pshufb nibble
 * lookup, which is the standard portable-AVX2 population count. */
static inline __m256i popcnt256(__m256i v) {
    const __m256i lut = _mm256_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
                                         0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i lo_mask = _mm256_set1_epi8(0x0f);
    __m256i lo = _mm256_and_si256(v, lo_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lo_mask);
    __m256i c  = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
                                 _mm256_shuffle_epi8(lut, hi));
    return _mm256_sad_epu8(c, _mm256_setzero_si256());
}
#endif

int32_t vsa_dot(const vsa_hv_t* a, const vsa_hv_t* b) {
#if defined(__AVX2__)
    const __m256i* as = (const __m256i*)a->sign; const __m256i* az = (const __m256i*)a->zero;
    const __m256i* bs = (const __m256i*)b->sign; const __m256i* bz = (const __m256i*)b->zero;
    /* Accumulate in-register; ONE horizontal reduction at the end. Reducing
     * inside the loop (as the Gate 1 draft did) costs ~40 extra reductions. */
    __m256i pacc = _mm256_setzero_si256(), nacc = _mm256_setzero_si256();
    for (size_t i = 0; i < VSA_DIMENSIONS / 256; ++i) {
        __m256i active = _mm256_and_si256(_mm256_loadu_si256(&az[i]), _mm256_loadu_si256(&bz[i]));
        __m256i sign   = _mm256_xor_si256(_mm256_loadu_si256(&as[i]), _mm256_loadu_si256(&bs[i]));
        pacc = _mm256_add_epi64(pacc, popcnt256(_mm256_andnot_si256(sign, active)));
        nacc = _mm256_add_epi64(nacc, popcnt256(_mm256_and_si256(sign, active)));
    }
    int64_t p[4], n[4];
    _mm256_storeu_si256((__m256i*)p, pacc);
    _mm256_storeu_si256((__m256i*)n, nacc);
    return (int32_t)((p[0]+p[1]+p[2]+p[3]) - (n[0]+n[1]+n[2]+n[3]));
#else
    int32_t s = 0;
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        uint64_t active = a->zero[w] & b->zero[w];
        uint64_t sign   = a->sign[w] ^ b->sign[w];
        s += __builtin_popcountll(active & ~sign) - __builtin_popcountll(active & sign);
    }
    return s;
#endif
}

/* ---------------------------------------------------------------- bundle */
void vsa_bundle(const vsa_hv_t* const* items, size_t n, vsa_hv_t* out) {
    /* Elementwise sum of {-1,0,+1}, then threshold. int16 is ample: |sum| <= n
     * and n is bounded by the 100-HV lane size in VSA_CONCURRENCY_MAP.md. */
    static _Thread_local int16_t acc[VSA_DIMENSIONS];
    memset(acc, 0, sizeof(acc));

    for (size_t k = 0; k < n; ++k) {
        const vsa_hv_t* h = items[k];
        for (size_t w = 0; w < VSA_WORDS; ++w) {
            uint64_t active = h->zero[w];
            uint64_t neg    = h->sign[w];
            while (active) {
                int b = __builtin_ctzll(active);
                active &= active - 1;
                acc[w * 64 + b] += ((neg >> b) & 1) ? -1 : 1;
            }
        }
    }

    memset(out, 0, sizeof(*out));
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        uint64_t sign = 0, zero = 0;
        for (int b = 0; b < 64; ++b) {
            int16_t v = acc[w * 64 + b];
            if (v > 0)      { zero |= 1ULL << b; }
            else if (v < 0) { zero |= 1ULL << b; sign |= 1ULL << b; }
            /* v == 0 stays in stasis: genuinely undecided, not silently +1 */
        }
        out->sign[w] = sign;
        out->zero[w] = zero;
    }
}

int32_t vsa_active_count(const vsa_hv_t* h) {
    int32_t c = 0;
    for (size_t w = 0; w < VSA_WORDS; ++w) c += __builtin_popcountll(h->zero[w]);
    return c;
}

/* --------------------------------------------------------------- seqlock */
void vsa_write_begin(uint64_t* seq) {
    __atomic_add_fetch(seq, 1, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_RELEASE);
}
void vsa_write_end(uint64_t* seq) {
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_add_fetch(seq, 1, __ATOMIC_RELAXED);
}
uint64_t vsa_read_begin(const uint64_t* seq) {
    uint64_t s;
    do { s = __atomic_load_n(seq, __ATOMIC_ACQUIRE); } while (s & 1); /* writer mid-update */
    return s;
}
int vsa_read_retry(const uint64_t* seq, uint64_t start) {
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    return __atomic_load_n(seq, __ATOMIC_ACQUIRE) != start;
}

/* ------------------------------------------------- deprecated single-plane */
void vsa_bind_avx2(const uint8_t* a, const uint8_t* b, uint8_t* out) {
    for (size_t i = 0; i < VSA_PLANE_BYTES; ++i) out[i] = a[i] ^ b[i];
}
