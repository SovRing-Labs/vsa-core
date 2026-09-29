#include "vsa_phase.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

/* ---------------------------------------------------------------------------
 * High-resolution timing
 * ------------------------------------------------------------------------- */
static inline uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static double cpu_freq_ghz = 0.0;

static void calibrate_cpu_freq(void) {
    struct timespec start, end;
    uint64_t tsc_start, tsc_end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    tsc_start = rdtsc();
    /* Busy wait ~10ms */
    volatile uint64_t sink = 0;
    for (uint64_t i = 0; i < 10000000ULL; i++) sink += i;
    tsc_end = rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &end);

    double secs = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
    cpu_freq_ghz = (double)(tsc_end - tsc_start) / secs / 1e9;
}

static inline double cycles_to_us(uint64_t cycles) {
    return (double)cycles / (cpu_freq_ghz * 1000.0);
}

/* ---------------------------------------------------------------------------
 * Independent RNG (different algorithm from xorshift64* in vsa_phase.c)
 * Using PCG32 for independence
 * ------------------------------------------------------------------------- */
typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_state_t;

static inline uint32_t pcg32_random(pcg32_state_t *rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static inline void pcg32_init(pcg32_state_t *rng, uint64_t seed) {
    rng->state = 0;
    rng->inc = (seed << 1) | 1;
    pcg32_random(rng);
    rng->state += seed;
    pcg32_random(rng);
}

/* ---------------------------------------------------------------------------
 * Generate random phase HV using independent RNG
 * ------------------------------------------------------------------------- */
static void gen_random_phase_hv(vsa_phase_hv_t *hv, uint8_t K, pcg32_state_t *rng) {
    hv->K = K;
    for (size_t i = 0; i < VSA_PHASE_DIM; i++) {
        hv->phases[i] = (uint8_t)(pcg32_random(rng) % K);
    }
}

/* ---------------------------------------------------------------------------
 * Scalar-only bind/unbind for reference comparison
 * ------------------------------------------------------------------------- */
static void bind_scalar_ref(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b,
                            vsa_phase_hv_t *c) {
    uint8_t K = a->K;
    c->K = K;
    for (size_t i = 0; i < VSA_PHASE_DIM; i++) {
        uint8_t sum = a->phases[i] + b->phases[i];
        c->phases[i] = (sum >= K) ? (sum - K) : sum;
    }
}

static void unbind_scalar_ref(const vsa_phase_hv_t *a, const vsa_phase_hv_t *b,
                              vsa_phase_hv_t *c) {
    uint8_t K = a->K;
    c->K = K;
    for (size_t i = 0; i < VSA_PHASE_DIM; i++) {
        int8_t diff = (int8_t)a->phases[i] - (int8_t)b->phases[i];
        c->phases[i] = (diff < 0) ? (diff + K) : diff;
    }
}

static float similarity_scalar_ref(const vsa_phase_hv_t *a,
                                   const vsa_phase_hv_t *b) {
    uint8_t K = a->K;
    const float two_pi_over_K = 2.0f * 3.14159265358979323846f / (float)K;
    float sum = 0.0f;
    for (size_t i = 0; i < VSA_PHASE_DIM; i++) {
        int8_t diff = (int8_t)a->phases[i] - (int8_t)b->phases[i];
        if (diff < 0) diff += K;
        sum += cosf((float)diff * two_pi_over_K);
    }
    return sum / (float)VSA_PHASE_DIM;
}

static vsa_phase_cleanup_result_t cleanup_scalar_ref(const vsa_phase_hv_t *query,
                                                      const vsa_phase_hv_t *codebook,
                                                      size_t codebook_size) {
    vsa_phase_cleanup_result_t result = {-1, -2.0f};
    for (size_t i = 0; i < codebook_size; i++) {
        float sim = similarity_scalar_ref(query, &codebook[i]);
        if (sim > result.best_score) {
            result.best_score = sim;
            result.best_index = (int)i;
        }
    }
    return result;
}

/* ---------------------------------------------------------------------------
 * Test scalar vs AVX2 bit-for-bit agreement
 * ------------------------------------------------------------------------- */
static int test_scalar_avx2_agreement(uint8_t K, uint64_t seed) {
    pcg32_state_t rng;
    pcg32_init(&rng, seed);

    vsa_phase_table_t table;
    vsa_phase_table_init(&table, K);

    vsa_phase_hv_t a, b, c_scalar, c_avx2;

    for (int trial = 0; trial < 3000; trial++) {
        gen_random_phase_hv(&a, K, &rng);
        gen_random_phase_hv(&b, K, &rng);

        /* Test bind */
        bind_scalar_ref(&a, &b, &c_scalar);
        vsa_phase_bind(&a, &b, &c_avx2);
        if (!vsa_phase_equal(&c_scalar, &c_avx2)) {
            printf("FAIL: bind disagreement trial %d K=%d\n", trial, K);
            return 0;
        }

        /* Test unbind */
        unbind_scalar_ref(&a, &b, &c_scalar);
        vsa_phase_unbind(&a, &b, &c_avx2);
        if (!vsa_phase_equal(&c_scalar, &c_avx2)) {
            printf("FAIL: unbind disagreement trial %d K=%d\n", trial, K);
            return 0;
        }

        /* Test similarity */
        float sim_scalar = similarity_scalar_ref(&a, &b);
        float sim_avx2 = vsa_phase_similarity(&a, &b, &table);
        if (fabsf(sim_scalar - sim_avx2) > 1e-6f) {
            printf("FAIL: similarity disagreement trial %d K=%d: scalar=%f avx2=%f\n",
                   trial, K, sim_scalar, sim_avx2);
            return 0;
        }
    }

    return 1;
}

/* ---------------------------------------------------------------------------
 * Test bind/unbind round-trip exactness
 * ------------------------------------------------------------------------- */
static int test_roundtrip_exactness(uint8_t K, uint64_t seed) {
    pcg32_state_t rng;
    pcg32_init(&rng, seed);

    vsa_phase_hv_t a, b, bound, unbound;

    for (int trial = 0; trial < 3000; trial++) {
        gen_random_phase_hv(&a, K, &rng);
        gen_random_phase_hv(&b, K, &rng);

        vsa_phase_bind(&a, &b, &bound);
        vsa_phase_unbind(&bound, &b, &unbound);

        if (!vsa_phase_equal(&a, &unbound)) {
            printf("FAIL: round-trip failure trial %d K=%d\n", trial, K);
            return 0;
        }
    }

    return 1;
}

/* ---------------------------------------------------------------------------
 * Test cleanup performance
 * ------------------------------------------------------------------------- */
static int test_cleanup_performance(uint8_t K, vsa_phase_table_t *table, double *us_per_codevector) {
    pcg32_state_t rng;
    pcg32_init(&rng, 0xDEADBEEF + K);

    const size_t codebook_size = 1000;
    vsa_phase_hv_t *codebook = malloc(codebook_size * sizeof(vsa_phase_hv_t));
    vsa_phase_hv_t query;

    if (!codebook) {
        printf("FAIL: OOM allocating codebook\n");
        return 0;
    }

    /* Generate codebook */
    for (size_t i = 0; i < codebook_size; i++) {
        gen_random_phase_hv(&codebook[i], K, &rng);
    }

    /* Generate query */
    gen_random_phase_hv(&query, K, &rng);

    /* Transpose codebook once (not counted in timing) */
    vsa_phase_codebook_t codebook_soa;
    vsa_phase_codebook_transpose(codebook, &codebook_soa, codebook_size);
    free(codebook);
    codebook = NULL;

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        vsa_phase_cleanup_avx2_pretransposed(&query, &codebook_soa, table);
    }

    /* Timed run: 100 iterations on pre-transposed codebook */
    const int iterations = 100;
    uint64_t start = rdtsc();
    for (int i = 0; i < iterations; i++) {
        vsa_phase_cleanup_avx2_pretransposed(&query, &codebook_soa, table);
    }
    uint64_t end = rdtsc();

    double total_us = cycles_to_us(end - start);
    *us_per_codevector = total_us / (iterations * codebook_size);

    vsa_phase_codebook_free(&codebook_soa);
    return 1;
}

/* ---------------------------------------------------------------------------
 * Main test runner
 * ------------------------------------------------------------------------- */
int main(void) {
    printf("================ VSA PHASE KERNEL CONFORMANCE ================\n");

    calibrate_cpu_freq();
    printf("CPU frequency: %.3f GHz\n", cpu_freq_ghz);

    int all_passed = 1;

    /* Test K = 8 and K = 16 */
    for (int k_idx = 0; k_idx < 2; k_idx++) {
        uint8_t K = (k_idx == 0) ? 8 : 16;
        printf("\n--- Testing K=%d ---\n", K);

        /* Initialize cos/sin table */
        vsa_phase_table_t table;
        vsa_phase_table_init(&table, K);

        /* Test 1: Scalar vs AVX2 bit-for-bit agreement (3000 trials) */
        printf("Test 1: Scalar vs AVX2 agreement (3000 trials)... ");
        fflush(stdout);
        if (!test_scalar_avx2_agreement(K, 0x12345678 + K)) {
            printf("KILL\n");
            all_passed = 0;
            break;
        }
        printf("PASS\n");

        /* Test 2: Bind/unbind round-trip exactness (3000 trials) */
        printf("Test 2: Bind/unbind round-trip (3000 trials)... ");
        fflush(stdout);
        if (!test_roundtrip_exactness(K, 0x87654321 + K)) {
            printf("KILL\n");
            all_passed = 0;
            break;
        }
        printf("PASS\n");

        /* Test 3: Cleanup performance */
        printf("Test 3: Cleanup performance (codebook=1000)... ");
        fflush(stdout);
        double us_per_codevector;
        if (!test_cleanup_performance(K, &table, &us_per_codevector)) {
            all_passed = 0;
            break;
        }
        printf("%.4f µs/codevector ", us_per_codevector);
        if (us_per_codevector > 1.25) {
            printf("KILL (threshold 1.25 µs)\n");
            all_passed = 0;
            break;
        }
        printf("PASS\n");
    }

    printf("\n==============================================================\n");
    if (all_passed) {
        printf("PASS: All conformance criteria met\n");
        return 0;
    } else {
        printf("KILL: Conformance failure\n");
        return 1;
    }
}