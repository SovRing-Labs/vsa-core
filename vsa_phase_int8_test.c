// vsa_phase_int8_test.c — 3000-trial conformance test for int8 pshufb cleanup
// RSPK Step 5 re-brief (VSAW2-S5-INT8)
// Compiles with: vsa_phase_int8_test.c + vsa_phase_int8.c + vsa_phase.c
#include "vsa_phase.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* --------------------------------------------------------------------------
 * PCG32 RNG — independent from xorshift64* in vsa_phase.c
 * -------------------------------------------------------------------------- */
typedef struct { uint64_t state; uint64_t inc; } pcg32_state_t;

static inline uint32_t pcg32_random(pcg32_state_t *rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static inline void pcg32_init(pcg32_state_t *rng, uint64_t seed) {
    rng->state = 0; rng->inc = (seed << 1) | 1;
    pcg32_random(rng); rng->state += seed; pcg32_random(rng);
}

static void gen_random_phase_hv(vsa_phase_hv_t *hv, uint8_t K, pcg32_state_t *rng) {
    hv->K = K;
    for (size_t i = 0; i < VSA_PHASE_DIM; i++)
        hv->phases[i] = (uint8_t)(pcg32_random(rng) % K);
}

/* --------------------------------------------------------------------------
 * Test: int8 scalar ref vs int8 AVX2 pshufb vs float reference
 *   - int8_scalar vs int8_avx2: must agree 100% (same uint8 math)
 *   - int8 vs float: must agree >= 95% (quantization tolerance)
 * -------------------------------------------------------------------------- */
static int test_int8_vs_float_ref(uint8_t K, uint64_t seed) {
    pcg32_state_t rng;
    pcg32_init(&rng, seed);

    vsa_phase_table_t table;
    vsa_phase_table_init(&table, K);

    const size_t codebook_size = 100;
    vsa_phase_hv_t *codebook = malloc(codebook_size * sizeof(vsa_phase_hv_t));
    vsa_phase_hv_t query;
    if (!codebook) { printf("FAIL: OOM\n"); return 0; }

    int agree_scalar_vs_float = 0;
    int agree_avx2_vs_float   = 0;
    int agree_both_vs_float   = 0;
    int agree_scalar_vs_avx2  = 0;

    for (int trial = 0; trial < 3000; trial++) {
        for (size_t i = 0; i < codebook_size; i++)
            gen_random_phase_hv(&codebook[i], K, &rng);
        gen_random_phase_hv(&query, K, &rng);

        vsa_phase_cleanup_result_t ref = vsa_phase_cleanup(&query, codebook, codebook_size, &table);
        vsa_phase_cleanup_result_t ir  = vsa_phase_cleanup_int8_scalar_ref(&query, codebook, codebook_size, &table);
        vsa_phase_cleanup_result_t ia  = vsa_phase_cleanup_int8_avx2_pshufb(&query, codebook, codebook_size, &table);

        if (ir.best_index == ref.best_index) agree_scalar_vs_float++;
        if (ia.best_index == ref.best_index) agree_avx2_vs_float++;
        if (ir.best_index == ref.best_index && ia.best_index == ref.best_index) agree_both_vs_float++;
        if (ir.best_index == ia.best_index) agree_scalar_vs_avx2++;
    }

    free(codebook);

    printf("  K=%d: int8_scalar==float %d/3000  int8_avx2==float %d/3000  both==float %d/3000\n",
           K, agree_scalar_vs_float, agree_avx2_vs_float, agree_both_vs_float);
    printf("       int8_scalar==int8_avx2  %d/3000 (must be 3000)\n", agree_scalar_vs_avx2);

    int int8_agrees = (agree_scalar_vs_avx2 == 3000);
    int int8_vs_float_ok = (agree_both_vs_float >= 2850);
    return (int8_agrees && int8_vs_float_ok);
}

/* --------------------------------------------------------------------------
 * Performance benchmark
 * -------------------------------------------------------------------------- */
static int benchmark_cleanup(uint8_t K, vsa_phase_table_t *table,
                             double *us_scalar, double *us_avx2) {
    pcg32_state_t rng;
    pcg32_init(&rng, 0xDEADBEEF + K);

    const size_t codebook_size = 1000;
    vsa_phase_hv_t *codebook = malloc(codebook_size * sizeof(vsa_phase_hv_t));
    vsa_phase_hv_t query;
    if (!codebook) return 0;

    for (size_t i = 0; i < codebook_size; i++)
        gen_random_phase_hv(&codebook[i], K, &rng);
    gen_random_phase_hv(&query, K, &rng);

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        vsa_phase_cleanup_int8_scalar_ref(&query, codebook, codebook_size, table);
        vsa_phase_cleanup_int8_avx2_pshufb(&query, codebook, codebook_size, table);
    }

    /* Scalar ref timing */
    struct timespec t0, t1;
    volatile int sink = 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < 50; r++)
        sink += vsa_phase_cleanup_int8_scalar_ref(&query, codebook, codebook_size, table).best_index;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    *us_scalar = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / 50.0 / codebook_size / 1000.0;

    /* AVX2 pshufb timing */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < 50; r++)
        sink += vsa_phase_cleanup_int8_avx2_pshufb(&query, codebook, codebook_size, table).best_index;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    *us_avx2 = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / 50.0 / codebook_size / 1000.0;

    free(codebook);
    (void)sink;
    return 1;
}

/* -------------------------------------------------------------------------- */
int main(void) {
    int all_pass = 1;

    printf("=== INT8 PSHUFB Cleanup Conformance ===\n");

    for (int ki = 0; ki < 2; ki++) {
        uint8_t K = (uint8_t)(ki == 0 ? 8 : 16);
        printf("Conformance test K=%d (3000 trials)...\n", K);
        if (!test_int8_vs_float_ref(K, 0xABCDEF01 + K)) {
            printf("  FAIL K=%d\n", K);
            all_pass = 0;
        } else {
            printf("  PASS K=%d\n", K);
        }
    }

    printf("\nPerformance benchmark K=16 (codebook=1000)...\n");
    vsa_phase_table_t table;
    vsa_phase_table_init(&table, 16);
    double us_scalar = 0, us_avx2 = 0;
    if (benchmark_cleanup(16, &table, &us_scalar, &us_avx2)) {
        printf("  int8 scalar ref: %.3f us/codevector\n", us_scalar);
        printf("  int8 avx2 pshufb: %.3f us/codevector (budget 1.25)\n", us_avx2);
    }

    printf("\n====================\n");
    if (all_pass) {
        printf("INT8-CONFORM-OK\n");
        return 0;
    } else {
        printf("INT8-CONFORM-FAIL\n");
        return 1;
    }
}
