#include "vsa_kernel.h"
#include "vsa_bundle_bitsliced.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#define VSA_HVS 1024
#define TRIALS  1000
#define SEED    0xDEADBEEF

static uint64_t rng_state;
static void rng_seed(uint64_t s) { rng_state = s; }
static uint64_t rng_u64(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static void rand_hv(vsa_hv_t* h) {
    for (int i = 0; i < VSA_WORDS; ++i) {
        h->zero[i] = rng_u64();
        h->sign[i] = rng_u64();
    }
}

static int hv_eq(const vsa_hv_t* a, const vsa_hv_t* b) {
    for (int i = 0; i < VSA_WORDS; ++i)
        if (a->sign[i] != b->sign[i] || a->zero[i] != b->zero[i]) return 0;
    return 1;
}

static double timespec_us(const struct timespec* t) {
    return t->tv_sec * 1e6 + t->tv_nsec / 1e3;
}

int main(void) {
    rng_seed(SEED);

    static vsa_hv_t pool[VSA_HVS];
    for (int i = 0; i < VSA_HVS; ++i) rand_hv(&pool[i]);

    int all_pass = 1;

    /* Correctness: 1000 random trials for each n in spec */
    size_t ns[] = {3, 8, 16, 64};
    for (int _i = 0; _i < 4; ++_i) {
        size_t n = ns[_i];
        /* Correctness: 1000 random trials */
        int trials_pass = 0;
        for (int t = 0; t < TRIALS; ++t) {
            vsa_hv_t* ptrs[64];
            for (size_t k = 0; k < n; ++k) ptrs[k] = &pool[(t * 7 + k * 13) % VSA_HVS];

            vsa_hv_t out_scalar, out_bitsliced;
            vsa_bundle((const vsa_hv_t* const*)ptrs, n, &out_scalar);
            vsa_bundle_bitsliced((const vsa_hv_t* const*)ptrs, n, &out_bitsliced);

            if (!hv_eq(&out_scalar, &out_bitsliced)) {
                printf("MISMATCH n=%zu trial=%d\n", n, t);
                all_pass = 0;
                break;
            }
            trials_pass++;
        }
        printf("correctness n=%-2zu: %d/%d PASS\n", n, trials_pass, TRIALS);
    }

    /* Benchmark: median µs/vector at n=16 and n=64 */
    for (size_t n = 16; n <= 64; n *= 4) {
        struct timespec ts1, ts2;
        double us_list[TRIALS];

        /* Warm cache */
        for (int t = 0; t < 100; ++t) {
            vsa_hv_t* ptrs[64];
            for (size_t k = 0; k < n; ++k) ptrs[k] = &pool[k % VSA_HVS];
            vsa_hv_t out; vsa_bundle(ptrs, n, &out);
        }

        /* Scalar benchmark */
        for (int t = 0; t < TRIALS; ++t) {
            vsa_hv_t* ptrs[64];
            for (size_t k = 0; k < n; ++k) ptrs[k] = &pool[(t * 7 + k * 13) % VSA_HVS];
            vsa_hv_t out;
            clock_gettime(CLOCK_MONOTONIC, &ts1);
            vsa_bundle(ptrs, n, &out);
            clock_gettime(CLOCK_MONOTONIC, &ts2);
            us_list[t] = (timespec_us(&ts2) - timespec_us(&ts1)) / n;
        }
        /* median */
        for (int i = 0; i < TRIALS - 1; ++i)
            for (int j = i + 1; j < TRIALS; ++j)
                if (us_list[j] < us_list[i]) { double tmp = us_list[i]; us_list[i] = us_list[j]; us_list[j] = tmp; }
        double scalar_us = us_list[TRIALS / 2];

        /* Bit-sliced benchmark */
        for (int t = 0; t < TRIALS; ++t) {
            vsa_hv_t* ptrs[64];
            for (size_t k = 0; k < n; ++k) ptrs[k] = &pool[(t * 7 + k * 13) % VSA_HVS];
            vsa_hv_t out;
            clock_gettime(CLOCK_MONOTONIC, &ts1);
            vsa_bundle_bitsliced(ptrs, n, &out);
            clock_gettime(CLOCK_MONOTONIC, &ts2);
            us_list[t] = (timespec_us(&ts2) - timespec_us(&ts1)) / n;
        }
        for (int i = 0; i < TRIALS - 1; ++i)
            for (int j = i + 1; j < TRIALS; ++j)
                if (us_list[j] < us_list[i]) { double tmp = us_list[i]; us_list[i] = us_list[j]; us_list[j] = tmp; }
        double bitsliced_us = us_list[TRIALS / 2];

        printf("benchmark n=%-2zu: scalar=%.2f µs/vec  bitsliced=%.2f µs/vec  speedup=%.1fx\n",
               n, scalar_us, bitsliced_us, scalar_us / bitsliced_us);
    }

    if (all_pass) {
        printf("\nPASS — bit-sliced bundle matches scalar on all trials.\n");
        return 0;
    } else {
        printf("\nKILL — correctness mismatch.\n");
        return 1;
    }
}
