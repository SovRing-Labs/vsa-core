#ifndef VSA_STOCHASTIC_H
#define VSA_STOCHASTIC_H

#include "vsa_kernel.h"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STOCHASTIC_DIM 10240

/* ---------------------------------------------------------------------------
 * 5-Bit Stochastic Ternary Hypervector
 * For gradient-free, local "confidence learning" on consumer CPUs:
 *   - weight: 2-bit ternary active state {-1, 0, +1}
 *   - confidence: 3-bit learning register {-3 to +3}
 * --------------------------------------------------------------------------- */
typedef struct {
    int8_t weight[STOCHASTIC_DIM];
    int8_t confidence[STOCHASTIC_DIM];
} vsa_stochastic_hv_t;

/* Initialize a stochastic hypervector to stasis (all zeros) */
static inline void vsa_stochastic_init(vsa_stochastic_hv_t* h) {
    memset(h->weight, 0, sizeof(h->weight));
    memset(h->confidence, 0, sizeof(h->confidence));
}

/* Update weights probabilistically based on target learning signal (gradient) */
static inline void vsa_stochastic_update(vsa_stochastic_hv_t* h, const vsa_hv_t* gradient, int probability_threshold) {
    for (size_t d = 0; d < STOCHASTIC_DIM; ++d) {
        size_t w = d / 64;
        size_t bit = d % 64;
        
        // Extract the ternary signal: active & !neg -> +1, active & neg -> -1, inactive -> 0
        int is_active = (gradient->zero[w] >> bit) & 1;
        int is_neg    = (gradient->sign[w] >> bit) & 1;
        int signal    = is_active * (1 - 2 * is_neg);
        
        if (signal == 0) continue;
        
        // Probability roll (nudge learning)
        if ((rand() % 100) < probability_threshold) {
            h->confidence[d] += signal;
            
            // Ternary snapping on overflow/underflow
            if (h->confidence[d] > 3) {
                if (h->weight[d] < 1) {
                    h->weight[d]++;
                }
                h->confidence[d] = 0;
            } else if (h->confidence[d] < -3) {
                if (h->weight[d] > -1) {
                    h->weight[d]--;
                }
                h->confidence[d] = 0;
            }
        }
    }
}

/* Quantize the 5-bit stochastic weights back into dual-plane ternary form for inference */
static inline void vsa_stochastic_quantize(const vsa_stochastic_hv_t* h, vsa_hv_t* out) {
    memset(out, 0, sizeof(vsa_hv_t));
    for (size_t d = 0; d < STOCHASTIC_DIM; ++d) {
        int8_t w_val = h->weight[d];
        if (w_val != 0) {
            size_t w = d / 64;
            size_t bit = d % 64;
            out->zero[w] |= (1ULL << bit);
            if (w_val < 0) {
                out->sign[w] |= (1ULL << bit);
            }
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* VSA_STOCHASTIC_H */
