#include "vsa_bundle_bitsliced.h"
#include <string.h>
#include <stdint.h>

#define BUNDLE_PLANES 7  /* ceil(log2(65)) — counts up to n=64 fit in 7 bits */

/* Add 'mask' to each of the 64 parallel binary counters represented by bit-planes.
 * For bit b: if mask has bit b set, add 1 to that counter; otherwise no change.
 * Carry propagates from plane i to plane i+1 within each counter b. */
static void add_to_counter(uint64_t planes[BUNDLE_PLANES], uint64_t mask) {
    uint64_t carry = mask;
    for (int i = 0; i < BUNDLE_PLANES; ++i) {
        uint64_t old = planes[i];
        planes[i] = old ^ carry;
        carry = old & carry;
    }
}

void vsa_bundle_bitsliced(const vsa_hv_t* const* items, size_t n, vsa_hv_t* out) {
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        uint64_t pos_planes[BUNDLE_PLANES];
        uint64_t neg_planes[BUNDLE_PLANES];
        for (int i = 0; i < BUNDLE_PLANES; ++i) {
            pos_planes[i] = 0;
            neg_planes[i] = 0;
        }

        for (size_t k = 0; k < n; ++k) {
            uint64_t active = items[k]->zero[w];
            uint64_t sign   = items[k]->sign[w];
            add_to_counter(pos_planes, active & ~sign);   /* bits contributing +1 */
            add_to_counter(neg_planes, active & sign);    /* bits contributing -1 */
        }

        /* Parallel per-bit comparison: find the MSB plane where pos and neg
         * counts differ for each bit position b. The side with 1 at that plane
         * has the larger count. */
        uint64_t higher_diff = 0;
        uint64_t pos_gt_neg  = 0;
        uint64_t neg_gt_pos  = 0;
        for (int i = BUNDLE_PLANES - 1; i >= 0; --i) {
            uint64_t d = pos_planes[i] ^ neg_planes[i];
            uint64_t new_diff = d & ~higher_diff;
            pos_gt_neg |= pos_planes[i] & new_diff;
            neg_gt_pos |= neg_planes[i] & new_diff;
            higher_diff |= d;
        }

        /* Threshold to ternary: pos>neg → +1 (zero=1, sign=0);
         * neg>pos → -1 (zero=1, sign=1); equal → 0 (stasis). */
        out->sign[w] = neg_gt_pos;
        out->zero[w] = pos_gt_neg | neg_gt_pos;
    }
}
