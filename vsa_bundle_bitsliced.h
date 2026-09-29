#ifndef VSA_BUNDLE_BITSLICED_H
#define VSA_BUNDLE_BITSLICED_H

#include "vsa_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bit-sliced majority bundle using carry-save counters across 64-bit words.
 * For each of the 160 words, maintains BUNDLE_PLANES bit-planes per counter
 * (pos and neg). Each bit-plane is a uint64_t where bit b represents bit b
 * of the counter for dimension (w*64 + b).
 *
 * Carry-save addition of a mask to the counter bit-planes:
 *   carry = mask; for each plane i: old=planes[i]; planes[i]=old^carry; carry=old&carry;
 *
 * Comparison (which count is larger) uses a parallel MSB-difference scan
 * across the bit-planes, producing per-bit pos>neg and neg>pos masks.
 *
 * Must produce output identical to vsa_bundle() for all inputs. */
void vsa_bundle_bitsliced(const vsa_hv_t* const* items, size_t n, vsa_hv_t* out);

#ifdef __cplusplus
}
#endif

#endif /* VSA_BUNDLE_BITSLICED_H */
