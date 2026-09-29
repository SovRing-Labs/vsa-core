#ifndef VSA_KERNEL_H
#define VSA_KERNEL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Ternary VSA hypervector: {-1, 0, +1} over 10,240 dimensions.
 *
 * Stored as TWO bitplanes, matching VSA_CONCURRENCY_MAP.md and every lane
 * process (control_head_*, worker_lane_*, worker_bugstomp):
 *
 *   sign[w] bit b == 1  ->  element is negative (-1)
 *   zero[w] bit b == 1  ->  element is active (-1 or +1); 0 means stasis (0)
 *
 * One hypervector is VSA_HV_BYTES = 2560 bytes. The previous single-plane
 * 1280-byte layout could only represent BINARY {0,1} and disagreed with the
 * lane processes by a factor of two.
 * ------------------------------------------------------------------------- */
#define VSA_DIMENSIONS 10240
#define VSA_WORDS      (VSA_DIMENSIONS / 64)   /* 160 uint64 per plane      */
#define VSA_PLANE_BYTES (VSA_DIMENSIONS / 8)   /* 1280 bytes per plane      */
#define VSA_HV_BYTES   (2 * VSA_PLANE_BYTES)   /* 2560 bytes per hypervector*/

/* 64-byte aligned so stack instances are cache-line aligned. The kernel uses
 * unaligned loads regardless, so correctness never depends on the caller
 * honouring this (malloc only guarantees 16 bytes on x86-64). */
typedef struct {
    uint64_t sign[VSA_WORDS];
    uint64_t zero[VSA_WORDS];
} __attribute__((aligned(64))) vsa_hv_t;

/* Binding (x): XOR the sign planes, AND the active planes. Self-inverse. */
void vsa_bind(const vsa_hv_t* a, const vsa_hv_t* b, vsa_hv_t* out);

/* Ternary dot product: popcount(active & ~sign) - popcount(active & sign).
 * Range [-10240, +10240]. Two independent random HVs score ~0 +/- 100. */
int32_t vsa_dot(const vsa_hv_t* a, const vsa_hv_t* b);

/* Bundling (superposition): elementwise SUM of the {-1,0,+1} values, then
 * threshold back to ternary. This is majority, NOT bitwise OR.
 *
 * OR saturates: it reaches 100% active density after ~12 bundles and member
 * recall collapses to 0% at k=7. Majority holds 100/100 recall at k=100.   */
void vsa_bundle(const vsa_hv_t* const* items, size_t n, vsa_hv_t* out);

/* Number of active (non-zero) elements. */
int32_t vsa_active_count(const vsa_hv_t* h);

/* --- torn-read protection -------------------------------------------------
 * A 2560-byte hypervector write is not atomic; a concurrent reader can see a
 * half-updated vector. These implement a seqlock over a caller-supplied
 * version cell held in a parallel array (see vsa_seq.h usage in the bus).
 * Writer: begin -> mutate -> end.  Reader: retry until the version is stable.
 * ------------------------------------------------------------------------- */
void     vsa_write_begin(uint64_t* seq);
void     vsa_write_end(uint64_t* seq);
uint64_t vsa_read_begin(const uint64_t* seq);
int      vsa_read_retry(const uint64_t* seq, uint64_t start);

/* Deprecated single-plane binary entry points. Kept as thin shims so existing
 * callers link, but they operate on ONE plane only and cannot represent 0. */
void vsa_bind_avx2(const uint8_t* a, const uint8_t* b, uint8_t* out);

#ifdef __cplusplus
}
#endif

#endif /* VSA_KERNEL_H */
