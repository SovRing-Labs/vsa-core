#define _DEFAULT_SOURCE
/* Cross-language layout test (2026-09-25). Reads two HVs written by Python from a shm segment
 * and prints what C sees: per-sign counts + cross dot. xlang_test.py compares these with numpy.
 * Usage: vsa_xlang_test <shm_name> <seq_name> <slot_a> <slot_b>   (shm_open names, e.g. /vsa_xlang_bus)
 * Unlike a self-dot check, a plane swap or byte-order bug changes pos/neg/dot and FAILS here. */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "vsa_kernel.h"

static int32_t count_sign(const vsa_hv_t* h, int neg) {
    int32_t c = 0;
    for (size_t w = 0; w < VSA_WORDS; ++w)
        c += __builtin_popcountll(h->zero[w] & (neg ? h->sign[w] : ~h->sign[w]));
    return c;
}

int main(int argc, char** argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s shm seq slot_a slot_b\n", argv[0]); return 2; }
    int a = atoi(argv[3]), b = atoi(argv[4]);
    int fd = shm_open(argv[1], O_RDONLY, 0);
    int sfd = shm_open(argv[2], O_RDONLY, 0);
    if (fd < 0 || sfd < 0) { perror("shm_open"); return 2; }
    const vsa_hv_t* bus = mmap(NULL, 1000 * sizeof(vsa_hv_t), PROT_READ, MAP_SHARED, fd, 0);
    const uint64_t* seq = mmap(NULL, 1000 * sizeof(uint64_t), PROT_READ, MAP_SHARED, sfd, 0);
    if (bus == MAP_FAILED || seq == MAP_FAILED) { perror("mmap"); return 2; }
    vsa_hv_t ha, hb;
    uint64_t s0;
    do { s0 = vsa_read_begin(&seq[a]); memcpy(&ha, &bus[a], sizeof ha); } while (vsa_read_retry(&seq[a], s0));
    do { s0 = vsa_read_begin(&seq[b]); memcpy(&hb, &bus[b], sizeof hb); } while (vsa_read_retry(&seq[b], s0));
    printf("{\"a_pos\":%d,\"a_neg\":%d,\"a_active\":%d,\"b_pos\":%d,\"b_neg\":%d,\"dot_ab\":%d,\"sizeof_hv\":%zu}\n",
           count_sign(&ha, 0), count_sign(&ha, 1), vsa_active_count(&ha),
           count_sign(&hb, 0), count_sign(&hb, 1), vsa_dot(&ha, &hb), sizeof(vsa_hv_t));
    return 0;
}
