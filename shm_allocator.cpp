#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" {
#include "vsa_kernel.h"
}

#define SHM_NAME     "/vsa_matrix_bus"
#define SHM_SEQ_NAME "/vsa_matrix_seq"

/* VSA_CONCURRENCY_MAP.md: 4 control heads + 6 worker lanes, 100 HVs each. */
static const size_t VSA_LANES        = 10;
static const size_t VSA_HV_PER_LANE  = 100;
static const size_t VSA_DEFAULT_N    = VSA_LANES * VSA_HV_PER_LANE;   /* 1000 */
static const size_t VSA_MAX_N        = 1u << 20;                      /* sanity ceiling */

int main(int argc, char* argv[]) {
    size_t N = VSA_DEFAULT_N;

    if (argc > 1) {
        char* end = nullptr;
        errno = 0;
        unsigned long long v = std::strtoull(argv[1], &end, 10);
        if (errno || !end || *end != '\0' || v == 0) {
            std::cerr << "error: argument must be a positive hypervector COUNT, got '"
                      << argv[1] << "'\n"
                      << "note:  this is a count of hypervectors, NOT a byte size.\n"
                      << "       the concurrency map specifies " << VSA_DEFAULT_N
                      << " (" << VSA_LANES << " lanes x " << VSA_HV_PER_LANE << " HVs).\n";
            return 2;
        }
        if (v > VSA_MAX_N) {
            std::cerr << "error: N=" << v << " exceeds the sanity ceiling of " << VSA_MAX_N << ".\n"
                      << "note:  N is a hypervector COUNT, not a byte size. N=" << v
                      << " would reserve " << (double)v * VSA_HV_BYTES / 1e9 << " GB.\n"
                      << "       did you mean " << VSA_DEFAULT_N << "?\n";
            return 2;
        }
        N = (size_t)v;
    }

    /* 2560 B/HV: two bitplanes. This is the stride every lane process uses
     * (HV_TOTAL_BYTES in control_head_*.cpp / worker_lane_*.cpp). The previous
     * 1280 B/HV was one plane and put every lane at twice its intended HV. */
    const size_t size     = N * VSA_HV_BYTES;
    const size_t seq_size = N * sizeof(uint64_t);

    std::cout << "VSA bus: N=" << N << " hypervectors x " << VSA_HV_BYTES
              << " B = " << size << " bytes (" << (double)size / 1e6 << " MB)\n";

    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd == -1) { perror("shm_open"); return 1; }
    if (ftruncate(fd, size) == -1) { perror("ftruncate"); close(fd); return 1; }

    void* ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
    std::memset(ptr, 0, size);            /* blank beats wrong: all HVs start in stasis */
    munmap(ptr, size);
    close(fd);

    /* Parallel version-counter array for seqlock torn-read protection. Kept in
     * its own segment so every existing lane byte offset stays exactly as-is. */
    int sfd = shm_open(SHM_SEQ_NAME, O_CREAT | O_RDWR, 0666);
    if (sfd == -1) { perror("shm_open seq"); return 1; }
    if (ftruncate(sfd, seq_size) == -1) { perror("ftruncate seq"); close(sfd); return 1; }
    void* sptr = mmap(0, seq_size, PROT_READ | PROT_WRITE, MAP_SHARED, sfd, 0);
    if (sptr == MAP_FAILED) { perror("mmap seq"); close(sfd); return 1; }
    std::memset(sptr, 0, seq_size);
    munmap(sptr, seq_size);
    close(sfd);

    std::cout << "seqlock array: " << seq_size << " bytes at " << SHM_SEQ_NAME << "\n";
    std::cout << "Lane map: " << VSA_LANES << " lanes x " << VSA_HV_PER_LANE
              << " HVs, lane stride " << VSA_HV_PER_LANE * VSA_HV_BYTES << " bytes.\n";
    std::cout << "Ready.\n";
    return 0;
}
