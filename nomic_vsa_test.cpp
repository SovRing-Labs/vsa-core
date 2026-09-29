#include "vsa_kernel.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <iomanip>

#define SHM_NAME "/vsa_matrix_bus"
#define SHM_SEQ_NAME "/vsa_matrix_seq"
#define HV_BYTES 2560

int main() {
    std::cout << "================ C++ NOMIC-VSA SHM INTEGRATION TEST ================\n";

    // 1. Open the VSA shared data bus
    int shm_fd = shm_open(SHM_NAME, O_RDONLY, 0666);
    if (shm_fd < 0) {
        std::cerr << "❌ Error: Failed to open shared memory segment: " << SHM_NAME << "\n";
        return 1;
    }
    
    // Map the shared memory (1000 HVs * 2560 bytes = 2.56 MB)
    void* shm_ptr = mmap(NULL, 1000 * HV_BYTES, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "❌ Error: Failed to mmap shared memory\n";
        close(shm_fd);
        return 1;
    }
    
    // 2. Open the seqlock sequence segment
    int seq_fd = shm_open(SHM_SEQ_NAME, O_RDONLY, 0666);
    if (seq_fd < 0) {
        std::cerr << "❌ Error: Failed to open sequence memory segment: " << SHM_SEQ_NAME << "\n";
        munmap(shm_ptr, 1000 * HV_BYTES);
        close(shm_fd);
        return 1;
    }
    
    // Map the sequence array (1000 * 8 bytes = 8000 bytes)
    volatile uint64_t* seq_ptr = (volatile uint64_t*)mmap(NULL, 1000 * sizeof(uint64_t), PROT_READ, MAP_SHARED, seq_fd, 0);
    if (seq_ptr == MAP_FAILED) {
        std::cerr << "❌ Error: Failed to mmap sequence array\n";
        close(seq_fd);
        munmap(shm_ptr, 1000 * HV_BYTES);
        close(shm_fd);
        return 1;
    }

    // 3. Read the hypervector from slot index 102 under seqlock control
    vsa_hv_t read_hv;
    const vsa_hv_t* bus_hvs = (const vsa_hv_t*)shm_ptr;
    size_t slot_idx = 102;
    
    std::cout << "Attempting to read Nomic Embed from slot index " << slot_idx << " under seqlock...\n";
    
    uint64_t s0, s1;
    do {
        s0 = seq_ptr[slot_idx];
        while (s0 % 2 != 0) {
            // Writer in progress, wait and reload
            usleep(1);
            s0 = seq_ptr[slot_idx];
        }
        
        // Copy the data
        memcpy(&read_hv, &bus_hvs[slot_idx], sizeof(vsa_hv_t));
        
        s1 = seq_ptr[slot_idx];
    } while (s0 != s1); // If mismatch, a concurrent write occurred -> retry

    std::cout << "✓ Read successful! (Sequence Version: " << s1 << ")\n";

    // 4. Run AVX2-optimized diagnostics in C++
    int32_t active_count = vsa_active_count(&read_hv);
    int32_t self_dot = vsa_dot(&read_hv, &read_hv);

    std::cout << "\n--- C++ AVX2 Analysis Results ---\n";
    std::cout << "Active dimensions read:   " << active_count << " / 10240\n";
    std::cout << "Self dot product (vsa_dot): " << self_dot << "\n";
    std::cout << "Sparsity:                 " << (1.0 - (double)active_count / 10240.0) * 100.0 << "%\n";
    
    // Self dot product of a ternary vector MUST equal its active dimension count!
    if (self_dot == active_count) {
        std::cout << "\n✓ Kernel self-consistency: self-dot == active count (" << active_count << "). NOTE: this holds for ANY bits — it does not prove Python/C layout agreement; run `make xlang-check` for that.\n";
    } else {
        std::cout << "\n❌ Error: Integrity mismatch. Self-dot product (" << self_dot << ") != active count (" << active_count << ").\n";
    }

    // Cleanup
    munmap((void*)seq_ptr, 1000 * sizeof(uint64_t));
    close(seq_fd);
    munmap(shm_ptr, 1000 * HV_BYTES);
    close(shm_fd);
    
    std::cout << "====================================================================\n";
    return 0;
}
