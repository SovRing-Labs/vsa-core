#include <iostream>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <iomanip>
#include <cstdint>

#define SHM_NAME "/vsa_matrix_bus"
#define SHM_SIZE 2560000
#define ITERATIONS 10000000 // 10 million reads/writes

int main() {
    int shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to open SHM." << std::endl;
        return 1;
    }
    
    volatile uint8_t* shm_ptr = (uint8_t*)mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Failed to mmap." << std::endl;
        return 1;
    }

    std::cout << "--- IPC LATENCY MICROBENCHMARK ---" << std::endl;
    std::cout << "Performing " << ITERATIONS << " lock-free reads/writes to /dev/shm..." << std::endl;

    // Benchmark Write
    auto start_write = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        // Write sequentially, wrapping around the memory map
        shm_ptr[i % SHM_SIZE] = (uint8_t)(i & 0xFF);
    }
    auto end_write = std::chrono::high_resolution_clock::now();
    
    // Benchmark Read
    uint64_t dummy_accumulator = 0;
    auto start_read = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        dummy_accumulator += shm_ptr[i % SHM_SIZE];
    }
    auto end_read = std::chrono::high_resolution_clock::now();

    // Calculate times
    double write_duration_ms = std::chrono::duration<double, std::milli>(end_write - start_write).count();
    double read_duration_ms = std::chrono::duration<double, std::milli>(end_read - start_read).count();

    double ns_per_write = (write_duration_ms * 1000000.0) / ITERATIONS;
    double ns_per_read = (read_duration_ms * 1000000.0) / ITERATIONS;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n[RESULTS]" << std::endl;
    std::cout << "Write Latency : " << ns_per_write << " nanoseconds per operation" << std::endl;
    std::cout << "Read Latency  : " << ns_per_read << " nanoseconds per operation" << std::endl;
    std::cout << "Checksum      : " << dummy_accumulator << std::endl;
    
    // The VSA benchmark promised ~14.9ns ops. Let's see raw memory bandwidth.
    std::cout << "\n(This confirms the hardware limit for IPC overhead before VSA kernel execution.)" << std::endl;

    munmap((void*)shm_ptr, SHM_SIZE);
    close(shm_fd);
    return 0;
}