// True cross-process IPC latency over POSIX shared memory.
// Two processes pinned to chosen CPUs ping-pong on separate cache lines.
// Test 1: bare 8-byte flag handoff (floor).  Test 2: seqlock publish of one
// 10,240-bit hypervector (1,280 B) + ack -- what a VSA bus message actually costs.
// Uses its own segment; never touches /vsa_matrix_bus.
#define _GNU_SOURCE
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define N 1000000
#define HV_BYTES 1280

typedef struct {
    _Alignas(64) _Atomic uint64_t ping;
    _Alignas(64) _Atomic uint64_t pong;
    _Alignas(64) _Atomic uint64_t seq;
    _Alignas(64) uint8_t hv[HV_BYTES];
} bus_t;

static void pin(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    if (sched_setaffinity(0, sizeof s, &s)) { perror("affinity"); exit(1); }
}
static double now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static void run(bus_t *b, int ca, int cb, int hv) {
    memset(b, 0, sizeof *b);
    pid_t p = fork();
    if (p == 0) {                       // responder
        pin(cb);
        uint8_t local[HV_BYTES];
        for (uint64_t i = 1; i <= N; i++) {
            if (!hv) {
                while (atomic_load_explicit(&b->ping, memory_order_acquire) != i) ;
            } else {                    // seqlock read of a full HV
                uint64_t s0, s1;
                do {
                    while ((s0 = atomic_load_explicit(&b->seq, memory_order_acquire)) != 2 * i) ;
                    memcpy(local, b->hv, HV_BYTES);
                    atomic_thread_fence(memory_order_acquire);
                    s1 = atomic_load_explicit(&b->seq, memory_order_relaxed);
                } while (s0 != s1);
                if (local[0] != (uint8_t)i) { fprintf(stderr, "torn read\n"); _exit(2); }
            }
            atomic_store_explicit(&b->pong, i, memory_order_release);
        }
        _exit(0);
    }
    pin(ca);
    uint8_t msg[HV_BYTES];
    double t0 = now_ns();
    for (uint64_t i = 1; i <= N; i++) {
        if (!hv) {
            atomic_store_explicit(&b->ping, i, memory_order_release);
        } else {                        // seqlock write
            memset(msg, (uint8_t)i, HV_BYTES);
            atomic_store_explicit(&b->seq, 2 * i - 1, memory_order_relaxed);
            atomic_thread_fence(memory_order_release);
            memcpy(b->hv, msg, HV_BYTES);
            atomic_store_explicit(&b->seq, 2 * i, memory_order_release);
        }
        while (atomic_load_explicit(&b->pong, memory_order_acquire) != i) ;
    }
    double rtt = (now_ns() - t0) / N;
    int st; waitpid(p, &st, 0);
    printf("  cpu%d<->cpu%d  %-18s round-trip %7.1f ns   one-way ~%6.1f ns   %s\n",
           ca, cb, hv ? "1280B HV seqlock" : "8B flag", rtt, rtt / 2,
           WEXITSTATUS(st) ? "FAIL" : "ok");
}

int main(int argc, char **argv) {
    int sib = argc > 1 ? atoi(argv[1]) : 4;   // hyperthread sibling of cpu0
    int far = argc > 2 ? atoi(argv[2]) : 2;   // different physical core
    bus_t *b = mmap(NULL, sizeof(bus_t), PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (b == MAP_FAILED) { perror("mmap"); return 1; }
    printf("Cross-process shared-memory latency, %d iterations each\n", N);
    run(b, 0, far, 0);
    run(b, 0, far, 1);
    run(b, 0, sib, 0);
    run(b, 0, sib, 1);
    return 0;
}
