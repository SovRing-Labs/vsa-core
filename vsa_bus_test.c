/* VSA Bus Test — Slot Registry + Writer/Reader (Prefix Summaries)
 *
 * Test requirements (from VSAW1-S3-BUS):
 * - One writer process + one reader process on temporary bus for 2s
 * - Zero torn reads
 * - Registry refuses a second writer (shared writer tracking)
 * - Bus path from env VSA_BUS_PATH
 * - Kill criterion: any torn read, or second writer accepted => KILL
 */

#include "vsa_slot_registry.h"
#include "vsa_bus_rw.h"
#include "vsa_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <errno.h>

#define TEST_SLOTS       1000
#define TEST_HV_BYTES    2560
#define TEST_PREFIX_DIMS 512
#define TEST_PREFIX_BYTES ((TEST_PREFIX_DIMS / 8) * 2)  /* 128 bytes */
#define TEST_DURATION_SEC 2
#define TEST_SLOT        42  /* Slot in worker_lane_1 range (400-499) */

static volatile sig_atomic_t g_stop = 0;

/* Shared synchronization structure in shared memory */
typedef struct {
    volatile uint32_t writer_ready;
    volatile uint32_t reader_ready;
    volatile uint32_t stop_flag;
    volatile int torn_reads;
    volatile int total_reads;
} sync_t;

static sync_t* g_sync = NULL;

static void sig_handler(int sig) {
    (void)sig;
    g_stop = 1;
    if (g_sync) g_sync->stop_flag = 1;
}

/* Generate a deterministic pseudo-random prefix summary with embedded counter */
static void make_prefix(uint8_t* buf, uint32_t seed, uint32_t iter) {
    uint64_t state = ((uint64_t)seed << 32) | iter;
    for (size_t i = 0; i < TEST_PREFIX_BYTES / 8; ++i) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((uint64_t*)buf)[i] = state;
    }
    /* Embed iteration counter in first word of each plane for consistency check */
    ((uint64_t*)buf)[0] = iter;      /* sign plane first word (byte 0) */
    ((uint64_t*)buf)[8] = iter;      /* zero plane first word (byte 64) */
}

/* Verify a prefix summary is self-consistent by checking embedded sequence number */
static int verify_prefix(const uint8_t* buf) {
    /* First 8 bytes (first uint64 of sign plane) hold the iteration counter.
     * Byte 64 (first uint64 of zero plane) holds the same counter.
     * A consistent read should have both matching. */
    uint64_t sign_counter = ((uint64_t*)buf)[0];
    uint64_t zero_counter = ((uint64_t*)buf)[8];  /* zero plane starts at byte 64 = 8*8 */
    return (sign_counter == zero_counter) ? 1 : 0;
}

/* Writer process */
static int writer_process(const char* bus_path, const char* seq_path, const char* wtr_path) {
    signal(SIGTERM, sig_handler);
    signal(SIGINT, sig_handler);

    /* Set env for bus paths */
    setenv("VSA_BUS_PATH", bus_path, 1);
    setenv("VSA_SEQ_PATH", seq_path, 1);
    setenv("VSA_WTR_PATH", wtr_path, 1);

    vsa_bus_t* bus = vsa_bus_open(TEST_PREFIX_DIMS);
    if (!bus) {
        fprintf(stderr, "Writer: failed to open bus\n");
        return 1;
    }

    /* Claim the test slot using shared writer tracking */
    if (vsa_bus_claim_slot(bus, TEST_SLOT) != 0) {
        fprintf(stderr, "Writer: failed to claim slot %d\n", TEST_SLOT);
        vsa_bus_close(bus);
        return 1;
    }

    uint8_t prefix[TEST_PREFIX_BYTES];
    uint32_t iter = 0;

    /* Signal ready via shared memory */
    __atomic_store_n(&g_sync->writer_ready, 1, __ATOMIC_RELEASE);

    /* Wait for reader to be ready */
    while (!g_sync->reader_ready && !g_sync->stop_flag) {
        usleep(1000);
    }

    struct timespec start_ts, now_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);

    while (!g_sync->stop_flag) {
        make_prefix(prefix, 0xDEADBEEF, iter++);
        int rc = vsa_bus_write_prefix(bus, TEST_SLOT, prefix);
        if (rc != 0) {
            fprintf(stderr, "Writer: write failed rc=%d\n", rc);
        }

        clock_gettime(CLOCK_MONOTONIC, &now_ts);
        double elapsed = (now_ts.tv_sec - start_ts.tv_sec) +
                         (now_ts.tv_nsec - start_ts.tv_nsec) / 1e9;
        if (elapsed >= TEST_DURATION_SEC) {
            g_sync->stop_flag = 1;
            break;
        }

        /* Small delay to allow reader to catch some writes */
        usleep(100);
    }

    vsa_bus_release_slot(bus, TEST_SLOT);
    vsa_bus_close(bus);
    return 0;
}

/* Reader process */
static int reader_process(const char* bus_path, const char* seq_path, const char* wtr_path) {
    signal(SIGTERM, sig_handler);
    signal(SIGINT, sig_handler);

    setenv("VSA_BUS_PATH", bus_path, 1);
    setenv("VSA_SEQ_PATH", seq_path, 1);
    setenv("VSA_WTR_PATH", wtr_path, 1);

    vsa_bus_t* bus = vsa_bus_open(TEST_PREFIX_DIMS);
    if (!bus) {
        fprintf(stderr, "Reader: failed to open bus\n");
        return 1;
    }

    uint8_t prefix[TEST_PREFIX_BYTES];

    /* Signal ready via shared memory */
    __atomic_store_n(&g_sync->reader_ready, 1, __ATOMIC_RELEASE);

    /* Wait for writer to be ready */
    while (!g_sync->writer_ready && !g_sync->stop_flag) {
        usleep(1000);
    }

    while (!g_sync->stop_flag) {
        int rc = vsa_bus_read_prefix(bus, TEST_SLOT, prefix);
        if (rc != 0) {
            fprintf(stderr, "Reader: read failed rc=%d\n", rc);
        } else {
            __atomic_add_fetch(&g_sync->total_reads, 1, __ATOMIC_RELAXED);
            if (!verify_prefix(prefix)) {
                __atomic_add_fetch(&g_sync->torn_reads, 1, __ATOMIC_RELAXED);
            }
        }
        usleep(50);  /* Read faster than writer writes */
    }

    vsa_bus_close(bus);
    return 0;
}

/* Test second writer rejection using shared writer tracking */
static int test_second_writer_rejection(const char* bus_path, const char* seq_path, const char* wtr_path) {
    setenv("VSA_BUS_PATH", bus_path, 1);
    setenv("VSA_SEQ_PATH", seq_path, 1);
    setenv("VSA_WTR_PATH", wtr_path, 1);

    vsa_bus_t* bus1 = vsa_bus_open(TEST_PREFIX_DIMS);
    if (!bus1) return 1;

    vsa_bus_t* bus2 = vsa_bus_open(TEST_PREFIX_DIMS);
    if (!bus2) {
        vsa_bus_close(bus1);
        return 1;
    }

    /* First writer claims slot */
    int rc1 = vsa_bus_claim_slot(bus1, TEST_SLOT);
    if (rc1 != 0) {
        fprintf(stderr, "Second writer test: first claim failed rc=%d\n", rc1);
        vsa_bus_close(bus1);
        vsa_bus_close(bus2);
        return 1;
    }

    /* Fork child to act as second writer */
    pid_t child = fork();
    if (child < 0) {
        perror("fork second writer");
        vsa_bus_release_slot(bus1, TEST_SLOT);
        vsa_bus_close(bus1);
        vsa_bus_close(bus2);
        return 1;
    }

    int child_rc = 0;
    if (child == 0) {
        /* Child: second writer */
        vsa_bus_t* bus3 = vsa_bus_open(TEST_PREFIX_DIMS);
        if (!bus3) {
            _exit(1);
        }
        int rc2 = vsa_bus_claim_slot(bus3, TEST_SLOT);
        vsa_bus_close(bus3);
        _exit(rc2 == -2 ? 0 : 1);  /* Exit 0 if refused, 1 if accepted */
    } else {
        /* Parent: wait for child */
        int status;
        waitpid(child, &status, 0);
        child_rc = WEXITSTATUS(status);
    }

    vsa_bus_release_slot(bus1, TEST_SLOT);
    vsa_bus_close(bus1);
    vsa_bus_close(bus2);

    if (child_rc == 0) {
        printf("  Second writer correctly refused (rc=-2)\n");
        return 0;  /* PASS: correctly refused */
    } else {
        fprintf(stderr, "  Second writer NOT refused (child exit=%d)\n", child_rc);
        return 1;  /* FAIL: second writer accepted */
    }
}

/* Create temporary bus files (data, seq, wtr, sync) */
static int create_test_bus(const char* bus_path, const char* seq_path, const char* wtr_path, const char* sync_path) {
    int fd = open(bus_path, O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) { perror("create bus file"); return -1; }
    size_t bus_size = TEST_SLOTS * TEST_HV_BYTES;
    if (ftruncate(fd, bus_size) < 0) { perror("ftruncate bus"); close(fd); return -1; }
    void* ptr = mmap(NULL, bus_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) { perror("mmap bus"); close(fd); return -1; }
    memset(ptr, 0, bus_size);
    munmap(ptr, bus_size);
    close(fd);

    fd = open(seq_path, O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) { perror("create seq file"); return -1; }
    size_t seq_size = TEST_SLOTS * sizeof(uint64_t);
    if (ftruncate(fd, seq_size) < 0) { perror("ftruncate seq"); close(fd); return -1; }
    ptr = mmap(NULL, seq_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) { perror("mmap seq"); close(fd); return -1; }
    memset(ptr, 0, seq_size);
    munmap(ptr, seq_size);
    close(fd);

    fd = open(wtr_path, O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) { perror("create wtr file"); return -1; }
    size_t wtr_size = TEST_SLOTS * sizeof(uint32_t);
    if (ftruncate(fd, wtr_size) < 0) { perror("ftruncate wtr"); close(fd); return -1; }
    ptr = mmap(NULL, wtr_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) { perror("mmap wtr"); close(fd); return -1; }
    memset(ptr, 0, wtr_size);
    munmap(ptr, wtr_size);
    close(fd);

    fd = open(sync_path, O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) { perror("create sync file"); return -1; }
    size_t sync_size = sizeof(sync_t);
    if (ftruncate(fd, sync_size) < 0) { perror("ftruncate sync"); close(fd); return -1; }
    ptr = mmap(NULL, sync_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) { perror("mmap sync"); close(fd); return -1; }
    memset(ptr, 0, sync_size);
    munmap(ptr, sync_size);
    close(fd);

    return 0;
}

/* Map the sync segment */
static int map_sync(const char* sync_path) {
    int fd = open(sync_path, O_RDWR);
    if (fd < 0) { perror("open sync"); return -1; }
    void* ptr = mmap(NULL, sizeof(sync_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) { perror("mmap sync"); return -1; }
    g_sync = (sync_t*)ptr;
    return 0;
}

static void unmap_sync(void) {
    if (g_sync) {
        munmap(g_sync, sizeof(sync_t));
        g_sync = NULL;
    }
}

static void cleanup_test_bus(const char* bus_path, const char* seq_path, const char* wtr_path, const char* sync_path) {
    unlink(bus_path);
    unlink(seq_path);
    unlink(wtr_path);
    unlink(sync_path);
}

int main(void) {
    printf("================ VSA BUS STEP 3 TEST ================\n");
    printf("Test: Slot registry + bus writer/reader (prefix summaries)\n");
    printf("Slot: %d (worker_lane_1 range 400-499)\n", TEST_SLOT);
    printf("Prefix: %d dims (%d bytes), Slot: %d bytes\n",
           TEST_PREFIX_DIMS, TEST_PREFIX_BYTES, TEST_HV_BYTES);
    printf("Duration: %d seconds\n", TEST_DURATION_SEC);
    printf("Kill criterion: ANY torn read OR second writer accepted => KILL\n\n");

    /* Create temporary bus files in /dev/shm with unique names */
    char bus_path[256], seq_path[256], wtr_path[256], sync_path[256];
    snprintf(bus_path, sizeof(bus_path), "/dev/shm/vsa_test_bus_%d", getpid());
    snprintf(seq_path, sizeof(seq_path), "/dev/shm/vsa_test_seq_%d", getpid());
    snprintf(wtr_path, sizeof(wtr_path), "/dev/shm/vsa_test_wtr_%d", getpid());
    snprintf(sync_path, sizeof(sync_path), "/dev/shm/vsa_test_sync_%d", getpid());

    if (create_test_bus(bus_path, seq_path, wtr_path, sync_path) != 0) {
        fprintf(stderr, "Failed to create test bus\n");
        return 1;
    }

    if (map_sync(sync_path) != 0) {
        fprintf(stderr, "Failed to map sync segment\n");
        cleanup_test_bus(bus_path, seq_path, wtr_path, sync_path);
        return 1;
    }

    int kill_reason = 0;
    char kill_msg[256] = {0};

    /* ---- Test 1: Shared writer tracking single-writer enforcement ---- */
    printf("Test 1: Registry refuses second writer...\n");
    if (test_second_writer_rejection(bus_path, seq_path, wtr_path) != 0) {
        kill_reason = 1;
        snprintf(kill_msg, sizeof(kill_msg), "Registry accepted second writer on slot %d", TEST_SLOT);
    } else {
        printf("  PASS\n");
    }

    /* ---- Test 2: Writer + Reader concurrent for 2s ---- */
    if (!kill_reason) {
        printf("\nTest 2: Writer + Reader concurrent for %ds...\n", TEST_DURATION_SEC);

        pid_t writer_pid = fork();
        if (writer_pid == 0) {
            _exit(writer_process(bus_path, seq_path, wtr_path));
        } else if (writer_pid < 0) {
            perror("fork writer");
            kill_reason = 1;
            snprintf(kill_msg, sizeof(kill_msg), "fork writer failed");
        } else {
            pid_t reader_pid = fork();
            if (reader_pid == 0) {
                _exit(reader_process(bus_path, seq_path, wtr_path));
            } else if (reader_pid < 0) {
                perror("fork reader");
                kill(writer_pid, SIGTERM);
                kill_reason = 1;
                snprintf(kill_msg, sizeof(kill_msg), "fork reader failed");
            } else {
                /* Parent: wait for both */
                int wstatus, rstatus;
                waitpid(writer_pid, &wstatus, 0);
                waitpid(reader_pid, &rstatus, 0);

                printf("  Writer exited: %s\n", WIFEXITED(wstatus) ? "normal" : "signal");
                printf("  Reader exited: %s\n", WIFEXITED(rstatus) ? "normal" : "signal");
                printf("  Total reads: %d\n", g_sync->total_reads);
                printf("  Torn reads:  %d\n", g_sync->torn_reads);

                if (g_sync->torn_reads > 0) {
                    kill_reason = 1;
                    snprintf(kill_msg, sizeof(kill_msg),
                             "%d torn reads detected out of %d total reads",
                             g_sync->torn_reads, g_sync->total_reads);
                } else if (g_sync->total_reads == 0) {
                    kill_reason = 1;
                    snprintf(kill_msg, sizeof(kill_msg), "Zero reads performed (reader failed)");
                } else {
                    printf("  PASS: Zero torn reads in %d reads\n", g_sync->total_reads);
                }
            }
        }
    }

    unmap_sync();
    cleanup_test_bus(bus_path, seq_path, wtr_path, sync_path);

    printf("\n====================================================\n");
    if (kill_reason) {
        printf("KILL: %s\n", kill_msg);
        return 0;  /* Exit 0 with KILL message per spec */
    } else {
        printf("PASS\n");
        return 0;
    }
}