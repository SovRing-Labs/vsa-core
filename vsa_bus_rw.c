#include "vsa_slot_registry.h"
#include "vsa_kernel.h"
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>

/* ---------------------------------------------------------------------------
 * VSA Bus Reader/Writer
 *
 * Provides write/read operations over the shared-memory bus using the
 * seqlock primitives from vsa_kernel.h. Slots hold prefix-width summaries
 * only (XP1: never materialise the full 10,240-D vector).
 *
 * Bus path comes from environment variable VSA_BUS_PATH (defaults to
 * /dev/shm/vsa_matrix_bus for data and /dev/shm/vsa_matrix_seq for seqlock).
 *
 * Writer tracking uses a third segment (_wtr suffix) with one uint32_t per slot
 * to enforce single-writer semantics across processes.
 * ------------------------------------------------------------------------- */

typedef struct {
    uint8_t* data_map;        /* Mapped bus data segment */
    uint64_t* seq_map;        /* Mapped seqlock version array */
    uint32_t* wtr_map;        /* Mapped writer tracking array (PID per slot) */
    size_t num_slots;         /* Number of slots in the bus */
    size_t slot_bytes;        /* Bytes per slot (VSA_HV_BYTES = 2560) */
    size_t prefix_bytes;      /* Bytes used for prefix summary */
    int data_fd;              /* File descriptor for data segment */
    int seq_fd;               /* File descriptor for seqlock segment */
    int wtr_fd;               /* File descriptor for writer tracking segment */
    vsa_registry_t* registry; /* Slot registry for writer enforcement (local cache) */
} vsa_bus_t;

/* Default bus paths */
#define VSA_DEFAULT_BUS_PATH  "/dev/shm/vsa_matrix_bus"
#define VSA_DEFAULT_SEQ_PATH  "/dev/shm/vsa_matrix_seq"
#define VSA_DEFAULT_WTR_PATH  "/dev/shm/vsa_matrix_wtr"

/* Forward declaration */
void vsa_bus_close(vsa_bus_t* bus);

/* Open the bus segments. Returns a bus handle or NULL on failure.
 * prefix_dims: number of dimensions in the prefix summary (default 512).
 * The bus path can be overridden with VSA_BUS_PATH env var. */
vsa_bus_t* vsa_bus_open(uint32_t prefix_dims) {
    const char* bus_path = getenv("VSA_BUS_PATH");
    if (!bus_path) bus_path = VSA_DEFAULT_BUS_PATH;

    char seq_path[256];
    snprintf(seq_path, sizeof(seq_path), "%s_seq", bus_path);
    const char* env_seq = getenv("VSA_SEQ_PATH");
    if (env_seq) strncpy(seq_path, env_seq, sizeof(seq_path) - 1);

    char wtr_path[256];
    snprintf(wtr_path, sizeof(wtr_path), "%s_wtr", bus_path);
    const char* env_wtr = getenv("VSA_WTR_PATH");
    if (env_wtr) strncpy(wtr_path, env_wtr, sizeof(wtr_path) - 1);

    vsa_prefix_summary_t summary = vsa_prefix_summary(prefix_dims);

    vsa_bus_t* bus = calloc(1, sizeof(vsa_bus_t));
    if (!bus) return NULL;

    bus->num_slots = 1000;
    bus->slot_bytes = VSA_HV_BYTES;
    bus->prefix_bytes = summary.prefix_bytes;
    bus->data_fd = -1;
    bus->seq_fd = -1;
    bus->wtr_fd = -1;
    bus->registry = vsa_registry_init();

    /* Open data segment */
    bus->data_fd = open(bus_path, O_RDWR);
    if (bus->data_fd < 0) {
        fprintf(stderr, "vsa_bus_open: failed to open %s: %s\n", bus_path, strerror(errno));
        goto fail;
    }

    struct stat st;
    if (fstat(bus->data_fd, &st) < 0) {
        fprintf(stderr, "vsa_bus_open: fstat failed: %s\n", strerror(errno));
        goto fail;
    }

    size_t expected_size = bus->num_slots * bus->slot_bytes;
    if ((size_t)st.st_size != expected_size) {
        fprintf(stderr, "vsa_bus_open: bus size mismatch: got %zu, expected %zu\n",
                (size_t)st.st_size, expected_size);
        goto fail;
    }

    bus->data_map = mmap(NULL, expected_size, PROT_READ | PROT_WRITE, MAP_SHARED, bus->data_fd, 0);
    if (bus->data_map == MAP_FAILED) {
        fprintf(stderr, "vsa_bus_open: mmap data failed: %s\n", strerror(errno));
        bus->data_map = NULL;
        goto fail;
    }

    /* Open seqlock segment */
    bus->seq_fd = open(seq_path, O_RDWR);
    if (bus->seq_fd < 0) {
        fprintf(stderr, "vsa_bus_open: failed to open %s: %s\n", seq_path, strerror(errno));
        goto fail;
    }

    if (fstat(bus->seq_fd, &st) < 0) {
        fprintf(stderr, "vsa_bus_open: fstat seq failed: %s\n", strerror(errno));
        goto fail;
    }

    size_t seq_expected = bus->num_slots * sizeof(uint64_t);
    if ((size_t)st.st_size != seq_expected) {
        fprintf(stderr, "vsa_bus_open: seq size mismatch: got %zu, expected %zu\n",
                (size_t)st.st_size, seq_expected);
        goto fail;
    }

    bus->seq_map = mmap(NULL, seq_expected, PROT_READ | PROT_WRITE, MAP_SHARED, bus->seq_fd, 0);
    if (bus->seq_map == MAP_FAILED) {
        fprintf(stderr, "vsa_bus_open: mmap seq failed: %s\n", strerror(errno));
        bus->seq_map = NULL;
        goto fail;
    }

    /* Open writer tracking segment */
    bus->wtr_fd = open(wtr_path, O_RDWR);
    if (bus->wtr_fd < 0) {
        fprintf(stderr, "vsa_bus_open: failed to open %s: %s\n", wtr_path, strerror(errno));
        goto fail;
    }

    if (fstat(bus->wtr_fd, &st) < 0) {
        fprintf(stderr, "vsa_bus_open: fstat wtr failed: %s\n", strerror(errno));
        goto fail;
    }

    size_t wtr_expected = bus->num_slots * sizeof(uint32_t);
    if ((size_t)st.st_size != wtr_expected) {
        fprintf(stderr, "vsa_bus_open: wtr size mismatch: got %zu, expected %zu\n",
                (size_t)st.st_size, wtr_expected);
        goto fail;
    }

    bus->wtr_map = mmap(NULL, wtr_expected, PROT_READ | PROT_WRITE, MAP_SHARED, bus->wtr_fd, 0);
    if (bus->wtr_map == MAP_FAILED) {
        fprintf(stderr, "vsa_bus_open: mmap wtr failed: %s\n", strerror(errno));
        bus->wtr_map = NULL;
        goto fail;
    }

    return bus;

fail:
    vsa_bus_close(bus);
    return NULL;
}

/* Close the bus and free resources. */
void vsa_bus_close(vsa_bus_t* bus) {
    if (!bus) return;

    if (bus->data_map && bus->data_map != MAP_FAILED) {
        munmap(bus->data_map, bus->num_slots * bus->slot_bytes);
    }
    if (bus->seq_map && bus->seq_map != MAP_FAILED) {
        munmap(bus->seq_map, bus->num_slots * sizeof(uint64_t));
    }
    if (bus->wtr_map && bus->wtr_map != MAP_FAILED) {
        munmap(bus->wtr_map, bus->num_slots * sizeof(uint32_t));
    }
    if (bus->data_fd >= 0) close(bus->data_fd);
    if (bus->seq_fd >= 0) close(bus->seq_fd);
    if (bus->wtr_fd >= 0) close(bus->wtr_fd);
    vsa_registry_destroy(bus->registry);
    free(bus);
}

/* Get the slot registry for writer claims. */
vsa_registry_t* vsa_bus_registry(vsa_bus_t* bus) {
    return bus ? bus->registry : NULL;
}

/* Shared writer tracking: claim a slot for the current PID.
 * Returns 0 on success, -1 if slot invalid, -2 if already claimed by another PID. */
int vsa_bus_claim_slot(vsa_bus_t* bus, uint32_t slot) {
    if (!bus || !bus->wtr_map || slot >= bus->num_slots) return -1;

    uint32_t my_pid = (uint32_t)getpid();
    uint32_t* wtr = &bus->wtr_map[slot];

    /* Atomic compare-and-swap: claim if unowned (0) or owned by us */
    uint32_t expected = 0;
    if (__atomic_compare_exchange_n(wtr, &expected, my_pid, 0,
                                     __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        return 0;  /* Successfully claimed */
    }

    /* Check if we already own it */
    if (expected == my_pid) {
        return 0;  /* Already owned by this process */
    }

    return -2;  /* Owned by another process */
}

/* Shared writer tracking: release a slot claimed by the current PID.
 * Returns 0 on success, -1 if slot invalid, -2 if not owned by caller. */
int vsa_bus_release_slot(vsa_bus_t* bus, uint32_t slot) {
    if (!bus || !bus->wtr_map || slot >= bus->num_slots) return -1;

    uint32_t my_pid = (uint32_t)getpid();
    uint32_t* wtr = &bus->wtr_map[slot];

    /* Atomic compare-and-swap: release only if we own it */
    uint32_t expected = my_pid;
    if (__atomic_compare_exchange_n(wtr, &expected, 0, 0,
                                     __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        return 0;  /* Successfully released */
    }

    return -2;  /* Not owned by this process */
}

/* Shared writer tracking: check who owns a slot.
 * Returns 0 if unclaimed, PID of writer if claimed, -1 if slot invalid. */
int vsa_bus_check_slot(vsa_bus_t* bus, uint32_t slot) {
    if (!bus || !bus->wtr_map || slot >= bus->num_slots) return -1;
    return (int)__atomic_load_n(&bus->wtr_map[slot], __ATOMIC_ACQUIRE);
}

/* Write a prefix summary to a slot under seqlock protection.
 * src: pointer to prefix data (prefix_bytes bytes, two bitplanes packed as sign then zero).
 * slot: target slot index (0-999).
 * Returns 0 on success, -1 on invalid slot, -2 if slot not claimed by caller,
 * -3 on seqlock/map error. */
int vsa_bus_write_prefix(vsa_bus_t* bus, uint32_t slot, const void* src) {
    if (!bus || !bus->data_map || !bus->seq_map || !src) return -3;
    if (slot >= bus->num_slots) return -1;

    /* Enforce single-writer: check shared writer tracking */
    int claim = vsa_bus_check_slot(bus, slot);
    if (claim < 0) return -1;  /* Invalid slot */
    if (claim != 0 && claim != (int)getpid()) return -2;  /* Not owned by caller */

    uint64_t* seq = &bus->seq_map[slot];
    uint8_t* dst = bus->data_map + slot * bus->slot_bytes;

    /* Seqlock write: begin -> write prefix -> end */
    vsa_write_begin(seq);

    /* Write prefix summary (sign plane then zero plane) */
    memcpy(dst, src, bus->prefix_bytes);

    /* Zero the remainder of the slot (prefix-width only, rest is stasis) */
    if (bus->prefix_bytes < bus->slot_bytes) {
        memset(dst + bus->prefix_bytes, 0, bus->slot_bytes - bus->prefix_bytes);
    }

    vsa_write_end(seq);
    return 0;
}

/* Read a prefix summary from a slot under seqlock torn-read protection.
 * dst: buffer to receive prefix data (must be at least prefix_bytes).
 * slot: source slot index (0-999).
 * Returns 0 on success, -1 on invalid slot, -3 on map error.
 * Retries until a consistent snapshot is obtained. */
int vsa_bus_read_prefix(vsa_bus_t* bus, uint32_t slot, void* dst) {
    if (!bus || !bus->data_map || !bus->seq_map || !dst) return -3;
    if (slot >= bus->num_slots) return -1;

    const uint64_t* seq = &bus->seq_map[slot];
    const uint8_t* src = bus->data_map + slot * bus->slot_bytes;

    /* Seqlock read with retry loop */
    uint64_t start;
    do {
        start = vsa_read_begin(seq);
        memcpy(dst, src, bus->prefix_bytes);
    } while (vsa_read_retry(seq, start));

    return 0;
}

/* Write a full hypervector to a slot (for compatibility with existing lanes).
 * This writes the full 2560 bytes. Use vsa_bus_write_prefix for prefix summaries.
 * src: pointer to vsa_hv_t (2560 bytes).
 * slot: target slot index (0-999).
 * Returns 0 on success, -1 on invalid slot, -2 if slot not claimed by caller. */
int vsa_bus_write_hv(vsa_bus_t* bus, uint32_t slot, const vsa_hv_t* src) {
    if (!bus || !bus->data_map || !bus->seq_map || !src) return -3;
    if (slot >= bus->num_slots) return -1;

    int claim = vsa_bus_check_slot(bus, slot);
    if (claim < 0) return -1;
    if (claim != 0 && claim != (int)getpid()) return -2;

    uint64_t* seq = &bus->seq_map[slot];
    uint8_t* dst = bus->data_map + slot * bus->slot_bytes;

    vsa_write_begin(seq);
    memcpy(dst, src, bus->slot_bytes);
    vsa_write_end(seq);
    return 0;
}

/* Read a full hypervector from a slot under seqlock protection.
 * dst: pointer to vsa_hv_t to receive data.
 * slot: source slot index (0-999).
 * Returns 0 on success, -1 on invalid slot. */
int vsa_bus_read_hv(vsa_bus_t* bus, uint32_t slot, vsa_hv_t* dst) {
    if (!bus || !bus->data_map || !bus->seq_map || !dst) return -3;
    if (slot >= bus->num_slots) return -1;

    const uint64_t* seq = &bus->seq_map[slot];
    const uint8_t* src = bus->data_map + slot * bus->slot_bytes;

    uint64_t start;
    do {
        start = vsa_read_begin(seq);
        memcpy(dst, src, bus->slot_bytes);
    } while (vsa_read_retry(seq, start));

    return 0;
}

/* Get the prefix summary layout for this bus. */
vsa_prefix_summary_t vsa_bus_prefix_summary(const vsa_bus_t* bus) {
    vsa_prefix_summary_t s = {0};
    if (bus) {
        s.prefix_dims = bus->prefix_bytes * 4;  /* 2 planes * 8 bits/byte = 16 dims/byte */
        s.prefix_bytes = bus->prefix_bytes;
        s.slot_bytes = bus->slot_bytes;
    }
    return s;
}