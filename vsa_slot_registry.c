#include "vsa_slot_registry.h"
#include "vsa_kernel.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>

/* Internal registry structure */
struct vsa_registry {
    vsa_slot_entry_t lanes[10];
    uint32_t lane_count;
    uint32_t total_slots;
};

/* Default concurrency map from VSA_CONCURRENCY_MAP.md (0-based slots) */
static const struct {
    uint32_t start;
    uint32_t end;
    const char* owner;
    const char* purpose;
} default_lanes[10] = {
    {0,   99,  "control_head_1", "Task Queue & Master State Consensus"},
    {100, 199, "control_head_2", "Health & Diagnostic Status (HUD)"},
    {200, 299, "control_head_3", "Orchestration & Scheduling"},
    {300, 399, "control_head_4", "Global Error Register / Circuit Breaker"},
    {400, 499, "worker_lane_1",  "Production Worker 1 (Dev)"},
    {500, 599, "worker_lane_2",  "Production Worker 2 (Dev)"},
    {600, 699, "worker_lane_3",  "Production Worker 3 (Oracle)"},
    {700, 799, "worker_lane_4",  "Production Worker 4 (Bugstomp)"},
    {800, 899, "worker_lane_5",  "Production Worker 5 (Infra)"},
    {900, 999, "worker_lane_6",  "Production Worker 6 (Research)"},
};

vsa_registry_t* vsa_registry_init(void) {
    vsa_registry_t* reg = calloc(1, sizeof(vsa_registry_t));
    if (!reg) return NULL;

    reg->lane_count = 10;
    reg->total_slots = 1000;

    for (uint32_t i = 0; i < 10; ++i) {
        reg->lanes[i].start_slot = default_lanes[i].start;
        reg->lanes[i].end_slot = default_lanes[i].end;
        strncpy(reg->lanes[i].owner, default_lanes[i].owner, VSA_REGISTRY_NAME_MAX - 1);
        reg->lanes[i].owner[VSA_REGISTRY_NAME_MAX - 1] = '\0';
        strncpy(reg->lanes[i].purpose, default_lanes[i].purpose, VSA_REGISTRY_NAME_MAX - 1);
        reg->lanes[i].purpose[VSA_REGISTRY_NAME_MAX - 1] = '\0';
        reg->lanes[i].writer_pid = 0;
    }

    return reg;
}

void vsa_registry_destroy(vsa_registry_t* reg) {
    if (reg) free(reg);
}

static int slot_in_range(const vsa_slot_entry_t* lane, uint32_t slot) {
    return (slot >= lane->start_slot && slot <= lane->end_slot);
}

int vsa_registry_claim_slot(vsa_registry_t* reg, uint32_t slot) {
    if (!reg || slot >= reg->total_slots) return -1;

    pid_t my_pid = getpid();

    for (uint32_t i = 0; i < reg->lane_count; ++i) {
        if (slot_in_range(&reg->lanes[i], slot)) {
            if (reg->lanes[i].writer_pid == 0) {
                reg->lanes[i].writer_pid = my_pid;
                return 0;
            }
            if (reg->lanes[i].writer_pid == my_pid) {
                return 0;  /* Already owned by this process */
            }
            return -2;  /* Owned by different process */
        }
    }
    return -1;  /* Slot not in any registered range */
}

int vsa_registry_release_slot(vsa_registry_t* reg, uint32_t slot) {
    if (!reg || slot >= reg->total_slots) return -1;

    pid_t my_pid = getpid();

    for (uint32_t i = 0; i < reg->lane_count; ++i) {
        if (slot_in_range(&reg->lanes[i], slot)) {
            if (reg->lanes[i].writer_pid == my_pid) {
                reg->lanes[i].writer_pid = 0;
                return 0;
            }
            return -2;  /* Not owned by this process */
        }
    }
    return -1;
}

int vsa_registry_check_slot(const vsa_registry_t* reg, uint32_t slot) {
    if (!reg || slot >= reg->total_slots) return -1;

    for (uint32_t i = 0; i < reg->lane_count; ++i) {
        if (slot_in_range(&reg->lanes[i], slot)) {
            return reg->lanes[i].writer_pid;
        }
    }
    return -1;
}

const vsa_slot_entry_t* vsa_registry_get_lane(const vsa_registry_t* reg, uint32_t lane_index) {
    if (!reg || lane_index >= reg->lane_count) return NULL;
    return &reg->lanes[lane_index];
}

const vsa_slot_entry_t* vsa_registry_find_slot(const vsa_registry_t* reg, uint32_t slot) {
    if (!reg || slot >= reg->total_slots) return NULL;

    for (uint32_t i = 0; i < reg->lane_count; ++i) {
        if (slot_in_range(&reg->lanes[i], slot)) {
            return &reg->lanes[i];
        }
    }
    return NULL;
}

uint32_t vsa_registry_lane_count(const vsa_registry_t* reg) {
    return reg ? reg->lane_count : 0;
}

uint32_t vsa_registry_total_slots(const vsa_registry_t* reg) {
    return reg ? reg->total_slots : 0;
}

vsa_prefix_summary_t vsa_prefix_summary(uint32_t prefix_dims) {
    vsa_prefix_summary_t summary = {0};

    if (prefix_dims == 0 || prefix_dims > VSA_DIMENSIONS) {
        prefix_dims = 512;  /* Default: 512 dims */
    }

    /* Round up to word boundary (64 dims) */
    prefix_dims = (prefix_dims + 63) & ~63u;
    if (prefix_dims > VSA_DIMENSIONS) prefix_dims = VSA_DIMENSIONS;

    summary.prefix_dims = prefix_dims;
    summary.prefix_bytes = (prefix_dims / 8) * 2;  /* Two bitplanes */
    summary.slot_bytes = VSA_HV_BYTES;

    return summary;
}