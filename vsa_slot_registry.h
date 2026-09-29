#ifndef VSA_SLOT_REGISTRY_H
#define VSA_SLOT_REGISTRY_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * VSA Slot Registry
 *
 * The bus has 1000 slots (0-999), each 2560 bytes (one ternary hypervector).
 * This registry maps slot ranges to owners/purposes and enforces single-writer
 * semantics: a second writer claiming the same slot is refused.
 *
 * Slot ranges per VSA_CONCURRENCY_MAP.md:
 *   0-99    (1-100)   Control Head 1: Task Queue & Master State Consensus
 *   100-199 (101-200) Control Head 2: Health & Diagnostic Status (HUD)
 *   200-299 (201-300) Control Head 3: Orchestration & Scheduling
 *   300-399 (301-400) Control Head 4: Global Error Register / Circuit Breaker
 *   400-499 (401-500) Worker Lane 1:  Production Worker 1 (Dev)
 *   500-599 (501-600) Worker Lane 2:  Production Worker 2 (Dev)
 *   600-699 (601-700) Worker Lane 3:  Production Worker 3 (Oracle)
 *   700-799 (701-800) Worker Lane 4:  Production Worker 4 (Bugstomp)
 *   800-899 (801-900) Worker Lane 5:  Production Worker 5 (Infra)
 *   900-999 (901-1000) Worker Lane 6: Production Worker 6 (Research)
 * ------------------------------------------------------------------------- */

/* Maximum length for owner/purpose strings (including null terminator) */
#define VSA_REGISTRY_NAME_MAX 64

/* Slot entry in the registry */
typedef struct {
    uint32_t start_slot;      /* Inclusive start slot (0-based) */
    uint32_t end_slot;        /* Inclusive end slot (0-based) */
    char owner[VSA_REGISTRY_NAME_MAX];
    char purpose[VSA_REGISTRY_NAME_MAX];
    uint32_t writer_pid;      /* PID of registered writer (0 = unclaimed) */
} vsa_slot_entry_t;

/* Registry handle (opaque to caller; allocated by vsa_registry_init) */
typedef struct vsa_registry vsa_registry_t;

/* ---------------------------------------------------------------------------
 * Registry lifecycle
 * ------------------------------------------------------------------------- */

/* Initialize the registry with the default 10-lane concurrency map.
 * Returns a registry handle on success, NULL on failure. */
vsa_registry_t* vsa_registry_init(void);

/* Destroy the registry and free all resources. */
void vsa_registry_destroy(vsa_registry_t* reg);

/* ---------------------------------------------------------------------------
 * Slot claim / release (writer enforcement)
 * ------------------------------------------------------------------------- */

/* Claim a slot for writing by the current process.
 * Returns 0 on success, -1 if slot is out of range, -2 if already claimed
 * by a different PID. The registry records the caller's PID.
 *
 * This is the single-writer enforcement: a second process calling
 * vsa_registry_claim_slot on the same slot will receive -2. */
int vsa_registry_claim_slot(vsa_registry_t* reg, uint32_t slot);

/* Release a slot claimed by the current process.
 * Returns 0 on success, -1 if slot out of range, -2 if not owned by caller. */
int vsa_registry_release_slot(vsa_registry_t* reg, uint32_t slot);

/* Check if a slot is claimed and by whom.
 * Returns 0 if unclaimed, PID of writer if claimed, -1 if slot out of range. */
int vsa_registry_check_slot(const vsa_registry_t* reg, uint32_t slot);

/* ---------------------------------------------------------------------------
 * Registry queries
 * ------------------------------------------------------------------------- */

/* Get the registry entry for a slot (by index 0-9 in the lane map).
 * Returns pointer to static entry, or NULL if index out of range.
 * The entry's start_slot/end_slot define the inclusive range. */
const vsa_slot_entry_t* vsa_registry_get_lane(const vsa_registry_t* reg, uint32_t lane_index);

/* Get the registry entry containing a specific slot.
 * Returns pointer to entry, or NULL if slot not in any registered range. */
const vsa_slot_entry_t* vsa_registry_find_slot(const vsa_registry_t* reg, uint32_t slot);

/* Number of registered lane ranges (always 10 for the default map). */
uint32_t vsa_registry_lane_count(const vsa_registry_t* reg);

/* Total number of slots in the registry (always 1000 for the default map). */
uint32_t vsa_registry_total_slots(const vsa_registry_t* reg);

/* ---------------------------------------------------------------------------
 * Prefix-width summary metadata (XP1: never materialise full vector)
 * ------------------------------------------------------------------------- */

/* A slot holds a prefix-width summary, not a full 10,240-D vector.
 * The prefix width is configurable at runtime (default: 512 dims = 128 bytes).
 * This struct describes the summary layout in a slot. */
typedef struct {
    uint32_t prefix_dims;     /* Number of active dimensions in the prefix */
    uint32_t prefix_bytes;    /* Bytes occupied by prefix (prefix_dims/8 * 2 planes) */
    uint32_t slot_bytes;      /* Total slot size (always VSA_HV_BYTES = 2560) */
} vsa_prefix_summary_t;

/* Get the prefix summary layout for the current configuration.
 * prefix_dims must be a multiple of 64 (word-aligned) and <= VSA_DIMENSIONS. */
vsa_prefix_summary_t vsa_prefix_summary(uint32_t prefix_dims);

#ifdef __cplusplus
}
#endif

#endif /* VSA_SLOT_REGISTRY_H */