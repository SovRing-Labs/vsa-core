#ifndef VSA_BUS_RW_H
#define VSA_BUS_RW_H

#include "vsa_kernel.h"
#include "vsa_slot_registry.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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

typedef struct vsa_bus vsa_bus_t;

/* Open the bus segments. Returns a bus handle or NULL on failure.
 * prefix_dims: number of dimensions in the prefix summary (default 512).
 * The bus path can be overridden with VSA_BUS_PATH env var. */
vsa_bus_t* vsa_bus_open(uint32_t prefix_dims);

/* Close the bus and free resources. */
void vsa_bus_close(vsa_bus_t* bus);

/* Get the slot registry for lane/owner queries (local cache). */
vsa_registry_t* vsa_bus_registry(vsa_bus_t* bus);

/* Shared writer tracking: claim a slot for the current PID.
 * Returns 0 on success, -1 if slot invalid, -2 if already claimed by another PID. */
int vsa_bus_claim_slot(vsa_bus_t* bus, uint32_t slot);

/* Shared writer tracking: release a slot claimed by the current PID.
 * Returns 0 on success, -1 if slot invalid, -2 if not owned by caller. */
int vsa_bus_release_slot(vsa_bus_t* bus, uint32_t slot);

/* Shared writer tracking: check who owns a slot.
 * Returns 0 if unclaimed, PID of writer if claimed, -1 if slot invalid. */
int vsa_bus_check_slot(vsa_bus_t* bus, uint32_t slot);

/* Write a prefix summary to a slot under seqlock protection.
 * src: pointer to prefix data (prefix_bytes bytes, two bitplanes packed as sign then zero).
 * slot: target slot index (0-999).
 * Returns 0 on success, -1 on invalid slot, -2 if slot not claimed by caller,
 * -3 on seqlock/map error. */
int vsa_bus_write_prefix(vsa_bus_t* bus, uint32_t slot, const void* src);

/* Read a prefix summary from a slot under seqlock torn-read protection.
 * dst: buffer to receive prefix data (must be at least prefix_bytes).
 * slot: source slot index (0-999).
 * Returns 0 on success, -1 on invalid slot, -3 on map error.
 * Retries until a consistent snapshot is obtained. */
int vsa_bus_read_prefix(vsa_bus_t* bus, uint32_t slot, void* dst);

/* Write a full hypervector to a slot (for compatibility with existing lanes).
 * This writes the full 2560 bytes. Use vsa_bus_write_prefix for prefix summaries.
 * src: pointer to vsa_hv_t (2560 bytes).
 * slot: target slot index (0-999).
 * Returns 0 on success, -1 on invalid slot, -2 if slot not claimed by caller. */
int vsa_bus_write_hv(vsa_bus_t* bus, uint32_t slot, const vsa_hv_t* src);

/* Read a full hypervector from a slot under seqlock protection.
 * dst: pointer to vsa_hv_t to receive data.
 * slot: source slot index (0-999).
 * Returns 0 on success, -1 on invalid slot. */
int vsa_bus_read_hv(vsa_bus_t* bus, uint32_t slot, vsa_hv_t* dst);

/* Get the prefix summary layout for this bus. */
vsa_prefix_summary_t vsa_bus_prefix_summary(const vsa_bus_t* bus);

#ifdef __cplusplus
}
#endif

#endif /* VSA_BUS_RW_H */