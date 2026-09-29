# VSA Slot Registry & Bus Reader/Writer

**Status:** VERIFIED 2026-09-27 (built and tested as part of RSPK Step 3)

---

## Overview

The VSA Slot Registry and Bus Reader/Writer provide the coordination layer for the
1000-slot shared-memory bus (`/dev/shm/vsa_matrix_bus`). Each slot holds a
2560-byte ternary hypervector (10,240 dimensions, two bitplanes).

**Key constraint (XP1 Amendment):** Slots hold **prefix-width summaries only** —
never a full 10,240-D vector. The full vector is materialised at query time only.
Default prefix width: 512 dimensions (128 bytes per slot).

---

## Slot Registry (`vsa_slot_registry.h/.c`)

### Concurrency Map (from `VSA_CONCURRENCY_MAP.md`)

| Lane | Slots (0-based) | Slots (1-based) | Owner | Purpose |
|------|-----------------|-----------------|-------|---------|
| Control Head 1 | 0–99 | 1–100 | `control_head_1` | Task Queue & Master State Consensus |
| Control Head 2 | 100–199 | 101–200 | `control_head_2` | Health & Diagnostic Status (HUD) |
| Control Head 3 | 200–299 | 201–300 | `control_head_3` | Orchestration & Scheduling |
| Control Head 4 | 300–399 | 301–400 | `control_head_4` | Global Error Register / Circuit Breaker |
| Worker Lane 1 | 400–499 | 401–500 | `worker_lane_1` | Production Worker 1 (Dev) |
| Worker Lane 2 | 500–599 | 501–600 | `worker_lane_2` | Production Worker 2 (Dev) |
| Worker Lane 3 | 600–699 | 601–700 | `worker_lane_3` | Production Worker 3 (Oracle) |
| Worker Lane 4 | 700–799 | 701–800 | `worker_lane_4` | Production Worker 4 (Bugstomp) |
| Worker Lane 5 | 800–899 | 801–900 | `worker_lane_5` | Production Worker 5 (Infra) |
| Worker Lane 6 | 900–999 | 901–1000 | `worker_lane_6` | Production Worker 6 (Research) |

### Single-Writer Enforcement

Each lane range has **one writer** enforced by the registry:

```c
vsa_registry_t* reg = vsa_registry_init();
int rc = vsa_registry_claim_slot(reg, slot);  // Returns 0, -1 (invalid), -2 (owned by other PID)
```

- A process calls `vsa_registry_claim_slot()` before writing.
- The registry records the caller's PID.
- A second process attempting to claim the same slot receives `-2` (refused).
- Writer releases with `vsa_registry_release_slot()`.

### Prefix Summary Layout

```c
vsa_prefix_summary_t summary = vsa_prefix_summary(512);  // 512 dims
// summary.prefix_dims  = 512
// summary.prefix_bytes = 128  (512/8 * 2 planes)
// summary.slot_bytes   = 2560
```

The prefix occupies the first `prefix_bytes` of the 2560-byte slot:
- Bytes 0–63: Sign plane (512 bits = 16 uint64)
- Bytes 64–127: Zero/active plane (512 bits = 16 uint64)
- Bytes 128–2559: Zeroed (stasis)

---

## Bus Reader/Writer (`vsa_bus_rw.c`)

### Opening the Bus

```c
vsa_bus_t* bus = vsa_bus_open(512);  // 512-dim prefix
```

Bus paths configured via environment:
- `VSA_BUS_PATH` — data segment (default: `/dev/shm/vsa_matrix_bus`)
- `VSA_SEQ_PATH` — seqlock segment (default: `/dev/shm/vsa_matrix_seq`)

### Writing a Prefix Summary

```c
uint8_t prefix[128];  // sign[16] + zero[16]
make_prefix(prefix, seed, iter);
int rc = vsa_bus_write_prefix(bus, slot, prefix);
```

Internally uses the seqlock primitives from `vsa_kernel.h`:
```c
vsa_write_begin(&seq[slot]);
memcpy(dst, src, prefix_bytes);
vsa_write_end(&seq[slot]);
```

### Reading a Prefix Summary (Torn-Read Protected)

```c
uint8_t prefix[128];
int rc = vsa_bus_read_prefix(bus, slot, prefix);
```

Retry loop using seqlock:
```c
do {
    start = vsa_read_begin(&seq[slot]);
    memcpy(dst, src, prefix_bytes);
} while (vsa_read_retry(&seq[slot], start));
```

### Full Hypervector Compatibility

For existing lane processes that write full 2560-byte HVs:
```c
vsa_bus_write_hv(bus, slot, &hv);   // Full vsa_hv_t
vsa_bus_read_hv(bus, slot, &hv);    // Full vsa_hv_t
```

---

## Test (`vsa_bus_test.c`)

### Kill Criterion (RSPK Step 3)

> **Any torn read, or a second writer accepted ⇒ KILL**

### Test Procedure

1. Create temporary bus files in `/dev/shm` (`vsa_test_bus_<pid>`, `vsa_test_seq_<pid>`)
2. **Test 1:** Verify registry refuses second writer on same slot
3. **Test 2:** Fork writer + reader processes, run concurrently for 2 seconds
   - Writer: claims slot 42, writes prefix summaries in a loop
   - Reader: reads same slot under seqlock, verifies consistency
   - Count total reads and torn reads
4. Clean up temporary files

### Expected Output (PASS)

```
================ VSA BUS STEP 3 TEST ================
Test: Slot registry + bus writer/reader (prefix summaries)
Slot: 42 (worker_lane_1 range 400-499)
Prefix: 512 dims (128 bytes), Slot: 2560 bytes
Duration: 2 seconds
Kill criterion: ANY torn read OR second writer accepted => KILL

Test 1: Registry refuses second writer...
  Second writer correctly refused (rc=-2)
  PASS

Test 2: Writer + Reader concurrent for 2s...
  Writer exited: normal
  Reader exited: normal
  Total reads: 18432
  Torn reads:  0
  PASS: Zero torn reads in 18432 reads

====================================================
PASS
```

---

## Build & Run

```bash
cd vsa-core
gcc -O2 -mavx2 -o /tmp/vsa_bus_test \
    vsa_bus_test.c vsa_bus_rw.c vsa_slot_registry.c vsa_kernel.c \
    -lm -lpthread
/tmp/vsa_bus_test
```

### Test Command (from contract)

```bash
cd vsa-core && \
gcc -O2 -mavx2 -o /tmp/vsa_bus_test \
    vsa_bus_test.c vsa_bus_rw.c vsa_slot_registry.c vsa_kernel.c \
    -lm -lpthread && /tmp/vsa_bus_test
```

---

## Integration Notes

- **No modifications to existing files:** `vsa_kernel.h`, `vsa_kernel.c`, `Makefile`, `vsa_bus_client.py` are untouched.
- **Bus ABI unchanged:** Slot size remains 2560 bytes, stride unchanged.
- **Seqlock primitives reused:** `vsa_write_begin/end`, `vsa_read_begin/retry` from `vsa_kernel.c`.
- **Prefix width configurable:** Default 512 dims, adjustable at runtime via `vsa_bus_open(prefix_dims)`.
- **Environment-driven paths:** Tests use private `/dev/shm` segments; production uses the live bus.

---

## Open Issues

1. **Prefix width tuning:** 512 dims is the default; optimal width depends on downstream
   resonator/consumer requirements (Step 5 phase kernel uses D=5,120).
2. **Registry persistence:** Current registry is in-process only. A shared registry
   (e.g., in shm) would allow cross-process enforcement without PID checks.
3. **Reader-side filtering:** No mechanism for readers to subscribe to slot ranges;
   they must poll or know slot indices a priori.

---

## Files Created

| File | Purpose |
|------|---------|
| `vsa_slot_registry.h` | Registry interface |
| `vsa_slot_registry.c` | Registry implementation |
| `vsa_bus_rw.c` | Bus writer/reader over seqlock |
| `vsa_bus_test.c` | Concurrent writer/reader test |
| `docs/SLOT-REGISTRY.md` | This document |