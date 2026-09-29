# VSA Matrix Core

Ternary Vector Symbolic Architecture kernel and shared-memory blackboard for the
sovereign fleet. C/C++ with a POSIX `/dev/shm` bus, AVX2 SIMD, no external deps.

> **Status convention.** Every claim below is marked **VERIFIED `<date>`** (measured
> this tree, reproducible via `make check`) or **UNVERIFIED** (carried from an earlier
> doc, not independently confirmed). Do not trust UNVERIFIED rows without checking.
> This convention exists because four defects survived for weeks behind green tests.

---

## What it is

A blackboard where 10 lanes (4 control heads + 6 workers) coordinate through a shared
hypervector matrix rather than message passing. Each hypervector is 10,240 ternary
dimensions `{-1, 0, +1}` stored as two bitplanes.

```
/dev/shm/vsa_matrix_bus   1000 HV x 2560 B = 2.56 MB   (the blackboard)
/dev/shm/vsa_matrix_seq   1000 x 8 B       = 8 KB      (seqlock version counters)
```

## Layout

| file | role |
|---|---|
| `vsa_kernel.{c,h}` | bind / dot / bundle / seqlock. The only place VSA math lives. |
| `shm_allocator.cpp` | creates both segments; validates its argument |
| `vsa_selftest.c` | the regression gate — `make check` |
| `vsa_bus_client.py` | Python bus client (write/read under seqlock). **Packs sign plane then zero plane** = `vsa_hv_t`. Copy lives at `ai-stack/bin/vsa_bus_client.py` (keep identical) |
| `nomic_vsa_bridge.py` | text → Nomic Embed 1.5 (768-D) → ±1 projection to 10,240-D → ternary (γ·σ deadband) → bus slot. `--backend ollama` (default: `nomic-embed-text` F16 GGUF on :11434, no torch) · `hf` (safetensors, unsloth venv) · `--no-publish` |
| `vsa_xlang_test.c` + `xlang_test.py` | **cross-language layout gate** — Python writes known HVs to a private segment, C reports +1/−1/active/cross-dot, must equal numpy; includes a mutation check (old packing must fail). `make xlang-check` |
| `nomic_vsa_test.cpp` | reads a published slot; self-dot == active is a kernel self-consistency check only |
| `vsa_trust.{c,h}` · `vsa_bus_rw.{c,h}` · `vsa_slot_registry.{c,h}` · `vsa_bundle_bitsliced.{c,h}` · `vsa_phase.{c,h}` · `vsa_phase_int8.c` · `vsa_resonator_min.{c,h}` | sealed trust result type · single-writer bus reader/writer + slot registry · bit-sliced majority bundle · K-level phase kernel + int8 pshufb cleanup · minimal resonator (see `TECHNICAL-DISCLOSURE.md`) |
| `docs/` | `SEALED-METRIC.md` (pre-registered metric), `SLOT-REGISTRY.md` |

## Build & verify

```bash
make          # kernel, allocator, selftest, lanes, bridge
make check        # ZEROES the live bus then self-tests; refuses if the bus holds data (FORCE=1 to override)
make xlang-check  # Python<->C layout gate on a private shm segment (safe any time)
```

## Verified behaviour — VERIFIED 2026-09-24

| property | measurement |
|---|---|
| dot product vs scalar reference | exact, 200/200 random vectors |
| bundling capacity | **100% member recall at k=100**, noise floor ~133 |
| bind latency (10,240-D, AVX2) | ~35 ns |
| dot latency (10,240-D, AVX2) | ~93 ns |
| full 1000-HV blackboard scan | 145 µs SIMD · 240 µs plain scalar C |
| torn reads under concurrency | 0 in 200,000 reads |

## Python ↔ C layout — VERIFIED 2026-09-25
| property | measurement |
|---|---|
| layout | `vsa_hv_t = {sign[160], zero[160]}`; Python client fixed 2026-09-25 (it packed zero first → C decoded every Python HV plane-swapped: only the −1s survived, all read as −1) |
| cross-language gate | `make xlang-check` PASS: +1 4644 / −1 1451 / active 6095 / cross-dot 85 identical in numpy and C; old packing detected |
| Nomic slot 102 | C reads 6297 active = 3142 (+1) / 3155 (−1), 38.5% zero — matches Python |
| HF vs GGUF embeddings | cosine 1.0000 on 4 test texts; ternary keeps similarity order (related 0.584 vs unrelated 0.275 cos, float 0.689 vs 0.317) |
| UNVERIFIED | "sub-microsecond" read of a published slot (never timed); semantic search / routing over the bus (not built) |

**Slot ranges:** control heads 1–100 / 101–200, worker lane 401–500; dispatcherd status publishes to 301. There is no slot registry yet — slot 102 (Nomic test) sits inside control_head_2's range.

## Design decisions worth knowing

**Stride is 2560 B/HV (two bitplanes).** An earlier kernel used 1280 B (one plane,
*binary*) while every lane process assumed 2560. Each lane therefore addressed twice
its intended hypervector. It went unnoticed because every test wrote to byte offset 0 —
the one address where a 2× stride error is invisible. `vsa_selftest` now stamps all ten
lanes at their real offsets.

**Bundling is majority, not OR.** Bitwise OR saturates: member recall collapses to 0%
at k=7 and scores go *negative*. Sum-then-threshold holds 100% recall at k=100. A
regression guard in the self-test fails if this is ever reverted.

**Reads never block on a partial write.** A 2560-byte write is not atomic. Writers
bracket with `vsa_write_begin/end`; readers retry on `vsa_read_retry`. Counters live in
a separate segment so every existing lane byte offset is unchanged.

**The kernel uses unaligned loads.** Dereferencing `__m256i*` emits aligned `vmovdqa`,
but `malloc` only guarantees 16 bytes on x86-64 — that combination segfaults at `-O2`
while ASan at `-O0` hides it. Correctness must not depend on caller alignment.

## What this is NOT — VERIFIED 2026-09-24

Claims in `docs/` that did not survive measurement:

- **"Sub-nanosecond binding."** Real figure is ~35 ns — off by ~35×. The AVX-512
  benchmark in `docs/VSA Matrices Proof.txt` **does not compile** (`hv_a.sign ^= ...`
  applies `^=` to a `uint64_t[160]`), and this host has AVX2 only.
- **"67M ops/sec / 14.9 ns per op."** That run's checksum decodes to exactly
  −5120 × 10,000,000 — every iteration returned the same degenerate value.
- **"Gate 1 secured."** Not by that benchmark. Capacity — the property that actually
  decides viability — was untested until `vsa_selftest` existed.
- **"Zero-token telemetry."** An LLM consumes embedding slots, not hypervectors.
  The viable form is a projection head feeding ~1–4 slots (a modality, like vision),
  not the elimination of tokens.

## Honest scope

VSA is **not** faster than the alternative at this scale: a full blackboard scan is
145 µs SIMD vs 240 µs in plain scalar C, ~1% of one core either way, and the fleet's
live intent daemon already does cosine similarity over MiniLM embeddings. Speed is not
a reason to use this.

Its real niche is **diagnosis and verification**: anomaly detection by Hamming distance
against a bundle of known-good states, failure isolation by unbinding a role, and
scoring an already-enumerated candidate set against a safety codebook. It retrieves and
verifies; it does not generate. Authoritative state stays in SQLite, where
`RAISE(FAIL)` constraints can be enforced — a hypervector space cannot express those.

## Licence and provenance

PolyForm Small Business 1.0.0 or PolyForm Noncommercial 1.0.0, your choice (`LICENSING.md`); data and shards: `DATA-AND-SHARDS.md`, `SHARD-FORMAT.md`. Cited prior art, patent design-around and checks: `PROVENANCE.md`. How it works, in full: `TECHNICAL-DISCLOSURE.md`.
