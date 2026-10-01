# vsa-core — Dossier

*History, specification, metrics, tests and limits. Companion to `TECHNICAL-DISCLOSURE.md` (what is disclosed) and
`README.md` (how to build). Guide to the whole stack: "Hyperdimensional Computing and You" in the OMNIRING research
corpus.*

## What it is

A ternary hyperdimensional blackboard: one block of shared memory (`/dev/shm`) holding 1,000 hypervectors of 10,240
ternary dimensions, which many processes read and write safely without message passing. In the stack it is the
**bus** — the substrate omni-ring's memory and the fleet's agents share. Stage (essay 07): primitive.

## Creation history

| Date (2026) | Milestone |
|---|---|
| Aug–Sep | Early "VSA matrix" kernel and design notes in the author's research folder; claims of sub-nanosecond binding and 67M ops/s later failed measurement (below) |
| 09-24 | VSA-MATRIX review panel decision; the kernel measured for the first time: bind ~35 ns, dot ~93 ns, 0 torn reads in 200,000 concurrent reads, bundling 100% recall at k = 100 |
| 09-24 | Four defects found that had survived behind green tests: a 1,280 vs 2,560-byte stride split (every lane addressed twice its intended vector — invisible at offset 0), OR-bundling that collapsed recall at k = 7, aligned loads that segfaulted at -O2, and an AVX-512 benchmark that did not compile |
| 09-25 | Python/C plane-order bug fixed (Python packed the zero plane first, so C read every Python vector plane-swapped); a real cross-language gate (`make xlang-check`) added with a mutation check; nomic → ternary bridge verified (HF vs GGUF cosine 1.0000) |
| 09-26 | Library hazard removed: `random.seed(text)` in the bus client reseeded the global RNG; replaced by a private generator, determinism re-verified |
| 09-27 | Bit-sliced majority bundle (carry-save counters across 64 dims at once) built as RSPK step 4; sealed trust result; slot registry; phase kernel with byte-shuffle clean-up |
| 09-29 | Release export prepared: ScanCode + gitleaks clean |
| 09-30 | Published: github.com/SovRing-Labs/vsa-core v0.1.0 (Software Heritage save succeeded) |

## Specification (summary; full text in the disclosure)

* **Vector:** 10,240 ternary dims as two bitplanes (`sign`, `active`) of 160 × 64-bit words = 2,560 bytes = one slot.
  Layout `vsa_hv_t = {sign[160], zero[160]}` — identical in C and Python (gated by `make xlang-check`).
* **Bus:** `/dev/shm/vsa_matrix_bus` (1,000 × 2,560 B) + `/dev/shm/vsa_matrix_seq` (1,000 × 8 B seqlock counters).
  Single writer per slot brackets writes with begin/end; readers retry if the counter is odd or changed.
* **Slot registry + prefix summaries:** writers claim slot ranges; a foreign claim is refused (−2); summaries let
  readers skip regions.
* **Trust result:** every retrieval returns scores plus a convergence class, a barycenter flag and an ambiguity bit —
  never a bare answer.
* **Bundle:** majority by bit-sliced carry-save counting (never OR).
* **Phase kernel:** 5,120 dims at K = 16; bind = add mod K; clean-up by `vpshufb` into a cos table, accumulated with
  `vpsadbw`.

## Metrics (measured on an Intel i5-8365U, AVX2)

| Property | Result | Date |
|---|---|---|
| Dot vs scalar reference | exact, 200/200 | 09-24 |
| Bundling capacity | 100% member recall at k = 100 (noise floor ~133) | 09-24 |
| Bind / dot (10,240-D) | ~35 ns / ~93 ns | 09-24 |
| Full 1,000-vector scan | 145 µs SIMD · 240 µs scalar | 09-24 |
| Torn reads under concurrency | 0 / 200,000 | 09-24 |
| Cross-language layout | +1 4,644 / −1 1,451 / active 6,095 / cross-dot 85 identical in numpy and C | 09-25 |
| Barycenter collapse detection | 100% specificity at 100% sensitivity, 4,000 trials | 09-27 |
| Phase argmax, 1,000-entry codebook | ≈ 0.525 µs per codevector (float path ≈ 8 µs); 200/200 agreement with float | 09-27 |

## Tests

* `make check` — self-test (zeroes the live bus; refuses if it holds data unless `FORCE=1`). Stamps all ten lanes at
  their real offsets, guards majority bundling, checks dot exactness and seqlock behaviour.
* `make xlang-check` — Python ↔ C layout gate on a private segment, with a mutation check (safe any time).
* Component tests: `vsa_trust_test`, `vsa_phase_test`, `vsa_phase_int8_test`, `vsa_bus_test`, `vsa_stochastic_test`.

## Limits and corrections

* "Sub-nanosecond binding" was false (~35 ns). "67M ops/s" came from a degenerate benchmark. "Zero-token telemetry"
  is not how language models consume vectors — a projection head feeding a few slots is.
* Not built in 0.1.0: semantic search or routing over the bus; timing of a single published-slot read.
* The bus is single-machine shared memory; multi-machine use goes through omni-ring shards (Speculative at scale).

## Lineage and credits

Kanerva (hyperdimensional computing), Plate (HRR), Frady et al. (resonators), BitNet (ternary), Nomic AI (embedding
model used by the bridge). Full list: `PROVENANCE.md`. Updates: weekly, recorded here with dates.
