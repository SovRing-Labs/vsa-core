# vsa-core — Technical Disclosure

**Title:** A ternary hyperdimensional shared-memory blackboard for multi-agent coordination on commodity CPUs: two-bitplane
10,240-D vectors in 2,560-byte slots, seqlock single-writer bus with a slot registry and prefix summaries, a sealed trust
result type, bit-sliced majority bundling, and a 16-level phase kernel whose cleanup is a byte-shuffle table lookup.
**First public disclosure:** the date of the first public commit of this file. **Code licence:** PolyForm Small Business
1.0.0 or PolyForm Noncommercial 1.0.0, user's choice (`LICENSING.md`). **Intent:** published so that everything described here is prior art; the authors claim no patent on it.
Companion: the OMNIRING disclosure (coprime residue rings, check ring, cued resonator, peeling, mark store) builds on this bus.

## 1. Vector format
* A hypervector has `D = 10,240` ternary dimensions `{−1, 0, +1}` stored as two bitplanes of 160 × 64-bit words each:
  `sign` (1 = negative) and `active` (1 = non-zero). Total 2,560 bytes = one slot.
* **Bind** = elementwise ternary product: `sign = sign_a XOR sign_b`, `active = active_a AND active_b`.
* **Dot** = `popcount(active_a & active_b & ~(sign_a ^ sign_b)) − popcount(active_a & active_b & (sign_a ^ sign_b))`,
  computed with AVX2 over the 160 words (≈ 93 ns); bind ≈ 35 ns.
* The same 2,560 bytes can be reinterpreted as 5,120 phases at 16 levels (4 bits each); see section 5.

## 2. Shared-memory blackboard
* Two POSIX shared-memory segments: the bus (1,000 slots × 2,560 B = 2.56 MB, small enough to stay in L3 cache, full scan
  145 µs with SIMD) and a sequence array (1,000 × 8 B).
* **Seqlock single writer per slot:** the writer increments the slot's counter to odd, writes, increments to even; a reader
  retries if the counter is odd or changed during its copy. Measured: 0 torn reads in 200,000 concurrent reads.
* **Slot registry:** ranges of slots are owned by named lanes (control heads, workers, status); a second writer claiming a
  slot is refused (return code −2). **Prefix summaries** let a reader skip regions without scanning every slot.
* A layout gate writes known vectors from Python into a private segment and checks the C reader bit-for-bit, including a
  mutation check that the old (wrong) plane order fails.

## 3. Sealed trust result
Every retrieval returns one small value instead of a bare answer: a **convergence class** (2 bits), a **barycenter flag**
(1 bit: "did this collapse to the global average?", readable without decoding the class), an **integrity** bit (set when
the deciding gap sits within one noise unit of its threshold, i.e. the class could flip), a **load-ratio bucket**, and the
raw **scores**. Packed form: `(class << 1) | flag` in one byte. All thresholds are pre-registered and derived from null
noise (see `docs/SEALED-METRIC.md`); none are fitted to test output. The design returns scores plus a class, never only a
hard decision. Measured: 100% specificity at 100% sensitivity for barycenter collapse over 4,000 trials (majority class
75%); 0 API-path disagreements in 4,000.

## 4. Bit-sliced majority bundle
To bundle `n` ternary vectors, keep for each 64-bit word a small binary counter per sign stored as bit-planes (7 planes
each for positive and negative counts). Adding a vector is a carry-save addition done with bitwise operations on all 64
lanes at once (`planes[i] ^= carry; carry = old & carry`, …); the majority is read from the top planes. Exact against a
scalar reference for n = 3, 8, 16, 64 (1,000/1,000 each), 6–9× faster per vector.

## 5. Phase kernel and table-lookup cleanup
* Phase vectors: 5,120 dimensions at `K = 16` (also `K = 8`) levels. Bind = add phases mod `K`; unbind = subtract;
  bundle = integer accumulation of unit phasors from a `K`-entry cos/sin table; similarity = mean cosine of the phase
  difference. Scalar and AVX2 paths agree bit-for-bit (3,000 trials); bind/unbind round-trips are exact.
* **Cleanup by byte shuffle:** because a phase difference is a 4-bit index, the cosine of the difference is a 16-entry table.
  Store one phase per byte; per 32 dimensions: subtract, AND 15, `vpshufb` into a table of `round(127·cos + 127)`, and
  accumulate with `vpsadbw`. Argmax over a 1,000-entry codebook costs ≈ 0.525 µs per codevector on a laptop CPU (the
  float path: ≈ 8 µs), with 200/200 argmax agreement against float on noisy queries. The integer score is cosine quantised
  to 1/127 per term; it is the adopted similarity metric.

## 6. Dense-to-ternary projection
A dense embedding (e.g. 768-D) is projected to 10,240-D with a fixed seeded ±1 matrix and ternarised with a deadband
(`|value| < γ·σ` → 0), so text can be placed on the bus and compared by popcount.

## 7. Minimal resonator and experiments
`vsa_resonator_min` factorises a bound phase vector over several codebooks (alternating unbind + cleanup). An asymmetry
experiment (`vsa_asym_test.c`) tested whether breaking symmetry (asymmetry control, a frozen mask, iterated function
system maps) reduces limit cycles; no arm reached the pre-registered 2× bar (largest 1.90×), and that negative result is
disclosed here as well.

## 8. Variations also disclosed
Any dimension and word size; binary, bipolar, ternary or multi-level phases; any `K` with table-lookup cleanup (`vpshufb`,
`tbl`, gather, or scalar tables) on AVX2, AVX-512, NEON, SVE, GPU shared memory or FPGA LUTs; any number of slots, segment
layouts and single-writer schemes (seqlock, epoch, RCU); per-slot owner registries; any counter width for bit-sliced
bundling; any trust encoding that returns scores with a class and an ambiguity bit; projection from any embedding model.
