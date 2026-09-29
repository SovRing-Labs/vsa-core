# SEALED METRIC — RSPK build-order step 2 (pre-registration)

**Sealed before `vsa_trust_test` was compiled or run.** Written first, on purpose.
If a number below had been chosen after seeing results it would not be a threshold,
it would be a fit. Every constant here is derived from an analytic argument or
measured elsewhere in this repo by another harness. No constant in this file is
derived from this step's own test output.

Parent lineage: RSPK CONVERGE build-order row 2 · XP1 CONVERGE amendment 4
("collapse / barycenter detection is a first-class output of step 2").
Patent shape held from RSPK: **return scores + a convergence class; never bipolarize.**
Every operation below is the existing ternary kernel on the existing two-plane layout.

---

## 1. The result type

3 bits on the confidence register, per the build order:

| bits | field | values |
|---|---|---|
| `[2:1]` | convergence class | `0` single-pattern · `1` metastable · `2` barycenter · `3` invalid |
| `[0]` | barycenter flag | `0` not collapsed to the global average · `1` collapsed |

`vsa_trust_pack3()` returns `(class << 1) | flag` — exactly 3 bits, one `uint8`. The flag
is the class at 1-bit resolution: a consumer that only needs to know "did this collapse?"
reads bit 0 and never decodes the class. (See §4b — the flag was once derived from a
`coherence` threshold; measurement rejected that and the correction is recorded there.)

The result struct also carries `load_ratio_bucket` (3 bits, 0–7). **It is not on the
register packet**, because it is a deterministic function of `(D_f, N)` which the reader
already holds; the register spends its 3 bits on the two quantities that cannot be
recomputed without re-scanning the memory. (Spec tension flagged in the STEP-2 report.)

Two further fields are part of the struct but not the packet:

- **`integrity`** — the write-side integrity channel XP1 asks for. It is **1 when the
  barycenter flag disagrees with the class**, i.e. two independent tests of the same
  physical question reached different answers. It is live, not decorative: the flag is
  computed from a statistic the class decision never looks at (§4).
- **scores** — `sim_barycenter`, `sim_best`, `sim_second`, `coherence`. Required by the
  patent design-around: return scores, return a class, never a hard decision alone.

## 2. Similarity

    sim(a,b) = vsa_dot(a,b) / sqrt(active(a) * active(b))

The true cosine of the dense `{-1,0,+1}` vectors — `vsa_dot` already returns that
inner product, and `sqrt(na*nb)` is its exact norm product. Defined as `0.0` when
either vector has no active dimension.

## 3. Null noise — derived, and cross-checked against an existing measurement

Two independent HVs at `p_active = 1/2` (the convention in `vsa_selftest.c:32`, `mkhv`):

- co-active dims: `n_ab = p^2 * D = D/4 = 2560`
- `vsa_dot` = sum of 2560 iid `±1` ⇒ `Var = D/4`, `sd(dot) = sqrt(D)/2 = 50.60`
- normalised: `Var(sim) = (D/4)/(D/2)^2 = 1/D` ⇒ **`sd(sim) = 1/sqrt(D) = 0.0098821`**

*Cross-check, independent of this file:* `sqrt(D)/2 * 2 = 101.2`, and
`vsa_kernel.h:42` states "two independent random HVs score ~0 +/- 100". The derivation
reproduces a number the kernel's own header already asserts. `vsa_selftest.c:65-70`
separately asserts that random pairs produce *varying* scores, not a constant.

**D = 10,240 is fixed by the bus ABI** (`vsa_kernel.h:27`, `VSA_HV_BYTES = 2560`), so
these constants are properties of the slot, not of this test.

## 4. The two decisions

### 4a. Class — similarity margin

    m = sim(query, barycenter)   s1 = top-1 sim over stored patterns   s2 = top-2

    if      (m - s1  >  M)  ->  BARYCENTER
    else if (s1 - s2  >  M)  ->  SINGLE
    else                     ->  METASTABLE

**Where the signal comes from (no constants involved).** For a *pattern* fixed point the
query equals one stored pattern: `s1 = 1`, `s2 ≈ 0`, and the barycenter is an average of
many near-orthogonal patterns so `m ≈ 1/sqrt(N)` — the margin is about `-1`. For a
*barycenter* fixed point the signs invert: `m = 1`, `s1 ≈ 1/sqrt(N)`, margin about `+1`.
For a *k-subset average* (metastable) the query is equidistant from all `k` members, so
`s1 ≈ s2 ≈ 1/sqrt(k)` and the margin is about `0`. The analytic crossing point is `0` in
both cases; the classifier is therefore a sign test, not a fitted boundary.

**Why a non-zero margin anyway.** Under the metastable hypothesis `s1` and `s2` are equal
*in expectation*, so a bare `> 0` test is a coin flip — the noise must be excluded
before the test. `sd(s1 - s2) = sqrt(2) * sd(sim) = sqrt(2/D) = 0.013975`.

**Margin derivation.** The kill criterion demands > 99% specificity, which bounds the
per-decision error at 1% ⇒ one-sided `z = 2.326`. We pre-register **`z = 4`** (per-trial
`3.17e-5`, three orders of headroom) because the error being bounded is a **missed
collapse** — the exact failure mode this step exists to catch (ADV-TOP: "the output
appears stable and converged, masking the underlying error"). So:

    M = 4 * sqrt(2/D) = 0.055900

Actual signal is ≈ 0.97, i.e. **17x the margin**, so the decision is insensitive to this
choice. The test therefore also prints the whole ROC over `z` — the decision point is
theory, the curve is disclosure, so nobody can claim the point was cherry-picked.

**Precedence.** Barycenter is tested first, making the single/metastable branch
unreachable for a genuine collapse. The more specific hypothesis wins.

## 4b. Coherence — reported, not decided on (CORRECTED 2026-09-27, after a KILL)

> **This section supersedes the original §4b, which was wrong. The correction was
> forced by measurement, and the failing run is preserved below rather than edited away.**

    phi = |{query-active dims whose sign equals the consensus sign over all N}|
           / |{query-active dims}|

The first pre-registration used this as a **second, independent opinion** on the
collapse: the flag was set by `phi >= 0.5 + z/(2*sqrt(nq))`, and `integrity` reported
flag/class disagreement. The first gate run **KILLED on that** — specificity 0.0000%,
integrity firing on all 3,000 negatives. Two distinct causes, one shared root.

### Cause 1 — the analytic null mean was wrong

The original threshold assumed `E[phi] = 0.5`. Measured: **0.5179**, so a threshold of
0.528 sat *below* the null and fired on everything.

The corrected derivation. `T` = sum of the other `N-1` ternary votes at one dim. `T` is
**lattice-symmetric**, not a continuous normal, and that is what the first version got
wrong. With `var(T) = (N-1)/2` and `k = 1/sqrt(pi*(N-1))`:

- consensus decided unless `T` is exactly 0 ⇒ `p_dec = 1 - k`
- the query's single vote is outvoted only when `|T| > 1`, which symmetry splits evenly ⇒
  `P(sign equal | co-active) = 1/2 + k`
- `phi_null = (1-k)(1/2+k)`

At `N = 256`: derived **0.516417**, measured **0.51792** — agreement to 0.0013. The
decomposition was verified directly rather than argued: measured `p_dec` 0.96490 x
measured `P(sign eq|co)` 0.53677 = 0.51793, against measured `phi` 0.51792. The
identity is exact and the closed form is right.

### Cause 2 — coherence is a continuum in `k`, not a separable cut

With the corrected threshold, the class was already 100% on 4,000 trials, but the flag
still failed: specificity 38.6%. The subset-size sweep says why:

| `k` | 1 | 4 | 8 | 16 | 32 | 64 | 128 | 200 | 250 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|
| `phi` | 0.554 | 0.542 | 0.555 | 0.579 | 0.614 | 0.673 | 0.753 | 0.847 | 0.950 | **1.000** |
| class | meta | meta | meta | meta | meta | **bary** | bary | bary | bary | bary |

`phi` rises smoothly and monotonically with `k` and **has no gap anywhere in it**. A
`k`-subset average genuinely is more consensus-like than a single pattern, so `phi`
cannot separate "metastable" from "barycenter" at any fixed cut: the metastable panels
sit at 0.552 / 0.558, only ~8 sd below `k=32` at 0.614, and the cut that cleared them
would have been a number chosen by looking at the measurements — exactly the fit this
document exists to forbid. Meanwhile `gap_bary` changes **sign** at the same place
(−0.030 at `k=32` → +0.176 at `k=64`): a real boundary, not an arbitrary one.

### The correction

- The **shipped flag is the class at 1-bit resolution** — a consumer wanting only
  "did it collapse?" reads bit 0 and never decodes the class. This is also what the
  build order literally specified: class (2 bits) + barycenter flag (1 bit).
- `integrity` no longer reports flag/class disagreement. It reports **ambiguity**: the
  gap that governed the class lies within one noise unit of its threshold,
  `|governing − M| <= sigma_gap`. Derived from sealed constants, no new value. It
  answers the ADV-TOP worry more usefully than a second opinion did — not "does another
  test agree?" but "here is exactly where this test must not be trusted."
- `coherence` stays in the published scores so a caller can see how far from the
  consensus the query sits, and the `vsa_trust_phi_null()` / `vsa_trust_phi_threshold()`
  helpers are retained — they are the correct analytic null for a **single-member** query
  and are what let a reader place their own query type on the `phi` scale.

**What did not change:** `M = 4*sqrt(2/D) = 0.055902`, `z = 4`, the class rule, the
similarity definitions, the load ladder, the stimulus parameters, and the gate
definitions. The class decision and its threshold are untouched from the first
pre-registration; what measurement rejected was a *redundant second opinion* that was
never part of the scope or the kill criterion. The kill criterion is evaluated on the
decision the reader actually ships.

### The failing run, preserved

First run, verbatim headline, before any change:

    G1 specificity  = 0/3000 = 0.0000%   (false positives 3000; >99% permits <= 30)
    G2 sensitivity  = 1000/1000 = 100.0000%
    G3 3-way class  = 3000/4000 = 75.0000%
    G4 integrity channel fired 3000 times in 4000 trials (want 0)
    KILL: G1 barycenter not separable at >99% specificity: 0.0000%

### A confound found and disclosed, not hidden

That first run used a member of the near-equal **pair** as the single-pattern query, so
the memory held a near-duplicate of it. One confound, both symptoms: the twin pinned
`sim_second` just below `sim_best` (`gap_solo` 0.025 instead of 0.97) *and* double-weighted
that vote in the consensus, inflating `phi` from 0.518 to 0.552. Correcting the
*stimulus* (the query is now an independent member; the pair stays in the memory to
create R2's metastable state) is experimental design, not threshold tuning — and the
confounded case is **kept in the gate output as disclosure panel D1**, because it is a
real and useful result: a near-duplicate in the memory produces a **confidently wrong**
class, and the ambiguity bit does not catch it. See the STEP-2 report.

## 5. Load-ratio bucket — `r = D_f / N`, 3 bits

`D_f` = active (decided, non-stasis) dimensions of the barycenter = `vsa_active_count`;
`N` = stored patterns. Geometric ladder, ratio 4 per step:

| bucket | range of `r = D_f/N` |
|---|---|
| 0 | `r >= 1536` |
| 1 | `384 <= r < 1536` |
| 2 | `96 <= r < 384` |
| 3 | `24 <= r < 96` |
| 4 | `6 <= r < 24` |
| 5 | `1.5 <= r < 6` |
| 6 | `0 < r < 1.5` |
| 7 | `r <= 0` (degenerate; reserved) |

**Why geometric, not linear.** Associative-memory capacity grows as `D / log2(D)`, so
the load count `N` spans decades across practical `D`. Linear buckets would put
essentially all real cases in one bucket. The ladder steps by 4, so a 2x change in load
is half a bucket.

**Where the edges are anchored — two constants measured elsewhere, by other harnesses:**

- `k* = 100`, 100% member recall, noise floor 133 — `README.md:54`, marked
  **VERIFIED 2026-09-24**, produced by `vsa_selftest.c:72` `test_capacity()`.
  At `D_f ≈ 6100` that is `r ≈ 61` ⇒ **bucket 3**.
- `k* = 8` — XP1 CONVERGE amendment 2, a measured fleet constant from the cost law,
  not from this test. That is `r ≈ 763` ⇒ **bucket 1**.

Check: `r(k*=8) / r(k*=100) = 12.5`, and the ladder assigns those two anchors 2 buckets
apart (`4^2 = 16`), consistent to within one ladder step. Neither anchor was produced by
this step's test, and the test reports where each `N` actually lands rather than tuning
the edges to make the table tidy.

## 6. Stimulus parameters — pre-registered, not thresholds

Pre-registered so the *stimulus* cannot be tuned either:

| parameter | value | why this value |
|---|---|---|
| `D` | 10,240 | the bus ABI; not a choice |
| `p_active` | 1/2 | matches `vsa_selftest.c:32`; the uncorrelated null. Live data runs *denser* (Nomic slot 102 = 0.595 active, xlang = 0.595), so the null is the conservative choice |
| seed | `0x5EED5EED0002` | fixed; xoshiro256\*\* from splitmix64, no libc RNG |
| trials | 1,000 per panel | the scope's number |
| memory `N` | 256 per trial (resampled) | costs one 256-pattern bundle per trial; hard-mode confirmation at `N = 1000` is run separately and reported |
| `rho` | 0.10 primary; sweep 0.05 / 0.20 | see below |

`rho` is the fraction of a pattern's *active* dims that are resampled when making the
near-equal partner. It is a **stimulus**, not a threshold — the decision boundary is `z`
(§4a), not `rho`. The primary panel reports `rho = 0.10`; the full sweep is reported
beside it so the reader can see where the case degrades.

**Uncorrelated synthetic patterns, stated as a limitation.** These test the null: it
proves separability against *noise*, and says nothing about correlated real corpora.
The corpus gate is the honest home for that (XP1 reopened it); this step cannot settle it.

## 7. Panels and the decision rule

Every panel: 1,000 independent trials, each trial resamples the whole memory.

| panel | memory | query | ground truth |
|---|---|---|---|
| R1 single | `N = 256` | one stored pattern | SINGLE |
| R2 metastable pair | `N = 256` | bundle of 2 near-equal patterns | METASTABLE |
| R2b metastable subset | `N = 256` | bundle of 8 stored patterns | METASTABLE |
| R3 barycenter | `N = 256` | bundle of **all** 256 | BARYCENTER |

**R2/R2b cannot be run with a 2-pattern memory, and that is a finding, not a workaround.**
If the memory *is* the near-equal pair, the 2-subset average **is** the global average —
the same point in space. "Metastable" and "barycenter" are then undecidable by
definition. A metastable state only exists relative to a memory it is a proper subset
of, so the memory must be larger than the subset. R2b (subset of 8 out of 256) is the
hard case the ADV cell named: same memory as the barycenter panel, so the *only*
difference between METASTABLE and BARYCENTER is whether the query is the average of all
patterns or of a few.

Also run, and reported: the subset-size boundary sweep (where does the class flip?), the
load-ratio sweep (the §5 table, measured), the `z` ROC, and the 48-combination
pack round-trip.

### Gates

**G1 — the kill criterion, verbatim: "Barycenter not separable at > 99% specificity ⇒ no
reader ships."** Counting rule pre-registered: negatives are the 3,000 non-barycenter
primary trials; strictly greater than 0.99 permits at most 29 false positives of 3,000.

**G2 — added, and strictly harder than G1:** sensitivity > 0.99 over the 1,000 positives.
G1 is one-sided, and a detector that never fires scores 100% specificity while being
useless. This guard exists only to close that degenerate pass; it can never rescue a
failure of G1.

**G3–G7 — supporting gates**, all pre-registered here: 3-way class accuracy ≥ 0.99
against a 25.0% majority-class rate; `integrity == 0` on all 4,000 primary trials (no
decision lands in the ambiguity band — see §4b); 48/48 pack round-trip;
`vsa_trust_score()` identical to `barycenter() + score_bary()` on all 4,000 (the two API
paths must not disagree); load bucket self-consistent and monotone in `N`.

**Disclosure panel D1** (the confounded twin, §4b) is reported every run and is
**excluded from the kill accounting** — it is a measured property of near-duplicate
memories, not one of the four pre-registered regimes.

**Majority-class rate is reported beside every accuracy**, as the build order requires:
75.0% for the binary barycenter predicate (3,000/4,000) and 25.0% for the 3-way class
(1,000/4,000). An accuracy is worthless without them.

### What a KILL would mean

Hitting a gate is a valid result and is reported honestly as `KILL: <reason>`. G1
failing means the reader does not ship and the resonator has no collapse detector, so
step 7's accuracy numbers cannot be trusted to distinguish converged-but-wrong from
converged-and-right — which is the whole point of the build order.

---

## AMENDMENT A (2026-09-28) — the cleanup similarity is now the int8 pshufb metric

**Added after step 5 re-brief; approved by J on 2026-09-28.** Recorded here because
Track A micromodels must not train against an undeclared metric.

Sections 1–4 above are UNCHANGED and remain sealed: the 3-bit result type
(`vsa_trust_pack3`), the convergence class, the barycenter flag, and every
pre-registered threshold. **This amendment does not alter the trust gate.**

What changed is the **similarity score on the `cleanup` path** (step 5):

| | before | now (ADOPTED) |
|---|---|---|
| implementation | float scalar / AVX2 gather, `cos_diff_table[16]` as floats | int8 AVX2 `_mm256_shuffle_epi8` into the same 16-entry table, quantised to `cos*127+127`, accumulated with `_mm256_sad_epu8` |
| cost | 9.67 µs/codevector (KILL vs 1.25 budget) | **0.535 µs/codevector** (2.4× under budget) |
| scalar↔AVX2 agreement | bit-for-bit (float) | **bit-for-bit (int8)** — 3000/3000 |
| agreement with the float score | n/a | **99.3%** of argmax decisions (2980/3000) |

**The honest statement of the metric:** the int8 score is
`Σ round(127·cos(2π·((q−c) mod 16)/16))` accumulated over D=5,120 dims, i.e. each cosine
term quantised to steps of **1/127**. It is therefore **a different measure** from the
float mean-cos, not a faster evaluation of it. Concretely:

- Ties are broken differently. The quantised score is an integer, so near-ties that
  float would separate collapse — which is precisely the 0.7% of cases where the argmax
  differs. **`argmax` is stable to 99.3%, and that 0.7% is a known, accepted, declared cost.**
- Any threshold quoted against "mean-cos" is **not** transferable to this score. Numbers
  must be re-derived, not converted.
- `vsa_trust` thresholds in §2–§4 are unaffected: they are computed on the float trust
  path, not on `cleanup`.

**Why 1M-context nemotron and 4-bit packing were not used:** the kernel is 1 byte/dim
here (5,120 B/vector); 4-bit packing would be faster again but was not measured, so it
is not claimed.

**Verification that must be re-run for any future change** (step 5 was verified twice, by
the peer review and by the Commander):
```
gcc -O3 -mavx2 -o /tmp/s5i vsa_phase_int8_test.c vsa_phase_int8.c vsa_phase.c -lm && /tmp/s5i
# must print INT8-CONFORM-OK, int8_scalar==int8_avx2 3000/3000, and < 1.25 us/codevector
```
