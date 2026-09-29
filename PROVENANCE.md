# vsa-core — Provenance

**Licence:** PolyForm Small Business 1.0.0 **or** PolyForm Noncommercial 1.0.0, at the user's choice (`LICENSING.md`). **Disclosure:** published deliberately as prior art; the authors
claim and file no patent on it (decision 2026-09-29). Full description: `TECHNICAL-DISCLOSURE.md`.

## How this code was written
Clean-room: written from published ideas and our own measurements, not from anyone's source code. No third-party source is
included; the vendored ONNX Runtime and the ONNX bridge of the development tree are **not** part of this release. C11/C++17
standard libraries, POSIX shared memory and x86 AVX2 intrinsics only; Python helpers use NumPy (BSD-3-Clause). The optional
`nomic_vsa_bridge.py` calls a locally served Nomic Embed v1.5 model (Apache-2.0) at run time; no model weights are shipped.

## Ideas this code builds on (cited, not copied)
| Idea | Source |
|---|---|
| Vector symbolic architectures / hyperdimensional computing (bind, bundle, cleanup) | Plate (holographic reduced representations); Kanerva (hyperdimensional computing); Gayler (VSA) |
| Resonator networks | Frady, Kent, Olshausen, Sommer, Neural Computation 2020 (arXiv:2007.03748); Kent et al. arXiv:1906.11684 |
| Quantised phase (FHRR-style) vectors | Plate (FHRR); later quantised-phase work (qFHRR) |
| Bit-sliced / carry-save counting | classical bit-slicing and carry-save adder techniques |
| Seqlock single-writer concurrency | Linux kernel seqlock pattern |
| Random projection of dense embeddings to high-dimensional ±1 codes | Johnson–Lindenstrauss style random projection |

## Patent design-around
Built to avoid the claims reviewed in the fleet's clean-room plan (IBM US 12,561,553 in-memory crossbar resonator;
US 12,306,870 N resonators with permuted-and-bundled inputs; US 12,518,150 share-based bundling; US 12,579,411 reviewed
separately): the trust type returns scores plus a class, never a hard bipolarised decision alone. A freedom-to-operate
opinion is still required before any commercial licence fee is charged.

## Checks run (2026-09-29)
| Check | Result |
|---|---|
| ScanCode 32.5.0 (`scancode -clpi`) | see `release-scan.txt` |
| gitleaks 8.30.1 (`--no-git`) | see `release-scan.txt` |
| Absolute home paths | stripped; the model path is `NOMIC_MODEL_PATH` |
| `make test` | trust 4,000/4,000 · bus 0 torn reads, second writer refused · phase scalar = AVX2 (3,000) · int8 cleanup conform (K = 8, 16) · bit-sliced bundle 1,000/1,000 per n, 6–9× faster than scalar · Python↔C layout gate PASS. The float phase path misses only its 1.25 µs speed bar; the int8 path is the adopted cleanup |
