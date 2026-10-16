> **Status: under development — pre-alpha.** Interfaces, numbers and file layout can change without notice.

# vsa-core — status

Last checked: 2026-10-16, Linux laptop (Intel i5-8365U, 8 threads), on a clean copy of this repo. 6 of 6 suite tests work today; 6 of 6 can run without special hardware. Each line below names its test id in `suite.json` and the evidence file it rests on.
Quick test `make -s && python3 xlang_test.py`: PASS (`XLANG-TEST PASS`, fresh copy, empty HOME).


## Works today

- **V1** 5-bit stochastic ternary learner converges on a noisy target (C++) — PASS 2026-10-16 (`stochastic OK accuracy@1000=99.96% target_active=5128`); evidence: `vsa_stochastic_test.cpp`, `vsa_stochastic.h`, `vsa_kernel.h`, `DOSSIER.md`.
- **V2** Python <-> C hypervector layout gate with a mutation check — PASS 2026-10-16 (`XLANG-TEST PASS`); evidence: `xlang_test.py`, `vsa_xlang_test.c`, `vsa_bus_client.py`, `README.md`, `DOSSIER.md`.
- **V3** Bus reader/writer: zero torn reads under concurrent seqlock access — PASS 2026-10-16 (`PASS: Zero torn reads in 17018 reads`); evidence: `vsa_bus_test.c`, `vsa_bus_rw.c`, `vsa_slot_registry.c`, `DOSSIER.md`, `docs/SLOT-REGISTRY.md`.
- **V4** Bit-sliced majority bundle equals the scalar bundle on all trials — PASS 2026-10-16 (`PASS — bit-sliced bundle matches scalar on all trials.`); evidence: `vsa_bundle_bench.c`, `vsa_bundle_bitsliced.c`, `vsa_bundle_bitsliced.h`, `vsa_kernel.c`.
- **V5** Phase kernel with int8 pshufb clean-up agrees with the float path — PASS 2026-10-16 (`INT8-CONFORM-OK`); evidence: `vsa_phase_int8_test.c`, `vsa_phase_int8.c`, `vsa_phase.c`, `docs/SEALED-METRIC.md`, `DOSSIER.md`.
- **V6** Trust kernel: barycenter collapse detected at 100% specificity and sensitivity — PASS 2026-10-16 (`PASS`); evidence: `vsa_trust_test.c`, `vsa_trust.c`, `vsa_trust.h`, `docs/SEALED-METRIC.md`, `DOSSIER.md`.

## Not yet tested

- Every suite row ran on the test laptop.

## Known failures and honest negatives

- The README names `make check` and `make xlang-check`; the Makefile has only `all`, `test`, `clean`. The quick test is `make -s && python3 xlang_test.py`; `make test` ran past 10 minutes on a loaded laptop.
- `nomic_vsa_test` and `ipc_speed_test` are compile-only in `tests/run_tests.sh`: running them writes to the live shared-memory bus `/dev/shm/vsa_matrix_bus`.

## Corrections

- No corrections touch this repo.

## How to re-check

- One file, any python3 >= 3.8: download [`sovring-suite.pyz`](https://github.com/SovRing-Labs/SovRing/releases/latest/download/sovring-suite.pyz), then `python3 sovring-suite.pyz --run vsa-core` (headless) or `python3 sovring-suite.pyz` (local page on 127.0.0.1).
- From this checkout: `bash install.sh` (checks python, prepares a venv, runs the suite headless, then opens the page; `--no-gui` to skip the page).
- Quick test only: `D=$(mktemp -d); trap 'rm -rf $D' EXIT; gcc -O2 -mavx2 -std=gnu11 -c vsa_kernel.c -o $D/k.o && g++ -O2 -mavx2 -std=c++17 -o $D/s vsa_stochastic_test.cpp $D/k.o && $D/s > $D/o.log && acc=$(awk -F'\t' '/^1000\t/{gsub(/%/,"",$NF);print $NF}' $D/o.log) && awk -v a=$acc 'BEGIN{exit !(a>=99.0)}' && echo "stochastic OK accuracy@1000=${acc}% target_active=$(grep -o 'active dimensions = [0-9]*' $D/o.log | grep -o '[0-9]*$')"`.

## Defensive publication

- `TECHNICAL-DISCLOSURE.md` — on public GitHub since 2026-09-29
- DOI: [10.5281/zenodo.23282124](https://doi.org/10.5281/zenodo.23282124) (Zenodo, see `CITATION.cff`)
- Licence: PolyForm Small Business 1.0.0 / PolyForm Noncommercial 1.0.0 (LICENSING.md); PolyForm Free Trial for evaluation
