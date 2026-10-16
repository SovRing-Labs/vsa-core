# vsa-core tests (proposed `tests/` folder)

Proposal for J to push into `vsa-core/tests/`. vsa-core currently has no `tests/` directory: its tests are `*_test.*` files at the repo root,
wired through the Makefile `test` target. This folder adds one runner. The test sources are NOT copied: `run_tests.sh` builds the root files
in place, so only `run_tests.sh` and this README are new.

## Run

```bash
bash tests/run_tests.sh              # from the repo root; prints PASS/FAIL per test, exit 1 on any failure
VSA_SKIP_SLOW=1 bash tests/run_tests.sh   # skip vsa_trust_test (about 2 minutes on an idle machine)
```

Needs gcc, g++ (C++17), python3 with numpy, and an x86-64 CPU with AVX2 (the sources use AVX2 intrinsics; no AVX-512). Everything is built
into a temp directory that is deleted at the end, so the repo stays clean. Tests use private `/dev/shm` segments and never touch the live bus.

## What each test checks

| Test | Built from (repo root) | Checks |
|---|---|---|
| `vsa_stochastic_test` (C++) | `vsa_stochastic_test.cpp`, `vsa_stochastic.h`, `vsa_kernel.c` | the 5-bit stochastic learner reaches >= 99% match with a noisy target at step 1000 (runner parses the table) |
| `nomic_vsa_test` (compile only) | `nomic_vsa_test.cpp` | builds; NOT run: it reads slot 102 of the live bus |
| `ipc_speed_test` (compile only) | `ipc_speed_test.cpp` | builds; NOT run: it writes over `/dev/shm/vsa_matrix_bus` |
| `xlang_test.py` | `xlang_test.py`, `vsa_bus_client.py`, `vsa_xlang_test.c`, `vsa_kernel.c/.h` | Python and C read the same hypervector layout; the old wrong packing is detected (mutation check) |
| `vsa_bus_test` | `vsa_bus_test.c`, `vsa_bus_rw.c`, `vsa_slot_registry.c`, `vsa_kernel.c` | zero torn reads with concurrent writer and reader on a private bus |
| `vsa_bundle_bench` | `vsa_bundle_bench.c`, `vsa_bundle_bitsliced.c`, `vsa_kernel.c` | bit-sliced bundle equals scalar on 1000 trials each at n = 3, 8, 16, 64 |
| `vsa_phase_int8_test` | `vsa_phase_int8_test.c`, `vsa_phase_int8.c`, `vsa_phase.c` | int8 scalar and AVX2 pshufb phase clean-up agree on 3,000 trials |
| `vsa_trust_test` | `vsa_trust_test.c`, `vsa_trust.c`, `vsa_kernel.c` | barycenter detection at over 99% specificity (pre-registered in `docs/SEALED-METRIC.md`) |

Not run on purpose: `vsa_selftest` (`make check`) zeroes the live bus; `vsa_phase_test` fails only its 1.25 us speed bar by design
(`docs/SEALED-METRIC.md`, Amendment A) and the int8 test is the adopted path.

Note: `vsa_kernel.c` uses `_Thread_local`, which g++ rejects, so the runner compiles it with gcc and links the object into the C++ tests.
