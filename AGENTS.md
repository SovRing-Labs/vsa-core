# AGENTS.md — vsa-core

For coding assistants (and people) setting this repository up. Plain steps; every command runs from the repo root.

**Status: pre-alpha, under development.** Interfaces, numbers and file layout can change without notice. Read `STATUS.md` first: what works today (each line names its test and evidence file), what is not yet tested, the known failures and the corrections.

## What it is
Ternary vector-symbolic kernel (10,240-dim, two bitplanes) and a POSIX shared-memory blackboard with a seqlock, AVX2 SIMD, phase kernel and trust classifier; C/C++ with a Python client.

## Install and check (3 commands)
```bash
git clone --depth 1 https://github.com/SovRing-Labs/vsa-core && cd vsa-core
python3 -m venv .venv && . .venv/bin/activate && python -m pip install -q numpy
make -s && python3 xlang_test.py   # expect: PASS
```
Needs: gcc, make, python3>=3.10, numpy, cpu:avx2.
Whole suite instead: `bash install.sh --no-gui` (python check → venv → every runnable `suite.json` test, headless; without `--no-gui` it then opens the local results page on 127.0.0.1).
Measured: 41.5s on a fresh clone (Intel i5-8365U laptop, 2026-10-16, exit 0).
Note: the quick test builds everything and runs the Python-C layout gate on a private shared-memory segment; never run `make check` on a machine with a live /dev/shm/vsa_matrix_bus (it zeroes it).
Full suite: `make test`.

## What the tests mean
`suite.json` lists every claim this repository makes with its verify command, the expected output, what passing means, how the number was obtained and how to use it in your own work (6 tests, in `suite.json` at the repository root). `STATUS.md` says what works today, what is not yet tested and the known failures; the umbrella's `STATUS.md` has one row per repository.

## Where the evidence is
- `DOSSIER.md` — history, metrics and limits
- `TECHNICAL-DISCLOSURE.md` — the method, as dated prior art
- `CITATION.cff` — DOI 10.5281/zenodo.23282124

## Honest limits (from DOSSIER.md)
"Sub-nanosecond binding" was false (~35 ns). "67M ops/s" came from a degenerate benchmark. "Zero-token telemetry" is not how language models consume vectors — a projection head feeding a few slots is. Not built in 0.1.0: semantic search or routing over the bus; timing of a single published-slot read. The bus is single-machine shared memory; multi-machine use goes through omni-ring shards (Speculative at scale).

## Licence
PolyForm Small Business 1.0.0 OR PolyForm Noncommercial 1.0.0 (free trial for evaluation). Details: `LICENSING.md`.

## Guided setup through MCP
The SovRing umbrella ships an MCP server that explains, checks readiness, clones and runs these tests: https://github.com/SovRing-Labs/SovRing (README → "Paste this to your assistant"). Zero-install docs endpoint: https://gitmcp.io/SovRing-Labs/vsa-core
