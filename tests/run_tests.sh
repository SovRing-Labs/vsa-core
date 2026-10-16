#!/usr/bin/env bash
# vsa-core test runner. Builds every test from the repo-root sources into a temp dir and runs it.
# Usage:  bash tests/run_tests.sh            (from the repo root; if cwd has no vsa_kernel.c it uses the parent of this folder)
#         VSA_ROOT=/path/to/vsa-core bash run_tests.sh
#         VSA_SKIP_SLOW=1 bash tests/run_tests.sh   (skips vsa_trust_test, the slowest)
# Needs: gcc, g++ (C++17), python3 + numpy, x86-64 with AVX2.
# Never touches the live bus /dev/shm/vsa_matrix_bus: tests use private segments (vsa_test_*_<pid>, vsa_xlang_*).
# vsa_selftest, nomic_vsa_test and ipc_speed_test are NOT run: they open the live bus (selftest zeroes it, ipc_speed_test writes
# over it). nomic_vsa_test and ipc_speed_test are compiled, so a build break is still caught.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${VSA_ROOT:-}" ]; then ROOT="$VSA_ROOT"; elif [ -f "$PWD/vsa_kernel.c" ]; then ROOT="$PWD"; else ROOT="$(cd "$HERE/.." && pwd)"; fi
cd "$ROOT" || exit 2
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
CFLAGS="-O2 -mavx2 -std=gnu11"; CXXFLAGS="-O2 -mavx2 -std=c++17"
export PYTHONDONTWRITEBYTECODE=1
FAILS=0
report() { if [ "$2" = 0 ]; then echo "PASS $1"; else echo "FAIL $1 ($3)"; FAILS=$((FAILS+1)); fi; }

# vsa_kernel.c uses _Thread_local, which g++ does not accept: build it as C once and link the objects into the C++ tests.
gcc $CFLAGS -c vsa_kernel.c -o "$OUT/vsa_kernel.o" || { echo "FAIL build vsa_kernel.c"; exit 1; }

# 1. stochastic training (C++): the 5-bit learner must reach >= 99% match with the target within 1000 noisy steps.
if g++ $CXXFLAGS -o "$OUT/stoch" vsa_stochastic_test.cpp "$OUT/vsa_kernel.o"; then
  "$OUT/stoch" > "$OUT/stoch.log" 2>&1; rc=$?
  acc=$(awk -F'\t' '/^1000\t/{gsub(/%/,"",$NF); print $NF}' "$OUT/stoch.log")
  ok=1; [ "$rc" = 0 ] && [ -n "$acc" ] && awk -v a="$acc" 'BEGIN{exit !(a>=99.0)}' && ok=0
  report vsa_stochastic_test "$ok" "exit=$rc accuracy@1000='${acc:-none}'"
else report vsa_stochastic_test 1 "compile error"; fi

# 2. compile-only: these two open the live bus when run.
for t in nomic_vsa_test ipc_speed_test; do
  g++ $CXXFLAGS -o "$OUT/$t" "$t.cpp" "$OUT/vsa_kernel.o" -lrt; report "$t (compile only)" $? "compile error"
done

# 3. Python <-> C layout gate on a private segment (with a mutation check).
mkdir "$OUT/x" && cp xlang_test.py vsa_bus_client.py vsa_xlang_test.c vsa_kernel.c vsa_kernel.h "$OUT/x/"
if gcc $CFLAGS -o "$OUT/x/vsa_xlang_test" "$OUT/x/vsa_xlang_test.c" "$OUT/x/vsa_kernel.c" -lrt; then
  (cd "$OUT/x" && python3 xlang_test.py) > "$OUT/xlang.log" 2>&1; rc=$?
  grep -q "XLANG-TEST PASS" "$OUT/xlang.log" && [ "$rc" = 0 ]; report xlang_test.py $? "see xlang_test.py output"
else report xlang_test.py 1 "compile error"; fi

# 4. C tests listed in the Makefile `test` target (private temp segments). Each prints its own verdict line.
run_c() {  # name, expected-marker, sources..., -- libs
  name="$1"; marker="$2"; shift 2
  if gcc $CFLAGS -o "$OUT/$name" "$@" > "$OUT/$name.build" 2>&1; then
    "$OUT/$name" > "$OUT/$name.log" 2>&1; rc=$?
    grep -q "$marker" "$OUT/$name.log" && [ "$rc" = 0 ]; report "$name" $? "exit=$rc, marker '$marker' not found"
  else report "$name" 1 "compile error"; fi
}
run_c vsa_bus_test "Zero torn reads" vsa_bus_test.c vsa_bus_rw.c vsa_slot_registry.c vsa_kernel.c -lm -lpthread
run_c vsa_bundle_bench "bit-sliced bundle matches scalar" vsa_bundle_bench.c vsa_bundle_bitsliced.c vsa_kernel.c -lm
run_c vsa_phase_int8_test "INT8-CONFORM-OK" vsa_phase_int8_test.c vsa_phase_int8.c vsa_phase.c -lm
if [ "${VSA_SKIP_SLOW:-0}" = 1 ]; then echo "SKIP vsa_trust_test (VSA_SKIP_SLOW=1)"; else
  run_c vsa_trust_test "^PASS" vsa_trust_test.c vsa_trust.c vsa_kernel.c -lm
fi
# vsa_phase_test is not gated: its float path misses only the 1.25 us speed bar on purpose (SEALED-METRIC Amendment A; the int8 path above is the adopted one).

echo "----"; [ "$FAILS" = 0 ] && echo "ALL PASS" || echo "$FAILS FAILED"
exit $((FAILS > 0))
