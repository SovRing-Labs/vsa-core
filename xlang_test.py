#!/usr/bin/env python3
"""Cross-language HV layout test (2026-09-25).
Python (vsa_bus_client) writes two known ternary HVs into a PRIVATE shm segment; C (vsa_xlang_test)
reads them with the kernel's vsa_hv_t and reports pos/neg/active/dot. Must match numpy exactly.
Mutation check: the pre-fix zero-first packing must be DETECTED (test fails on it), proving the
test can fail — unlike the old self-dot 'cohesion' check, which passes for any bytes.
Usage: python3 xlang_test.py   (needs `make vsa_xlang_test`; never touches /dev/shm/vsa_matrix_bus)"""
import json, os, struct, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vsa_bus_client import VSABusClient

BUS, SEQ = "/vsa_xlang_bus", "/vsa_xlang_seq"
HERE = os.path.dirname(os.path.abspath(__file__))

def pack(hv):
    z, s = [], []
    for w in range(160):
        zw = sw = 0
        for bit in range(64):
            v = hv[w * 64 + bit]
            if v:
                zw |= 1 << bit
                if v < 0:
                    sw |= 1 << bit
        z.append(zw); s.append(sw)
    return z, s

def c_view(slot_a, slot_b):
    out = subprocess.run([os.path.join(HERE, "vsa_xlang_test"), BUS, SEQ, str(slot_a), str(slot_b)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)

def main():
    for name, size in ((BUS, 1000 * 2560), (SEQ, 1000 * 8)):
        with open("/dev/shm" + name, "wb") as f:
            f.truncate(size)
    try:
        rng = np.random.default_rng(7)
        # asymmetric on purpose: ~45% +1, ~15% -1, rest 0 -> a plane swap cannot preserve the counts
        a = rng.choice([1, -1, 0], size=10240, p=[0.45, 0.15, 0.40]).astype(np.int8)
        b = rng.choice([1, -1, 0], size=10240, p=[0.30, 0.30, 0.40]).astype(np.int8)
        exp = {"a_pos": int((a == 1).sum()), "a_neg": int((a == -1).sum()), "a_active": int((a != 0).sum()),
               "b_pos": int((b == 1).sum()), "b_neg": int((b == -1).sum()),
               "dot_ab": int(a.astype(np.int32) @ b.astype(np.int32)), "sizeof_hv": 2560}
        cli = VSABusClient(shm_name="/dev/shm" + BUS, seq_name="/dev/shm" + SEQ)
        for slot, hv in ((10, a), (11, b)):
            assert cli.write_hv(slot, *pack(hv))
        # Python round trip
        rz, rs = cli.read_hv(10)
        assert (list(rz), list(rs)) == pack(a), "python read_hv round trip failed"
        got = c_view(10, 11)
        ok = got == exp
        print("expected:", exp); print("C sees:  ", got)
        # mutation: old zero-first packing must be caught
        z, s = pack(a)
        cli.shm_map[12 * 2560:13 * 2560] = struct.pack("<160Q160Q", *z, *s)
        caught = c_view(12, 11) != {**exp}
        print("mutation (old zero-first packing) detected:", caught)
        cli.close()
        ok = ok and caught
        print("XLANG-TEST", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        for name in (BUS, SEQ):
            try:
                os.unlink("/dev/shm" + name)
            except OSError:
                pass

if __name__ == "__main__":
    sys.exit(main())
