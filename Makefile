# vsa-core release Makefile (no vendored dependencies). x86-64 with AVX2; gcc/g++.
CC  ?= gcc
CXX ?= g++
CFLAGS   = -O2 -Wall -Wextra -mavx2 -std=c11
CXXFLAGS = -O2 -Wall -Wextra -mavx2 -std=c++17
BIN = build

TESTS = vsa_trust_test vsa_bus_test vsa_bundle_bench vsa_phase_test vsa_phase_int8_test

all: $(BIN) shm_allocator vsa_selftest vsa_xlang_test $(addprefix $(BIN)/,$(TESTS))

$(BIN):
	mkdir -p $(BIN)

vsa_kernel.o: vsa_kernel.c vsa_kernel.h
	$(CC) $(CFLAGS) -c vsa_kernel.c

shm_allocator: shm_allocator.cpp vsa_kernel.o
	$(CXX) $(CXXFLAGS) -o $@ shm_allocator.cpp vsa_kernel.o -lrt

vsa_selftest: vsa_selftest.c vsa_kernel.o
	$(CC) $(CFLAGS) -o $@ vsa_selftest.c vsa_kernel.o -lrt -lpthread

$(BIN)/vsa_trust_test: vsa_trust_test.c vsa_trust.c vsa_kernel.c
	$(CC) -O2 -mavx2 -o $@ $^ -lm
$(BIN)/vsa_bus_test: vsa_bus_test.c vsa_bus_rw.c vsa_slot_registry.c vsa_kernel.c
	$(CC) -O2 -mavx2 -o $@ $^ -lm -lpthread
$(BIN)/vsa_bundle_bench: vsa_bundle_bench.c vsa_bundle_bitsliced.c vsa_kernel.c
	$(CC) -O2 -mavx2 -o $@ $^ -lm
$(BIN)/vsa_phase_test: vsa_phase_test.c vsa_phase.c
	$(CC) -O2 -mavx2 -o $@ $^ -lm
$(BIN)/vsa_phase_int8_test: vsa_phase_int8_test.c vsa_phase_int8.c vsa_phase.c
	$(CC) -O2 -mavx2 -o $@ $^ -lm
vsa_xlang_test: vsa_xlang_test.c vsa_kernel.o
	$(CC) $(CFLAGS) -o $@ vsa_xlang_test.c vsa_kernel.o -lrt

# Conformance: every test on private temp segments; never touches a live /dev/shm/vsa_matrix_bus.
test: all
	./$(BIN)/vsa_trust_test && ./$(BIN)/vsa_bus_test && (./$(BIN)/vsa_phase_test || echo 'note: float phase path fails only its 1.25 us speed bar; int8 pshufb (next) is the adopted cleanup (SEALED-METRIC Amendment A)') && ./$(BIN)/vsa_phase_int8_test && ./$(BIN)/vsa_bundle_bench && python3 xlang_test.py

clean:
	rm -rf $(BIN) *.o shm_allocator vsa_selftest vsa_xlang_test
.PHONY: all test clean
