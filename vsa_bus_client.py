# ┌─ CORE FILE · governed by control-center/core-changelog.jsonl ──────────
# │ ⚠  LOAD-BEARING. 1 logged change(s). Read the ledger before editing:
# │     bin/core-changelog show bin/vsa_bus_client.py
# │ Last: 2026-09-26T18:43Z · claude-seat · 7bc55eb2d · FIX-VSA-BUS-CLIENT-0926 closed: private-RNG fix moved into ~/vsa_...
# └─────────────────────────────────────────────
import mmap
import os
import struct
import random

SHM_NAME = "/dev/shm/vsa_matrix_bus"  # os.open needs the tmpfs path; "/vsa_matrix_bus" is only the shm_open() name
SHM_SEQ_NAME = "/dev/shm/vsa_matrix_seq"
HV_BYTES = 2560
VSA_WORDS = 160

class VSABusClient:
    def __init__(self, num_vectors=1000, shm_name=None, seq_name=None):
        # shm_name/seq_name: override for tests (a private segment); default = the live bus
        self.shm_name = shm_name or SHM_NAME
        self.seq_name = seq_name or SHM_SEQ_NAME
        self.num_vectors = num_vectors
        self.shm_size = num_vectors * HV_BYTES
        self.seq_size = num_vectors * 8
        
        # 1. Map the VSA shared data bus
        try:
            self.shm_fd = os.open(self.shm_name, os.O_RDWR)
            self.shm_map = mmap.mmap(self.shm_fd, self.shm_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        except Exception as e:
            print(f"VSABusClient Warning: Failed to map data bus {self.shm_name}. Error: {e}")
            self.shm_map = None
            
        # 2. Map the seqlock version array
        try:
            self.seq_fd = os.open(self.seq_name, os.O_RDWR)
            self.seq_map = mmap.mmap(self.seq_fd, self.seq_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        except Exception as e:
            print(f"VSABusClient Warning: Failed to map seqlock segment {self.seq_name}. Error: {e}")
            self.seq_map = None

    def close(self):
        if self.shm_map:
            self.shm_map.close()
            os.close(self.shm_fd)
        if self.seq_map:
            self.seq_map.close()
            os.close(self.seq_fd)

    def write_hv(self, index, zero_words, sign_words):
        """
        Write a 2560-byte ternary hypervector to the bus under a seqlock barrier.
        zero_words: list/tuple of 160 uint64
        sign_words: list/tuple of 160 uint64
        """
        if not self.shm_map or not self.seq_map:
            return False
            
        if index < 0 or index >= self.num_vectors:
            raise ValueError(f"Index {index} out of range [0, {self.num_vectors-1}]")
            
        # 1. Build binary packet (2560 bytes)
        # SIGN plane first, then ZERO (active) plane — the canonical layout is vsa_kernel.h
        # `typedef struct { uint64_t sign[160]; uint64_t zero[160]; } vsa_hv_t`. (2026-09-25 fix:
        # this used to pack zero first, so every C reader decoded Python-written HVs plane-swapped.)
        packet = struct.pack("<160Q160Q", *sign_words, *zero_words)
        
        offset = index * HV_BYTES
        seq_offset = index * 8
        
        # 2. Acquire seqlock (increment by 1, making it odd to signal writers are active)
        current_seq = struct.unpack("<Q", self.seq_map[seq_offset:seq_offset+8])[0]
        self.seq_map[seq_offset:seq_offset+8] = struct.pack("<Q", current_seq + 1)
        
        # Memory barrier / synchronization (mmap handles sharing across threads/processes automatically)
        
        # 3. Write data
        self.shm_map[offset:offset+HV_BYTES] = packet
        
        # 4. Release seqlock (increment by 1, making it even to signal complete)
        self.seq_map[seq_offset:seq_offset+8] = struct.pack("<Q", current_seq + 2)
        return True

    def read_hv(self, index):
        """
        Read a 2560-byte ternary hypervector from the bus under seqlock torn-read protection.
        Returns (zero_words, sign_words) as lists of 160 uint64s each.
        """
        if not self.shm_map or not self.seq_map:
            return None
            
        if index < 0 or index >= self.num_vectors:
            raise ValueError(f"Index {index} out of range [0, {self.num_vectors-1}]")
            
        offset = index * HV_BYTES
        seq_offset = index * 8
        
        # Retry loop to protect against torn reads (seqlock)
        while True:
            # Read start sequence number (must be even)
            s0 = struct.unpack("<Q", self.seq_map[seq_offset:seq_offset+8])[0]
            if s0 % 2 != 0:
                # Writer is currently modifying; sleep a tiny bit and retry
                continue
                
            # Read the hypervector data
            packet = self.shm_map[offset:offset+HV_BYTES]
            
            # Read end sequence number; must match start number
            s1 = struct.unpack("<Q", self.seq_map[seq_offset:seq_offset+8])[0]
            if s0 == s1:
                # Success! No concurrent write occurred
                unpacked = struct.unpack("<160Q160Q", packet)  # sign plane, then zero plane
                return unpacked[160:], unpacked[:160]  # API unchanged: (zero_words, sign_words)

    def publish_text_status(self, index, text):
        """
        Encode raw text status into a VSA ternary hypervector and publish it to the bus.
        Each character is hashed into the vector space using our deterministic generator.
        """
        # Simple deterministic hashing of status text into 10,240-dimensional space.
        # FIX (2026-09-26): this was random.seed(text) on the MODULE-LEVEL RNG, which
        # reseeds every other `random` consumer in the importing process. Both call
        # sites in dispatcherd are disabled (ST-7, lines ~1460/~1895), but the hazard
        # lived here in the library: any future re-enable would silently corrupt
        # unrelated code sharing the interpreter. A private Random instance is
        # byte-for-byte deterministic for the same text and touches nothing global.
        rng = random.Random(text)
        zero_words = []
        sign_words = []
        for _ in range(VSA_WORDS):
            zero_words.append(rng.getrandbits(64))
            sign_words.append(rng.getrandbits(64))
            
        return self.write_hv(index, zero_words, sign_words)
