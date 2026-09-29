#!/usr/bin/env python3
"""Nomic Embed 1.5 -> ternary 10,240-D HV -> /dev/shm/vsa_matrix_bus.
Backends (2026-09-25): `ollama` (default; the nomic-embed-text F16 GGUF already served on :11434 —
no torch needed) or `hf` (safetensors via transformers, needs the unsloth venv + einops).
Verified identical: HF vs GGUF cosine 1.0000 on test texts."""
import argparse
import os
import json
import sys
import urllib.request
import numpy as np
from vsa_bus_client import VSABusClient

MODEL_PATH = os.environ.get("NOMIC_MODEL_PATH", "models/nomic-embed-text-v1.5")
OLLAMA_URL = "http://127.0.0.1:11434/api/embed"
OLLAMA_MODEL = "nomic-embed-text"

class NomicVSABridge:
    def __init__(self, seed=42, gamma=0.5, backend="ollama", publish=True):
        self.seed = seed
        self.gamma = gamma
        self.backend = backend

        # 1. Embedding backend
        if backend == "hf":
            print("Loading Nomic Embed 1.5 (HF safetensors, CPU)...", flush=True)
            try:
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                self.model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
                self.model.eval()
            except Exception as e:
                print(f"Error loading model: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Using {OLLAMA_MODEL} (GGUF) via ollama {OLLAMA_URL}", flush=True)
            
        # 2. Generate Fixed, Deterministic Random Projection Matrix (768 -> 10240)
        print("Generating deterministic Locality-Sensitive Projection Matrix...", flush=True)
        rng = np.random.default_rng(seed)
        # Bipolar {-1, 1} random matrix preserves angular/cosine distance perfectly
        self.proj_matrix = rng.choice([-1, 1], size=(768, 10240))
        
        # 3. Connect to the live VSA shared data bus (skipped for --no-publish)
        self.vsa_bus = VSABusClient() if publish else None

    def get_float_embedding(self, text: str) -> np.ndarray:
        """Extract L2-normalized 768-D float embedding from Nomic 1.5."""
        # Nomic Embed expects search queries to have the 'search_query: ' prefix
        if not text.startswith("search_query:") and not text.startswith("search_document:"):
            text = f"search_query: {text}"

        if self.backend != "hf":
            req = urllib.request.Request(OLLAMA_URL, data=json.dumps({"model": OLLAMA_MODEL, "input": text}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                v = np.asarray(json.load(r)["embeddings"][0], dtype=np.float64)
            return v / np.linalg.norm(v)

        import torch
        import torch.nn.functional as F
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=2048)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Mean Pooling over token embeddings
        token_embeddings = outputs[0]
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        # L2 Normalization
        normalized = F.normalize(mean_pooled, p=2, dim=1)
        return normalized[0].numpy()

    def project_and_quantize(self, embedding: np.ndarray) -> np.ndarray:
        """Project 768-D embedding to 10,240-D and threshold to ternary {-1, 0, +1}."""
        # 1. Linear random projection
        hv_float = np.matmul(embedding, self.proj_matrix)
        
        # 2. Normalize to stabilize the deadband threshold
        hv_float = (hv_float - np.mean(hv_float)) / (np.std(hv_float) + 1e-8)
        
        # 3. Apply Deadband Thresholding to derive Ternary hypervector
        hv_ternary = np.zeros(10240, dtype=np.int8)
        hv_ternary[hv_float > self.gamma] = 1
        hv_ternary[hv_float < -self.gamma] = -1
        
        return hv_ternary

    def pack_to_bitplanes(self, hv_ternary: np.ndarray) -> tuple[list[int], list[int]]:
        """Pack 10,240-D ternary vector into dual-plane uint64 format (160 words each)."""
        zero_words = []
        sign_words = []
        
        for w in range(160):
            zero_word = 0
            sign_word = 0
            for b in range(64):
                idx = w * 64 + b
                val = hv_ternary[idx]
                if val != 0:
                    zero_word |= (1 << b)
                    if val < 0:
                        sign_word |= (1 << b)
            zero_words.append(zero_word)
            sign_words.append(sign_word)
            
        return zero_words, sign_words

    def process_and_publish(self, text: str, slot_index: int) -> bool:
        """Process text and publish its ternary VSA hypervector directly to the VSA bus."""
        print(f"\nProcessing text: '{text}'")
        
        # Step 1: Extract 768-D float embedding
        embedding = self.get_float_embedding(text)
        print(f"✓ Extracted Nomic float embedding (L2 Norm matches: {np.linalg.norm(embedding):.4f})")
        
        # Step 2: Project and Threshold to Ternary
        hv_ternary = self.project_and_quantize(embedding)
        active_count = np.sum(hv_ternary != 0)
        sparsity = (1.0 - (active_count / 10240.0)) * 100.0
        print(f"✓ Projected and Quantized to 10,240-D ternary hypervector")
        print(f"  Sparsity: {sparsity:.2f}% ({active_count} active dimensions, {10240 - active_count} null dimensions)")
        
        # Step 3: Pack into 2 bitplanes
        zero_words, sign_words = self.pack_to_bitplanes(hv_ternary)
        print(f"✓ Packed into dual-plane 2560-byte stride (160 uint64 words per plane)")
        
        # Step 4: Publish to /vsa_matrix_bus under seqlock control
        if self.vsa_bus is None:
            print("(--no-publish: bus not written)")
            return True
        success = self.vsa_bus.write_hv(slot_index, zero_words, sign_words)
        if success:
            print(f"✅ Successfully published Nomic concept to /vsa_matrix_bus slot index {slot_index}!")
        else:
            print("❌ Failure: Could not write to VSA Shared Memory Bus. Is /dev/shm/vsa_matrix_bus unmapped?")
        return success

def main():
    parser = argparse.ArgumentParser(description="Ternary Nomic Embed 1.5 VSA Projection and Packing Bridge")
    parser.add_argument("text", help="The text / query to process and project")
    parser.add_argument("slot", type=int, help="VSA Shared Memory Bus slot index (0-999) to write to")
    parser.add_argument("--gamma", type=float, default=0.5, help="Deadband threshold for sparsity tuning (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for projection matrix generation (default: 42)")
    parser.add_argument("--backend", choices=["ollama", "hf"], default="ollama",
                        help="ollama = nomic-embed-text GGUF on :11434 (default, no torch); hf = safetensors via transformers")
    parser.add_argument("--no-publish", action="store_true", help="compute + report only; do not write the bus")

    args = parser.parse_args()

    bridge = NomicVSABridge(seed=args.seed, gamma=args.gamma, backend=args.backend, publish=not args.no_publish)
    ok = bridge.process_and_publish(args.text, args.slot)
    if bridge.vsa_bus:
        bridge.vsa_bus.close()
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
