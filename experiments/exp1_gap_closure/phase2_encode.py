#!/usr/bin/env python3
"""Phase 2: SAE-only feature encoding from saved activations.
Loads SAE weights + activations from disk (no model in memory).

Usage: python phase2_encode.py LAYER_IDX"""

import sys, time
from pathlib import Path
import numpy as np
from safetensors import safe_open

SAE_PATH = Path("saes/sae-Llama-3.2-1B-131k")
OUTPUT_DIR = Path("experiments/exp1_gap_closure/results")
K = 32

layer_idx = int(sys.argv[1])
print(f"Phase 2: SAE encoding L{layer_idx}", flush=True)

# Load activations
act_file = OUTPUT_DIR / f"activations_L{layer_idx}.npz"
activations = np.load(act_file)["activations"]  # [n_q, d_model]
n_q, d_model = activations.shape
print(f"  Activations: {n_q} × {d_model}", flush=True)

# Load SAE
t0 = time.time()
layer_dir = SAE_PATH / f"layers.{layer_idx}.mlp" / "sae.safetensors"
with safe_open(str(layer_dir), framework="np") as f:
    W_enc = f.get_tensor("encoder.weight").T  # [d_model, n_latents]
    b_enc = f.get_tensor("encoder.bias")     # [n_latents]
    b_dec = f.get_tensor("b_dec")            # [d_model]

n_latents = W_enc.shape[1]
print(f"  SAE: {n_latents:,} latents ({time.time()-t0:.1f}s)", flush=True)

# Encode all questions
features = np.zeros((n_q, n_latents), dtype=np.float32)
t0 = time.time()

for i in range(n_q):
    mlp_out = activations[i]
    centered = mlp_out - b_dec
    latents = centered @ W_enc + b_enc  # [n_latents]
    # Top-K sparsification
    top_idx = np.argsort(np.abs(latents))[-K:]
    feats = np.zeros(n_latents, dtype=np.float32)
    feats[top_idx] = latents[top_idx]
    features[i] = feats

# Save
out = OUTPUT_DIR / f"features_L{layer_idx}.npz"
np.savez_compressed(out, features=features)
print(f"  Features: {features.shape} → {out} ({features.nbytes/1024/1024:.0f}MB) in {time.time()-t0:.0f}s", flush=True)
print(f"  ✓ Done", flush=True)
