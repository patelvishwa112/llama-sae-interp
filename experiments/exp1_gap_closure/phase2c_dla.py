#!/usr/bin/env python3
"""Phase 2c: DLA-based steering direction computation.
Scores SAE features by W_dec[f] · embedding[target_token] instead of activation magnitude.
This directly measures how much amplifying feature f pushes logits toward the correct answer.

Usage: python phase2c_dla.py LAYER_IDX"""

import sys, json, time
from pathlib import Path
import numpy as np
from safetensors import safe_open
from mlx_lm import load
from config import MODEL_PATH, SAE_PATH, QUESTIONS_PATH, OUTPUT_DIR
TOP_K = 5

layer_idx = int(sys.argv[1])
print(f"Phase 2c: DLA steering for L{layer_idx}", flush=True)

# Load tokenizer + embedding weights (not full model)
print("  Loading tokenizer + embeddings...", flush=True)
model, tokenizer = load(MODEL_PATH)
embed_weight = np.array(model.model.embed_tokens.weight.tolist(), dtype=np.float32)  # [vocab, d_model]
d_model = embed_weight.shape[1]
del model  # Free model memory
print(f"  Embeddings: {embed_weight.shape}", flush=True)

# Load features
features_file = OUTPUT_DIR / f"features_L{layer_idx}.npz"
features = np.load(features_file)["features"]
n_q, n_latents = features.shape
print(f"  Features: {n_q} questions × {n_latents:,} latents", flush=True)

# Load SAE decoder
t0 = time.time()
layer_dir = SAE_PATH / f"layers.{layer_idx}.mlp" / "sae.safetensors"
with safe_open(str(layer_dir), framework="np") as f:
    W_dec = f.get_tensor("W_dec")  # [n_latents, d_model]
print(f"  W_dec: {W_dec.shape} ({time.time()-t0:.1f}s)", flush=True)

# Load questions for answer tokens
with open(QUESTIONS_PATH) as f:
    questions = [json.loads(line.strip()) for line in f]

# Compute DLA-based steering per question
steering = np.zeros((n_q, d_model), dtype=np.float32)
dla_top_features = []

for i in range(n_q):
    q = questions[i]
    answer = q["answer"]
    
    # Tokenize the answer to get its embedding
    answer_ids = tokenizer.encode(answer)
    # Skip BOS token (128000), use first actual content token
    target_id = answer_ids[1] if len(answer_ids) > 1 and answer_ids[0] == 128000 else answer_ids[0]
    target_emb = embed_weight[target_id]  # [d_model]
    
    # DLA score for every feature: dot product of decoder vector with target embedding
    # Score(f) = W_dec[f] · target_emb
    dla_scores = W_dec @ target_emb  # [n_latents] — matrix-vector product
    dla_scores = np.abs(dla_scores)  # magnitude matters, sign is direction
    
    # Top-K features by DLA score
    top_idx = np.argsort(dla_scores)[-TOP_K:]
    
    # Steering direction: sum of feature_value × decoder_vector for top features
    feats = features[i]
    steer = np.zeros(d_model, dtype=np.float32)
    for fid in top_idx:
        steer += W_dec[fid] * feats[fid]
    norm = np.linalg.norm(steer)
    if norm > 1e-8:
        steer = steer / norm
    steering[i] = steer
    
    dla_top_features.append({
        "answer": answer,
        "target_token": tokenizer.decode([target_id]),
        "top_dla_features": top_idx.tolist(),
        "top_dla_scores": dla_scores[top_idx].tolist(),
    })

# Save steering
out = OUTPUT_DIR / f"steering_dla_L{layer_idx}.npz"
np.savez_compressed(out, steering=steering)
print(f"  Saved {steering.nbytes/1024:.0f}KB → {out}", flush=True)

# Show DLA vs activation comparison for first question
print(f"\n  First question analysis:")
print(f"    Answer: {dla_top_features[0]['answer']}")
print(f"    Target token: '{dla_top_features[0]['target_token']}'")
print(f"    Top DLA features: {dla_top_features[0]['top_dla_features'][:5]}")
print(f"    DLA scores: {[f'{s:.2f}' for s in dla_top_features[0]['top_dla_scores']]}")

# Save analysis
analysis_file = OUTPUT_DIR / f"dla_analysis_L{layer_idx}.json"
with open(analysis_file, "w") as f:
    json.dump({"layer": layer_idx, "dla_features": dla_top_features[:5]}, f, indent=2)
print(f"  Analysis → {analysis_file}", flush=True)
print(f"  ✓ Done", flush=True)
