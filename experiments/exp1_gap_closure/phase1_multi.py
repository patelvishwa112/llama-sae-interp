#!/usr/bin/env python3
"""Phase 1 for multiple layers: model-only MLP activation capture.
Saves activations for specified layers in one forward pass.

Usage: python phase1_multi.py LAYER1 LAYER2 ..."""

import sys, json, time
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx_lm import load
from config import MODEL_PATH, QUESTIONS_PATH, OUTPUT_DIR

target_layers = [int(a) for a in sys.argv[1:]]
print(f"Phase 1: Capture activations for layers {target_layers}", flush=True)

model, tokenizer = load(MODEL_PATH)
questions = []
with open(QUESTIONS_PATH) as f:
    for line in f:
        questions.append(json.loads(line.strip()))

# Store activations: {layer: [activations]}
activations = {l: [] for l in target_layers}
t0 = time.time()

for i, q in enumerate(questions[:50]):
    prompt = f"{q['question']} Answer:"
    tokens = mx.array(tokenizer.encode(prompt))[None, :]
    h = model.model.embed_tokens(tokens)
    for l, layer in enumerate(model.model.layers):
        r = layer.self_attn(layer.input_layernorm(h)); h = h + r
        m = layer.mlp(layer.post_attention_layernorm(h))
        if l in target_layers:
            mlp_out = np.array(m[0, -1, :].tolist(), dtype=np.float32)
            activations[l].append(mlp_out)
        h = h + m
        if l % 4 == 0:
            mx.eval(h)

    if (i + 1) % 25 == 0:
        print(f"  {i+1}/50 ({time.time()-t0:.0f}s)", flush=True)

for l in target_layers:
    arr = np.array(activations[l])
    out = OUTPUT_DIR / f"activations_L{l}.npz"
    np.savez_compressed(out, activations=arr)
    print(f"  L{l}: {arr.shape} → {out} ({arr.nbytes/1024:.0f}KB)", flush=True)

print(f"  ✓ Done in {time.time()-t0:.0f}s", flush=True)
