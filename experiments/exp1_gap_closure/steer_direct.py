#!/usr/bin/env python3
"""Direct Decoder Steering — add α · W_dec[f] to residual stream.
No SAE encode/decode. No reconstruction artifacts.
Tests features with known token associations.

Usage: python steer_direct.py"""

import time, sys
import mlx.core as mx
import numpy as np
from safetensors import safe_open
from mlx_lm import load
from config import MODEL_PATH, SAE_PATH
MAX_TOKENS = 40

# Verified features with strong DLA (>0.3)
FEATURES = {
    "enthusiasm": {"layer": 13, "feature": 69262, "scale": 20.0, "token": "love", "dla": 0.531},
    "bullet_points": {"layer": 12, "feature": 105206, "scale": 15.0, "token": "-", "dla": 0.431},
    "spanish": {"layer": 13, "feature": 92551, "scale": 20.0, "token": "Hola", "dla": 0.525},
    "refusal": {"layer": 11, "feature": 121502, "scale": 15.0, "token": "cannot", "dla": 0.375},
}

PROMPTS = [
    "Tell me a story about a dragon.",
    "What is the meaning of life?",
]

model, tokenizer = load(MODEL_PATH)

# Pre-load all SAE decoders
decs = {}
for name, cfg in FEATURES.items():
    l = cfg["layer"]
    if l not in decs:
        with safe_open(str(SAE_PATH / f"layers.{l}.mlp" / "sae.safetensors"), framework="np") as sf:
            decs[l] = mx.array(sf.get_tensor("W_dec"))  # [131072, 2048]

def generate(prompt, steer_name=None, scale=None):
    """Generate with optional direct decoder vector addition."""
    ids = tokenizer.encode(prompt)
    generated = []
    cfg = FEATURES.get(steer_name) if steer_name else None
    
    for step in range(MAX_TOKENS):
        tokens = mx.array(ids)[None, :]
        h = model.model.embed_tokens(tokens)
        
        for l, layer in enumerate(model.model.layers):
            r = layer.self_attn(layer.input_layernorm(h))
            h = h + r
            m = layer.mlp(layer.post_attention_layernorm(h))
            
            if cfg and l == cfg["layer"]:
                # Direct decoder vector addition
                vec = decs[l][cfg["feature"]]  # [d_model]
                vec = vec / (mx.linalg.norm(vec) + 1e-8)  # normalize
                m = m + vec * scale
            
            h = h + m
            if l % 8 == 0:
                mx.eval(h)
        
        logits = model.model.embed_tokens.as_linear(model.model.norm(h))
        next_id = int(mx.argmax(logits[0, -1, :]).item())
        
        if next_id == tokenizer.eos_token_id:
            break
        
        ids.append(next_id)
        generated.append(next_id)
    
    return tokenizer.decode(generated)

for prompt in PROMPTS:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    
    t0 = time.time()
    baseline = generate(prompt)
    print(f"\nBaseline ({time.time()-t0:.0f}s):\n  {baseline[:200]}")
    
    for name, cfg in FEATURES.items():
        t0 = time.time()
        steered = generate(prompt, name, cfg["scale"])
        changed = "⚡" if steered != baseline else "="
        print(f"\n{changed} {name} L{cfg['layer']} f{cfg['feature']} ({cfg['token']}, DLA={cfg['dla']}, {cfg['scale']}×) ({time.time()-t0:.0f}s):\n  {steered[:200]}")
