#!/usr/bin/env python3
"""Autoregressive SAE Feature Steering — Anthropic-style.
Clamps a target SAE feature at every generation step, not just the last prompt token.

This is the positive control: can we make the model talk about dogs
by clamping a dog-DLA feature during autoregressive generation?

Usage: python steer_autoregressive.py"""

import sys, time
from pathlib import Path
import mlx.core as mx
import numpy as np
from safetensors import safe_open
from mlx_lm import load
from config import MODEL_PATH, SAE_PATH

# Configuration
LAYER = 10
FEATURE_IDX = 7494  # Top DLA feature for "dog" (0.178 score)
DOG_NATURAL = 0.394  # Natural activation on "My favorite animal is a"
CLAMP_MULTIPLIER = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
MAX_TOKENS = 30

PROMPTS = [
    "Write a sentence about France.",
    "Tell me about your day.",
    "What is the weather like?",
]

print(f"Autoregressive Steering: L{LAYER} feature {FEATURE_IDX}", flush=True)
print(f"  Clamp: {DOG_NATURAL * CLAMP_MULTIPLIER:.2f} ({CLAMP_MULTIPLIER:.0f}x natural dog-context value)")
print(f"  Max tokens: {MAX_TOKENS}", flush=True)

# Load model
print("Loading model...", flush=True)
model, tokenizer = load(MODEL_PATH)

# Load SAE
print("Loading SAE...", flush=True)
t0 = time.time()
layer_dir = SAE_PATH / f"layers.{LAYER}.mlp" / "sae.safetensors"
with safe_open(str(layer_dir), framework="np") as f:
    W_enc = mx.array(f.get_tensor("encoder.weight").T)
    W_dec = mx.array(f.get_tensor("W_dec"))
    b_enc = mx.array(f.get_tensor("encoder.bias"))
    b_dec = mx.array(f.get_tensor("b_dec"))
print(f"  SAE ready ({time.time()-t0:.1f}s)", flush=True)

CLAMP_VAL = DOG_NATURAL * CLAMP_MULTIPLIER

def generate_with_steering(prompt, steer=False):
    """Generate tokens with optional SAE feature clamping at every step."""
    ids = tokenizer.encode(prompt)
    generated = []
    
    for step in range(MAX_TOKENS):
        tokens = mx.array(ids)[None, :]
        h = model.model.embed_tokens(tokens)
        
        for l, layer in enumerate(model.model.layers):
            r = layer.self_attn(layer.input_layernorm(h))
            h = h + r
            m = layer.mlp(layer.post_attention_layernorm(h))
            
            if steer and l == LAYER:
                # Apply SAE clamp at ALL positions (not just last)
                mlp_out = m[0, :, :]  # [seq_len, d_model]
                centered = mlp_out - b_dec  # [seq_len, d_model]
                latents = centered @ W_enc + b_enc  # [seq_len, n_latents]
                
                # Clamp the feature at every position
                latents[:, FEATURE_IDX] = mx.array(CLAMP_VAL, dtype=latents.dtype)
                
                # Decode back
                reconstructed = latents @ W_dec + b_dec  # [seq_len, d_model]
                steer_delta = reconstructed - mlp_out  # [seq_len, d_model]
                m = m + steer_delta
            
            h = h + m
            if l % 4 == 0:
                mx.eval(h)
        
        # Get next token
        logits = model.model.embed_tokens.as_linear(model.model.norm(h))
        next_id = int(mx.argmax(logits[0, -1, :]).item())
        
        # Stop at EOS
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
    baseline = generate_with_steering(prompt, steer=False)
    print(f"Baseline ({time.time()-t0:.0f}s): {baseline}")
    
    t0 = time.time()
    steered = generate_with_steering(prompt, steer=True)
    print(f"Steered ({time.time()-t0:.0f}s): {steered}")
    
    if baseline != steered:
        print(f"  ⚡ OUTPUT CHANGED!")
    else:
        print(f"  (identical)")
