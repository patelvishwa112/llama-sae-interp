#!/usr/bin/env python3
"""Batch test multiple SAE features with direct decoder steering.
Tests at scales 1-4x, shows before/after for successful features."""

import mlx.core as mx, numpy as np, time
from safetensors import safe_open
from mlx_lm import load
from config import MODEL_PATH, SAE_PATH, QUESTIONS_PATH, OUTPUT_DIR
MAX_TOKENS = 60

FEATURES = [
    ("enthusiasm", 13, 109639, "great", 0.545, "What do you think about pizza?"),
    ("casual",     13, 55592,  "cool",  0.491, "What's your opinion on renewable energy?"),
    ("exclamation",13, 90138,  "!",     0.418, "Describe a sunset."),
    ("enthusiasm", 12, 28124,  "great", 0.398, "What do you think about pizza?"),
    ("compliance", 11, 29679,  "Sure",  0.376, "Can you help me with a difficult math problem?"),
]

model, tokenizer = load(MODEL_PATH)

# Pre-load all decoders
decs = {}
for _, l, f, _, _, _ in FEATURES:
    if l not in decs:
        with safe_open(str(SAE_PATH / f"layers.{l}.mlp" / "sae.safetensors"), framework="np") as sf:
            decs[l] = mx.array(sf.get_tensor("W_dec"))

print("=" * 70)
print("SAE Feature Steering — Batch Test")
print("=" * 70)

for name, layer, feat, token, dla, prompt in FEATURES:
    print(f"\n{'─'*70}")
    print(f"Feature: {name} | L{layer} f{feat} | DLA={dla} token=\"{token}\"")
    print(f"Prompt: \"{prompt}\"")
    print(f"{'─'*70}")
    
    vec = decs[layer][feat]
    vec = vec / (mx.linalg.norm(vec) + 1e-8)
    
    # Baseline
    ids = tokenizer.encode(prompt); gen = []
    for _ in range(MAX_TOKENS):
        tokens = mx.array(ids)[None, :]
        h = model.model.embed_tokens(tokens)
        for l, lyr in enumerate(model.model.layers):
            r = lyr.self_attn(lyr.input_layernorm(h)); h = h + r
            h = h + lyr.mlp(lyr.post_attention_layernorm(h))
        logits = model.model.embed_tokens.as_linear(model.model.norm(h))
        nid = int(mx.argmax(logits[0,-1,:]).item())
        if nid == tokenizer.eos_token_id: break
        ids.append(nid); gen.append(nid)
    baseline = tokenizer.decode(gen)[:250]
    print(f"\n  Before: {baseline[:150]}")
    
    # Sweep scales
    best_scale = None
    best_text = ""
    best_score = 0
    
    for scale in [1.0, 2.0, 2.5, 3.0, 4.0]:
        ids = tokenizer.encode(prompt); gen = []
        for _ in range(MAX_TOKENS):
            tokens = mx.array(ids)[None, :]
            h = model.model.embed_tokens(tokens)
            for l, lyr in enumerate(model.model.layers):
                r = lyr.self_attn(lyr.input_layernorm(h)); h = h + r
                m = lyr.mlp(lyr.post_attention_layernorm(h))
                if l == layer:
                    m = m + vec * scale
                h = h + m
            logits = model.model.embed_tokens.as_linear(model.model.norm(h))
            nid = int(mx.argmax(logits[0,-1,:]).item())
            if nid == tokenizer.eos_token_id: break
            ids.append(nid); gen.append(nid)
        text = tokenizer.decode(gen)[:250]
        
        # Simple heuristics for each behavior
        score = 0
        if name == "enthusiasm":
            score = sum(1 for w in ['love', 'great', 'amazing', 'wonderful', 'fantastic', 'excellent', '!'] if w.lower() in text.lower())
            score += text.count('!')
        elif name == "casual":
            score = sum(1 for w in ['yeah', 'cool', 'dude', 'pretty', 'kinda', 'stuff'] if w.lower() in text.lower())
        elif name == "exclamation":
            score = text.count('!')
        elif name == "compliance":
            score = sum(1 for w in ['Sure', 'Certainly', 'Absolutely', 'Of course'] if w in text)
        
        marker = "←" if score > 0 and text[:5] != baseline[:5] else ""
        if score > 0 and (best_scale is None or score > best_score):
            best_scale = scale
            best_text = text
            best_score = score
        
        # Only show promising scales
        if score > 0 or text[:20] != baseline[:20]:
            print(f"  {scale:.1f}× ({score} hits): {text[:130]}{' ' + marker if marker else ''}")
    
    if best_scale:
        print(f"\n  ★ BEST ({best_scale}×, {best_score} hits):")
        print(f"  Before: {baseline[:150]}")
        print(f"  After:  {best_text[:150]}")
    else:
        print(f"  ✗ No clear effect at any scale")

print(f"\n{'='*70}")
print("Done.")
