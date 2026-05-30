#!/usr/bin/env python3
"""Phase 3: Causal steering with DLA-based directions + multiplier sweep.
Loads model only + steering vectors (no SAE weights in memory).
Baseline vs steered generation per question.

Usage: python phase3_causal.py LAYER_IDX [STEERING_STRENGTH] [--dla]"""

import sys, json, time
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx_lm import load
from config import MODEL_PATH, QUESTIONS_PATH, OUTPUT_DIR

layer_idx = int(sys.argv[1])

# Parse optional steering strength and DLA flag
use_dla = "--dla" in sys.argv
strength_arg = None
for a in sys.argv[2:]:
    try:
        strength_arg = float(a)
        break
    except ValueError:
        continue

if strength_arg is None:
    # Sweep multiple multipliers if none specified
    multipliers = [5.0, 10.0, 20.0, 50.0]
else:
    multipliers = [strength_arg]

steer_file = OUTPUT_DIR / (f"steering_dla_L{layer_idx}.npz" if use_dla else f"steering_L{layer_idx}.npz")
if not steer_file.exists():
    print(f"ERROR: {steer_file} not found", flush=True)
    sys.exit(1)

print(f"Phase 3: Causal steering L{layer_idx} {'(DLA)' if use_dla else '(activation)'}", flush=True)

# Load model only
print("  Loading model...", flush=True)
model, tokenizer = load(MODEL_PATH)
print("  Model ready", flush=True)

# Load questions
with open(QUESTIONS_PATH) as f:
    questions = [json.loads(line.strip()) for line in f]

# Load pre-computed steering directions
steering = np.load(steer_file)["steering"]  # [50, d_model]
print(f"  Steering: {steering.shape}", flush=True)

test_qs = questions[:15]

# Measure residual stream magnitude for scaling diagnostic
print("  Measuring residual stream norm...", flush=True)
sample_tokens = mx.array(tokenizer.encode(test_qs[0]["question"] + " Answer:"))[None, :]
h = model.model.embed_tokens(sample_tokens)
for l, layer in enumerate(model.model.layers):
    r = layer.self_attn(layer.input_layernorm(h)); h = h + r
    m = layer.mlp(layer.post_attention_layernorm(h))
    h = h + m
    if l == layer_idx:
        resid_norm = float(mx.linalg.norm(h[0, -1, :]).item())
        print(f"  Residual L2 norm at L{layer_idx}: {resid_norm:.1f}", flush=True)
        break

for STEERING_STRENGTH in multipliers:
    print(f"\n  --- Multiplier: {STEERING_STRENGTH:.0f}x ---", flush=True)
    steer_norm_frac = STEERING_STRENGTH / (resid_norm + 1e-8)
    print(f"  Steering = {steer_norm_frac:.1%} of residual stream", flush=True)

    effects = []
    t0 = time.time()

    for i, q in enumerate(test_qs):
        prompt = f"{q['question']} Answer:"
        answer = q["answer"]
        tokens = mx.array(tokenizer.encode(prompt))[None, :]
        steer_mx = mx.array(steering[i] * STEERING_STRENGTH)

        try:
            # --- Baseline ---
            h = model.model.embed_tokens(tokens)
            for l, layer in enumerate(model.model.layers):
                r = layer.self_attn(layer.input_layernorm(h)); h = h + r
                m = layer.mlp(layer.post_attention_layernorm(h))
                h = h + m
                if l % 4 == 0:
                    mx.eval(h)
            logits = model.model.embed_tokens.as_linear(model.model.norm(h))
            mx.eval(logits)
            baseline = tokenizer.decode([int(mx.argmax(logits[0, -1, :]))])

            # --- Steered ---
            h = model.model.embed_tokens(tokens)
            for l, layer in enumerate(model.model.layers):
                r = layer.self_attn(layer.input_layernorm(h)); h = h + r
                m = layer.mlp(layer.post_attention_layernorm(h))
                if l == layer_idx:
                    m = m + steer_mx
                h = h + m
                if l % 4 == 0:
                    mx.eval(h)
            logits = model.model.embed_tokens.as_linear(model.model.norm(h))
            mx.eval(logits)
            steered = tokenizer.decode([int(mx.argmax(logits[0, -1, :]))])

            b_ok = answer.lower() in baseline.lower()
            s_ok = answer.lower() in steered.lower()
            effects.append({
                "answer": answer, "baseline": baseline, "steered": steered,
                "baseline_correct": b_ok, "steered_correct": s_ok,
                "improved": s_ok and not b_ok,
            })
        except Exception as e:
            if i < 2: print(f"    Error Q{i}: {e}", flush=True)
            continue

        if (i + 1) % 5 == 0:
            print(f"    {i+1}/15 ({time.time()-t0:.0f}s)", flush=True)

    n = len(effects)
    n_base = sum(1 for e in effects if e["baseline_correct"])
    n_steer = sum(1 for e in effects if e["steered_correct"])
    n_imp = sum(1 for e in effects if e["improved"])
    n_changed = sum(1 for e in effects if e["baseline"] != e["steered"])
    eff = max(0, (n_steer - n_base) / max(n, 1))

    print(f"    Basel={n_base}/{n} Steer={n_steer}/{n} Changed={n_changed} Eff={eff:.1%} Imp={n_imp}", flush=True)

    result = {
        "layer": layer_idx, "n": n, "multiplier": STEERING_STRENGTH,
        "steer_fraction": round(steer_norm_frac, 4),
        "dla": use_dla,
        "baseline_accuracy": n_base / max(n, 1),
        "steered_accuracy": n_steer / max(n, 1),
        "causal_efficacy": eff,
        "improvement_rate": n_imp / max(n, 1),
        "tokens_changed": n_changed,
        "examples": effects[:5],
    }

    tag = "dla" if use_dla else "act"
    out_file = OUTPUT_DIR / f"causal_{tag}_L{layer_idx}_x{int(STEERING_STRENGTH)}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    → {out_file}", flush=True)

print(f"\n  ✓ Done", flush=True)
