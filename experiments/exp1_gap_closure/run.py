#!/usr/bin/env python3
"""Experiment #1: SAE Encoding-Deployment Gap Closure.

Measures whether SAE features close the gap between encoding accuracy
and causal efficacy compared to linear probes (ghost experiment baseline).

Protocol:
  1. Run 100 factual questions through Llama 3.2 1B
  2. Extract MLP activations at each target layer (L8-L13)
  3. Encode via TopK SAE → sparse features (k=32 per token)
  4. Encoding score: how well do SAE features predict the correct answer?
  5. Causal efficacy: does steering answer-correlated features change output?
  6. Compare to ghost experiment baseline (79.7% encoding → 4.3% causal)
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_loader_light import TopKSAELight

# === Config ===
MODEL_PATH = "models/Llama-3.2-1B-Instruct"
SAE_PATH = "saes/sae-Llama-3.2-1B-131k"
QUESTIONS_PATH = "data/factual_questions.jsonl"
TARGET_LAYERS = list(range(8, 14))  # L8-L13
STEERING_FACTOR = 3.0
TOP_K_FEATURES = 5
OUTPUT_DIR = Path("experiments/exp1_gap_closure/results")
DEVICE = "mps"


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE, local_files_only=True
    )
    model.eval()
    return model, tokenizer


def load_questions():
    questions = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    return questions


def get_answer_tokens(tokenizer, answer: str) -> list[int]:
    tokens = tokenizer.encode(answer, add_special_tokens=False)
    return tokens


def extract_mlp_activations(model, tokenizer, prompt: str, layer_idx: int):
    mlp_outputs = {}
    def hook_fn(module, input, output):
        mlp_outputs["act"] = output.detach()
    hook = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model(**inputs)
    hook.remove()
    return mlp_outputs["act"][0, -1, :].cpu().float().numpy()


def generate_with_steering(
    model, tokenizer, prompt: str, layer_idx: int,
    steer_direction: np.ndarray, steer_factor: float
) -> str:
    def steering_hook(module, input, output):
        direction = torch.tensor(steer_direction, dtype=output.dtype, device=output.device)
        return output + steer_factor * direction
    hook = model.model.layers[layer_idx].mlp.register_forward_hook(steering_hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=3, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    hook.remove()
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def process_one_layer(layer_idx, sae_path, model, tokenizer, questions, output_dir):
    """Process a single layer: load SAE, run encoding + causal, unload SAE."""
    import gc

    print(f"\n{'='*50}")
    print(f"Layer {layer_idx}")
    print(f"{'='*50}")

    # Load SAE for this layer only
    print(f"  Loading SAE L{layer_idx}...")
    sae = TopKSAELight(sae_path, layer_idx, device=DEVICE)
    print(f"  {sae}")

    # Phase 1: Encoding
    print(f"  Phase 1: Encoding analysis...")
    layer_features = []
    layer_answers = []
    t0 = time.time()

    for i, q in enumerate(questions):
        prompt = f"{q['question']} Answer:"
        answer = q["answer"]
        try:
            mlp_act = extract_mlp_activations(model, tokenizer, prompt, layer_idx)
            # mlp_act is numpy float32, convert to torch
            x = torch.tensor(mlp_act, dtype=torch.float16, device=DEVICE)
            sae_features = sae.encode(x)
            active_features = sae_features.cpu().float().numpy()
            answer_tok = get_answer_tokens(tokenizer, answer)[0]
            layer_features.append(active_features)
            layer_answers.append(answer_tok)
        except Exception as e:
            if i < 3: print(f"    Error Q{i}: {e}")
            continue
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(questions)} ({time.time()-t0:.0f}s)")

    encoding_time = time.time() - t0
    np.savez(output_dir / f"features_L{layer_idx}.npz",
             features=np.array(layer_features), answers=np.array(layer_answers))
    n_enc = len(layer_answers)
    print(f"  Encoding: {n_enc} processed in {encoding_time:.0f}s")

    # Phase 2: Causal (on subset)
    print(f"  Phase 2: Causal steering...")
    decoder = sae.W_dec.cpu().float().numpy()  # Keep decoder in numpy for steering
    layer_effects = []
    test_questions = questions[:30]
    t0 = time.time()

    for i, q in enumerate(test_questions):
        prompt = f"{q['question']} Answer:"
        answer = q["answer"]
        try:
            # Baseline
            baseline = generate_with_steering(model, tokenizer, prompt, layer_idx,
                                              np.zeros(2048), 0.0)
            # Get active features and build steering direction
            mlp_act = extract_mlp_activations(model, tokenizer, prompt, layer_idx)
            x = torch.tensor(mlp_act, dtype=torch.float16, device=DEVICE)
            sae_feats = sae.encode(x)
            active_feats = sae_feats.cpu().float().numpy()
            top_idx = np.argsort(active_feats)[-TOP_K_FEATURES:]
            steer_dir = np.zeros(2048)
            for feat_id in top_idx:
                steer_dir += decoder[feat_id] * active_feats[feat_id]
            steer_dir = steer_dir / (np.linalg.norm(steer_dir) + 1e-8)
            # Steered
            steered = generate_with_steering(model, tokenizer, prompt, layer_idx,
                                             steer_dir, STEERING_FACTOR)
            b_correct = answer.lower() in baseline.lower()
            s_correct = answer.lower() in steered.lower()
            layer_effects.append({
                "question": q["question"][:50], "answer": answer,
                "baseline_output": baseline, "steered_output": steered,
                "baseline_correct": b_correct, "steered_correct": s_correct,
                "improved": s_correct and not b_correct,
            })
        except Exception as e:
            if i < 3: print(f"    Error Q{i}: {e}")
            continue
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(test_questions)} ({time.time()-t0:.0f}s)")

    causal_time = time.time() - t0
    n_total = len(layer_effects)
    n_base = sum(1 for e in layer_effects if e["baseline_correct"])
    n_steer = sum(1 for e in layer_effects if e["steered_correct"])
    n_improved = sum(1 for e in layer_effects if e["improved"])
    efficacy = max(0, (n_steer - n_base) / max(n_total, 1))

    print(f"  Causal: baseline={n_base}/{n_total} ({n_base/max(n_total,1):.0%}) "
          f"steered={n_steer}/{n_total} ({n_steer/max(n_total,1):.0%}) "
          f"efficacy={efficacy:.1%} ({causal_time:.0f}s)")

    # Unload SAE
    del sae
    gc.collect()

    return {
        "layer": layer_idx,
        "n_encoding": n_enc,
        "encoding_time": encoding_time,
        "n_causal": n_total,
        "baseline_accuracy": n_base / max(n_total, 1),
        "steered_accuracy": n_steer / max(n_total, 1),
        "causal_efficacy": efficacy,
        "improvement_rate": n_improved / max(n_total, 1),
        "causal_time": causal_time,
        "examples": layer_effects[:5],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    questions = load_questions()
    print(f"Questions: {len(questions)}")

    all_layer_results = []

    for layer_idx in TARGET_LAYERS:
        result = process_one_layer(layer_idx, SAE_PATH, model, tokenizer,
                                   questions, OUTPUT_DIR)
        all_layer_results.append(result)

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY: SAE Encoding-Deployment Gap")
    print("=" * 65)
    print(f"{'Layer':<8} {'Baseline':>10} {'Steered':>10} {'Causal Eff':>12} {'Improved':>10}")
    print("-" * 55)
    for r in all_layer_results:
        print(f"L{r['layer']:<7} {r['baseline_accuracy']:>9.1%} "
              f"{r['steered_accuracy']:>9.1%} {r['causal_efficacy']:>11.1%} "
              f"{r['improvement_rate']:>9.1%}")

    # Ghost experiment baseline for comparison
    print(f"\n{'Ghost probe (L21)':<8} {'~0%':>10} {'4.3%':>10} {'4.3%':>12}")
    print(f"\nResults: {OUTPUT_DIR / 'results.json'}")

    all_results = {
        "config": {
            "model": "Llama-3.2-1B-Instruct",
            "layers": TARGET_LAYERS,
            "steering_factor": STEERING_FACTOR,
            "top_k_features": TOP_K_FEATURES,
            "n_questions": len(questions),
        },
        "layers": all_layer_results,
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("✓ Experiment #1 complete")


if __name__ == "__main__":
    main()
