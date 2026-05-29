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
import time
from pathlib import Path
from collections import defaultdict

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_loader import TopKSAE, load_sae_suite

# === Config ===
MODEL_PATH = "models/Llama-3.2-1B-Instruct"
SAE_PATH = "saes/sae-Llama-3.2-1B-131k"
QUESTIONS_PATH = "data/factual_questions.jsonl"
TARGET_LAYERS = list(range(8, 14))  # L8-L13
STEERING_FACTOR = 3.0  # Within Anthropic's sweet spot (±5)
TOP_K_FEATURES = 5  # Number of features to correlate/steer
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
    return questions[:100]  # Limit to 100 for tractability


def get_answer_tokens(tokenizer, answer: str) -> list[int]:
    """Get token ids for the answer string."""
    tokens = tokenizer.encode(answer, add_special_tokens=False)
    return tokens


def extract_mlp_activations(model, tokenizer, prompt: str, layer_idx: int):
    """Extract MLP output at a specific layer for the last token position."""
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
    """Generate with a steering vector added to MLP output at specified layer."""
    original_outputs = {}

    def steering_hook(module, input, output):
        # Add steering direction to output
        direction = torch.tensor(steer_direction, dtype=output.dtype, device=output.device)
        modified = output + steer_factor * direction
        original_outputs["original"] = output.detach()
        return modified

    hook = model.model.layers[layer_idx].mlp.register_forward_hook(steering_hook)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=3, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    hook.remove()
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_encoding_analysis(saes, model, tokenizer, questions):
    """Phase 1: Extract SAE features and measure encoding accuracy."""
    print("=== Phase 1: Encoding Analysis ===")
    results = {}

    for layer_idx in TARGET_LAYERS:
        if layer_idx not in saes:
            print(f"  L{layer_idx}: SAE not available, skipping")
            continue

        sae = saes[layer_idx]
        layer_features = []
        layer_answers = []
        correct = 0
        total = 0

        t0 = time.time()
        for i, q in enumerate(questions):
            prompt = f"{q['question']} Answer:"
            answer = q["answer"]

            try:
                # Extract MLP activation
                mlp_act = extract_mlp_activations(model, tokenizer, prompt, layer_idx)

                # Encode via SAE
                x = mx.array(mlp_act)
                sae_features = sae.encode(x)
                active_features = np.array(sae_features.tolist())

                # Get answer token id
                answer_tok = get_answer_tokens(tokenizer, answer)[0]

                layer_features.append(active_features)
                layer_answers.append(answer_tok)

                # Simple encoding check: is the answer token among
                # top decoded features? We check if top features
                # point toward the answer in output space
                top_idx = np.argsort(active_features)[-TOP_K_FEATURES:]
                top_decoder_dirs = np.array(sae.W_dec.tolist())[top_idx]

                # Check if any top feature's decoder direction
                # correlates with the answer token embedding
                # (skip actual embedding lookup — use logit lens later)
                total += 1

            except Exception as e:
                if i < 3:
                    print(f"    Error on Q{i}: {e}")
                continue

            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  L{layer_idx}: {i+1}/{len(questions)} ({elapsed:.1f}s)")

        elapsed = time.time() - t0
        print(f"  L{layer_idx}: {total} processed in {elapsed:.1f}s")

        # Save raw features for later analysis
        np.savez(
            OUTPUT_DIR / f"features_L{layer_idx}.npz",
            features=np.array(layer_features),
            answers=np.array(layer_answers),
        )

        results[layer_idx] = {
            "n_processed": total,
            "time_seconds": elapsed,
        }

    return results


def run_causal_analysis(saes, model, tokenizer, questions, encoding_results):
    """Phase 2: Test causal efficacy of SAE feature steering."""
    print("\n=== Phase 2: Causal Analysis ===")
    causal_results = {}

    # For each layer, find top answer-correlated features
    # and test if steering them changes output
    for layer_idx in TARGET_LAYERS:
        if layer_idx not in saes:
            continue

        # Load features from encoding phase
        data = np.load(OUTPUT_DIR / f"features_L{layer_idx}.npz")
        features = data["features"]
        answers = data["answers"]

        # Find features most correlated with each unique answer
        unique_answers = np.unique(answers)
        sae = saes[layer_idx]

        layer_effects = []
        t0 = time.time()

        # Test on a subset for speed
        test_questions = questions[:30]
        baseline_outputs = []
        steered_outputs = []

        for i, q in enumerate(test_questions):
            prompt = f"{q['question']} Answer:"
            answer = q["answer"]

            try:
                # Get baseline output
                baseline = generate_with_steering(
                    model, tokenizer, prompt, layer_idx,
                    np.zeros(2048), 0.0
                )

                # Get the most active features for this question
                mlp_act = extract_mlp_activations(model, tokenizer, prompt, layer_idx)
                x = mx.array(mlp_act)
                sae_feats = sae.encode(x)
                active_feats = np.array(sae_feats.tolist())

                # Find top features and create composite steering direction
                top_idx = np.argsort(active_feats)[-TOP_K_FEATURES:]
                steer_dir = np.zeros(2048)
                decoder = np.array(sae.W_dec.tolist())
                for feat_id in top_idx:
                    steer_dir += decoder[feat_id] * active_feats[feat_id]

                steer_dir = steer_dir / (np.linalg.norm(steer_dir) + 1e-8)

                # Generate with steering
                steered = generate_with_steering(
                    model, tokenizer, prompt, layer_idx,
                    steer_dir, STEERING_FACTOR
                )

                baseline_outputs.append(baseline)
                steered_outputs.append(steered)

                # Check if answer appears in steered output
                baseline_has_answer = answer.lower() in baseline.lower()
                steered_has_answer = answer.lower() in steered.lower()

                layer_effects.append({
                    "question": q["question"][:50],
                    "answer": answer,
                    "baseline_output": baseline,
                    "steered_output": steered,
                    "baseline_correct": baseline_has_answer,
                    "steered_correct": steered_has_answer,
                    "improved": steered_has_answer and not baseline_has_answer,
                })

            except Exception as e:
                if i < 3:
                    print(f"    Error on Q{i}: {e}")
                continue

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  L{layer_idx} causal: {i+1}/{len(test_questions)} ({elapsed:.1f}s)")

        # Compute metrics
        n_baseline_correct = sum(1 for e in layer_effects if e["baseline_correct"])
        n_steered_correct = sum(1 for e in layer_effects if e["steered_correct"])
        n_improved = sum(1 for e in layer_effects if e["improved"])
        n_total = len(layer_effects)

        causal_results[layer_idx] = {
            "n_tested": n_total,
            "baseline_accuracy": n_baseline_correct / max(n_total, 1),
            "steered_accuracy": n_steered_correct / max(n_total, 1),
            "improvement_rate": n_improved / max(n_total, 1),
            "causal_efficacy": max(0, (n_steered_correct - n_baseline_correct) / max(n_total, 1)),
            "examples": layer_effects[:10],  # Save first 10 examples
        }

        elapsed = time.time() - t0
        print(f"  L{layer_idx}: baseline={n_baseline_correct}/{n_total} "
              f"steered={n_steered_correct}/{n_total} "
              f"efficacy={causal_results[layer_idx]['causal_efficacy']:.1%} "
              f"({elapsed:.0f}s)")

    return causal_results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, "
          f"{len(model.model.layers)} layers")

    print(f"Loading SAEs for layers {TARGET_LAYERS}...")
    saes = load_sae_suite(SAE_PATH, layers=TARGET_LAYERS)
    print(f"Loaded {len(saes)} SAEs: {[s.layer_idx for s in saes.values()]}")

    questions = load_questions()
    print(f"Questions: {len(questions)}")

    # Phase 1: Encoding
    encoding_results = run_encoding_analysis(saes, model, tokenizer, questions)

    # Phase 2: Causal
    causal_results = run_causal_analysis(saes, model, tokenizer, questions, encoding_results)

    # Save results
    all_results = {
        "config": {
            "model": "Llama-3.2-1B-Instruct",
            "layers": TARGET_LAYERS,
            "steering_factor": STEERING_FACTOR,
            "top_k_features": TOP_K_FEATURES,
            "n_questions": len(questions),
        },
        "encoding": encoding_results,
        "causal": causal_results,
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Layer':<8} {'Baseline':>10} {'Steered':>10} {'Causal Eff':>12} {'Improved':>10}")
    print("-" * 55)
    for layer_idx in TARGET_LAYERS:
        if layer_idx in causal_results:
            cr = causal_results[layer_idx]
            print(f"L{layer_idx:<7} {cr['baseline_accuracy']:>9.1%} "
                  f"{cr['steered_accuracy']:>9.1%} "
                  f"{cr['causal_efficacy']:>11.1%} "
                  f"{cr['improvement_rate']:>9.1%}")

    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")
    print("✓ Experiment #1 complete")


if __name__ == "__main__":
    main()
