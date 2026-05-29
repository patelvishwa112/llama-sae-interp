#!/usr/bin/env python3
"""Experiment #1 Phase 2: Process SAEs on saved activations + causal steering.

Loads activations from disk, runs SAE encode one layer at a time,
then tests causal steering on a subset.
"""

import json, sys, time, gc
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from sae_loader_light import TopKSAELight

MODEL_PATH = "models/Llama-3.2-1B-Instruct"
SAE_PATH = "saes/sae-Llama-3.2-1B-131k"
QUESTIONS_PATH = "data/factual_questions.jsonl"
OUTPUT_DIR = Path("experiments/exp1_gap_closure/results")
LAYERS = list(range(8, 14))
STEERING_FACTOR = 3.0
TOP_K = 5
DEVICE = "mps"


def process_layer_encoding(layer_idx):
    """Load saved activations, run SAE, save features."""
    data = np.load(OUTPUT_DIR / f"activations_L{layer_idx}.npz")
    activations = data["activations"]
    answers = data["answers"]
    data.close()

    print(f"  Loading SAE L{layer_idx}...", flush=True)
    sae = TopKSAELight(SAE_PATH, layer_idx, device=DEVICE)
    print(f"  SAE loaded: {torch.mps.current_allocated_memory()/1e9:.1f}GB", flush=True)

    features = []
    t0 = time.time()
    for i, act in enumerate(activations):
        x = torch.tensor(act, dtype=torch.float16, device=DEVICE)
        feat = sae.encode(x).cpu().float().numpy()
        features.append(feat)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(activations)} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(OUTPUT_DIR / f"features_L{layer_idx}.npz",
             features=np.array(features), answers=answers)
    print(f"  L{layer_idx} features saved: {len(features)}", flush=True)

    # Find top most-active features across all questions
    all_feats = np.array(features)
    mean_activation = all_feats.mean(axis=0)
    top_global = np.argsort(mean_activation)[-10:][::-1]
    print(f"  Top global features: {top_global.tolist()}", flush=True)

    del sae
    gc.collect()

    return len(features)


def process_layer_causal(layer_idx, model, tokenizer, questions):
    """Test causal steering at this layer."""
    print(f"\n  Causal steering L{layer_idx}...", flush=True)
    
    sae = TopKSAELight(SAE_PATH, layer_idx, device=DEVICE)
    decoder = sae.W_dec.cpu().float().numpy()
    
    effects = []
    test_qs = questions[:20]  # Test on 20 questions
    t0 = time.time()
    
    for i, q in enumerate(test_qs):
        prompt = f"{q['question']} Answer:"
        answer = q["answer"]
        try:
            # Baseline
            baseline = generate(model, tokenizer, prompt, layer_idx, np.zeros(2048), 0.0)
            
            # Get features and build steering direction
            inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
            mlp_out = {}
            def hook(m, inp, out): mlp_out['act'] = out.detach()
            h = model.model.layers[layer_idx].mlp.register_forward_hook(hook)
            with torch.no_grad(): model(**inputs)
            h.remove()
            
            act = mlp_out['act'][0, -1, :]
            feats = sae.encode(act).cpu().float().numpy()
            top_idx = np.argsort(feats)[-TOP_K:]
            
            steer_dir = np.zeros(2048)
            for fid in top_idx:
                steer_dir += decoder[fid] * feats[fid]
            steer_dir = steer_dir / (np.linalg.norm(steer_dir) + 1e-8)
            
            steered = generate(model, tokenizer, prompt, layer_idx, steer_dir, STEERING_FACTOR)
            
            b_ok = answer.lower() in baseline.lower()
            s_ok = answer.lower() in steered.lower()
            effects.append({
                "question": q["question"][:50], "answer": answer,
                "baseline": baseline, "steered": steered,
                "baseline_correct": b_ok, "steered_correct": s_ok,
                "improved": s_ok and not b_ok,
            })
        except Exception as e:
            if i < 2: print(f"    Error: {e}", flush=True)
            continue
        
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(test_qs)} ({time.time()-t0:.0f}s)", flush=True)
    
    n = len(effects)
    n_base = sum(1 for e in effects if e["baseline_correct"])
    n_steer = sum(1 for e in effects if e["steered_correct"])
    n_imp = sum(1 for e in effects if e["improved"])
    efficacy = max(0, (n_steer - n_base) / max(n, 1))
    
    print(f"    Basel={n_base}/{n} ({n_base/max(n,1):.0%}) Steer={n_steer}/{n} ({n_steer/max(n,1):.0%}) "
          f"Eff={efficacy:.1%} Imp={n_imp}", flush=True)
    
    del sae
    gc.collect()
    
    return {
        "layer": layer_idx, "n": n,
        "baseline_accuracy": n_base / max(n, 1),
        "steered_accuracy": n_steer / max(n, 1),
        "causal_efficacy": efficacy,
        "improvement_rate": n_imp / max(n, 1),
        "examples": effects[:5],
    }


def generate(model, tokenizer, prompt, layer_idx, steer_dir, steer_factor):
    """Generate with optional steering."""
    def steering_hook(m, inp, out):
        d = torch.tensor(steer_dir, dtype=out.dtype, device=out.device)
        return out + steer_factor * d
    
    h = model.model.layers[layer_idx].mlp.register_forward_hook(steering_hook)
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=3, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    h.remove()
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 2a: SAE encoding (offline from saved activations)
    print("=" * 60)
    print("Phase 2a: SAE Encoding (offline)")
    print("=" * 60)
    for layer in LAYERS:
        n = process_layer_encoding(layer)
        print(f"  L{layer}: {n} features saved", flush=True)

    # Phase 2b: Causal steering (needs model loaded)
    print("\n" + "=" * 60)
    print("Phase 2b: Causal Steering")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE, local_files_only=True
    )
    model.eval()
    
    questions = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    
    results = []
    for layer in LAYERS:
        r = process_layer_causal(layer, model, tokenizer, questions)
        results.append(r)
    
    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY: SAE Encoding-Deployment Gap")
    print("=" * 65)
    print(f"{'Layer':<8} {'Baseline':>10} {'Steered':>10} {'Causal Eff':>12} {'Improved':>10}")
    print("-" * 55)
    for r in results:
        print(f"L{r['layer']:<7} {r['baseline_accuracy']:>9.1%} "
              f"{r['steered_accuracy']:>9.1%} {r['causal_efficacy']:>11.1%} "
              f"{r['improvement_rate']:>9.1%}")
    
    print(f"\nGhost probe (L21): basel~0%  steer=4.3%  eff=4.3%")
    
    all_results = {"config": {"model": "Llama-3.2-1B-Instruct", "layers": LAYERS,
                               "steering_factor": STEERING_FACTOR, "top_k": TOP_K},
                   "layers": results}
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR / 'results.json'}")
    print("✓ Experiment #1 complete")


if __name__ == "__main__":
    main()
