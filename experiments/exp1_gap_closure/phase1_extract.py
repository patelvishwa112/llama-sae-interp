#!/usr/bin/env python3
"""Experiment #1 Phase 1: Extract MLP activations for all layers in one pass.

Saves activations to disk so Phase 2 can process SAEs one layer at a time
without holding both model and SAE in memory simultaneously.
"""

import json, sys, time, gc
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

MODEL_PATH = "models/Llama-3.2-1B-Instruct"
QUESTIONS_PATH = "data/factual_questions.jsonl"
OUTPUT_DIR = Path("experiments/exp1_gap_closure/results")
LAYERS = list(range(8, 14))
DEVICE = "mps"

print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE, local_files_only=True
)
model.eval()
print(f"Model ready: {torch.mps.current_allocated_memory()/1e9:.1f}GB", flush=True)

questions = []
with open(QUESTIONS_PATH) as f:
    for line in f:
        questions.append(json.loads(line.strip()))
questions = questions[:50]  # 50 questions for Phase 1
print(f"Questions: {len(questions)}", flush=True)

# Collect activations for all layers
all_activations = {layer: [] for layer in LAYERS}
all_answers = []

t0 = time.time()
for i, q in enumerate(questions):
    prompt = f"{q['question']} Answer:"
    answer = q["answer"]
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Hook all target layers simultaneously
    hooks = []
    layer_outputs = {}
    for layer in LAYERS:
        def make_hook(l):
            def hook(m, inp, out):
                layer_outputs[l] = out.detach()
            return hook
        h = model.model.layers[layer].mlp.register_forward_hook(make_hook(layer))
        hooks.append(h)
    
    with torch.no_grad():
        model(**inputs)
    
    for h in hooks:
        h.remove()
    
    answer_tok = tokenizer.encode(answer, add_special_tokens=False)[0]
    all_answers.append(answer_tok)
    
    for layer in LAYERS:
        act = layer_outputs[layer][0, -1, :].cpu().float().numpy()
        all_activations[layer].append(act)
    
    if (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{len(questions)} ({elapsed:.0f}s, {torch.mps.current_allocated_memory()/1e9:.1f}GB)", flush=True)

elapsed = time.time() - t0
print(f"Extraction complete: {len(questions)} in {elapsed:.0f}s", flush=True)

# Save to disk
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for layer in LAYERS:
    np.savez(
        OUTPUT_DIR / f"activations_L{layer}.npz",
        activations=np.array(all_activations[layer]),
        answers=np.array(all_answers),
    )
    print(f"  Saved L{layer}: {len(all_activations[layer])} activations", flush=True)

print("✓ Phase 1 complete — activations saved to disk", flush=True)
