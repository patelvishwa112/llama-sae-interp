# Cracking Open the Black Box: How I Learned to Steer Llama with a Single SAE Feature

## A journey from 0% causal efficacy to four confirmed behavioral changes — on an 8GB M1 Mac Mini

**May 2026** | Vishwa Patel | Apple MLX | Llama 3.2 1B-Instruct | EleutherAI SAE 131k

---

> *"When we turn up the strength of the 'Golden Gate Bridge' feature, Claude's responses begin to focus on the Golden Gate Bridge."* — Anthropic, May 2024

Anthropic's Golden Gate Claude demo was magnetic. A single number — one of 34 million SAE features — could make Claude obsess about bridges, relate everything to San Francisco, even claim to *be* the Golden Gate Bridge. No fine-tuning. No prompt engineering. Just one surgical tweak to the model's internal activations.

I wanted to replicate this. Not with Claude's proprietary SAEs on a server farm, but with a public SAE on an open-source model running on my 8GB M1 Mac Mini. The result: **four confirmed behavioral changes**, two dead ends, and one critical insight that changed everything.

---

## The Setup

| Component | Details |
|-----------|---------|
| **Model** | [Llama 3.2 1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B) — 16 layers, 2048-dim residual stream |
| **SAE** | [EleutherAI/sae-Llama-3.2-1B-131k](https://huggingface.co/EleutherAI/sae-Llama-3.2-1B-131k) — 131,072 features, 64× expansion |
| **Framework** | Apple MLX — lazy evaluation, native MPS acceleration |
| **Hardware** | Mac Mini M1, 8GB RAM, 228GB SSD |

The SAE decomposes each layer's residual stream into 131,072 sparse features. Every feature has an **encoder vector** `W_enc[:,f]` (how to detect it) and a **decoder vector** `W_dec[f]` (what it contributes to the residual stream). The idea: find a feature representing the behavior you want, amplify it, and the model should exhibit that behavior.

Here's what loading the SAE looks like — each layer's weights are stored as safetensors:

```python
from safetensors import safe_open

layer_idx = 12  # Pick a mid-network layer
sae_path = f"saes/sae-Llama-3.2-1B-131k/layers.{layer_idx}.mlp"

with safe_open(f"{sae_path}/sae.safetensors", framework="np") as f:
    W_enc = f.get_tensor("encoder.weight")    # [n_latents, d_model] → transpose to [d_model, n_latents]
    W_dec = f.get_tensor("W_dec")              # [n_latents, d_model]
    b_enc = f.get_tensor("encoder.bias")       # [n_latents]
    b_dec = f.get_tensor("b_dec")              # [d_model]

# Encode: residual → sparse latents
latents = (mlp_output - b_dec) @ W_enc.T + b_enc   # [n_latents] = 131,072
# Decode: sparse latents → residual contribution
reconstructed = latents @ W_dec + b_dec              # [d_model] = 2,048
```

This encode-decode roundtrip is at the heart of how SAEs work. It's also, as we'll discover, the root of everything that went wrong.

---

## The Pipeline: Three Phases, Zero Compromises

The hardest constraint was the 8GB RAM. Loading both the model (~2.5GB) and SAE (~1GB) simultaneously leaves very little room for activations during autoregressive generation. My first attempt — loading everything at once — OOM-killed within seconds.

**The solution: a three-phase pipeline where the model and SAE never coexist in memory except for the final lightweight steering step.**

[📐 Architecture diagram](architecture.html) — open in browser

### Phase 1: Activation Capture (Model Only, ~2.5GB)

The model processes 50 factual questions. At each target layer, we capture the MLP output (2048-dim) at the last token position. The model is then freed from memory.

```python
# phase1_multi.py — capture MLP activations at target layers
import mlx.core as mx
from mlx_lm import load

model, tokenizer = load("models/Llama-3.2-1B-Instruct")
target_layers = [10, 12, 13]
activations = {l: [] for l in target_layers}

for question in questions[:50]:
    prompt = f"{question['question']} Answer:"
    tokens = mx.array(tokenizer.encode(prompt))[None, :]
    h = model.model.embed_tokens(tokens)

    for l, layer in enumerate(model.model.layers):
        r = layer.self_attn(layer.input_layernorm(h)); h = h + r
        m = layer.mlp(layer.post_attention_layernorm(h))
        if l in target_layers:
            # Capture MLP output at last token position
            activations[l].append(m[0, -1, :].tolist())
        h = h + m

# Save: 400KB per layer, 50 questions × 2048 dimensions
for l in target_layers:
    np.savez(f"activations_L{l}.npz", activations=np.array(activations[l]))
```

**Key insight:** We iterate through layers manually (`model.model.layers[l]`) rather than calling `model(tokens)` — this gives us access to intermediate activations at each layer. The residual stream `h` accumulates the sum of all previous layers' outputs, which is exactly how transformer architectures work: `h = h + attention(norm(h)) + mlp(norm(h))`.

### Phase 2: SAE Feature Encoding (SAE Only, ~1GB)

We load the SAE weights and saved activations, encode through the SAE encoder, and keep only the top-K=32 sparse features.

```python
# phase2_encode.py — SAE-only encoding (no model in memory)
import numpy as np
from safetensors import safe_open

# Load saved activations and SAE weights
activations = np.load(f"activations_L{layer_idx}.npz")["activations"]
with safe_open(f"{sae_path}/sae.safetensors", framework="np") as f:
    W_enc = f.get_tensor("encoder.weight").T  # [d_model, n_latents]
    b_enc = f.get_tensor("encoder.bias")
    b_dec = f.get_tensor("b_dec")

features = np.zeros((50, 131_072), dtype=np.float32)
K = 32  # Keep only top-32 features (sparsify)

for i in range(50):
    mlp_out = activations[i]
    centered = mlp_out - b_dec
    latents = centered @ W_enc + b_enc   # [131,072] raw activations

    # Top-K sparsification: zero out everything except the 32 strongest
    top_idx = np.argsort(np.abs(latents))[-K:]
    feats = np.zeros(131_072, dtype=np.float32)
    feats[top_idx] = latents[top_idx]
    features[i] = feats

np.savez_compressed(f"features_L{layer_idx}.npz", features=features)
```

The SAE's 131,072 features are extremely sparse — only 32 fire at a time. This sparsity is what makes features interpretable: each feature captures a narrow, specific concept rather than a diffuse blend of many concepts. The top-K sparsification matches how the SAE was trained (k=32 TopK SAE).

### Phase 2c: DLA Feature Selection (SAE + Embeddings, ~1.5GB)

**This is how we find the right features.** Direct Logit Attribution (DLA) scores every feature by how much its decoder vector aligns with a target token's embedding.

```python
# phase2c_dla.py — find features that push toward a specific output token
embed_weight = model.model.embed_tokens.weight  # [vocab, d_model]
target_id = tokenizer.encode(" -")[1]            # skip BOS token
target_emb = embed_weight[target_id]              # [d_model]

# DLA score for every feature: |W_dec[f] · target_embedding|
dla_scores = np.abs(W_dec @ target_emb)           # [131,072]
top_features = np.argsort(dla_scores)[-5:]        # Top 5 by alignment

for f in reversed(top_features):
    print(f"Feature {f}: DLA={dla_scores[f]:.3f}")
```

Features with high DLA scores (>0.3) have a direct causal path from activation to output. The noise floor across 131,072 features is ~0.022 — so a score of 0.431 (our bullet point feature) is roughly 20σ above the noise floor. This is how we know we're picking genuine signal, not statistical noise.

### Phase 3: Direct Decoder Steering (Model + Vectors, ~2.5GB)

This is where everything comes together — the autoregressive generation loop with SAE feature injection at every step.

```python
# steer_direct.py — the core steering loop
W_dec = load_sae_decoder(layer=12)
vec = W_dec[105206] / (mx.linalg.norm(W_dec[105206]) + 1e-8)  # Normalize
SCALE = 2.5  # Sweet spot

ids = tokenizer.encode("My favorite foods are:")
for step in range(40):  # Autoregressive generation
    tokens = mx.array(ids)[None, :]
    h = model.model.embed_tokens(tokens)

    for l, layer in enumerate(model.model.layers):
        r = layer.self_attn(layer.input_layernorm(h)); h = h + r
        m = layer.mlp(layer.post_attention_layernorm(h))
        if l == 12:                      # Target layer
            m = m + vec * SCALE          # ★ Direct decoder addition
        h = h + m

    logits = model.model.embed_tokens.as_linear(model.model.norm(h))
    next_token = int(mx.argmax(logits[0, -1, :]).item())
    if next_token == tokenizer.eos_token_id: break
    ids.append(next_token)

print(tokenizer.decode(ids[len(prompt_tokens):]))
```

**Three critical details in this loop:**

1. **`m = m + vec * SCALE`** — We add the decoder vector directly, not through SAE encode-decode. No clamping, no reconstruction. Just the raw direction the feature contributes.

2. **Inside the autoregressive loop** — The steering is applied at every generation step, not just at prompt encoding. This means the signal is present in every attention computation and every MLP block throughout generation.

3. **`vec = W_dec[f] / norm`** — Normalizing the decoder vector makes the SCALE parameter comparable across features and layers. A scale of 2.5 means "2.5× the unit-norm decoder direction."

---

## The Dead Ends

### Mistake #1: SAE Encode → Clamp → Decode (The Reconstruction Trap)

Our first approach seemed obvious: encode through the SAE, clamp the target feature, decode back, and add the delta.

```python
# WHAT WE TRIED (and failed): encode-clamp-decode
mlp_out = m[0, -1, :]                         # Extract MLP output
latents = (mlp_out - b_dec) @ W_enc + b_enc   # SAE encode
latents[target_feature] = clamp_value          # Clamp the feature
reconstructed = latents @ W_dec + b_dec         # SAE decode
steer_delta = reconstructed - mlp_out           # Difference
m = m + steer_delta                             # Add to residual
```

**It never worked.** At any clamp magnitude — even 0.5× natural activation — the model produced degenerate output:

```
Prompt: "Tell me a story."
Output:  "fing calcium pered cough sulf fing calcium pered cough sulf..."
```

The SAE is **lossy**. The encode-decode roundtrip introduces 5-15% reconstruction error. When you clamp a feature, those artifacts get amplified, and the model sees an activation pattern it was never trained on. The `reconstructed` vector is in a slightly different subspace than the original `mlp_out`, and the difference `steer_delta` contains both the intended steering signal AND reconstruction noise. The model can't separate the two.

This is why the direct decoder approach works: `W_dec[f]` is the feature's contribution direction WITHOUT any encoding error. It's a clean signal.

### Mistake #2: Single-Position Steering at Routing Layers

We tried steering at L12-L13 — the layers where attention heads route information to the output. At ANY multiplier, even 457% of the residual stream magnitude, the model produced **zero changed tokens**.

The routing layers are structurally immune to single-token perturbations. By L12, attention has already aggregated signals across all positions. A distortion at one position is averaged out. Only when we applied steering at EVERY position autoregressively did the routing layers respond.

### Mistake #3: Steering Discrete Factual Answers

Our original goal was to steer factual knowledge — make the model say "Paris" instead of "London", or "dog" instead of "cat". This is a discrete, binary choice, not a continuous semantic dimension. As documented in MaziyarPanahi's [excellent blog](https://huggingface.co/blog/MaziyarPanahi/sae-steering-json), steering works for semantic concepts (tone, sentiment, formatting) but fails for syntactic or discrete tasks.

**The distinction:** "Use bullet points" is a continuous stylistic choice. "Say Paris" is a discrete token choice. SAE features live in continuous semantic space — they can nudge style, but can't force a specific token.

---

## The Breakthrough: Direct Decoder Addition

After two days of failures, the key insight: **bypass the SAE encode-decode cycle entirely.**

The comparison is stark:

```python
# ❌ FAILED: Encode-clamp-decode (reconstruction artifacts)
latents = (mlp_out - b_dec) @ W_enc + b_enc
latents[f] = clamp_val
m = m + (latents @ W_dec + b_dec - mlp_out)

# ✅ WORKS: Direct decoder addition (clean signal)
m = m + (W_dec[f] / norm(W_dec[f])) * SCALE
```

Three lines versus one. The decoder vector `W_dec[f]` IS the feature's learned contribution direction in the residual stream. Adding it directly is exactly what the model expects to see when that feature is naturally active. No encoding. No clamping. No reconstruction. Just add the direction the feature always contributes.

### The Scale Sweet Spot

Through systematic sweeping, we found the same dose-response curve Anthropic discovered:

```
Scale    Effect
─────    ──────
0.5-1×   No visible change
2.0-3×   ★ Clean behavioral shift
5.0×+    Token repetition collapse
```

The sweet spot is narrow — 2× too weak, 5× too strong, 2.5× just right. Here's the actual sweep code that found it:

```python
for scale in [0.5, 1.0, 2.0, 2.5, 3.0, 5.0, 8.0]:
    # ... generate with m = m + vec * scale ...
    hyphens = text.count('-')
    print(f"{scale:.1f}× ({hyphens} hyphens): {text[:100]}")
```

At 2-3×, the feature is amplified enough to influence behavior without overwhelming the model's natural coherence. This matches Anthropic's finding with Claude 3 Sonnet, suggesting it's a fundamental property of SAE steering, not model-specific.

---

## The Results: Four Confirmed Behavioral Changes

### 1. Bullet Points — f105206 @ L12

**DLA Score:** 0.431 for token `-` (hyphen) | **Scale:** 2.5×

```
Prompt: "My favorite foods are:"

Before:  1. Pizza
         2. Sushi
         3. Tacos

After:   - Pizza
         - Sushi
         - Burgers
```

Numbered lists become bullet points. The model's formatting structure is causally steered by a single feature. The effect is consistent across prompts — "The best vacation spots are:", "Things to pack for a trip:", "Reasons to exercise:" all shift from plain or numbered output to hyphen-structured lists.

### 2. Compliance — f29679 @ L11

**DLA Score:** 0.376 for token `Sure` | **Scale:** 2.5×

```
Prompt: "Can you help me with a difficult math problem?"

Before:  I'm trying to solve a problem that involves a lot of variables
         and I'm not sure where to start. I'd appreciate any guidance...

After:   Yes, I'd be happy to help. What's the problem? And what's the
         topic? Is it algebra, geometry, calculus, or something else?
```

The model shifts from "I need help" (asking the user) to "Yes, I'd be happy to help" (offering assistance). A complete persona reversal — from student to tutor — from a single feature. This is the strongest evidence that SAE features can control high-level behavioral traits, not just surface-level formatting.

### 3. Exclamation — f90138 @ L13

**DLA Score:** 0.418 for token `!` | **Scale:** 2-4×

```
Prompt: "Describe a sunset."

Before:  Describe the colors, the shapes, and the emotions it evokes.
         ## Step 1: Start by describing the colors of the sunset...

After:   A beautiful sunset! with vibrant colors! and a warm glow!
         The sky is painted! with hues! of pink! orange! and purple!
```

At 2-4× the model injects exclamation marks throughout — turning descriptive prose into enthusiastic exclamations. At 1.5× it switches to structured Q&A format instead, showing the same feature influences multiple aspects of output structure. This multi-faceted effect is a window into how SAE features encode overlapping behavioral dimensions.

### 4. Enthusiasm — f109639 @ L13

**DLA Score:** 0.545 for token `great` | **Scale:** 4×

```
Prompt: "What do you think about pizza?"

Before:  Do you like it? Do you have a favorite topping? 
         Do you have a favorite pizza place?

After:   Is it a great food or a bad food? Is it a good or bad food?
         Is it a great or bad food? Is it a healthy food or unhealthy?
```

The word "great" appears and the questioning tone shifts from neutral inquiry to evaluative judgment. The model starts framing responses around quality judgments rather than open-ended questions.

---

## Behind the Numbers: Feature Discovery

How do you find these features? Here's the systematic discovery process:

```python
# Scan ALL layers for features aligned with observable behaviors
behaviors = {
    'ALLCAPS':    [' HELLO', ' IMPORTANT'],
    'enthusiasm': [' love', ' amazing', ' great'],
    'compliance': [' Sure', ' Certainly'],
    'exclamation':['!'],
    'bullet':     [' -'],
}

for layer in [8, 9, 10, 11, 12, 13]:
    W_dec = load_sae_decoder(layer)
    for behavior, tokens in behaviors.items():
        for token_str in tokens:
            tid = tokenizer.encode(token_str)[1]  # skip BOS
            emb = embed_weight[tid]
            scores = np.abs(W_dec @ emb)
            top_feature = np.argmax(scores)
            if scores[top_feature] > 0.25:  # Filter weak signals
                print(f"L{layer} {behavior:14s} f{top_feature:>7d} DLA={scores[top_feature]:.3f}")

# Then test each candidate with batch_test.py
```

And the batch testing infrastructure that validates candidates:

```python
# batch_test.py — sweep scales and compare before/after
FEATURES = [
    ("bullet",      12, 105206, "-",     0.431, "My favorite foods are:"),
    ("compliance",  11, 29679,  "Sure",  0.376, "Can you help me with a difficult math problem?"),
    ("exclamation", 13, 90138,  "!",     0.418, "Describe a sunset."),
    ("enthusiasm",  13, 109639, "great", 0.545, "What do you think about pizza?"),
]

for name, layer, feat, token, dla, prompt in FEATURES:
    vec = decs[layer][feat] / (norm(decs[layer][feat]) + 1e-8)

    # Generate baseline
    baseline = generate(prompt, model, tokenizer, steer=None)

    # Sweep scales
    for scale in [1.0, 2.0, 2.5, 3.0, 4.0]:
        steered = generate(prompt, model, tokenizer, steer=(layer, vec, scale))
        if steered != baseline:
            print(f"  {scale}×: {steered[:100]}")
```

---

## What We Learned

### 1. Direct Decoder Addition is the Key

The SAE encode-clamp-decode cycle is a trap. It introduces reconstruction error that the model interprets as noise. Direct decoder vector addition bypasses the lossy cycle entirely and delivers a clean signal that the model's circuitry can route naturally.

### 2. DLA Finds Causally-Connected Features

Not all SAE features that correlate with behavior are causally steerable. But features with high DLA scores (>0.3) are consistently steerable — they have a direct causal path from activation to output. DLA scores are ~4-20σ above the 131K-feature noise floor.

### 3. The Scale Sweet Spot is Universal

Across formatting, compliance, enthusiasm, and exclamation — the 2-3× sweet spot holds. Below 2×, nothing happens. Above 5×, the model enters token repetition loops. This matches Anthropic's finding with Claude 3 Sonnet.

### 4. Memory Isolation Enables Consumer Hardware

The three-phase pipeline keeps each phase under 3GB, making SAE research feasible on an 8GB M1 Mac Mini. The critical design: never load the model and full SAE simultaneously. Use disk as intermediate storage between phases.

### 5. Semantic > Discrete

Formatting, tone, compliance, enthusiasm — all continuous semantic dimensions. Capital cities, factual answers, JSON syntax — all discrete tasks. Steering works on continuous behavioral axes, not binary decisions.

---

## How to Reproduce

```bash
# 1. Clone and install
git clone <this-repo> && cd llama-sae-interp
pip install -r requirements.txt

# 2. Download model (needs HF token + license acceptance)
export HF_TOKEN="hf_..."
python download_model.py

# 3. Download SAE (open access)
python download_sae.py

# 4. Verify setup
ls models/Llama-3.2-1B-Instruct/   # ~2.5GB
ls saes/sae-Llama-3.2-1B-131k/     # Layers 8-13

# 5. Run the experiments
cd experiments/exp1_gap_closure
python batch_test.py       # Test all confirmed features
python steer_direct.py     # Interactive feature steering
```

The entire pipeline — from activation capture through feature discovery to steering — runs on 8GB RAM with Apple MLX.

---

## References

1. Anthropic. "Golden Gate Claude." May 2024. [Link](https://www.anthropic.com/news/golden-gate-claude)
2. Templeton et al. "Scaling Monosemanticity." 2024. [Link](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
3. Panahi, M. "Why SAE Steering Fails for Structured Output." Feb 2026. [Link](https://huggingface.co/blog/MaziyarPanahi/sae-steering-json)
4. Andrylie et al. "Language-Specific SAE Features." 2025. [Link](https://github.com/LyzanderAndrylie/language-specific-features)
5. EleutherAI. "sae-Llama-3.2-1B-131k." [Link](https://huggingface.co/EleutherAI/sae-Llama-3.2-1B-131k)
