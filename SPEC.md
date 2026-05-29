# SPEC: Llama 3.2 1B SAE Interpretability

**Project:** `~/Projects/llama-sae-interp/`
**Created:** 2026-05-29
**Status:** Setup phase — SAEs downloading, model pending HF token

---

## 1. Objective

Use EleutherAI's pre-trained TopK SAEs on Llama 3.2 1B Instruct to understand what the model is "trying to do" — not just decode static features, but trace behavioral circuits, identify competing mechanisms, and test causal steering. Build on the ghost experiment's central finding: the encoding-deployment gap exists at small scale and may be a circuit-level competition rather than a representation failure.

---

## 2. Model & SAE Details

| Property | Value |
|----------|-------|
| Model | `meta-llama/Llama-3.2-1B-Instruct` |
| Parameters | 1.23B |
| Precision | BF16 (~2.4GB) |
| Layers | 16 transformer layers |
| Hidden dim | 2,048 |
| MLP intermediate | 8,192 |
| Context | 128K tokens |
| Architecture | GQA, RoPE, SwiGLU |

| SAE Property | Value |
|-------------|-------|
| Source | `EleutherAI/sae-Llama-3.2-1B-131k` |
| Type | TopK (k=32), MLP output hookpoints |
| Features | 131,072 per layer (64× expansion) |
| Layers | All 16 (layers.0.mlp through layers.15.mlp) |
| Training data | RedPajama v2, 8.2B tokens |
| Format | Safetensors, float32 |
| Memory | ~2GB per layer (load one at a time) |

| Total memory budget | |
|-------------------|---|
| Model (BF16) | ~2.4 GB |
| SAE (1 layer, float32) | ~2.0 GB |
| Activations + KV cache | ~0.5-1.0 GB |
| **Total** | **~5-6 GB** ✓ fits in 8GB |

---

## 3. Which Layers to Target — Evidence from Anthropic and Prior Work

### 3.1 What the Ghost Experiment Tells Us (Qwen2.5-0.5B, 24 layers)

| Layer Range | What Happens | Experiment Implication |
|------------|--------------|----------------------|
| L0-10 (early) | Token-level features, syntax. Not probed in ghost experiment | Not initially targeted — low semantic content |
| L14 (mid) | **Persona peaks** (31% probe accuracy). Model decides "who it is" | **Target for Experiment #3** (persona steering) |
| L20-23 (late-mid) | **Factual encoding peaks** (79.7% at L21). Answer computed but NOT deployed (4.3% causal) | **Primary target** — where the encoding-deployment gap lives |
| L23-24 (late) | Slight causal efficacy uptick (6.5%). The answer starts leaking through | **Target for Experiment #4** — what changes between L21 and L24? |

**Key insight:** The encoding-deployment gap is maximal at L20-22. Something between L21 (knowing) and L24 (saying) is suppressing or rerouting the answer signal. This is the circuit we need to trace.

### 3.2 What Anthropic's Circuit Tracing Tells Us (Claude 3.5 Haiku)

From the **Circuit Tracing methods paper** (Anthropic, Mar 2025) — the most detailed source on which layers contain which types of features:

#### Anthropic's Feature Taxonomy by Layer

Anthropic identifies FOUR distinct feature types, each with a specific layer distribution:

| Feature Type | Description | Layer Distribution | Example |
|-------------|------------|-------------------|---------|
| **Input features** | Represent low-level text properties (tokens, phrases) | Primarily **early** layers, but also present in mid/late | "digital" token feature |
| **Abstract features** | Represent conceptual properties of context | **Middle and later** layers | "danger of mixing cleaning chemicals" |
| **Functional features** | Perform computations, transform information | **Middle and later** layers | "add 9" feature that causes output = input + 9 |
| **Output features** | Promote specific output tokens | **Late** layers | features that promote "the", "and", etc. |

**This is the most explicit layer guidance Anthropic has published.** The paper states directly:

> *"Input features...are common in early layers, and output features...are common in late layers."*
> *"Features whose activations represent more abstract properties...appear in middle and later layers."*
> *"Features that perform functions...tend to be found in middle and later layers."*

#### Why Anthropic Replaces MLPs (and why our MLP SAEs are well-targeted)

Anthropic's circuit tracing method uses **Cross-Layer Transcoders (CLTs)** that replace ALL MLP layers. Each CLT feature reads from the residual stream at one layer and contributes to all subsequent MLP outputs. The key insight:

> *"We substitute a more interpretable component (a 'cross-layer transcoder') for the multi-layer perceptrons."*

They replace MLPs — not attention — because **MLPs are where the interpretable computation lives.** Attention layers handle routing (which tokens to look at); MLPs handle transformation (what to do with that information). Our EleutherAI SAEs, trained on MLP outputs at every layer, are positioned exactly where Anthropic finds the most interpretable features.

#### The Steering Plateau Effect

A critical finding for our experiments: when Anthropic steered an "Analytics" feature, the effect **plateaued before layer 13 and then DECREASED toward the final layer.** This is because:
- Features that promote an output (like "say an acronym") compete with each other
- If you amplify only one feature, the other competing features dilute the effect in late layers
- **Successful steering requires understanding the full circuit, not just one feature**

This directly explains the ghost experiment's encoding-deployment gap: the probe finds the answer feature, but competing output-routing features in late layers override it.

### 3.3 Community Findings: MLP vs Attention vs Residual Stream SAEs

From independent replications and extensions of Anthropic's work:

| SAE Type | Interpretability | Circuit Analysis Utility | Source |
|----------|-----------------|------------------------|--------|
| **Residual stream SAEs** | Highest interpretability in early-mid layers. Drops in late layers | Captures full information flow but features are mixtures of MLP+attention | OpenReview finding: "works less well for later model layers" |
| **MLP output SAEs** (ours) | Good for functional and abstract features | "MLP layers are most relevant to computing a residual stream feature" | Alignment Forum: attention-output SAE paper |
| **Attention output SAEs** | Also produce interpretable features | Reveal attention-level features that MLP SAEs miss | LessWrong: "Sparse Autoencoders Work on Attention Layer Outputs" |
| **Cross-Layer Transcoders** (Anthropic) | Best for circuit tracing | Reads from residual at one layer, writes to ALL subsequent MLPs | Circuit Tracing methods paper |

**Key limitation of our MLP-only SAEs:** We cannot trace attention-level features (induction heads, copy patterns, token-position routing). For experiments #2 (refusal circuit) and #4 (override tracing), attention-level features may be crucial — refusal might be an attention pattern, not an MLP computation. Mitigation: focus on experiments where MLP features are most informative (#1 gap closure, #3 persona steering).

### 3.4 What Anthropic's Feature Steering Tells Us (Claude 3 Sonnet)

| Finding | Layer Implication |
|---------|------------------|
| Steering sweet spot: ±5 factor doesn't degrade MMLU | There's slack in the system — features can be amplified without breaking the model |
| Off-target effects are real and unpredictable | Steering a "gender bias" feature affects age bias. Features are correlated, not isolated |
| Neutrality feature reduces bias across ALL 9 dimensions | Some features have broad, beneficial effects — finding analogous features in Llama 3.2 is high-value |
| Feature activation context ≠ resulting behavior | Don't trust what a feature fires on to predict what steering it will do. Test causally |

### 3.5 Layer Targeting Strategy for Llama 3.2 1B (16 layers)

Mapping from Qwen 0.5B (24 layers) to Llama 3.2 1B (16 layers), approximate scale factor ~0.67:

| Llama Layer | Anthropic Feature Type | Equivalent Qwen Range | Expected Content | Priority |
|------------|----------------------|----------------------|------------------|----------|
| **L0-4** | Input features (token-level) | Early (0-7) | Token/syntax, local patterns | P4 (future) |
| **L5-8** | Abstract features begin | Mid-early (8-13) | Semantic composition, entity binding, concepts | P3 |
| **L9-10** | Abstract + Functional features | Mid (~14) | **Persona formation, abstract concepts** (≈L14 peak) | **HIGH — Exp #3** |
| **L11-13** | Functional features dominate | Late-mid (17-21) | **Factual computation, functional transformations** (≈L20-21) | **HIGHEST — Exp #1, #2** |
| **L14-15** | Output features dominate | Late (22-24) | **Output routing, token promotion, deployment** (≈L23-24) | **HIGH — Exp #4** |

**For the first experiment (#1 — Gap Closure), we target L11-13** where the factual encoding-deployment gap is expected to peak. Then expand outward.

---

## 4. Experiment Plan (Prioritized)

### Phase 1: Setup & Validation (CURRENT)

- [ ] Download SAEs (in progress: 11/16 SAE layers)
- [ ] Download Llama 3.2 1B Instruct (needs HF token)
- [ ] Implement activation extraction from Llama 3.2 (MLX forward pass with hook points)
- [ ] Verify SAE features are meaningful on real activations (not just random noise)
- [ ] Build factual question dataset (port from ghost experiment: 500 questions)

### Phase 2: Experiment #1 — Gap Closure (EST: 4-6 hrs)

**Question:** Do SAE features close the encoding-deployment gap vs linear probes?

**Target layers:** L9-L15 (6 layers spanning persona through output)

**Approach:**
1. Run 500 factual questions through Llama 3.2, extract MLP outputs at each target layer
2. For each layer's SAE: encode activations, find features that correlate with correct answer
3. Compute encoding score: how well do top-k SAE features predict the answer?
4. Compute causal efficacy: clamp/amplify answer-correlated features, measure output change
5. Compare SAE feature gap to linear probe gap from ghost experiment
6. Control: random SAE baseline (shuffle feature indices)

**Success:** SAE features show >20% causal efficacy (vs linear probe's 4.3%)

### Phase 3: Experiment #2 — Default Refusal Circuit (EST: 4-6 hrs)

**Question:** Does Llama 3.2 have a "default refusal" or "hesitation" circuit?

**Target layers:** L11-L15 (where the encoding-deployment gap lives)

**Approach:**
1. Run easy factual (should answer) vs impossible (should refuse) questions
2. Compare SAE feature activation patterns between conditions
3. Identify "refusal" features: active on impossible Qs, inactive on easy Qs
4. Ablate refusal features on impossible Qs → does model hallucinate?
5. Amplify "known entity" features on impossible Qs → does model answer confidently?

**Success:** Finding a feature whose ablation causes hallucination — proving a default behavior circuit

### Phase 4: Experiment #3 — Persona Steering (EST: 3-5 hrs)

**Question:** Can SAE features steer persona more precisely than linear probes?

**Target layers:** L8-L11 (persona formation region)

**Approach:**
1. Run helpful vs unhelpful persona prompts
2. Identify SAE features that differentially activate between personas
3. Test steering: amplify helpful features on unhelpful-prompted model
4. Measure specificity: does persona change while preserving factual accuracy?

**Success:** >50% persona compliance while preserving >70% factual accuracy

### Phase 5: Experiment #4 — Planning/Override (EST: 4-8 hrs)

**Question:** Is there an active "suppress" feature between encoding and output?

**Target layers:** L11-L15 (spanning encoding peak to output)

**Approach:**
1. For a single high-gap factual prompt, trace SAE features across L11→L15
2. Identify: when does the answer feature activate? When does a "suppress/hedge" feature activate?
3. Ablate suppress feature → does "The." become "2."?

**Success:** Finding a specific feature whose ablation causes the answer to deploy correctly

---

## 5. Technical Approach

### 5.1 Activation Extraction

Use MLX's forward pass with hook points at each MLP output. For Llama 3.2:

```python
# Hook into each transformer layer's MLP output
# Layer structure: attention → add_residual → MLP → add_residual
# We hook at the MLP output (before residual add), matching EleutherAI's hookpoint
```

### 5.2 SAE Forward Pass

```python
sae = TopKSAE(sae_dir, layer_idx=N)
features = sae.encode(mlp_output)     # [batch, 131072] — sparse (k=32)
reconstruction = sae.decode(features)  # [batch, 2048] — back to residual stream
```

### 5.3 Causal Intervention

For steering: add/subtract feature direction to activations.

```python
# Amplify feature F by factor S at layer L
mlp_output_modified = mlp_output + S * sae.W_dec[F, :]
```

For ablation: project out feature direction.

### 5.4 Evaluation

- Encoding score: `accuracy(probe(features) == correct_answer)`
- Causal efficacy: `ΔP(correct_answer | steered) - ΔP(correct_answer | baseline)`
- Output analysis: compare first-token distribution changes

---

## 6. File Structure

```
llama-sae-interp/
├── README.md
├── SPEC.md                    ← this file
├── .gitignore
├── sae_loader.py              ← TopKSAE class (verified)
├── verify_sae.py              ← SAE verification tests (5/6 passing)
├── download_model.py          ← Llama 3.2 downloader (needs HF token)
├── saes/                      ← gitignored (~32GB)
│   ├── sae-Llama-3.2-1B-131k/
│   └── skip-transcoder-Llama-3.2-1B-131k/
├── models/                    ← gitignored (~2.4GB)
│   └── Llama-3.2-1B-Instruct/
├── data/
│   ├── factual_questions.jsonl    ← 500 factual Qs (from ghost experiment)
│   ├── impossible_questions.jsonl  ← 100 impossible Qs
│   ├── persona_prompts.jsonl      ← helpful/unhelpful persona prompts
│   └── reasoning_questions.jsonl  ← multi-step reasoning Qs
└── experiments/
    ├── exp1_gap_closure/
    │   ├── run.py
    │   └── results/
    ├── exp2_default_refusal/
    ├── exp3_persona_steering/
    └── exp4_planning_override/
```

---

## 7. Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| SAEs are MLP-output only (not residual stream) | Limits circuit tracing but still captures MLP-level features. Anthropic found MLP features are sufficient for many behavioral circuits |
| Llama 3.2 1B Instruct may not show the same encoding-deployment gap as Qwen 0.5B | This is a finding either way — if gap doesn't exist at 1B, that tells us something about scale |
| TopK SAE (k=32) may miss important features with <32 active | Anthropic used TopK with similar k. The 131K feature dictionary is large enough that important features should be in top-32 |
| Off-target effects from feature steering (Anthropic's finding) | Test specificity explicitly in every experiment. Don't assume a "factual answer" feature only affects factual answers |
| M1 8GB memory ceiling | Load one SAE layer at a time. Batch size 1-4. Use MLX's memory-efficient operations |

---

## 8. References

- EleutherAI SAE: https://huggingface.co/EleutherAI/sae-Llama-3.2-1B-131k
- Anthropic "Tracing the Thoughts": https://www.anthropic.com/research/tracing-thoughts-language-model
- Anthropic "Evaluating Feature Steering": https://www.anthropic.com/research/evaluating-feature-steering
- Ghost experiment: `~/Projects/ghost-in-residual-stream/`
- Wiki proposals: `~/Documents/Obsidian/wiki/beyond-ghost-sae-experiment-proposals.md`
- Qwen-Scope: https://qwen.ai/blog?id=qwen-scope
