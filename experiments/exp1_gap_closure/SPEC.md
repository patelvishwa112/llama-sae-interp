# Exp1: SAE Gap Closure — Spec & Findings

## Status: 🟢 First Successful Steering Achieved

Feature 105206 (bullet point, L12) successfully steers model formatting at 2-2.5× scale using direct decoder vector addition. Numbered lists become bullet points. Comma-separated text becomes hyphen-structured.

## Pipeline Files (5 scripts)

| File | Purpose | Memory |
|------|---------|--------|
| `phase1_multi.py` | Capture MLP activations at target layers | ~2.5GB |
| `phase2_encode.py` | SAE-only feature encoding from saved activations | ~1GB |
| `phase2c_dla.py` | DLA-based steering direction computation | ~1.5GB |
| `phase3_causal.py` | Single-position DLA steering with multiplier sweep | ~2.5GB |
| `steer_direct.py` | **Autoregressive direct decoder steering** (what works) | ~3.5GB |

## What Works

- **Direct decoder addition:** `h += α · W_dec[f]` — no SAE encode/decode
- **DLA feature selection:** `score(f) = |W_dec[f] · embedding[token]|`
- **Scale sweet spot:** 2-3× for formatting features
- **Autoregressive application:** steer at every generation step
- **Formatting behaviors:** bullet points, lists, hyphen structure

## What Failed

- SAE encode→clamp→decode: reconstruction artifacts → degenerate collapse
- Single-position steering at L12-L13: attention ignores it
- Language features (SAE-LAPE): correlate but don't causally control
- Factual knowledge steering: discrete answers aren't on continuous semantic dimensions
- DLA-only features (dog, capital cities): push toward token but can't overcome model's natural output

## Key Design Decision

**Direct decoder addition instead of SAE clamping.** The SAE is lossy — encoding and decoding introduces 5-15% reconstruction error. When you clamp a feature, those artifacts get amplified. Adding W_dec[f] directly bypasses the lossy encode-decode cycle entirely.

## Open Questions

1. Can we find features for other behaviors? (ALL CAPS, sentiment, refusal, verbosity)
2. Does multi-feature simultaneous steering work? (bullet + enthusiasm)
3. Can we steer factual answers with a DIFFERENT method? (multi-layer, feature ablation)
4. Is Parameter Decomposition (Goodfire) better for discrete tasks than SAEs?
