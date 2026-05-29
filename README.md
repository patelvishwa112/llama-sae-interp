# Llama 3.2 1B SAE Interpretability

Experiments using EleutherAI sparse autoencoders on Llama 3.2 1B Instruct
for mechanistic interpretability and behavioral steering.

## Setup

- Model: meta-llama/Llama-3.2-1B-Instruct (BF16)
- SAEs: EleutherAI/sae-Llama-3.2-1B-131k (16 layers, MLP outputs, 131K features)
- Framework: MLX + PyTorch (SAEs in safetensors)
- Hardware: Apple M1 Mac Mini, 8GB RAM

## Structure

```
data/           — prompts, evaluation datasets
experiments/    — per-experiment scripts and results
sae_loader.py   — SAE loading and forward pass
verify.py       — verification script
```

## Experiments

See [[../wiki/beyond-ghost-sae-experiment-proposals|wiki proposals]] for full details.

