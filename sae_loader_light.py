"""Lightweight SAE forward pass in PyTorch (no MLX dependency).

Loads EleutherAI TopK SAE weights and runs encode/decode on MPS.
Memory: ~1GB per layer (float16) vs 2GB (float32 in MLX).
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


class TopKSAELight:
    """Single-layer TopK SAE using PyTorch tensors on MPS."""

    def __init__(self, sae_dir: Path, layer_idx: int, device: str = "mps"):
        self.layer_idx = layer_idx
        self.device = device

        sae_dir = Path(sae_dir)
        with open(sae_dir / "config.json") as f:
            self.config = json.load(f)

        self.k = self.config["sae"]["k"]
        self.num_latents = self.config["sae"]["num_latents"]

        layer_dir = sae_dir / f"layers.{layer_idx}.mlp"
        weights_path = layer_dir / "sae.safetensors"

        weights = {}
        with safe_open(str(weights_path), framework="np") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

        # Convert to torch float16 on device
        self.W_enc = torch.tensor(weights["encoder.weight"].T, dtype=torch.float16, device=device)
        self.W_dec = torch.tensor(weights["W_dec"], dtype=torch.float16, device=device)
        self.b_enc = torch.tensor(weights["encoder.bias"], dtype=torch.float16, device=device)
        self.b_dec = torch.tensor(weights["b_dec"], dtype=torch.float16, device=device)

        self.d_model = self.W_enc.shape[0]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to sparse features (TopK)."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to(dtype=self.W_enc.dtype, device=self.device)
        x_centered = x - self.b_dec
        latents = x_centered @ self.W_enc + self.b_enc
        # TopK
        _, top_idx = torch.topk(latents, k=self.k, dim=-1)
        features = torch.zeros(latents.shape[0], self.num_latents,
                               dtype=latents.dtype, device=self.device)
        features.scatter_(1, top_idx, latents.gather(1, top_idx))
        if features.shape[0] == 1:
            features = features[0]
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to residual stream space."""
        if features.ndim == 1:
            features = features.unsqueeze(0)
        reconstruction = features @ self.W_dec + self.b_dec
        if reconstruction.shape[0] == 1:
            reconstruction = reconstruction[0]
        return reconstruction

    def get_top_features(self, x: torch.Tensor, top_n: int = 10):
        """Top-N active feature indices and values."""
        features = self.encode(x)
        if features.ndim == 2:
            features = features[0]
        nonzero_mask = features > 0
        n_nonzero = int(nonzero_mask.sum())
        if n_nonzero == 0:
            return []
        actual_n = min(top_n, n_nonzero)
        _, top_idx = torch.topk(features.float(), k=actual_n)
        return [(int(i), float(features[i])) for i in top_idx]
