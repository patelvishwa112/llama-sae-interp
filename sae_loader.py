"""SAE loader and forward pass for EleutherAI TopK SAEs on Llama 3.2 1B.

SAE format (per layer, from EleutherAI/sae-Llama-3.2-1B-131k):
  - TopK activation: only top k=32 latents are active
  - normalize_decoder: True (decoder columns are unit-norm)
  - expansion_factor: 64x (131,072 latents / 2,048 d_model)
  - Hookpoint: layers.{N}.mlp (MLP output, residual stream dimension)
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open


class TopKSAE:
    """Single-layer TopK Sparse Autoencoder loaded from safetensors."""

    def __init__(self, sae_dir: Path, layer_idx: int):
        self.layer_idx = layer_idx
        self.sae_dir = Path(sae_dir)

        # Load config
        with open(self.sae_dir / "config.json") as f:
            self.config = json.load(f)

        self.k = self.config["sae"]["k"]
        self.num_latents = self.config["sae"]["num_latents"]
        self.normalize_decoder = self.config["sae"]["normalize_decoder"]

        # Load weights from safetensors
        layer_dir = self.sae_dir / f"layers.{layer_idx}.mlp"
        weights_path = layer_dir / "sae.safetensors"

        weights = {}
        with safe_open(str(weights_path), framework="np") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

        # SAE weight matrices (stored as [n_latents, d_model] in safetensors)
        # W_enc: [d_model, num_latents] — encoder (transposed from stored)
        # W_dec: [num_latents, d_model] — decoder (column-normalized)
        # b_dec: [d_model] — decoder bias (also used as pre-encoder bias)
        # b_enc: [num_latents] — encoder bias
        self.W_enc = mx.array(weights["encoder.weight"].T)  # [n_latents, d_model] -> [d_model, n_latents]
        self.W_dec = mx.array(weights["W_dec"])              # [n_latents, d_model]
        self.b_enc = mx.array(weights["encoder.bias"])       # [n_latents]
        self.b_dec = mx.array(weights["b_dec"])              # [d_model]

        self.d_model = self.W_enc.shape[0]

    def encode(self, x: mx.array) -> mx.array:
        """Encode activations to sparse feature activations (TopK).

        Args:
            x: [batch, d_model] or [d_model] — residual stream activations

        Returns:
            features: [batch, num_latents] — sparse (k nonzeros per row)
        """
        if x.ndim == 1:
            x = x[None, :]  # [d_model] -> [1, d_model]

        # Pre-encoder bias subtraction
        x_centered = x - self.b_dec  # [batch, d_model]

        # Encoder: latents = x_centered @ W_enc + b_enc
        latents = x_centered @ self.W_enc + self.b_enc  # [batch, n_latents]

        # TopK activation (mx.topk returns values only in MLX 0.31, use argsort)
        top_idx = mx.argsort(latents, axis=-1)[:, -self.k:]  # [batch, k] — indices of top k
        # Gather top values
        batch_indices = mx.arange(latents.shape[0])[:, None]
        top_vals = latents[batch_indices, top_idx]  # [batch, k]

        # Create sparse mask (init zeros, then add values at TopK indices)
        features = mx.zeros((latents.shape[0], self.num_latents))
        features = features.at[batch_indices, top_idx].add(top_vals)

        if features.shape[0] == 1:
            features = features[0]

        return features

    def decode(self, features: mx.array) -> mx.array:
        """Decode feature activations back to residual stream space.

        Args:
            features: [batch, num_latents] or [num_latents]

        Returns:
            reconstruction: [batch, d_model] or [d_model]
        """
        if features.ndim == 1:
            features = features[None, :]

        reconstruction = features @ self.W_dec + self.b_dec  # [batch, d_model]

        if reconstruction.shape[0] == 1:
            reconstruction = reconstruction[0]

        return reconstruction

    def forward(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Full SAE forward pass: encode → decode.

        Returns:
            features: sparse latent activations
            reconstruction: decoded approximation of x
        """
        features = self.encode(x)
        reconstruction = self.decode(features)
        return features, reconstruction

    def get_top_features(self, x: mx.array, top_n: int = 10) -> list[tuple[int, float]]:
        """Get the top-N active feature indices and their activation values."""
        features = self.encode(x)
        if features.ndim == 2:
            features = features[0]

        # Get top-k indices among active features (the only nonzero ones)
        nonzero_mask = features > 0
        n_nonzero = int(nonzero_mask.sum())
        if n_nonzero == 0:
            return []

        actual_n = min(top_n, n_nonzero)
        # argsort descending: sort indices by feature value
        sort_idx = mx.argsort(features, axis=-1)[::-1]
        top_feat_idx = sort_idx[:actual_n]
        top_feat_vals = features[top_feat_idx]

        return [(int(i), float(v)) for i, v in zip(top_feat_idx.tolist(), top_feat_vals.tolist())]

    def __repr__(self):
        return f"TopKSAE(layer={self.layer_idx}, d_model={self.d_model}, n_latents={self.num_latents}, k={self.k})"


def load_sae_suite(sae_dir: str | Path, layers: list[int] | None = None) -> dict[int, TopKSAE]:
    """Load SAEs for specified layers.

    Args:
        sae_dir: path to downloaded SAE directory
        layers: list of layer indices to load (default: all 16)

    Returns:
        dict mapping layer_idx -> TopKSAE
    """
    sae_dir = Path(sae_dir)
    with open(sae_dir / "config.json") as f:
        config = json.load(f)

    total_layers = max(
        int(h.split(".")[1]) for h in config["hookpoints"]
    ) + 1

    if layers is None:
        layers = list(range(total_layers))

    saes = {}
    for layer_idx in layers:
        saes[layer_idx] = TopKSAE(sae_dir, layer_idx)

    return saes
