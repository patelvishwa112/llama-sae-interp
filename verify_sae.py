"""Verify SAE loading and basic operations.

Tests:
  1. SAE weight loading from safetensors
  2. SAE dimensions match expected values
  3. Forward pass (encode + decode) on random activations
  4. TopK sparsity correct (k=32 active features)
  5. Memory usage estimate
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from sae_loader import TopKSAE, load_sae_suite


def test_single_layer():
    """Test loading and forward pass for a single layer SAE."""
    sae_dir = Path("saes/sae-Llama-3.2-1B-131k")

    print("=== Test 1: Loading layer 0 SAE ===")
    sae = TopKSAE(sae_dir, layer_idx=0)
    print(f"  {sae}")
    print(f"  W_enc shape: {sae.W_enc.shape}")
    print(f"  W_dec shape: {sae.W_dec.shape}")
    print(f"  b_enc shape: {sae.b_enc.shape}")
    print(f"  b_dec shape: {sae.b_dec.shape}")

    # Verify dimensions
    d_model = sae.W_enc.shape[0]
    n_latents = sae.W_enc.shape[1]
    assert d_model == 2048, f"Expected d_model=2048, got {d_model}"
    assert n_latents == 131072, f"Expected n_latents=131072, got {n_latents}"
    print(f"  ✓ Dimensions correct: d_model={d_model}, n_latents={n_latents}")

    # Verify decoder normalization
    dec_norms = mx.linalg.norm(sae.W_dec, axis=1)  # [n_latents]
    norm_mean = float(dec_norms.mean())
    norm_close = float((mx.abs(dec_norms - 1.0) < 0.01).mean())
    print(f"  Decoder norm: mean={norm_mean:.4f}, within 1% of 1.0: {norm_close:.1%}")
    assert norm_close > 0.95, f"Decoder columns not normalized"

    print("\n=== Test 2: Forward pass on random activations ===")
    # Simulate MLP output activations (residual stream dimension)
    x = mx.random.normal((1, d_model))

    features, reconstruction = sae.forward(x)

    n_active = int((features > 0).sum())
    print(f"  Input shape: {x.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Active features: {n_active}")
    print(f"  Reconstruction shape: {reconstruction.shape}")

    # Verify sparsity
    if features.ndim == 2:
        n_active_per_sample = (features > 0).sum(axis=1)
        for i, n in enumerate(n_active_per_sample.tolist()):
            print(f"  Sample {i}: {int(n)} active features")
            assert int(n) == sae.k, f"Expected k={sae.k} active, got {int(n)}"

    print(f"  ✓ Forward pass successful, sparsity correct")

    print("\n=== Test 3: Top feature extraction ===")
    top = sae.get_top_features(x, top_n=5)
    print(f"  Top 5 features:")
    for feat_id, activation in top:
        print(f"    Feature {feat_id}: {activation:.4f}")

    print("\n=== Test 4: Reconstruction fidelity ===")
    mse = float(((x - reconstruction) ** 2).mean())
    var = float(x.var())
    explained = 1.0 - mse / var
    print(f"  MSE: {mse:.6f}")
    print(f"  Variance explained: {explained:.2%}")

    print("\n=== Test 5: Memory estimate ===")
    w_enc_mb = sae.W_enc.size * 4 / (1024 * 1024)  # float32 in MLX
    w_dec_mb = sae.W_dec.size * 4 / (1024 * 1024)
    print(f"  W_enc: {w_enc_mb:.0f} MB")
    print(f"  W_dec: {w_dec_mb:.0f} MB")
    print(f"  Total SAE (one layer): {(w_enc_mb + w_dec_mb):.0f} MB")
    print(f"  Total SAE (16 layers): {(w_enc_mb + w_dec_mb) * 16:.0f} MB")

    return True


def test_multi_layer():
    """Test loading multiple layers."""
    sae_dir = Path("saes/sae-Llama-3.2-1B-131k")

    print("\n=== Test 6: Loading layers 0, 7, 15 ===")
    saes = load_sae_suite(sae_dir, layers=[0, 7, 15])
    for idx, sae in saes.items():
        print(f"  Layer {idx}: {sae}")
        # Quick forward pass
        x = mx.random.normal((1, sae.d_model))
        f, _ = sae.forward(x)
        n_active = int((f > 0).sum())
        print(f"    Forward pass OK, {n_active} active features")

    print("  ✓ Multi-layer loading successful")
    return True


if __name__ == "__main__":
    try:
        test_single_layer()
        test_multi_layer()
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED")
        print("=" * 50)
    except FileNotFoundError as e:
        print(f"\n⚠ SAE files not found: {e}")
        print("Downloads may still be in progress. Run again when complete.")
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
