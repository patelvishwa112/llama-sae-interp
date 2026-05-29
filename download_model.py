#!/usr/bin/env python3
"""Download Llama 3.2 1B Instruct model.

REQUIRES:
  1. Accept the license: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
  2. Set HF_TOKEN env var or run: huggingface-cli login

Usage:
  python3 download_model.py
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
LOCAL_DIR = Path(__file__).parent / "models" / "Llama-3.2-1B-Instruct"


def main():
    # Check for token
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        # Check if logged in via CLI
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()

    if not token:
        print("ERROR: No HuggingFace token found.")
        print("\nSteps:")
        print("1. Accept the Llama 3.2 license at:")
        print("   https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
        print("2. Create a token at https://huggingface.co/settings/tokens")
        print("3. Run: export HF_TOKEN='hf_...'")
        print("   Or:  huggingface-cli login")
        sys.exit(1)

    print(f"Downloading {MODEL_ID}...")
    print(f"Destination: {LOCAL_DIR}")

    try:
        path = snapshot_download(
            MODEL_ID,
            local_dir=str(LOCAL_DIR),
            token=token,
            ignore_patterns=["original/*", "*.pth"],  # skip original Meta format
        )
        print(f"\n✓ Model downloaded to: {path}")

        # List downloaded files
        total_size = sum(f.stat().st_size for f in LOCAL_DIR.rglob("*") if f.is_file())
        print(f"Total size: {total_size / (1024**3):.1f} GB")
        print(f"Files:")
        for f in sorted(LOCAL_DIR.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(LOCAL_DIR)} ({f.stat().st_size / (1024**2):.0f} MB)")

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nCommon issues:")
        print("- License not accepted (visit the model page and click 'Agree')")
        print("- Invalid token")
        print("- Rate limiting (try again in a few minutes)")
        sys.exit(1)


if __name__ == "__main__":
    main()
