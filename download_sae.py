#!/usr/bin/env python3
"""Download the SAE model from HuggingFace.
The SAE is open-access — no license required.

Usage:
  python3 download_sae.py
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

SAE_ID = "EleutherAI/sae-Llama-3.2-1B-131k"
LOCAL_DIR = Path(__file__).parent / "saes" / "sae-Llama-3.2-1B-131k"

def main():
    print(f"Downloading {SAE_ID}...")
    print(f"Destination: {LOCAL_DIR}")

    try:
        path = snapshot_download(
            SAE_ID,
            local_dir=str(LOCAL_DIR),
        )
        print(f"\n✓ SAE downloaded to: {path}")

        total_size = sum(f.stat().st_size for f in LOCAL_DIR.rglob("*") if f.is_file())
        print(f"Total size: {total_size / (1024**2):.0f} MB")
        print(f"Files:")
        for f in sorted(LOCAL_DIR.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(LOCAL_DIR)} ({f.stat().st_size / (1024**2):.0f} MB)")

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
