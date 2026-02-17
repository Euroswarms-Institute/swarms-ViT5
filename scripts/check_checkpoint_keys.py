"""
Load an official ViT-5 checkpoint (.pth) and print or save its state_dict keys.
Usage: python scripts/check_checkpoint_keys.py [path_to_checkpoint.pth]
If no path given, prints where to download (Hugging Face FengWang3211/ViT-5).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect ViT-5 checkpoint keys")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Path to .pth checkpoint (e.g. vit5_base_patch16_224.pth)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional JSON file to write key list",
    )
    args = parser.parse_args()

    if not args.checkpoint:
        print("No checkpoint path given.")
        print("Download from: https://huggingface.co/FengWang3211/ViT-5")
        print("  e.g. vit5_base_patch16_224.pth")
        print("Usage: python scripts/check_checkpoint_keys.py <path_to.pth> [-o keys.json]")
        return 0

    path = Path(args.checkpoint)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        sd = getattr(ckpt, "state_dict", lambda: ckpt)()

    keys = sorted(sd.keys())
    print(f"Total keys: {len(keys)}")
    for k in keys:
        print(k)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(keys, f, indent=2)
        print(f"Wrote keys to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
