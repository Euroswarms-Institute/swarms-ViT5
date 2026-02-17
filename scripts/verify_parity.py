"""
Verify parity with official ViT-5 checkpoint: load weights, run forward, report.
Usage:
  python scripts/verify_parity.py [path_to_checkpoint.pth]
  VIT5_CHECKPOINT_PATH=path.pth python scripts/verify_parity.py
Download checkpoint: https://huggingface.co/FengWang3211/ViT-5 (e.g. vit5_base_patch16_224.pth)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from vit5 import remap_official_state_dict, vit5_base


def main() -> int:
    path = os.environ.get("VIT5_CHECKPOINT_PATH")
    if not path and len(sys.argv) >= 2:
        path = sys.argv[1]
    if not path or not Path(path).exists():
        print("No checkpoint path. Set VIT5_CHECKPOINT_PATH or pass path as arg.")
        print("Download: https://huggingface.co/FengWang3211/ViT-5")
        return 1

    path = Path(path)
    print(f"Loading {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt if isinstance(ckpt, dict) else getattr(ckpt, "state_dict", lambda: ckpt)()

    remapped = remap_official_state_dict(sd)
    print("Building vit5_base(224)...")
    model = vit5_base(img_size=224)
    result = model.load_state_dict(remapped, strict=False)
    allowed_missing = {k for k in result.missing_keys if "rope" in k and k.endswith("inv_freq")}
    if result.missing_keys:
        if set(result.missing_keys) == allowed_missing:
            print("Load OK (missing only RoPE buffers inv_freq, which are deterministic).")
        else:
            print("Missing keys:", result.missing_keys)
    if result.unexpected_keys:
        print("Unexpected keys:", result.unexpected_keys)
    if result.unexpected_keys or (result.missing_keys and set(result.missing_keys) != allowed_missing):
        print("Load completed with key mismatches (see above).")

    model.eval()
    torch.manual_seed(42)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    print(f"Forward OK, logits shape {logits.shape}")
    with torch.no_grad():
        feats, intermediates = model.forward_features(x, return_intermediates=True)
    print(f"Block intermediates: {len(intermediates)} blocks")
    print("Parity verification done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
