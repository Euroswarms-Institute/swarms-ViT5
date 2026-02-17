"""
Compare our ViT-5 with the original repo (ViT-5/) on the same checkpoint and input.
Requires: ViT-5/ in the workspace, timm, and the original deps (see ViT-5/).
Usage: python scripts/compare_with_original.py <checkpoint.pth> [-v|--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Original repo (ViT-5/) must be on path for timm create_model + their models_vit5
ORIGINAL_DIR = ROOT / "ViT-5"
if not ORIGINAL_DIR.is_dir():
    print("Original repo not found at ViT-5/. Skipping comparison.")
    sys.exit(0)

sys.path.insert(0, str(ORIGINAL_DIR))

import torch


def _log(verbose: bool, msg: str) -> None:
    """Print only when verbose."""
    if verbose:
        print(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare our ViT-5 with original (ViT-5/) on same checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint .pth")
    parser.add_argument("-v", "--verbose", action="store_true", help="Expanded step-by-step logging")
    args = parser.parse_args()
    verbose = args.verbose
    ckpt_path = args.checkpoint
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    _log(verbose, f"Checkpoint: {ckpt_path}")

    try:
        import models_vit5  # noqa: F401 — registers vit5_* with timm
        from timm.models import create_model
        # Original ViT-5 rope.py uses .cuda() hardcoded; patch to use input device so it works on CPU too
        import rope
        _orig_rope_forward = rope.VisionRotaryEmbedding.forward

        def _rope_forward_device(self, x):
            import numpy as np
            from einops import repeat
            ft_seq_len = int(np.sqrt(x.shape[1]))
            t = torch.arange(ft_seq_len, device=x.device, dtype=x.dtype) / ft_seq_len * self.pt_seq_len
            f = self.freqs.to(x.device).to(x.dtype)
            freqs = torch.einsum("..., f -> ... f", t, f)
            freqs = repeat(freqs, "... n -> ... (n r)", r=2)
            freqs = rope.broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)
            freqs_cos = freqs.cos().view(-1, 1, freqs.shape[-1])
            freqs_sin = freqs.sin().view(-1, 1, freqs.shape[-1])
            return x * freqs_cos + rope.rotate_half(x) * freqs_sin

        rope.VisionRotaryEmbedding.forward = _rope_forward_device
        _log(verbose, "Loaded ViT-5/timm + patched RoPE for device-agnostic forward.")
    except ImportError as e:
        print(f"Need ViT-5/ on path and timm. {e}")
        return 1

    # Our model + remap
    from vit5 import remap_official_state_dict, vit5_base

    _log(verbose, "Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
        _log(verbose, "State dict from ckpt['model']")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        _log(verbose, "State dict from ckpt['state_dict']")
    else:
        sd = ckpt if isinstance(ckpt, dict) else getattr(ckpt, "state_dict", lambda: ckpt)()
        _log(verbose, "State dict from ckpt root / state_dict()")

    _log(verbose, f"State dict keys: {len(sd)}")
    our_model = vit5_base(img_size=224)
    remapped = remap_official_state_dict(sd)
    our_keys = set(our_model.state_dict().keys())
    filtered = {k: v for k, v in remapped.items() if k in our_keys}
    _log(verbose, f"Remapped: {len(remapped)} keys -> {len(filtered)} matched our model")
    result = our_model.load_state_dict(filtered, strict=False)
    # RoPE buffers (inv_freq) are deterministic from (dim, theta) and often not in checkpoint
    allowed_missing = {k for k in result.missing_keys if "rope" in k and k.endswith("inv_freq")}
    if set(result.missing_keys) != allowed_missing:
        raise RuntimeError(f"Unexpected missing keys: {set(result.missing_keys) - allowed_missing}")
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected keys: {result.unexpected_keys}")
    _log(verbose, f"Our model loaded; missing (allowed RoPE inv_freq): {len(allowed_missing)}")
    our_model.eval()

    # Original model (from ViT-5/, uses timm)
    try:
        _log(verbose, "Creating original timm vit5_base(224)...")
        orig_model = create_model("vit5_base", img_size=224, num_classes=1000)
    except Exception as e:
        print(f"Could not create original model (timm + ViT-5): {e}")
        return 1
    # Original checkpoint uses their key names; load without remap
    orig_model.load_state_dict(sd, strict=False)
    orig_model.eval()
    _log(verbose, "Original model loaded and eval().")

    torch.manual_seed(42)
    x = torch.randn(2, 3, 224, 224)
    _log(verbose, f"Input: x.shape={tuple(x.shape)}")
    with torch.no_grad():
        logits_ours = our_model(x)
        logits_orig = orig_model(x)

    # Verify we are comparing two different outputs (not the same tensor or hardcoded)
    assert logits_ours.data_ptr() != logits_orig.data_ptr(), "Bug: same tensor for both models"
    assert logits_ours.shape == logits_orig.shape == (2, 1000), "Unexpected logit shape"
    _log(verbose, f"Logits shape: {tuple(logits_ours.shape)}")

    cos = (logits_ours.float() * logits_orig.float()).sum() / (
        logits_ours.float().norm() * logits_orig.float().norm() + 1e-8
    )
    cos_val = cos.item()
    # Sanity: cosine must be in [-1, 1] (allow tiny numerical error)
    assert -1.0 - 1e-5 <= cos_val <= 1.0 + 1e-5, f"Cosine out of range: {cos_val}"

    if cos_val >= 0.999:
        if verbose:
            print(f"Logit cosine (ours vs original): {cos_val:.6f}")
            print("Parity OK (>= 0.999).")
        else:
            print(f"OK  cos={cos_val:.6f}  parity (ours vs original)")
    else:
        print(f"FAIL  cos={cos_val:.6f}  (expected >= 0.999 — check RoPE / token order / key mapping)")
        return 1

    # Sanity check: different input must yield different outputs (catch faked comparison)
    with torch.no_grad():
        logits_ours_other = our_model(x + 1.0)  # different input
    cos_wrong = (logits_ours.float() * logits_ours_other.float()).sum() / (
        logits_ours.float().norm() * logits_ours_other.float().norm() + 1e-8
    ).item()
    # Require strictly < 1 so we're not comparing identical tensors; 0.999 was too strict
    # (constant pixel shift can leave logit direction very similar)
    assert cos_wrong < 1.0 - 1e-5, "Sanity failed: different input should not match (comparison may be faked)"
    if verbose:
        print(f"Sanity: same model, different input -> cos={cos_wrong:.4f} (expected < 1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
