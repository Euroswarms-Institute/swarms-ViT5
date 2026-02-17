"""
Parity and checkpoint compatibility tests for ViT-5.
Requires an official checkpoint for full tests; skips if not found.
Set VIT5_CHECKPOINT_PATH to a .pth file (e.g. vit5_base_patch16_224.pth).
Download: https://huggingface.co/FengWang3211/ViT-5
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from vit5 import remap_official_state_dict, vit5_base


def _get_checkpoint_path() -> Path | None:
    path = os.environ.get("VIT5_CHECKPOINT_PATH")
    if path and Path(path).exists():
        return Path(path)
    # Default locations
    for candidate in [
        Path(__file__).resolve().parent.parent / "vit5_base_patch16_224.pth",
        Path(__file__).resolve().parent / "fixtures" / "vit5_base_patch16_224.pth",
    ]:
        if candidate.exists():
            return candidate
    return None


def _load_checkpoint_state_dict(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    return getattr(ckpt, "state_dict", lambda: ckpt)()


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    a_flat = a.float().flatten(1)
    b_flat = b.float().flatten(1)
    return (a_flat * b_flat).sum(dim=dim) / (
        a_flat.norm(dim=dim, p=2) * b_flat.norm(dim=dim, p=2) + 1e-8
    )


@pytest.fixture(scope="module")
def checkpoint_path():
    return _get_checkpoint_path()


@pytest.fixture(scope="module")
def loaded_state_dict(checkpoint_path):
    if checkpoint_path is None:
        return None
    return _load_checkpoint_state_dict(checkpoint_path)


@pytest.fixture(scope="module")
def remapped_state_dict(loaded_state_dict):
    if loaded_state_dict is None:
        return None
    return remap_official_state_dict(loaded_state_dict)


class TestCheckpointLoad:
    """Load official checkpoint with key remapping."""

    @pytest.mark.skipif(
        _get_checkpoint_path() is None,
        reason="No official checkpoint; set VIT5_CHECKPOINT_PATH or place vit5_base_patch16_224.pth",
    )
    def test_load_remapped_checkpoint_strict(self, remapped_state_dict):
        model = vit5_base(img_size=224)
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in remapped_state_dict.items() if k in model_keys}
        result = model.load_state_dict(filtered, strict=False)
        # RoPE buffers (inv_freq) are deterministic and often not saved in checkpoint
        allowed_missing = {k for k in result.missing_keys if "rope" in k and k.endswith("inv_freq")}
        assert set(result.missing_keys) == allowed_missing, f"unexpected missing: {set(result.missing_keys) - allowed_missing}"
        assert len(result.unexpected_keys) == 0, f"unexpected: {result.unexpected_keys}"


class TestLogitParity:
    """Fixed-seed logit comparison (determinism and/or vs reference)."""

    @pytest.mark.skipif(
        _get_checkpoint_path() is None,
        reason="No official checkpoint for logit test",
    )
    def test_logits_deterministic_with_loaded_weights(self, remapped_state_dict):
        torch.manual_seed(42)
        x = torch.randn(2, 3, 224, 224)
        model = vit5_base(img_size=224)
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in remapped_state_dict.items() if k in model_keys}
        model.load_state_dict(filtered, strict=False)
        model.eval()
        with torch.no_grad():
            logits1 = model(x)
        torch.manual_seed(42)
        x2 = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits2 = model(x2)
        assert torch.allclose(x, x2), "Same seed should give same input"
        cos = _cosine_similarity(logits1, logits2).mean().item()
        assert cos >= 0.999, f"Determinism check: cos_sim={cos}"

    @pytest.mark.skipif(
        _get_checkpoint_path() is None,
        reason="No official checkpoint for logit test",
    )
    def test_logits_identical_same_input_twice(self, remapped_state_dict):
        model = vit5_base(img_size=224)
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in remapped_state_dict.items() if k in model_keys}
        model.load_state_dict(filtered, strict=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits1 = model(x)
            logits2 = model(x)
        assert torch.allclose(logits1, logits2), "Same input should give same logits"


class TestBlockIntermediates:
    """Intermediate block output cosine similarity (structural parity)."""

    @pytest.mark.skipif(
        _get_checkpoint_path() is None,
        reason="No official checkpoint for intermediate test",
    )
    def test_block_intermediates_cosine_similarity_self(self, remapped_state_dict):
        torch.manual_seed(42)
        model = vit5_base(img_size=224)
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in remapped_state_dict.items() if k in model_keys}
        model.load_state_dict(filtered, strict=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            _, intermediates1 = model.forward_features(x, return_intermediates=True)
        with torch.no_grad():
            _, intermediates2 = model.forward_features(x, return_intermediates=True)
        for i, (a, b) in enumerate(zip(intermediates1, intermediates2)):
            cos = _cosine_similarity(a, b).mean().item()
            assert cos >= 0.999, f"Block {i} cos_sim={cos}"

    @pytest.mark.skipif(
        _get_checkpoint_path() is None,
        reason="No official checkpoint for intermediate test",
    )
    def test_block_intermediates_deterministic_across_runs(self, remapped_state_dict):
        model = vit5_base(img_size=224)
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in remapped_state_dict.items() if k in model_keys}
        model.load_state_dict(filtered, strict=False)
        model.eval()
        torch.manual_seed(123)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            _, inter1 = model.forward_features(x, return_intermediates=True)
        torch.manual_seed(123)
        x2 = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            _, inter2 = model.forward_features(x2, return_intermediates=True)
        for i, (a, b) in enumerate(zip(inter1, inter2)):
            cos = _cosine_similarity(a, b).mean().item()
            assert cos >= 0.999, f"Block {i} determinism cos_sim={cos}"
