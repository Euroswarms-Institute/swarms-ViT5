"""
Tests for ViT-5 single-file implementation.
Run: pytest tests/ -v
"""

from __future__ import annotations

import pytest
import torch

from vit5 import ViT5, vit5_small, vit5_base, vit5_large, vit5_xlarge


# Expected param counts (paper / official)
PARAM_COUNTS = {
    "small": 22e6,
    "base": 87e6,
    "large": 304e6,
    "xlarge": 449e6,
}


@pytest.fixture(params=["small", "base", "large", "xlarge"])
def model_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def builders():
    return {
        "small": vit5_small,
        "base": vit5_base,
        "large": vit5_large,
        "xlarge": vit5_xlarge,
    }


@pytest.fixture
def batch():
    return torch.randn(2, 3, 224, 224)


class TestForward:
    """Forward pass and output shape."""

    def test_forward_shape(self, model_name: str, builders: dict, batch: torch.Tensor) -> None:
        model = builders[model_name]()
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (2, 1000), f"expected (2, 1000), got {out.shape}"

    def test_forward_single_sample(self, builders: dict) -> None:
        model = vit5_base()
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1000)

    def test_forward_features_shape(self, builders: dict, batch: torch.Tensor) -> None:
        model = vit5_base()
        model.eval()
        with torch.no_grad():
            feats = model.forward_features(batch)
        num_patches = (224 // 16) ** 2
        num_registers = 4
        expected_seq = 1 + num_patches + num_registers
        assert feats.shape == (2, expected_seq, 768), f"got {feats.shape}"


class TestParamCount:
    """Parameter counts vs paper."""

    def test_param_count_close(self, model_name: str, builders: dict) -> None:
        model = builders[model_name]()
        n = sum(p.numel() for p in model.parameters())
        ref = PARAM_COUNTS[model_name]
        # Allow ~2% tolerance (e.g. 87M vs 86.5M)
        assert abs(n - ref) < 0.02 * ref, f"{model_name}: got {n/1e6:.2f}M, ref ~{ref/1e6:.0f}M"


class TestGradient:
    """Gradients flow and backward pass."""

    def test_backward(self, builders: dict, batch: torch.Tensor) -> None:
        model = vit5_small()
        model.train()
        out = model(batch)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"nan grad in {name}"

    def test_grad_flow_to_embed(self, batch: torch.Tensor) -> None:
        model = vit5_small()
        model.train()
        out = model(batch)
        out.sum().backward()
        assert model.patch_embed.proj.weight.grad is not None
        assert not torch.isnan(model.patch_embed.proj.weight.grad).any()


class TestCompile:
    """torch.compile (PyTorch 2.x). Skips if inductor needs C++ compiler and it is missing."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"),
        reason="torch.compile requires PyTorch 2.0+",
    )
    def test_compile_forward(self, batch: torch.Tensor) -> None:
        model = vit5_small()
        model.eval()
        compiled = torch.compile(model, mode="default")
        try:
            with torch.no_grad():
                out = compiled(batch)
        except Exception as e:
            if "Compiler" in str(e) or "cl" in str(e) or "Inductor" in str(e):
                pytest.skip(f"torch.compile needs C++ compiler: {e}")
            raise
        assert out.shape == (2, 1000)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"),
        reason="torch.compile requires PyTorch 2.0+",
    )
    def test_compile_no_crash(self, batch: torch.Tensor) -> None:
        model = vit5_base()
        model.eval()
        compiled = torch.compile(model, mode="default")
        try:
            with torch.no_grad():
                _ = compiled(batch)
        except Exception as e:
            if "Compiler" in str(e) or "cl" in str(e) or "Inductor" in str(e):
                pytest.skip(f"torch.compile needs C++ compiler: {e}")
            raise


class TestStateDict:
    """Save/load and state_dict."""

    def test_state_dict_roundtrip(self, builders: dict) -> None:
        model = vit5_small()
        sd = model.state_dict()
        model2 = vit5_small()
        model2.load_state_dict(sd, strict=True)
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2 and torch.allclose(p1, p2), f"mismatch at {n1}"

    def test_state_dict_keys(self) -> None:
        model = vit5_base()
        keys = set(model.state_dict().keys())
        assert "patch_embed.proj.weight" in keys
        assert "cls_token" in keys
        assert "reg_token" in keys
        assert "pos_embed" in keys
        assert "blocks.0.norm1.weight" in keys
        assert "blocks.0.attn.qkv.weight" in keys
        assert "blocks.0.ls1.gamma" in keys and "blocks.0.ls2.gamma" in keys
        assert "norm.weight" in keys
        assert "head.weight" in keys


class TestDeterminism:
    """Eval determinism (no dropout)."""

    def test_eval_deterministic(self, batch: torch.Tensor) -> None:
        model = vit5_small()
        model.eval()
        with torch.no_grad():
            out1 = model(batch)
            out2 = model(batch)
        assert torch.allclose(out1, out2), "eval forward should be deterministic"


class TestNumClasses:
    """Custom num_classes (head output dim)."""

    def test_num_classes_zero(self) -> None:
        model = ViT5(embed_dim=384, depth=2, num_heads=6, num_classes=0)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 384)

    def test_num_classes_custom(self) -> None:
        model = vit5_small(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)
