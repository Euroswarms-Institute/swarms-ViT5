"""
Stress tests for ViT-5: resolutions, large batch, mixed precision.
"""

from __future__ import annotations

import pytest
import torch

from vit5 import vit5_base, vit5_small


class TestResolution:
    """Different fixed input resolutions (model built for that resolution)."""

    def test_forward_resolution_224(self):
        model = vit5_base(img_size=224)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1000)

    def test_forward_resolution_384(self):
        model = vit5_base(img_size=384)
        model.eval()
        x = torch.randn(2, 3, 384, 384)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1000)


class TestLargeBatch:
    """Large-batch inference (no OOM, correct shape, no NaNs)."""

    def test_forward_large_batch(self):
        model = vit5_small()
        model.eval()
        batch_size = 32
        x = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, 1000)
        assert not torch.isnan(out).any() and not torch.isinf(out).any()


class TestMixedPrecision:
    """Autocast fp16 / bf16 forward pass."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_autocast_fp16(self):
        model = vit5_small().cuda()
        model.eval()
        x = torch.randn(2, 3, 224, 224, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                out = model(x)
        assert out.shape == (2, 1000)
        assert not torch.isnan(out).any() and not torch.isinf(out).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_autocast_bf16(self):
        model = vit5_small().cuda()
        model.eval()
        x = torch.randn(2, 3, 224, 224, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                out = model(x)
        assert out.shape == (2, 1000)
        assert not torch.isnan(out).any() and not torch.isinf(out).any()

    def test_forward_autocast_cpu_bf16(self):
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                pass
        except (RuntimeError, AttributeError):
            pytest.skip("CPU autocast bfloat16 not available")
        model = vit5_small()
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            with torch.no_grad():
                out = model(x)
        assert out.shape == (2, 1000)
        assert not torch.isnan(out).any() and not torch.isinf(out).any()
