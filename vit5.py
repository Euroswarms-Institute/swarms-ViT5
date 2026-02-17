"""
ViT-5: Vision Transformers for The Mid-2020s — single-file PyTorch implementation.

Model definition only (no training scripts). Strict parity with official repo for
checkpoint loading and training. Dependencies: PyTorch, einops.

Checkpoint loading: Parameter names match timm-style (patch_embed.proj.*, cls_token,
reg_token, pos_embed, blocks.i.norm1.weight, blocks.i.attn.qkv.weight, q_norm/k_norm,
gamma_1/gamma_2, norm2, mlp.fc1/fc2, norm.weight, head.*). If loading official/HF
checkpoints saved from the timm-based repo, key names should align; use
load_state_dict(ckpt, strict=False) and inspect missing/unexpected keys if needed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# -----------------------------------------------------------------------------
# 1. RMSNorm
# -----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no recentering)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


# -----------------------------------------------------------------------------
# 2. LayerScale
# -----------------------------------------------------------------------------


class LayerScale(nn.Module):
    """Learnable per-channel scale after residual branch (init 1e-4)."""

    def __init__(self, dim: int, init_value: float = 1e-4) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


# -----------------------------------------------------------------------------
# 3. PatchEmbed
# -----------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Patch embedding via Conv2d: (B,C,H,W) -> (B, N, D)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, D, H', W') -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# -----------------------------------------------------------------------------
# 4. 2D Rotary Position Embedding
# -----------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the channels for RoPE."""
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def _broadcat(freqs_list: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """Broadcast and concatenate frequency tensors along dim."""
    num = len(freqs_list)
    shapes = [f.shape for f in freqs_list]
    ndim = len(shapes[0])
    dim = (dim + ndim) if dim < 0 else dim
    expanded = []
    for i, f in enumerate(freqs_list):
        target_shape = list(f.shape)
        for d in range(ndim):
            if d != dim:
                target_shape[d] = max(s[d] for s in shapes)
        expanded.append(f.expand(target_shape))
    return torch.cat(expanded, dim=dim)


class RotaryEmbedding2D(nn.Module):
    """
    2D rotary position embedding for vision.
    dim: head_dim // 2 (half dim per axis).
    pt_seq_len: spatial grid size (e.g. 14 for 14x14 patches, or 2 for 2x2 registers).
    theta: frequency base (10000 for patches, 100 for registers).
    """

    def __init__(
        self,
        dim: int,
        pt_seq_len: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.theta = theta
        # Official: inv_freq = 1/(theta ** (arange(0, dim, 2).float() / dim)); dim = head_dim//2
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        if inv_freq.numel() == 0:
            inv_freq = torch.ones(1)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, num_heads, head_dim); seq_len = pt_seq_len ** 2
        B, seq_len, num_heads, head_dim = x.shape
        n = int(math.isqrt(seq_len))
        assert n * n == seq_len, "seq_len must be a perfect square"
        device = x.device
        t = torch.arange(n, device=device, dtype=x.dtype) / n * self.pt_seq_len
        # t: (n,), inv_freq: (n_freqs,) -> freqs: (n, n_freqs)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.dtype))
        # repeat for rotation pairs -> (n, head_dim//2)
        freqs = repeat(freqs, "n f -> n (f r)", r=2)
        # 2D grid: (n, n, head_dim)
        freqs = _broadcat([freqs[:, None, :], freqs[None, :, :]], dim=-1)
        freqs_cos = freqs.cos().view(-1, 1, head_dim)
        freqs_sin = freqs.sin().view(-1, 1, head_dim)
        return x * freqs_cos + _rotate_half(x) * freqs_sin


# -----------------------------------------------------------------------------
# 5. Attention (QK-RMSNorm, dual RoPE, no QKV bias)
# -----------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head self-attention with QK-RMSNorm and 2D RoPE (patch + register)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = True,
        rope: Optional[RotaryEmbedding2D] = None,
        rope_reg: Optional[RotaryEmbedding2D] = None,
        num_registers: int = 0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_registers = num_registers
        self.rope = rope
        self.rope_reg = rope_reg

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        reg_idx = N - self.num_registers

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # q, k, v: (B, N, num_heads, head_dim)

        if self.qk_norm:
            qk_dtype = q.dtype
            q = self.q_norm(q).to(qk_dtype)
            k = self.k_norm(k).to(qk_dtype)

        if self.rope is not None:
            # cls: :1, patches: 1:reg_idx, regs: reg_idx:
            q_cls, q_patch, q_reg = q[:, :1], q[:, 1:reg_idx], q[:, reg_idx:]
            k_cls, k_patch, k_reg = k[:, :1], k[:, 1:reg_idx], k[:, reg_idx:]
            q_patch = self.rope(q_patch)
            k_patch = self.rope(k_patch)
            if self.rope_reg is not None and q_reg.numel() > 0:
                q_reg = self.rope_reg(q_reg)
                k_reg = self.rope_reg(k_reg)
            q = torch.cat([q_cls, q_patch, q_reg], dim=1)
            k = torch.cat([k_cls, k_patch, k_reg], dim=1)

        q = q.transpose(1, 2) * self.scale
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


# -----------------------------------------------------------------------------
# 6. MLP (GeLU, no SwiGLU)
# -----------------------------------------------------------------------------


class MLP(nn.Module):
    """Feed-forward: Linear -> GELU -> Linear."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


# -----------------------------------------------------------------------------
# 7. DropPath
# -----------------------------------------------------------------------------


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    mask = mask.div(keep_prob)
    return x * mask


class DropPath(nn.Module):
    """Drop paths per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# -----------------------------------------------------------------------------
# 8. Block
# -----------------------------------------------------------------------------


class Block(nn.Module):
    """Transformer block: Norm -> Attn -> LayerScale -> residual; same for MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float = 1e-4,
        rope: Optional[RotaryEmbedding2D] = None,
        rope_reg: Optional[RotaryEmbedding2D] = None,
        num_registers: int = 0,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
            rope_reg=rope_reg,
            num_registers=num_registers,
        )
        self.ls1 = LayerScale(dim, init_value=init_values)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.ls2 = LayerScale(dim, init_value=init_values)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# -----------------------------------------------------------------------------
# 9. ViT5
# -----------------------------------------------------------------------------


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02) -> None:
    """Truncated normal init."""
    nn.init.trunc_normal_(tensor, mean=mean, std=std)


class ViT5(nn.Module):
    """
    ViT-5: [cls, patches, registers], APE on patches only, classify from cls.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type = RMSNorm,
        init_scale: float = 1e-4,
        num_registers: int = 4,
        rope: bool = True,
        reg_theta: float = 100.0,
        qk_norm: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_registers = num_registers

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, num_registers, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # RoPE: patch grid and register grid
        rope_patch_size = img_size // patch_size
        rope_reg_size = int(num_registers ** 0.5)
        assert rope_reg_size * rope_reg_size == num_registers

        head_dim = embed_dim // num_heads
        dim_rope = head_dim // 2

        rope_module: Optional[RotaryEmbedding2D] = None
        rope_reg_module: Optional[RotaryEmbedding2D] = None
        if rope:
            rope_module = RotaryEmbedding2D(dim_rope, rope_patch_size, theta=10000.0)
            rope_reg_module = RotaryEmbedding2D(dim_rope, rope_reg_size, theta=reg_theta)

        dpr = [drop_path_rate] * depth
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=0.0,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                init_values=init_scale,
                rope=rope_module,
                rope_reg=rope_reg_module,
                num_registers=num_registers,
                qk_norm=qk_norm,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.drop_rate = drop_rate

        self._init_weights()

    def _init_weights(self) -> None:
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.reg_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, RMSNorm):
            nn.init.constant_(m.weight, 1.0)

    def forward_features(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        reg_tokens = self.reg_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x, reg_tokens], dim=1)
        intermediates: List[torch.Tensor] = []
        for blk in self.blocks:
            x = blk(x)
            if return_intermediates:
                intermediates.append(x)
        x = self.norm(x)
        if return_intermediates:
            return x, intermediates
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = x[:, 0]
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.head(x)


# -----------------------------------------------------------------------------
# 10. Factory builders
# -----------------------------------------------------------------------------


def vit5_small(
    img_size: int = 224,
    patch_size: int = 16,
    num_classes: int = 1000,
    **kwargs: object,
) -> ViT5:
    """ViT-5-Small: 12 layers, 384 dim, 6 heads, 22M params."""
    return ViT5(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=num_classes,
        qkv_bias=False,
        num_registers=4,
        init_scale=1e-4,
        rope=True,
        reg_theta=100.0,
        qk_norm=True,
        **kwargs,
    )


def vit5_base(
    img_size: int = 224,
    patch_size: int = 16,
    num_classes: int = 1000,
    **kwargs: object,
) -> ViT5:
    """ViT-5-Base: 12 layers, 768 dim, 12 heads, 87M params."""
    return ViT5(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=num_classes,
        qkv_bias=False,
        num_registers=4,
        init_scale=1e-4,
        rope=True,
        reg_theta=100.0,
        qk_norm=True,
        **kwargs,
    )


def vit5_large(
    img_size: int = 224,
    patch_size: int = 16,
    num_classes: int = 1000,
    **kwargs: object,
) -> ViT5:
    """ViT-5-Large: 24 layers, 1024 dim, 16 heads, 304M params."""
    return ViT5(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=num_classes,
        qkv_bias=False,
        num_registers=4,
        init_scale=1e-4,
        rope=True,
        reg_theta=100.0,
        qk_norm=True,
        **kwargs,
    )


def vit5_xlarge(
    img_size: int = 224,
    patch_size: int = 16,
    num_classes: int = 1000,
    **kwargs: object,
) -> ViT5:
    """ViT-5-XL: 28 layers, 1152 dim, 16 heads, 449M params."""
    return ViT5(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=num_classes,
        qkv_bias=False,
        num_registers=4,
        init_scale=1e-4,
        rope=True,
        reg_theta=100.0,
        qk_norm=True,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# Checkpoint loading: remap official (timm-style) state_dict to our names
# -----------------------------------------------------------------------------


def remap_official_state_dict(sd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remap official ViT-5 (timm-style) checkpoint keys to our model's names.
    Official Block uses gamma_1 / gamma_2; we use ls1.gamma / ls2.gamma.
    Strips "module." prefix if present (e.g. from DataParallel).
    """
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[7:]
        if ".gamma_1" in new_k:
            new_k = new_k.replace(".gamma_1", ".ls1.gamma")
        elif ".gamma_2" in new_k:
            new_k = new_k.replace(".gamma_2", ".ls2.gamma")
        out[new_k] = v
    return out
