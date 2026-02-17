"""
Resolution-scaling benchmark: ViT-5-base vs vanilla ViT-B/16.

Empirically verifies the paper's resolution robustness claim (APE + 2D RoPE):
train at 224, evaluate at 224, 256, 384, 448. Compare ViT-5 with vanilla ViT-B/16
so that structural differences (2D RoPE + APE) are the main variable.

Data:
  - With --data-path: ImageNet-1k validation (path must contain val/ or be val/).
  - With --max-samples N: use only N samples for faster runs.
  - Without --data-path: synthetic mode (random inputs); metrics are not meaningful.

Checkpoints:
  - --vit5-ckpt: ViT-5-base 224 checkpoint (e.g. vit5_base_patch16_224.pth).
  - --vitb-ckpt: ViT-B/16 224 checkpoint (timm-style state dict).

Output: printed table (model, train_res, eval_res, acc1, acc5, loss) and optional
--output-csv. Success = ViT-5 retains better accuracy at higher resolutions than ViT-B/16.

Usage:
  # Full eval with both models (ImageNet val required):
  python scripts/benchmark_resolution.py \\
    --vit5-ckpt vit5_base_patch16_224.pth \\
    --vitb-ckpt /path/to/vit_b_16_224.pth \\
    --data-path /path/to/imagenet

  # Subset for quick run:
  python scripts/benchmark_resolution.py --vit5-ckpt ckpt.pth --vitb-ckpt vitb.pth \\
    --data-path /data/imagenet --max-samples 2000

  # Synthetic sanity run (no real metrics):
  python scripts/benchmark_resolution.py --vit5-ckpt ckpt.pth --vitb-ckpt vitb.pth
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

# -----------------------------------------------------------------------------
# Helpers: pos_embed interpolation
# -----------------------------------------------------------------------------

TRAIN_RES = 224
PATCH_SIZE = 16


def interpolate_pos_embed_vit5(
    pos_embed: torch.Tensor,
    target_num_patches: int,
) -> torch.Tensor:
    """
    ViT-5: pos_embed is patch-only (1, num_patches, dim). No extra tokens.
    Reshape to 2D grid, bicubic interpolate to target grid, flatten.
    """
    _, n, dim = pos_embed.shape
    orig_size = int(n ** 0.5)
    if orig_size * orig_size != n:
        raise ValueError(f"pos_embed length {n} is not a perfect square")
    new_size = int(target_num_patches ** 0.5)
    if new_size * new_size != target_num_patches:
        raise ValueError(f"target_num_patches {target_num_patches} is not a perfect square")
    # (1, n, dim) -> (1, orig_size, orig_size, dim) -> interpolate -> (1, new_size*new_size, dim)
    pe = pos_embed.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
    pe = F.interpolate(pe, size=(new_size, new_size), mode="bicubic", align_corners=False)
    pe = pe.permute(0, 2, 3, 1).flatten(1, 2)
    return pe


def interpolate_pos_embed_timm(
    pos_embed: torch.Tensor,
    target_num_patches: int,
    num_extra_tokens: int = 1,
) -> torch.Tensor:
    """
    Timm ViT: pos_embed = [class_token, patch_tokens], shape (1, 1 + num_patches, dim).
    Interpolate only the patch part; keep extra token(s) unchanged.
    """
    _, seq_len, dim = pos_embed.shape
    num_patch_tokens = seq_len - num_extra_tokens
    if num_patch_tokens <= 0:
        raise ValueError(f"pos_embed seq_len {seq_len} <= num_extra_tokens {num_extra_tokens}")
    orig_size = int(num_patch_tokens ** 0.5)
    if orig_size * orig_size != num_patch_tokens:
        raise ValueError(f"num_patch_tokens {num_patch_tokens} is not a perfect square")
    new_size = int(target_num_patches ** 0.5)
    if new_size * new_size != target_num_patches:
        raise ValueError(f"target_num_patches {target_num_patches} is not a perfect square")
    extra = pos_embed[:, :num_extra_tokens, :]
    patch_tokens = pos_embed[:, num_extra_tokens:, :]
    patch_tokens = patch_tokens.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
    patch_tokens = F.interpolate(
        patch_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
    )
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    return torch.cat([extra, patch_tokens], dim=1)


# -----------------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------------


def _check_not_lfs_pointer(path: Path) -> None:
    """Raise if path is a Git LFS pointer (small text) instead of real weights."""
    path = Path(path)
    if not path.exists() or path.stat().st_size > 1000:
        return
    try:
        with open(path, "rb") as f:
            head = f.read(100)
    except OSError:
        return
    if b"git-lfs" in head or head.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer; the real weights were not downloaded. "
            "Run in that repo: git lfs pull (requires git-lfs installed)"
        )


def _parse_ckpt_to_state_dict(ckpt: Any) -> Dict[str, Any]:
    """Extract state dict from loaded checkpoint (PyTorch or wrapper)."""
    if isinstance(ckpt, dict) and "model" in ckpt:
        return dict(ckpt["model"])
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return dict(ckpt["state_dict"])
    if isinstance(ckpt, dict):
        return dict(ckpt)
    return dict(getattr(ckpt, "state_dict", lambda: ckpt)())


def load_checkpoint_state_dict(ckpt_path: Path) -> Dict[str, Any]:
    """Load state dict from checkpoint; support .pth/.bin (pickle) and .safetensors."""
    ckpt_path = Path(ckpt_path)
    # SafeTensors (e.g. Hugging Face model.safetensors)
    if ckpt_path.suffix == ".safetensors" or ckpt_path.name == "model.safetensors":
        _check_not_lfs_pointer(ckpt_path)
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError("Loading .safetensors requires: pip install safetensors") from None
        try:
            return dict(load_file(str(ckpt_path)))
        except Exception as e:
            if "header too large" in str(e) or "deserializing header" in str(e):
                raise RuntimeError(
                    f"{ckpt_path} may be a Git LFS pointer (weights not pulled). "
                    "Run in the cloned repo: git lfs pull"
                ) from e
            raise
    # PyTorch pickle
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        if "UnpicklingError" in type(e).__name__ or "invalid load key" in str(e):
            raise RuntimeError(
                f"Checkpoint is not PyTorch pickle format (e.g. Hugging Face may use SafeTensors). "
                f"Use the .safetensors file instead: --vitb-ckpt vit_base_patch16_224.augreg2_in21k_ft_in1k/model.safetensors "
                f"(requires: pip install safetensors)"
            ) from e
        raise
    return _parse_ckpt_to_state_dict(ckpt)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_val_transform(resolution: int) -> torch.nn.Module:
    """Validation transform: Resize to resolution, CenterCrop, ToTensor, normalize."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_imagenet_val(
    data_path: Path,
    resolution: int,
    max_samples: Optional[int],
    batch_size: int,
) -> DataLoader:
    """Build ImageFolder val loader at given resolution."""
    from torchvision.datasets import ImageFolder
    transform = build_val_transform(resolution)
    dataset = ImageFolder(str(data_path), transform=transform)
    if max_samples is not None and max_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:max_samples].tolist()
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


class SyntheticDataset(Dataset):
    """Random tensors + dummy labels for sanity runs."""

    def __init__(self, resolution: int, num_samples: int = 1000, num_classes: int = 1000):
        self.resolution = resolution
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        x = torch.randn(3, self.resolution, self.resolution)
        # Normalize like ImageNet for model input
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std.clamp(min=1e-6)
        y = index % self.num_classes
        return x, y


def build_synthetic_loader(
    resolution: int,
    num_samples: int,
    batch_size: int,
) -> DataLoader:
    dataset = SyntheticDataset(resolution, num_samples=num_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> List[torch.Tensor]:
    """Top-k accuracy; returns list of scalar tensors (one per k)."""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.unsqueeze(0).expand_as(pred))
        return [correct[:k].float().sum().mul_(100.0 / output.size(0)) for k in topk]


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    synthetic: bool,
    verbose: bool = False,
    run_label: str = "",
) -> Dict[str, float]:
    """Run evaluation; return acc1, acc5, loss. If synthetic, metrics are meaningless but computed."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    n = 0
    total_batches = len(loader)
    log_every = max(1, total_batches // 10) if verbose else 0
    for batch_idx, (images, target) in enumerate(loader):
        if verbose and log_every and batch_idx % log_every == 0 and batch_idx > 0:
            print(f"  {run_label} batch {batch_idx}/{total_batches} ({n} samples)", flush=True)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        out = model(images)
        loss = criterion(out, target)
        acc1, acc5 = accuracy(out, target, topk=(1, 5))
        b = images.size(0)
        total_loss += loss.item() * b
        total_acc1 += acc1.item() * b
        total_acc5 += acc5.item() * b
        n += b
    if verbose and run_label:
        print(f"  {run_label} done: {n} samples, acc1={total_acc1/n:.2f} acc5={total_acc5/n:.2f}", flush=True)
    if n == 0:
        return {"acc1": 0.0, "acc5": 0.0, "loss": 0.0}
    return {
        "acc1": total_acc1 / n,
        "acc5": total_acc5 / n,
        "loss": total_loss / n,
    }


# -----------------------------------------------------------------------------
# Model loading at resolution
# -----------------------------------------------------------------------------


def load_vit5_at_resolution(
    ckpt_path: Path,
    eval_res: int,
    device: torch.device,
    train_res: int = TRAIN_RES,
) -> torch.nn.Module:
    """Build vit5_base at eval_res, load 224 checkpoint with pos_embed interpolation."""
    from vit5 import remap_official_state_dict, vit5_base

    sd = load_checkpoint_state_dict(ckpt_path)
    remapped = remap_official_state_dict(sd)
    model = vit5_base(img_size=eval_res)
    our_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in remapped.items() if k in our_keys}

    # Interpolate pos_embed if resolution changed
    if eval_res != train_res and "pos_embed" in filtered:
        num_patches_train = (train_res // PATCH_SIZE) ** 2
        num_patches_eval = (eval_res // PATCH_SIZE) ** 2
        pe = filtered["pos_embed"]
        if pe.shape[1] != num_patches_eval:
            filtered["pos_embed"] = interpolate_pos_embed_vit5(pe, num_patches_eval)

    result = model.load_state_dict(filtered, strict=False)
    allowed_missing = {k for k in result.missing_keys if "rope" in k and k.endswith("inv_freq")}
    if set(result.missing_keys) != allowed_missing:
        raise RuntimeError(f"Unexpected missing keys: {set(result.missing_keys) - allowed_missing}")
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected keys: {result.unexpected_keys}")
    model.to(device)
    model.eval()
    return model


def load_vitb_at_resolution(
    ckpt_path: Path,
    eval_res: int,
    device: torch.device,
    train_res: int = TRAIN_RES,
) -> torch.nn.Module:
    """Build timm vit_base_patch16_224 at eval_res, load checkpoint with pos_embed interpolation."""
    from timm.models import create_model

    sd = load_checkpoint_state_dict(ckpt_path)
    # Strip module. prefix if present
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model = create_model("vit_base_patch16_224", img_size=eval_res, num_classes=1000, pretrained=False)
    num_patches_eval = (eval_res // PATCH_SIZE) ** 2
    num_patches_train = (train_res // PATCH_SIZE) ** 2
    if "pos_embed" in sd and sd["pos_embed"].shape[1] != model.pos_embed.shape[1]:
        # Timm: 1 class token + patch tokens
        sd["pos_embed"] = interpolate_pos_embed_timm(
            sd["pos_embed"],
            target_num_patches=num_patches_eval,
            num_extra_tokens=1,
        )
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run_benchmark(
    vit5_ckpt: Optional[Path],
    vitb_ckpt: Optional[Path],
    data_path: Optional[Path],
    resolutions: List[int],
    batch_size: int,
    device: torch.device,
    max_samples: Optional[int],
    output_csv: Optional[Path],
    synthetic_num_samples: int,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Run evaluation for each model and resolution; return list of result rows."""
    synthetic = data_path is None
    val_path: Optional[Path] = None
    if synthetic:
        print("Synthetic mode (no --data-path): metrics are not meaningful.")
    else:
        val_path = data_path / "val" if (data_path / "val").is_dir() else data_path
        if not val_path.is_dir():
            raise FileNotFoundError(f"Data path not found or missing val dir: {val_path}")

    results: List[Dict[str, Any]] = []

    for res in resolutions:
        if synthetic:
            loader = build_synthetic_loader(res, num_samples=synthetic_num_samples, batch_size=batch_size)
        else:
            assert val_path is not None
            loader = build_imagenet_val(val_path, res, max_samples, batch_size)
        num_samples = len(loader.dataset)
        if verbose:
            print(f"\n[Resolution {res}] dataset size: {num_samples} samples, {len(loader)} batches (batch_size={batch_size})", flush=True)

        if vit5_ckpt is not None and vit5_ckpt.exists():
            if verbose:
                print(f"  Loading ViT-5-base @ {res}...", flush=True)
            model = load_vit5_at_resolution(vit5_ckpt, res, device)
            if verbose:
                print(f"  Eval ViT-5-base @ {res}...", flush=True)
            stats = evaluate_loader(
                model, loader, device, synthetic,
                verbose=verbose,
                run_label=f"ViT-5 @{res}",
            )
            results.append({
                "model": "ViT-5-base",
                "train_res": TRAIN_RES,
                "eval_res": res,
                "acc1": stats["acc1"],
                "acc5": stats["acc5"],
                "loss": stats["loss"],
            })
            if synthetic:
                results[-1]["note"] = "synthetic"

        if vitb_ckpt is not None and vitb_ckpt.exists():
            if verbose:
                print(f"  Loading ViT-B/16 @ {res}...", flush=True)
            model = load_vitb_at_resolution(vitb_ckpt, res, device)
            if verbose:
                print(f"  Eval ViT-B/16 @ {res}...", flush=True)
            stats = evaluate_loader(
                model, loader, device, synthetic,
                verbose=verbose,
                run_label=f"ViT-B @{res}",
            )
            results.append({
                "model": "ViT-B/16",
                "train_res": TRAIN_RES,
                "eval_res": res,
                "acc1": stats["acc1"],
                "acc5": stats["acc5"],
                "loss": stats["loss"],
            })
            if synthetic:
                results[-1]["note"] = "synthetic"

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolution-scaling benchmark: ViT-5 vs ViT-B/16 at 224, 256, 384, 448.",
    )
    parser.add_argument("--vit5-ckpt", type=Path, default=None, help="Path to ViT-5-base 224 checkpoint")
    parser.add_argument("--vitb-ckpt", type=Path, default=None, help="Path to ViT-B/16 224 checkpoint")
    parser.add_argument("--data-path", type=Path, default=None, help="ImageNet root (must contain val/). Omit for synthetic.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit val samples (subset eval)")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[224, 256, 384, 448], help="Eval resolutions")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-csv", type=Path, default=None, help="Write results to CSV")
    parser.add_argument("--synthetic-samples", type=int, default=500, help="Samples in synthetic mode")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print progress (per resolution, per model, batch progress)")
    args = parser.parse_args()

    if args.vit5_ckpt is None and args.vitb_ckpt is None:
        print("Provide at least one of --vit5-ckpt or --vitb-ckpt.")
        return 1
    if args.vit5_ckpt is not None and not args.vit5_ckpt.exists():
        print(f"ViT-5 checkpoint not found: {args.vit5_ckpt}")
        return 1
    if args.vitb_ckpt is not None and not args.vitb_ckpt.exists():
        print(f"ViT-B checkpoint not found: {args.vitb_ckpt}")
        return 1

    device = torch.device(args.device)
    if args.verbose:
        print(f"Device: {device}", flush=True)
        print(f"Resolutions: {args.resolutions}", flush=True)
    results = run_benchmark(
        vit5_ckpt=args.vit5_ckpt,
        vitb_ckpt=args.vitb_ckpt,
        data_path=args.data_path,
        resolutions=args.resolutions,
        batch_size=args.batch_size,
        device=device,
        max_samples=args.max_samples,
        output_csv=args.output_csv,
        synthetic_num_samples=args.synthetic_samples,
        verbose=args.verbose,
    )

    # Print table
    print("\nResolution scaling benchmark")
    print("-" * 80)
    print(f"{'model':<12} {'train_res':<10} {'eval_res':<10} {'acc1':<8} {'acc5':<8} {'loss':<8}")
    print("-" * 80)
    for r in results:
        note = f" ({r.get('note', '')})" if r.get("note") else ""
        print(f"{r['model']:<12} {r['train_res']:<10} {r['eval_res']:<10} {r['acc1']:.3f}    {r['acc5']:.3f}    {r['loss']:.4f}{note}")
    print("-" * 80)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "train_res", "eval_res", "acc1", "acc5", "loss"])
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k) for k in w.fieldnames if k in r and r[k] is not None})
        print(f"Wrote {args.output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
