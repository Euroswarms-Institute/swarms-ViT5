"""
Train linear probe once at 224, then evaluate at 224 and higher resolutions.

Recommended immediate workflow:
  1. Freeze backbone, replace head with Linear(768 → 100).
  2. Train linear probe for 20 epochs on ImageNet-100 at 224.
  3. Compare top-1 at 224.
  4. Evaluate the same probe at 384 (and optionally 256, 448).

Produces meaningful resolution-robustness signal within hours. Requires ImageNet-100
(data path with train/ and val/ and 100 classes).

Usage:
  python scripts/benchmark_linear_probe_train_once.py \\
    --vit5-ckpt vit5_base_patch16_224.pth \\
    --vitb-ckpt path/to/vitb.safetensors \\
    --data-path /path/to/imagenet100 \\
    --probe-epochs 20
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

NUM_CLASSES = 100
TRAIN_RES = 224
PATCH_SIZE = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEAT_DIM = 768

# Reuse checkpoint + pos_embed helpers from benchmark_resolution_linear_probe
def _check_not_lfs_pointer(path: Path) -> None:
    if not path.exists() or path.stat().st_size > 1000:
        return
    with open(path, "rb") as f:
        head = f.read(100)
    if b"git-lfs" in head or head.startswith(b"version https://git-lfs"):
        raise RuntimeError(f"{path} is a Git LFS pointer. Run: git lfs pull")


def _parse_ckpt_to_state_dict(ckpt: Any) -> Dict[str, Any]:
    if isinstance(ckpt, dict) and "model" in ckpt:
        return dict(ckpt["model"])
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return dict(ckpt["state_dict"])
    if isinstance(ckpt, dict):
        return dict(ckpt)
    return dict(getattr(ckpt, "state_dict", lambda: ckpt)())


def load_checkpoint_state_dict(ckpt_path: Path) -> Dict[str, Any]:
    ckpt_path = Path(ckpt_path)
    if ckpt_path.suffix == ".safetensors" or ckpt_path.name == "model.safetensors":
        _check_not_lfs_pointer(ckpt_path)
        from safetensors.torch import load_file
        return dict(load_file(str(ckpt_path)))
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        if "UnpicklingError" in type(e).__name__ or "invalid load key" in str(e):
            raise RuntimeError(f"Use .safetensors or fix checkpoint. {e}") from e
        raise
    return _parse_ckpt_to_state_dict(ckpt)


def interpolate_pos_embed_vit5(pos_embed: torch.Tensor, target_num_patches: int) -> torch.Tensor:
    _, n, dim = pos_embed.shape
    orig_size = int(n ** 0.5)
    new_size = int(target_num_patches ** 0.5)
    pe = pos_embed.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
    pe = F.interpolate(pe, size=(new_size, new_size), mode="bicubic", align_corners=False)
    return pe.permute(0, 2, 3, 1).flatten(1, 2)


def interpolate_pos_embed_timm(
    pos_embed: torch.Tensor,
    target_num_patches: int,
    num_extra_tokens: int = 1,
) -> torch.Tensor:
    _, seq_len, dim = pos_embed.shape
    num_patch_tokens = seq_len - num_extra_tokens
    orig_size = int(num_patch_tokens ** 0.5)
    new_size = int(target_num_patches ** 0.5)
    extra = pos_embed[:, :num_extra_tokens, :]
    patch_tokens = pos_embed[:, num_extra_tokens:, :].reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
    patch_tokens = F.interpolate(patch_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    return torch.cat([extra, patch_tokens], dim=1)


def load_vit5_backbone_at_res(ckpt_path: Path, eval_res: int, device: torch.device) -> nn.Module:
    """Load ViT-5-base at eval_res, head replaced by Identity (penultimate features)."""
    from vit5 import remap_official_state_dict, vit5_base
    sd = load_checkpoint_state_dict(ckpt_path)
    remapped = remap_official_state_dict(sd)
    model = vit5_base(img_size=eval_res, num_classes=1000)
    our_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in remapped.items() if k in our_keys}
    if eval_res != TRAIN_RES and "pos_embed" in filtered:
        num_patches_eval = (eval_res // PATCH_SIZE) ** 2
        filtered["pos_embed"] = interpolate_pos_embed_vit5(filtered["pos_embed"], num_patches_eval)
    result = model.load_state_dict(filtered, strict=False)
    allowed_missing = {k for k in result.missing_keys if "rope" in k and k.endswith("inv_freq")}
    if set(result.missing_keys) != allowed_missing:
        raise RuntimeError(f"Unexpected missing keys: {set(result.missing_keys) - allowed_missing}")
    model.head = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()
    return model


def load_vitb_backbone_at_res(ckpt_path: Path, eval_res: int, device: torch.device) -> nn.Module:
    """Load ViT-B/16 at eval_res, head replaced by Identity."""
    from timm.models import create_model
    sd = load_checkpoint_state_dict(ckpt_path)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model = create_model("vit_base_patch16_224", img_size=eval_res, num_classes=1000, pretrained=False)
    num_patches_eval = (eval_res // PATCH_SIZE) ** 2
    if "pos_embed" in sd and sd["pos_embed"].shape[1] != model.pos_embed.shape[1]:
        sd["pos_embed"] = interpolate_pos_embed_timm(sd["pos_embed"], num_patches_eval, num_extra_tokens=1)
    model.load_state_dict(sd, strict=False)
    model.head = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()
    return model


def build_transform(resolution: int, is_train: bool) -> Any:
    from torchvision import transforms
    t = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if is_train:
        t.insert(1, transforms.RandomHorizontalFlip())
    return transforms.Compose(t)


def build_imagenet100_loaders(
    data_path: Path,
    resolution: int,
    batch_size: int,
    max_train: Optional[int],
    max_val: Optional[int],
) -> Tuple[DataLoader, DataLoader]:
    from torchvision.datasets import ImageFolder
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"ImageNet-100 train dir not found: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"ImageNet-100 val dir not found: {val_dir}")
    train_ds = ImageFolder(str(train_dir), transform=build_transform(resolution, is_train=True))
    val_ds = ImageFolder(str(val_dir), transform=build_transform(resolution, is_train=False))
    if len(train_ds.classes) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes; got {len(train_ds.classes)}")
    if max_train is not None and len(train_ds) > max_train:
        train_ds = Subset(train_ds, torch.randperm(len(train_ds))[:max_train].tolist())
    if max_val is not None and len(val_ds) > max_val:
        val_ds = Subset(val_ds, torch.randperm(len(val_ds))[:max_val].tolist())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader


class FrozenBackbonePlusProbe(nn.Module):
    """Backbone (frozen) + Linear(768, 100) probe."""

    def __init__(self, backbone: nn.Module, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.probe = nn.Linear(FEAT_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if out.dim() == 3:
            out = out[:, 0]
        return self.probe(out)


@torch.no_grad()
def eval_top1(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(1) == target).sum().item()
        total += target.size(0)
    return 100.0 * correct / total if total else 0.0


def train_probe_20_epochs(
    model: FrozenBackbonePlusProbe,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 0.01,
    verbose: bool = True,
) -> float:
    """Train only model.probe; return top-1 on val after training."""
    opt = torch.optim.SGD(model.probe.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    for ep in range(epochs):
        model.train()
        for images, target in train_loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if verbose and (ep + 1) % 5 == 0:
            acc = eval_top1(model, val_loader, device)
            print(f"    epoch {ep+1}/{epochs}  val_top1={acc:.2f}", flush=True)
    return eval_top1(model, val_loader, device)


def run_train_once_eval_multi(
    vit5_ckpt: Optional[Path],
    vitb_ckpt: Optional[Path],
    data_path: Path,
    eval_resolutions: List[int],
    batch_size: int,
    device: torch.device,
    max_train: Optional[int],
    max_val: Optional[int],
    probe_epochs: int,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """Train probe at 224 only; eval at 224 and eval_resolutions (e.g. 384)."""
    results: List[Dict[str, Any]] = []
    train_loader_224, val_loader_224 = build_imagenet100_loaders(
        data_path, TRAIN_RES, batch_size, max_train, max_val
    )
    if verbose:
        print(f"Train samples: {len(train_loader_224.dataset)}, val samples: {len(val_loader_224.dataset)}", flush=True)

    for name, ckpt, load_backbone in [
        ("ViT-5-base", vit5_ckpt, load_vit5_backbone_at_res),
        ("ViT-B/16", vitb_ckpt, load_vitb_backbone_at_res),
    ]:
        if ckpt is None or not ckpt.exists():
            continue
        if verbose:
            print(f"\n--- {name} ---", flush=True)
        backbone_224 = load_backbone(ckpt, TRAIN_RES, device)
        model = FrozenBackbonePlusProbe(backbone_224, NUM_CLASSES).to(device)
        if verbose:
            print(f"  Training linear probe at 224 for {probe_epochs} epochs...", flush=True)
        top1_224 = train_probe_20_epochs(
            model, train_loader_224, val_loader_224, device,
            epochs=probe_epochs, verbose=verbose,
        )
        if verbose:
            print(f"  Top-1 @ 224: {top1_224:.2f}", flush=True)
        row: Dict[str, Any] = {"model": name, "top1_224": top1_224}
        probe_state = {k: v.cpu().clone() for k, v in model.probe.state_dict().items()}

        for res in eval_resolutions:
            if res == TRAIN_RES:
                row["top1_224"] = top1_224
                continue
            if verbose:
                print(f"  Eval same probe @ {res}...", flush=True)
            backbone_res = load_backbone(ckpt, res, device)
            model_res = FrozenBackbonePlusProbe(backbone_res, NUM_CLASSES).to(device)
            model_res.probe.load_state_dict(probe_state)
            _, val_loader_res = build_imagenet100_loaders(data_path, res, batch_size, None, max_val)
            top1 = eval_top1(model_res, val_loader_res, device)
            row[f"top1_{res}"] = top1
            if verbose:
                print(f"  Top-1 @ {res}: {top1:.2f}", flush=True)
        results.append(row)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train linear probe at 224 (20 epochs), eval at 224 and 384. Freeze backbone."
    )
    parser.add_argument("--vit5-ckpt", type=Path, default=None)
    parser.add_argument("--vitb-ckpt", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, required=True, help="ImageNet-100 root (train/ and val/)")
    parser.add_argument("--eval-resolutions", type=int, nargs="+", default=[224, 384],
                        help="Resolutions to evaluate after training (default: 224 384)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--probe-epochs", type=int, default=20, help="Linear probe epochs at 224")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.vit5_ckpt is None and args.vitb_ckpt is None:
        print("Provide at least one of --vit5-ckpt or --vitb-ckpt.")
        return 1
    for p in (args.vit5_ckpt, args.vitb_ckpt):
        if p is not None and not p.exists():
            print(f"Checkpoint not found: {p}")
            return 1

    device = torch.device(args.device)
    if args.verbose:
        print(f"Device: {device}", flush=True)
    results = run_train_once_eval_multi(
        vit5_ckpt=args.vit5_ckpt,
        vitb_ckpt=args.vitb_ckpt,
        data_path=args.data_path,
        eval_resolutions=args.eval_resolutions,
        batch_size=args.batch_size,
        device=device,
        max_train=args.max_train,
        max_val=args.max_val,
        probe_epochs=args.probe_epochs,
        verbose=args.verbose,
    )

    all_res = sorted(set([TRAIN_RES] + args.eval_resolutions))
    print("\nLinear probe: train at 224, eval at multiple resolutions")
    print("-" * 60)
    for r in results:
        parts = [f"{r['model']}:"]
        for res in all_res:
            key = f"top1_{res}"
            if key in r:
                parts.append(f"top1_{res}={r[key]:.2f}")
        print("  ".join(parts))
    print("-" * 60)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["model"] + [f"top1_{res}" for res in all_res]
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in results:
                row = {k: r.get(k) for k in fieldnames if k in r}
                w.writerow(row)
        print(f"Wrote {args.output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
