"""
Resolution-scaling benchmark via linear probe on penultimate features (ImageNet-100).

Isolates representation quality: extract features before the classifier head,
train a linear probe on ImageNet-100, compare ViT-5 vs ViT-B/16 across resolutions.
No use of pretrained head; measures how well the backbone preserves information.

Requires ImageNet-100: data path must contain train/ and val/ with 100 classes each
(same folder names as in imagenet100.txt, or 100 class subdirs in any order).

Usage:
  python scripts/benchmark_resolution_linear_probe.py \\
    --vit5-ckpt vit5_base_patch16_224.pth \\
    --vitb-ckpt path/to/vitb.safetensors \\
    --data-path /path/to/imagenet100 \\
    --resolutions 224 256 384 448
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

# ImageNet-100 WNIDs (CMC subset)
IMAGENET_100_WNIDS = [
    "n02869837", "n01749939", "n02488291", "n02107142", "n13037406", "n02091831",
    "n04517823", "n04589890", "n03062245", "n01773797", "n01735189", "n07831146",
    "n07753275", "n03085013", "n04485082", "n02105505", "n01983481", "n02788148",
    "n03530642", "n04435653", "n02086910", "n02859443", "n13040303", "n03594734",
    "n02085620", "n02099849", "n01558993", "n04493381", "n02109047", "n04111531",
    "n02877765", "n04429376", "n02009229", "n01978455", "n02106550", "n01820546",
    "n01692333", "n07714571", "n02974003", "n02114855", "n03785016", "n03764736",
    "n03775546", "n02087046", "n07836838", "n04099969", "n04592741", "n03891251",
    "n02701002", "n03379051", "n02259212", "n07715103", "n03947888", "n04026417",
    "n02326432", "n03637318", "n01980166", "n02113799", "n02086240", "n03903868",
    "n02483362", "n04127249", "n02089973", "n03017168", "n02093428", "n02804414",
    "n02396427", "n04418357", "n02172182", "n01729322", "n02113978", "n03787032",
    "n02089867", "n02119022", "n03777754", "n04238763", "n02231487", "n03032252",
    "n02138441", "n02104029", "n03837869", "n03494278", "n04136333", "n03794056",
    "n03492542", "n02018207", "n04067472", "n03930630", "n03584829", "n02123045",
    "n04229816", "n02100583", "n03642806", "n04336792", "n03259280", "n02116738",
    "n02108089", "n03424325", "n01855672", "n02090622",
]
NUM_CLASSES = 100
TRAIN_RES = 224
PATCH_SIZE = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEAT_DIM = 768  # ViT-5-base and ViT-B/16


# -----------------------------------------------------------------------------
# Reuse checkpoint loading and pos_embed interpolation from benchmark_resolution
# -----------------------------------------------------------------------------

def _check_not_lfs_pointer(path: Path) -> None:
    if not path.exists() or path.stat().st_size > 1000:
        return
    with open(path, "rb") as f:
        head = f.read(100)
    if b"git-lfs" in head or head.startswith(b"version https://git-lfs"):
        raise RuntimeError(f"{path} is a Git LFS pointer. Run in that repo: git lfs pull")


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


def load_vit5_backbone(ckpt_path: Path, eval_res: int, device: torch.device) -> nn.Module:
    """Load ViT-5-base at eval_res; returns model that outputs penultimate features (no head)."""
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
    model.to(device)
    model.eval()
    return model


def load_vitb_backbone(ckpt_path: Path, eval_res: int, device: torch.device) -> nn.Module:
    """Load ViT-B/16 at eval_res; returns model that outputs penultimate features (no head)."""
    from timm.models import create_model
    sd = load_checkpoint_state_dict(ckpt_path)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model = create_model("vit_base_patch16_224", img_size=eval_res, num_classes=1000, pretrained=False)
    num_patches_eval = (eval_res // PATCH_SIZE) ** 2
    if "pos_embed" in sd and sd["pos_embed"].shape[1] != model.pos_embed.shape[1]:
        sd["pos_embed"] = interpolate_pos_embed_timm(sd["pos_embed"], num_patches_eval, num_extra_tokens=1)
    model.load_state_dict(sd, strict=False)
    model.head = nn.Identity()
    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Penultimate feature extraction (cls token only)
# -----------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device, verbose: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract penultimate (cls) features and labels. Returns features (N, D), labels (N,)."""
    feats_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    total = len(loader)
    for i, (images, target) in enumerate(loader):
        if verbose and (i + 1) % max(1, total // 10) == 0:
            print(f"    extract batch {i+1}/{total}", flush=True)
        images = images.to(device, non_blocking=True)
        # ViT-5 and timm ViT: forward returns (B, seq, dim); cls token is first
        out = model(images)
        if out.dim() == 3:
            out = out[:, 0]
        feats_list.append(out.cpu())
        labels_list.append(target)
    return torch.cat(feats_list, dim=0), torch.cat(labels_list, dim=0)


# -----------------------------------------------------------------------------
# ImageNet-100 data
# -----------------------------------------------------------------------------

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
) -> Tuple[DataLoader, DataLoader, int]:
    """Build train and val loaders for ImageNet-100. data_path must contain train/ and val/."""
    from torchvision.datasets import ImageFolder
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"ImageNet-100 train dir not found: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"ImageNet-100 val dir not found: {val_dir}")
    train_ds = ImageFolder(str(train_dir), transform=build_transform(resolution, is_train=True))
    val_ds = ImageFolder(str(val_dir), transform=build_transform(resolution, is_train=False))
    num_classes = len(train_ds.classes)
    if num_classes != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes (ImageNet-100); got {num_classes}. Check train/ layout.")
    if max_train is not None and len(train_ds) > max_train:
        idx = torch.randperm(len(train_ds))[:max_train].tolist()
        train_ds = Subset(train_ds, idx)
    if max_val is not None and len(val_ds) > max_val:
        idx = torch.randperm(len(val_ds))[:max_val].tolist()
        val_ds = Subset(val_ds, idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, num_classes


# -----------------------------------------------------------------------------
# Linear probe
# -----------------------------------------------------------------------------

def train_linear_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    num_classes: int,
    feat_dim: int,
    epochs: int = 30,
    lr: float = 0.01,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
) -> Tuple[float, float]:
    """Train linear classifier; return val acc1, acc5."""
    probe = nn.Linear(feat_dim, num_classes).to(device)
    opt = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    batch_size = 256
    n = X_train.size(0)
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(n, device=X_train.device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = probe(X_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if verbose and (ep + 1) % 10 == 0:
            with torch.no_grad():
                acc = (probe(X_val).argmax(1) == y_val).float().mean().item() * 100
            print(f"    probe epoch {ep+1}/{epochs} val_acc1={acc:.2f}", flush=True)
    probe.eval()
    with torch.no_grad():
        logits = probe(X_val)
        acc1 = (logits.argmax(1) == y_val).float().mean().item() * 100
        _, top5 = logits.topk(5, dim=1)
        acc5 = (top5 == y_val.unsqueeze(1)).any(1).float().mean().item() * 100
    return acc1, acc5


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_linear_probe_benchmark(
    vit5_ckpt: Optional[Path],
    vitb_ckpt: Optional[Path],
    data_path: Path,
    resolutions: List[int],
    batch_size: int,
    device: torch.device,
    max_train: Optional[int],
    max_val: Optional[int],
    probe_epochs: int,
    verbose: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for res in resolutions:
        if verbose:
            print(f"\n[Resolution {res}]", flush=True)
        train_loader, val_loader, num_classes = build_imagenet100_loaders(
            data_path, res, batch_size, max_train, max_val
        )
        if verbose:
            print(f"  Train samples: {len(train_loader.dataset)}, val samples: {len(val_loader.dataset)}", flush=True)

        if vit5_ckpt is not None and vit5_ckpt.exists():
            if verbose:
                print(f"  Loading ViT-5-base @ {res}...", flush=True)
            model = load_vit5_backbone(vit5_ckpt, res, device)
            if verbose:
                print(f"  Extracting train features (ViT-5 @ {res})...", flush=True)
            X_train, y_train = extract_features(model, train_loader, device, verbose)
            if verbose:
                print(f"  Extracting val features (ViT-5 @ {res})...", flush=True)
            X_val, y_val = extract_features(model, val_loader, device, verbose)
            if verbose:
                print(f"  Training linear probe (ViT-5 @ {res})...", flush=True)
            acc1, acc5 = train_linear_probe(
                X_train, y_train, X_val, y_val, num_classes, FEAT_DIM,
                epochs=probe_epochs, device=device, verbose=verbose,
            )
            results.append({"model": "ViT-5-base", "train_res": TRAIN_RES, "eval_res": res, "acc1": acc1, "acc5": acc5})

        if vitb_ckpt is not None and vitb_ckpt.exists():
            if verbose:
                print(f"  Loading ViT-B/16 @ {res}...", flush=True)
            model = load_vitb_backbone(vitb_ckpt, res, device)
            if verbose:
                print(f"  Extracting train features (ViT-B @ {res})...", flush=True)
            X_train, y_train = extract_features(model, train_loader, device, verbose)
            if verbose:
                print(f"  Extracting val features (ViT-B @ {res})...", flush=True)
            X_val, y_val = extract_features(model, val_loader, device, verbose)
            if verbose:
                print(f"  Training linear probe (ViT-B @ {res})...", flush=True)
            acc1, acc5 = train_linear_probe(
                X_train, y_train, X_val, y_val, num_classes, FEAT_DIM,
                epochs=probe_epochs, device=device, verbose=verbose,
            )
            results.append({"model": "ViT-B/16", "train_res": TRAIN_RES, "eval_res": res, "acc1": acc1, "acc5": acc5})

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolution robustness via linear probe on ImageNet-100 (penultimate features).")
    parser.add_argument("--vit5-ckpt", type=Path, default=None)
    parser.add_argument("--vitb-ckpt", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, required=True, help="ImageNet-100 root (train/ and val/ with 100 classes)")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[224, 256, 384, 448])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train", type=int, default=None, help="Cap train samples (faster)")
    parser.add_argument("--max-val", type=int, default=None, help="Cap val samples")
    parser.add_argument("--probe-epochs", type=int, default=30, help="Linear probe training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
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
    results = run_linear_probe_benchmark(
        vit5_ckpt=args.vit5_ckpt,
        vitb_ckpt=args.vitb_ckpt,
        data_path=args.data_path,
        resolutions=args.resolutions,
        batch_size=args.batch_size,
        device=device,
        max_train=args.max_train,
        max_val=args.max_val,
        probe_epochs=args.probe_epochs,
        verbose=args.verbose,
    )

    print("\nResolution scaling (linear probe on ImageNet-100, penultimate features)")
    print("-" * 70)
    print(f"{'model':<12} {'train_res':<10} {'eval_res':<10} {'acc1':<8} {'acc5':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<12} {r['train_res']:<10} {r['eval_res']:<10} {r['acc1']:.2f}    {r['acc5']:.2f}")
    print("-" * 70)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "train_res", "eval_res", "acc1", "acc5"])
            w.writeheader()
            for r in results:
                w.writerow({k: r[k] for k in w.fieldnames})
        print(f"Wrote {args.output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
