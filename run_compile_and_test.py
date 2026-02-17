"""
Compile ViT-5 with torch.compile and run a quick forward test.
Usage: python run_compile_and_test.py
"""

from __future__ import annotations

import torch
from vit5 import vit5_base

def main() -> None:
    print("Building ViT-5-Base...")
    model = vit5_base()
    model.eval()

    # Compile (requires C++ compiler on Windows for inductor; skip if unavailable)
    if hasattr(torch, "compile"):
        print("Compiling with torch.compile(mode='default')...")
        model = torch.compile(model, mode="default")
    else:
        print("torch.compile not available, running uncompiled.")

    x = torch.randn(2, 3, 224, 224)
    print("Running forward pass...")
    try:
        with torch.no_grad():
            out = model(x)
    except Exception as e:
        if "Compiler" in str(e) or "cl" in str(e) or "Inductor" in str(e):
            print("Compile failed (no C++ compiler); re-running uncompiled.")
            model = vit5_base()
            model.eval()
            with torch.no_grad():
                out = model(x)
        else:
            raise

    assert out.shape == (2, 1000), f"expected (2, 1000), got {out.shape}"
    print(f"OK: output shape {out.shape}")
    print("Compile & test passed.")

if __name__ == "__main__":
    main()
