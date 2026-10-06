import hashlib
import platform
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import models

from src.config import DATASET_ROOT, TABLES_DIR
from src.data.dataset import TomatoDataset
from src.preprocessing.transforms import get_eval_transform, IMG_SIZE
from src.utils.seed import set_seed, SEED


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Environment ===")
    print("Python:", sys.version.split()[0], "| OS:", platform.platform())
    print("PyTorch:", torch.__version__, "| torchvision:", torchvision.__version__)
    print("Device:", device, "|", torch.cuda.get_device_name(0) if device == "cuda" else "CPU only")
    print("Seed:", SEED, "| data root:", DATASET_ROOT)

    print("\n=== Dataset integrity (all files, byte level) ===")
    man = pd.read_csv(TABLES_DIR / "split_manifest.csv")
    missing = bad = 0
    for rel, h in zip(man["rel_path"], man["md5"]):
        p = DATASET_ROOT / rel
        if not p.exists():
            missing += 1
        elif md5(p) != h:
            bad += 1
    print(f"Manifest rows: {len(man)} | missing: {missing} | hash mismatches: {bad}")
    print(man["split"].value_counts().to_dict())

    print("\n=== Loading one batch (train split) ===")
    ds = TomatoDataset("train", get_eval_transform())
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2 if device == "cuda" else 0)
    xb, yb = next(iter(dl))
    print("Batch:", tuple(xb.shape), "| labels:", tuple(yb.shape))

    if "--no-bench" in sys.argv:
        print("\n(benchmark skipped)")
        return

    print(f"\n=== Training-step benchmark on {device} (batch 32, {IMG_SIZE}x{IMG_SIZE}, all layers) ===")
    n_train = len(ds)
    for name, fn in [("ResNet-18", models.resnet18), ("EfficientNet-B0", models.efficientnet_b0),
                     ("MobileNetV3-Small", models.mobilenet_v3_small),
                     ("MobileNetV3-Large", models.mobilenet_v3_large)]:
        model = fn(weights=None, num_classes=10).to(device).train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(32, 3, IMG_SIZE, IMG_SIZE, device=device)
        y = torch.randint(0, 10, (32,), device=device)
        times = []
        for _ in range(6):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(x), y).backward()
            opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t0)
        sec = sum(times[1:]) / 5
        mem = f" | peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if device == "cuda" else ""
        print(f"{name:18s} {sec:5.2f} s/batch -> about {sec * n_train / 32 / 60:5.1f} min/epoch (estimate){mem}")
        del model, opt


if __name__ == "__main__":
    main()