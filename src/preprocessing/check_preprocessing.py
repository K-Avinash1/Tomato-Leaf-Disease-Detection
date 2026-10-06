import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.config import PROJECT_ROOT
from src.data.dataset import TomatoDataset, CLASS_NAMES
from src.preprocessing.transforms import get_eval_transform, denormalize, IMG_SIZE
from src.utils.seed import set_seed

FIG_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    set_seed()
    tf = get_eval_transform()
    train = TomatoDataset("train", tf)      # the test split is intentionally not loaded
    val = TomatoDataset("val", tf)
    print(f"Train: {len(train)} | Val: {len(val)} | Image size: {IMG_SIZE}")

    # 1. one sample
    x, y = train[0]
    print(f"\nOne sample: shape={tuple(x.shape)} dtype={x.dtype} "
          f"min={x.min():.2f} max={x.max():.2f} label={y} ({CLASS_NAMES[y]})")

    # 2. batch statistics after normalization (should be near 0 mean, near 1 std)
    loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=0)
    t0 = time.time()
    xb, yb = next(iter(loader))
    print(f"\nBatch shape: {tuple(xb.shape)} | labels shape: {tuple(yb.shape)} "
          f"| load time for 64 images: {time.time() - t0:.2f}s")
    print("Per-channel mean after normalization:", [round(v, 3) for v in xb.mean(dim=(0, 2, 3)).tolist()])
    print("Per-channel std  after normalization:", [round(v, 3) for v in xb.std(dim=(0, 2, 3)).tolist()])

    # 3. figure: original vs preprocessed, one training image per class
    fig, axes = plt.subplots(2, 10, figsize=(25, 6))
    for c, name in enumerate(CLASS_NAMES):
        i = next(k for k, lab in enumerate(train.labels) if lab == c)
        axes[0][c].imshow(Image.open(train.paths[i]).convert("RGB"))
        axes[0][c].set_title(name, fontsize=9)
        axes[1][c].imshow(denormalize(train[i][0]).permute(1, 2, 0))
        axes[0][c].axis("off")
        axes[1][c].axis("off")
    axes[0][0].text(-0.1, 0.5, "original", transform=axes[0][0].transAxes, rotation=90, va="center", ha="right")
    axes[1][0].text(-0.1, 0.5, "preprocessed", transform=axes[1][0].transAxes, rotation=90, va="center", ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "preprocessing_check.png", dpi=110)
    plt.close()

    # 4. speed benchmark: one training step per model on this CPU (random weights, no download)
    print(f"\n--- CPU training-step benchmark (batch of 16 at {IMG_SIZE}x{IMG_SIZE}) ---")
    print(f"CPU threads used by PyTorch: {torch.get_num_threads()}")
    candidates = [
        ("ResNet-18", models.resnet18),
        ("EfficientNet-B0", models.efficientnet_b0),
        ("MobileNetV3-Small", models.mobilenet_v3_small),
        ("MobileNetV3-Large", models.mobilenet_v3_large),
    ]
    x16 = torch.randn(16, 3, IMG_SIZE, IMG_SIZE)
    y16 = torch.randint(0, 10, (16,))
    for name, fn in candidates:
        model = fn(weights=None, num_classes=10)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        times = []
        for it in range(4):                       # first iteration is warm-up
            t0 = time.time()
            opt.zero_grad()
            loss_fn(model(x16), y16).backward()
            opt.step()
            times.append(time.time() - t0)
        sec = sum(times[1:]) / 3
        est_min = sec * (len(train) / 16) / 60
        print(f"{name:18s} {sec:5.2f} s/batch  -> about {est_min:5.1f} min per training epoch (estimate)")

    print("\nSaved: reports/figures/preprocessing_check.png")


if __name__ == "__main__":
    main()