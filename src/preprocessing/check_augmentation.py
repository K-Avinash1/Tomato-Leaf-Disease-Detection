import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from src.config import DATASET_ROOT, PROJECT_ROOT, TABLES_DIR
from src.preprocessing.transforms import (AUG_CONFIG, denormalize, get_eval_transform,
                                          get_train_transform)
from src.utils.seed import SEED, set_seed

FIG_DIR = PROJECT_ROOT / "reports" / "figures"


def black_fraction(t):
    """Fraction of pixels that are almost pure black (all channels below 8 of 255)."""
    x = denormalize(t) * 255
    return float((x.max(dim=0).values < 8).float().mean())


def main():
    set_seed()
    print("Seed:", SEED)
    print("Augmentation config:", AUG_CONFIG)

    man = pd.read_csv(TABLES_DIR / "split_manifest.csv")
    train = man[man["split"] == "train"]          # only training images are used here
    picks = [(c, train[train["class"] == c].iloc[0]["rel_path"])
             for c in ["Septoria Leaf Spot", "Bacterial Spot", "Early Blight"]]
    dark = train[(train["class"] == "Late Blight") & train["dark"]].iloc[0]
    picks.append(("Late Blight (black background)", dark["rel_path"]))
    picks.append(("Healthy", train[train["class"] == "Healthy"].iloc[0]["rel_path"]))

    tr, ev = get_train_transform(), get_eval_transform()

    # 1. determinism checks
    img = Image.open(DATASET_ROOT / picks[0][1]).convert("RGB")
    print("\nEval transform gives identical output twice:", torch.equal(ev(img), ev(img)))
    print("Train transform gives different output each time:", not torch.equal(tr(img), tr(img)))
    print("Output shape:", tuple(tr(img).shape))

    # 2. black-border check on a normal (non-dark) image
    normal = Image.open(DATASET_ROOT / picks[1][1]).convert("RGB")
    orig_black = black_fraction(ev(normal))
    worst = max(black_fraction(tr(normal)) for _ in range(300))
    print(f"\nBlack-pixel fraction, original: {orig_black:.4f} | worst of 300 augmented: {worst:.4f}")

    # 3. figure: original + 7 random augmentations per image
    fig, axes = plt.subplots(len(picks), 8, figsize=(20, 2.6 * len(picks)))
    for r, (name, rel) in enumerate(picks):
        im = Image.open(DATASET_ROOT / rel).convert("RGB")
        axes[r][0].imshow(im)
        axes[r][0].set_title(f"{name}\n(original)", fontsize=8)
        for c in range(1, 8):
            axes[r][c].imshow(denormalize(tr(im)).permute(1, 2, 0))
            axes[r][c].set_title(f"aug {c}", fontsize=8)
        for c in range(8):
            axes[r][c].axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig06_augmentation_examples.png", dpi=100)
    plt.close()
    print("\nSaved: reports/figures/fig06_augmentation_examples.png")


if __name__ == "__main__":
    main()