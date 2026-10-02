import hashlib
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.config import DATASET_ROOT, PROJECT_ROOT, TABLES_DIR, CLASS_FOLDERS

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
GOOD_EXT = {".jpg", ".jpeg", ".png"}
SEED = 42


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_file(path):
    info = {"width": None, "height": None, "mode": None, "format": None, "corrupted": False}
    try:
        with Image.open(path) as im:
            im.verify()                 # structure check
        with Image.open(path) as im:
            im.load()                   # full decode check
            info.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
    except Exception:
        info["corrupted"] = True
    return info


def main():
    rows = []
    for folder, cls in CLASS_FOLDERS.items():
        for p in sorted((DATASET_ROOT / folder).rglob("*")):
            if not p.is_file():
                continue
            row = {"class": cls, "path": str(p), "name": p.name, "ext": p.suffix.lower(),
                   "size_bytes": p.stat().st_size, "md5": md5(p)}
            row.update(inspect_file(p))
            rows.append(row)

    df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "image_inventory.csv", index=False)

    print("Files inspected:", len(df))

    # 1. Odd files (anything that is not a plain .jpg)
    odd = df[df["ext"] != ".jpg"]
    print("\n--- Files that are not .jpg ---")
    print(odd[["class", "path", "ext", "size_bytes", "format", "corrupted"]].to_string(index=False))

    # 2. Corrupted
    bad = df[df["corrupted"]]
    print("\n--- Corrupted / unreadable files:", len(bad))
    if len(bad):
        print(bad[["class", "path"]].to_string(index=False))

    good = df[~df["corrupted"]]

    # 3. Size, mode, format
    print("\n--- Image sizes (top 5) ---")
    print(good.groupby(["width", "height"]).size().sort_values(ascending=False).head(5))
    print("\n--- Color modes ---")
    print(good["mode"].value_counts())
    print("\n--- Formats ---")
    print(good["format"].value_counts())
    print("\nMin side:", int(good[["width", "height"]].min().min()),
          "| Max side:", int(good[["width", "height"]].max().max()))

    # 4. Exact duplicates (same file content)
    groups = good.groupby("md5")
    dup_groups = [g for _, g in groups if len(g) > 1]
    cross = [g for g in dup_groups if g["class"].nunique() > 1]
    print("\n--- Exact duplicate check ---")
    print("Duplicate groups (same content):", len(dup_groups))
    print("Files involved:", sum(len(g) for g in dup_groups))
    print("Groups spanning different classes:", len(cross))
    for g in dup_groups[:5]:
        print(" ", list(g["class"]), list(g["name"]))

    # 5. Figure 3: class distribution
    counts = good.groupby("class").size().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    ax = counts.plot(kind="bar", color="seagreen")
    ax.set_title("Class distribution (PlantVillage tomato, local dataset)")
    ax.set_ylabel("Number of images")
    for i, v in enumerate(counts):
        ax.text(i, v + 20, str(v), ha="center", fontsize=8)
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_class_distribution.png", dpi=150)
    plt.close()

    # 6. Figure 4: sample grid, 4 random images per class
    rnd = random.Random(SEED)
    classes = list(CLASS_FOLDERS.values())
    fig, axes = plt.subplots(4, 10, figsize=(30, 12))
    for c, cls in enumerate(classes):
        paths = good[good["class"] == cls]["path"].tolist()
        for r, p in enumerate(rnd.sample(paths, 4)):
            axes[r][c].imshow(Image.open(p).convert("RGB"))
            axes[r][c].axis("off")
            if r == 0:
                axes[r][c].set_title(cls, fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig04_sample_images.png", dpi=100)
    plt.close()
    print("\nSaved: image_inventory.csv, fig03_class_distribution.png, fig04_sample_images.png")


if __name__ == "__main__":
    main()