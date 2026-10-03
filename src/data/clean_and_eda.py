import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT, TABLES_DIR

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)  # bit-count table


def main():
    inv = pd.read_csv(TABLES_DIR / "image_inventory.csv")
    n_all = len(inv)

    # 1. Remove corrupted files
    inv = inv[~inv["corrupted"]].copy()
    print(f"Files in inventory: {n_all} | after removing corrupted: {len(inv)}")

    # 2. Remove exact duplicates (keep the first file by path, so the choice is deterministic)
    inv = inv.sort_values("path").reset_index(drop=True)
    dup = inv.duplicated(subset="md5", keep="first")
    print(f"Exact duplicate copies removed: {int(dup.sum())}")
    inv = inv[~dup].reset_index(drop=True)
    print(f"Images after exact-duplicate removal: {len(inv)}")

    # 3. One pass: perceptual hash, brightness, blur, pixel statistics
    hashes, bright, blur = [], [], []
    ch_sum, ch_sq, n_pix = np.zeros(3), np.zeros(3), 0
    for i, p in enumerate(inv["path"]):
        with Image.open(p) as im:
            im = im.convert("RGB")
            hashes.append(np.packbits(imagehash.phash(im).hash.flatten()))
            arr = np.asarray(im)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        bright.append(float(gray.mean()))
        blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        a = arr.reshape(-1, 3).astype(np.float64) / 255.0
        ch_sum += a.sum(0)
        ch_sq += (a ** 2).sum(0)
        n_pix += a.shape[0]
        if (i + 1) % 4000 == 0:
            print(f"  processed {i + 1} images")
    inv["brightness"] = bright
    inv["blur_score"] = blur

    mean = ch_sum / n_pix
    std = np.sqrt(ch_sq / n_pix - mean ** 2)
    print("\nDataset channel mean (R,G,B):", np.round(mean, 4))
    print("Dataset channel std  (R,G,B):", np.round(std, 4))

    # 4. Near-duplicate analysis (perceptual hash, Hamming distance on 64-bit hashes)
    H = np.stack(hashes)
    N = len(H)
    thresholds = [0, 2, 4, 6]
    counts = {t: 0 for t in thresholds}
    pairs = []
    for i in range(N - 1):
        d = POP[np.bitwise_xor(H[i + 1:], H[i])].sum(axis=1)
        for t in thresholds:
            counts[t] += int((d <= t).sum())
        for j in np.where(d <= 4)[0]:
            pairs.append((i, i + 1 + int(j), int(d[j])))

    print("\n--- Near-duplicate pairs by Hamming distance (smaller = more similar) ---")
    for t in thresholds:
        print(f"  distance <= {t}: {counts[t]} pairs")

    pdf = pd.DataFrame(pairs, columns=["i", "j", "distance"])
    if len(pdf):
        pdf["class_a"] = inv.loc[pdf["i"], "class"].values
        pdf["class_b"] = inv.loc[pdf["j"], "class"].values
        pdf["path_a"] = inv.loc[pdf["i"], "path"].values
        pdf["path_b"] = inv.loc[pdf["j"], "path"].values
        same = int((pdf["class_a"] == pdf["class_b"]).sum())
        print(f"At distance <= 4: {len(pdf)} pairs | same class: {same} | across classes: {len(pdf) - same}")
        imgs_involved = len(set(pdf["i"]) | set(pdf["j"]))
        print(f"Distinct images involved in at least one pair: {imgs_involved} of {N}")
        print("Pairs per class (distance <= 4):")
        print(pdf[pdf["class_a"] == pdf["class_b"]].groupby("class_a").size())
        pdf.drop(columns=["i", "j"]).to_csv(TABLES_DIR / "near_duplicate_pairs.csv", index=False)
    else:
        print("No near-duplicate pairs at distance <= 4")

    # 5. Quality statistics
    print("\n--- Brightness and blur (all images) ---")
    print(inv[["brightness", "blur_score"]].describe(percentiles=[.01, .05, .5, .95, .99]).round(2))
    print("\n--- Median blur and brightness per class ---")
    print(inv.groupby("class")[["brightness", "blur_score"]].median().round(2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(inv["brightness"], bins=50, color="goldenrod")
    ax[0].set_title("Brightness (mean gray level)")
    ax[1].hist(np.log10(inv["blur_score"] + 1), bins=50, color="steelblue")
    ax[1].set_title("Sharpness: log10(variance of Laplacian + 1)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_brightness_blur.png", dpi=150)
    plt.close()

    # 6. Save the cleaned manifest
    manifest = inv[["class", "path", "md5", "brightness", "blur_score"]]
    manifest.to_csv(TABLES_DIR / "clean_manifest.csv", index=False)
    print("\n--- Clean dataset: images per class ---")
    print(manifest["class"].value_counts().to_string())
    print("Total clean images:", len(manifest))
    print("Saved: clean_manifest.csv, near_duplicate_pairs.csv, eda_brightness_blur.png")


if __name__ == "__main__":
    main()