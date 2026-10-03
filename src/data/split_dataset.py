import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT, TABLES_DIR
from src.utils.seed import SEED

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
N_FOLDS = 20          # each fold is 5% of the data
TEST_FOLDS = 3        # 15% test
VAL_FOLDS = 3         # 15% validation, the remaining 14 folds (70%) are training
DARK_THRESHOLD = 60   # brightness below this = black-background image (see Phase 3)


def build_groups(man):
    """Link same-class near-duplicate pairs (hash distance <= 4) into groups."""
    parent = {p: p for p in man["path"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = pd.read_csv(TABLES_DIR / "near_duplicate_pairs.csv")
    same = pairs[pairs["class_a"] == pairs["class_b"]]
    for a, b in zip(same["path_a"], same["path_b"]):
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    return man["path"].map(find)


def main():
    man = pd.read_csv(TABLES_DIR / "clean_manifest.csv")
    man["group"] = build_groups(man)
    man["dark"] = man["brightness"] < DARK_THRESHOLD

    # Stratify by class, and separate dark Late Blight (the only class with many dark images)
    man["stratum"] = man["class"]
    man.loc[(man["class"] == "Late Blight") & man["dark"], "stratum"] = "Late Blight|dark"

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    man["fold"] = -1
    for k, (_, idx) in enumerate(sgkf.split(man, man["stratum"], groups=man["group"])):
        man.loc[man.index[idx], "fold"] = k

    man["split"] = np.where(man["fold"] < TEST_FOLDS, "test",
                    np.where(man["fold"] < TEST_FOLDS + VAL_FOLDS, "val", "train"))

    # ---------- verification ----------
    order = ["train", "val", "test"]
    print("Seed:", SEED, "| total images:", len(man))
    sizes = man["split"].value_counts().reindex(order)
    print("\n--- Split sizes ---")
    for s in order:
        print(f"{s:6s}: {sizes[s]:6d}  ({sizes[s] / len(man) * 100:.1f}%)")

    counts = man.pivot_table(index="class", columns="split", values="path",
                             aggfunc="count", fill_value=0)[order]
    counts["total"] = counts.sum(axis=1)
    for s in order:
        counts[f"{s}_%"] = (counts[s] / counts["total"] * 100).round(1)
    print("\n--- Images per class per split ---")
    print(counts.sort_values("total", ascending=False).to_string())

    leaked_groups = int((man.groupby("group")["split"].nunique() > 1).sum())
    leaked_md5 = int((man.groupby("md5")["split"].nunique() > 1).sum())
    print("\n--- Leakage checks ---")
    print("Near-duplicate groups spread over more than one split:", leaked_groups)
    print("Identical files (MD5) appearing in more than one split:", leaked_md5)
    print("Images in groups of size > 1:", int((man.groupby("group")["path"].transform("count") > 1).sum()))

    print("\n--- Dark (black-background) images per split ---")
    dark = man[man["dark"]].groupby(["class", "split"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    print(dark.to_string())

    # ---------- save ----------
    man[["path", "class", "split", "group", "dark", "md5"]].to_csv(TABLES_DIR / "split_manifest.csv", index=False)
    counts.to_csv(TABLES_DIR / "split_counts.csv")

    pct = counts[order].div(counts[order].sum(axis=1), axis=0) * 100
    pct = pct.loc[counts["total"].sort_values(ascending=False).index]
    ax = pct.plot(kind="bar", stacked=True, figsize=(10, 5), color=["seagreen", "goldenrod", "steelblue"])
    ax.set_ylabel("Percent of class")
    ax.set_title("Train / validation / test share per class")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "split_distribution.png", dpi=150)
    plt.close()
    print("\nSaved: split_manifest.csv, split_counts.csv, split_distribution.png")


if __name__ == "__main__":
    main() 