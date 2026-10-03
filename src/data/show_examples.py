import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.config import PROJECT_ROOT, TABLES_DIR

FIG_DIR = PROJECT_ROOT / "reports" / "figures"


def show_pairs(df, title, out, max_pairs=4):
    df = df.head(max_pairs)
    if len(df) == 0:
        print("No pairs for:", title)
        return
    fig, axes = plt.subplots(len(df), 2, figsize=(6, 3.2 * len(df)))
    if len(df) == 1:
        axes = [axes]
    for r, (_, row) in enumerate(df.iterrows()):
        for c, side in enumerate(("a", "b")):
            ax = axes[r][c]
            ax.imshow(Image.open(row[f"path_{side}"]).convert("RGB"))
            ax.set_title(f"{row[f'class_{side}']} (dist {row['distance']})", fontsize=9)
            ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()


pairs = pd.read_csv(TABLES_DIR / "near_duplicate_pairs.csv")
cross = pairs[pairs["class_a"] != pairs["class_b"]]
same = pairs[pairs["class_a"] == pairs["class_b"]].sort_values("distance")
show_pairs(cross, "Near-duplicate pairs ACROSS classes", FIG_DIR / "pairs_cross_class.png")
show_pairs(same, "Near-duplicate pairs in the SAME class (closest first)", FIG_DIR / "pairs_same_class.png")

# Sharpness and darkness investigation
m = pd.read_csv(TABLES_DIR / "clean_manifest.csv")
ylcv = m[m["class"] == "Yellow Leaf Curl Virus"]
healthy = m[m["class"] == "Healthy"]
sets = [
    ("Yellow Leaf Curl: lowest sharpness score", ylcv.nsmallest(5, "blur_score")),
    ("Yellow Leaf Curl: random", ylcv.sample(5, random_state=42)),
    ("Healthy: random", healthy.sample(5, random_state=42)),
    ("Darkest images overall", m.nsmallest(5, "brightness")),
]
fig, axes = plt.subplots(len(sets), 5, figsize=(15, 12))
for r, (name, sub) in enumerate(sets):
    for c, (_, row) in enumerate(sub.iterrows()):
        ax = axes[r][c]
        ax.imshow(Image.open(row["path"]).convert("RGB"))
        ax.axis("off")
        ax.set_title(f"{name}\n{row['class']}\nsharp {row['blur_score']:.0f} | bright {row['brightness']:.0f}", fontsize=7)
plt.tight_layout()
plt.savefig(FIG_DIR / "quality_examples.png", dpi=100)
plt.close()
print("Saved: pairs_cross_class.png, pairs_same_class.png, quality_examples.png")