import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.config import DATASET_ROOT, PROJECT_ROOT, TABLES_DIR
from src.data.dataset import CLASS_NAMES

TAG = "E010_mnv3_deep_seed7"
TEST_DIR = PROJECT_ROOT / "reports" / "test_results" / TAG
VAL_CSV = PROJECT_ROOT / "reports" / "experiments" / TAG / "val_predictions.csv"
FIG = PROJECT_ROOT / "reports" / "figures"
OUT = PROJECT_ROOT / "reports" / "error_analysis"


def load_quality():
    sm = pd.read_csv(TABLES_DIR / "split_manifest.csv")[["rel_path", "path"]]
    cm = pd.read_csv(TABLES_DIR / "clean_manifest.csv")[["path", "brightness", "blur_score"]]
    return sm.merge(cm, on="path")[["rel_path", "brightness", "blur_score"]]


def montage(df, path, title, ncols=5):
    nrows = int(np.ceil(len(df) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 4.1 * nrows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), df.iterrows()):
        ax.imshow(Image.open(DATASET_ROOT / r["rel_path"]).convert("RGB"))
        cap = f"true: {r['class']}\npred: {r['pred_class']} ({r['confidence']:.2f})"
        if "extra" in df.columns:
            cap += f"\n{r['extra']}"
        ax.set_title(cap, fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    q = load_quality()
    test = pd.read_csv(TEST_DIR / "test_predictions.csv").merge(q, on="rel_path", how="left")
    val = pd.read_csv(VAL_CSV).merge(q, on="rel_path", how="left")
    for d in (test, val):
        d["wrong"] = d["true_idx"] != d["pred_idx"]
    train_n = pd.read_csv(TABLES_DIR / "split_manifest.csv").query("split == 'train'")["class"].value_counts()
    pd.set_option("display.width", 220)

    rows = []
    for c in CLASS_NAMES:
        t = test[test["class"] == c]
        rows.append({"class": c, "train_images": int(train_n[c]), "test_images": len(t),
                     "test_errors": int(t["wrong"].sum()), "error_rate_pct": round(100 * t["wrong"].mean(), 2),
                     "wrongly_predicted_as_this_class": int(((test["pred_class"] == c) & test["wrong"]).sum())})
    pc = pd.DataFrame(rows)
    print("=== Per class (test) ===")
    print(pc.to_string(index=False))
    pc.to_csv(OUT / "per_class_errors_test.csv", index=False)

    pairs = test[test["wrong"]].groupby(["class", "pred_class"]).size().sort_values(ascending=False)
    print("\n=== Test errors by (true class, predicted class) ===")
    print(pairs.to_string())
    pairs.to_csv(OUT / "confusion_pairs_test.csv")

    err = test[test["wrong"]].copy()

    def pctile(r, col):
        same = test[(test["class"] == r["class"]) & ~test["wrong"]][col]
        return round(100 * float((same < r[col]).mean()))

    err["blur_pctile_in_class"] = err.apply(lambda r: pctile(r, "blur_score"), axis=1)
    err["brightness_pctile_in_class"] = err.apply(lambda r: pctile(r, "brightness"), axis=1)
    err = err.sort_values("confidence", ascending=False)
    print("\n=== The test errors (percentile = where the error sits among correct images of its true class; 50 = typical) ===")
    print(err[["class", "pred_class", "confidence", "blur_pctile_in_class", "brightness_pctile_in_class"]].round(3).to_string())
    print(f"Mean blur percentile of errors: {err['blur_pctile_in_class'].mean():.0f} | mean brightness percentile: {err['brightness_pctile_in_class'].mean():.0f}")
    err[["rel_path", "class", "pred_class", "confidence", "blur_score", "brightness",
         "blur_pctile_in_class", "brightness_pctile_in_class"]].to_csv(OUT / "test_errors_detail.csv", index=False)

    print("\n=== Confidence ===")
    for name, d in [("validation", val), ("test", test)]:
        e, c = d[d["wrong"]], d[~d["wrong"]]
        print(f"{name}: errors {len(e)} | mean confidence wrong {e['confidence'].mean():.3f} vs correct {c['confidence'].mean():.3f} | "
              f"errors below 0.95: {int((e['confidence'] < 0.95).sum())} of {len(e)} | images below 0.95: {int((d['confidence'] < 0.95).sum())} of {len(d)}")

    print("\n=== Confident errors (confidence >= 0.90): look at these images; the label itself may be doubtful ===")
    print(err[err["confidence"] >= 0.90][["rel_path", "class", "pred_class", "confidence"]].round(3).to_string(index=False))

    montage(err, FIG / "error_test_misclassified.png", f"{TAG}: all test errors, most confident first")

    hard = pd.read_csv(TABLES_DIR / "hard_images_val.csv")
    hard = hard.rename(columns={hard.columns[0]: "rel_path"}).drop(columns=["class"])
    hard = hard.merge(val[["rel_path", "class", "pred_class", "confidence"]], on="rel_path")
    hard["extra"] = "wrong in " + hard["runs_wrong"].astype(str) + " of 10 runs"
    montage(hard.sort_values("runs_wrong", ascending=False), FIG / "error_hard_validation.png",
            f"Validation images that at least 3 of 10 runs got wrong (caption shows {TAG}'s prediction)")

    ok = test[~test["wrong"]]
    order = {c: i for i, c in enumerate(CLASS_NAMES)}
    top = ok.sort_values("confidence", ascending=False).groupby("class").head(1).sort_values("class", key=lambda s: s.map(order))
    low = ok.sort_values("confidence").groupby("class").head(1).sort_values("class", key=lambda s: s.map(order))
    fig, axes = plt.subplots(2, 10, figsize=(30, 7))
    for row, (label, d) in enumerate([("most confident correct", top), ("least confident correct", low)]):
        for col, (_, r) in enumerate(d.iterrows()):
            axes[row][col].imshow(Image.open(DATASET_ROOT / r["rel_path"]).convert("RGB"))
            axes[row][col].set_title(f"{r['class']}\n{label}: {r['confidence']:.3f}", fontsize=8)
            axes[row][col].axis("off")
    plt.tight_layout()
    plt.savefig(FIG / "error_examples_correct.png", dpi=90)
    plt.close()
    print("\nSaved figures: error_test_misclassified.png, error_hard_validation.png, error_examples_correct.png")


if __name__ == "__main__":
    main()