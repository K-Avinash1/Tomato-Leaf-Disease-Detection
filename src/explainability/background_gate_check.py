import numpy as np
import pandas as pd
from PIL import Image

from src.config import DATASET_ROOT, PROJECT_ROOT
from src.data.dataset import CLASS_NAMES
from src.explainability.gradcam_analysis import leaf_mask, load_arr
from src.preprocessing.transforms import get_eval_transform
from src.reliability.decision import ReliabilityGate
from src.reliability.features import load_final_model
from src.reliability.quality_experiment import forward

TAG = "E010_mnv3_deep_seed7"
VAL_CSV = PROJECT_ROOT / "reports" / "experiments" / TAG / "val_predictions.csv"
MODELS = PROJECT_ROOT / "models"
LB = CLASS_NAMES.index("Late Blight")
GREY = (138, 132, 133)       # usual background colour, from experiment (b)


def variant(a, m, kind):
    if kind == "original":
        return a
    b = a.copy()
    b[m == 0] = (0, 0, 0) if kind == "black" else GREY
    return b


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    val = pd.read_csv(VAL_CSV)
    rng = np.random.default_rng(42)
    idx = np.concatenate([rng.choice(np.where(val["class"] == c)[0], min(60, int((val["class"] == c).sum())), replace=False)
                          for c in CLASS_NAMES])
    sub = val.iloc[idx].reset_index(drop=True)
    arrs = [load_arr(DATASET_ROOT / r) for r in sub["rel_path"]]
    masks = [leaf_mask(a) for a in arrs]
    y = sub["true_idx"].to_numpy()
    keep = [i for i in range(len(sub)) if y[i] != LB and 0.1 <= 1 - masks[i].mean() <= 0.9]
    print(f"{len(keep)} validation images of the 9 classes other than Late Blight (usable leaf mask)")

    model, _ = load_final_model()
    gate = ReliabilityGate.from_files(MODELS / "reliability_config.json", MODELS / "knn_bank.npz")
    tf = get_eval_transform()
    rows, where = [], {}
    for kind in ["original", "grey", "black"]:
        imgs = [Image.fromarray(variant(arrs[i], masks[i], kind)) for i in keep]
        L, F = forward(model, imgs, tf)
        status, _, _ = gate.decide(L, F)
        pred, yk = L.argmax(1), y[keep]
        rel = status == "reliable"
        rows.append({"background": kind, "accuracy": round(float((pred == yk).mean()), 3),
                     "predicted_late_blight": round(float((pred == LB).mean()), 3),
                     "reliable": round(float(rel.mean()), 3),
                     "uncertain": round(float((status == "uncertain").mean()), 3),
                     "unsupported": round(float((status == "unsupported").mean()), 3),
                     "accuracy_among_reliable": round(float((pred[rel] == yk[rel]).mean()), 3) if rel.any() else float("nan"),
                     "wrongly_accepted_as_late_blight": int((rel & (pred == LB)).sum())})
        where[kind] = pd.Series([CLASS_NAMES[p] for p in pred]).value_counts().head(5).to_dict()
    print("\n=== The same images with the background left, turned grey (control) or turned black ===")
    print(pd.DataFrame(rows).to_string(index=False))
    for kind, d in where.items():
        print(f"Top predicted classes with {kind} background: {d}")


if __name__ == "__main__":
    main()