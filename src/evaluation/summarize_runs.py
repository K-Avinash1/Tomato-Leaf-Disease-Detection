import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import PROJECT_ROOT, TABLES_DIR
from src.data.dataset import CLASS_NAMES
from src.evaluation.calibration import nll_and_ece

EXP = PROJECT_ROOT / "reports" / "experiments"
EB = CLASS_NAMES.index("Early Blight")


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def change_label(cfg):
    if cfg.get("model") == "custom_cnn":
        return "from scratch, 30 epochs"
    parts = [f"unfreeze={cfg.get('unfreeze', 'partial')}"]
    if cfg.get("lr_b", 1e-4) != 1e-4:
        parts.append(f"lr_b={cfg['lr_b']}")
    if cfg.get("aug", "default") != "default":
        parts.append(f"aug={cfg['aug']}")
    if cfg.get("class_weights"):
        parts.append("class-weights")
    if cfg.get("seed", 42) != 42:
        parts.append(f"seed={cfg['seed']}")
    return ", ".join(parts)


def main():
    rows, wrong, labels = [], {}, None
    for d in sorted(EXP.iterdir()):
        f = d / "val_predictions.csv"
        if not (d.is_dir() and d.name.startswith("E") and f.exists()):
            continue
        df = pd.read_csv(f)
        y, p = df["true_idx"].to_numpy(), df["pred_idx"].to_numpy()
        logits = df[[f"logit_{c}" for c in CLASS_NAMES]].to_numpy()
        nll, ece = nll_and_ece(logits, y)
        cfg, hist, met = read_json(d / "config.json"), read_json(d / "history.json"), read_json(d / "metrics_val.json")
        rows.append({
            "run": d.name, "model": cfg.get("model", "?"), "setup": change_label(cfg),
            "errors": int((y != p).sum()), "acc": round(float((y == p).mean()), 4),
            "macroF1": round(float(f1_score(y, p, average="macro")), 4),
            "lastF1": round(hist[-1]["val_macro_f1"], 4) if hist else np.nan,
            "NLL": round(nll, 4), "ECE": round(ece, 4),
            "EarlyBlightRecall": round(float(((p == EB) & (y == EB)).sum() / (y == EB).sum()), 3),
            "params_M": round(cfg.get("params", 0) / 1e6, 2),
            "ckpt_MB": round(met["checkpoint_mb"], 1) if "checkpoint_mb" in met else np.nan,
            "minutes": round(met.get("train_seconds", np.nan) / 60, 1),
        })
        wrong[d.name] = pd.Series(y != p, index=df["rel_path"])
        labels = df.set_index("rel_path")["class"]

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.to_string(index=False))
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "run_comparison_val.csv", index=False)

    W = pd.DataFrame(wrong)
    n_wrong = W.sum(axis=1)
    print(f"\nRuns compared: {W.shape[1]}")
    print("Images misclassified by k runs (k: number of images):")
    print(n_wrong[n_wrong > 0].value_counts().sort_index().to_string())
    hard = pd.DataFrame({"class": labels[n_wrong.index], "runs_wrong": n_wrong})
    hard = hard[hard["runs_wrong"] >= 3].sort_values("runs_wrong", ascending=False)
    hard.to_csv(TABLES_DIR / "hard_images_val.csv")
    print(f"\nImages misclassified by at least 3 runs: {len(hard)}")
    print(hard["class"].value_counts().to_string())
    print("Saved: run_comparison_val.csv, hard_images_val.csv")


if __name__ == "__main__":
    main()