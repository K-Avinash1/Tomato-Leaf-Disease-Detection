import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT
from src.data.dataset import CLASS_NAMES, TomatoDataset
from src.evaluation.calibration import nll_and_ece, softmax_np
from src.models.checkpoint import load_checkpoint
from src.preprocessing.transforms import get_eval_transform
from src.utils.seed import SEED, set_seed

K = len(CLASS_NAMES)
LABELS = list(range(K))
EXP = PROJECT_ROOT / "reports" / "experiments"
OUT_ROOT = PROJECT_ROOT / "reports" / "test_results"
FIG = PROJECT_ROOT / "reports" / "figures"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.inference_mode()
def get_logits(model, ds):
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    return torch.cat([model(x).float() for x, _ in dl]).numpy()


def plot_confusion(cm, path, title):
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, data, sub, fmt in [(axes[0], cm, "counts", "d"), (axes[1], cmn, "row-normalized (recall per class)", ".2f")]:
        ax.imshow(data, cmap="Blues")
        ax.set_xticks(LABELS)
        ax.set_yticks(LABELS)
        ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix, {sub}")
        for i in range(K):
            for j in range(K):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > data.max() / 2 else "black")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_reliability(conf, correct, path, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
            ns.append(int(m.sum()))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax[0].plot(xs, ys, "o-", color="seagreen", label="model")
    for x_, y_, n_ in zip(xs, ys, ns):
        ax[0].annotate(str(n_), (x_, y_), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
    ax[0].set_xlabel("mean confidence in bin")
    ax[0].set_ylabel("accuracy in bin")
    ax[0].set_title("Reliability diagram (numbers = images in bin)")
    ax[0].legend()
    ax[1].hist(conf, bins=edges, color="steelblue")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("confidence")
    ax[1].set_ylabel("images (log scale)")
    ax[1].set_title("Confidence distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_curves(history, path):
    ep = [h["epoch"] for h in history]
    first = history[0]["stage"]
    boundary = max(h["epoch"] for h in history if h["stage"] == first)
    two_stage = any(h["stage"] != first for h in history)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a_, key, title in [(ax[0], "loss", "Loss"), (ax[1], "acc", "Accuracy")]:
        a_.plot(ep, [h[f"train_{key}"] for h in history], label="train")
        a_.plot(ep, [h[f"val_{key}"] for h in history], label="validation")
        if two_stage:
            a_.axvline(boundary + 0.5, color="gray", linestyle="--", label="stage A | stage B")
        a_.set_title(title)
        a_.set_xlabel("epoch")
        a_.grid(alpha=0.3)
        a_.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--role", default="final", choices=["final", "baseline_posthoc", "other_posthoc"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    set_seed()
    model, ck = load_checkpoint(a.ckpt)
    tag = ck["exp_id"]
    out = OUT_ROOT / tag
    if (out / "test_metrics.json").exists() and not a.force:
        print(f"STOP: test results for {tag} already exist in {out}.")
        print("The test set is evaluated once. Use --force only if you will report the re-run.")
        sys.exit(1)

    digest = sha256(a.ckpt)
    if Path(a.ckpt).name == "best_model.pth":
        info = json.loads((PROJECT_ROOT / "models" / "model_info.json").read_text())
        if info["sha256"] != digest:
            print("STOP: best_model.pth does not match the SHA-256 in model_info.json.")
            sys.exit(1)
        print("Check 1: SHA-256 matches model_info.json")

    tf = get_eval_transform()
    val_ds = TomatoDataset("val", tf)
    stored = pd.read_csv(EXP / tag / "val_predictions.csv")
    mism = int((get_logits(model, val_ds).argmax(1) != stored["pred_idx"].to_numpy()).sum())
    print(f"Check 2: validation predictions differ from the stored Colab predictions on {mism} of {len(stored)} images")
    if mism > 2:
        print("STOP: the model does not reproduce its validation predictions.")
        sys.exit(1)

    print("\nEvaluating the TEST set (once) ...")
    test_ds = TomatoDataset("test", tf)
    t0 = time.time()
    logits = get_logits(model, test_ds)
    secs = time.time() - t0
    y, p = np.array(test_ds.labels), logits.argmax(1)
    n = len(y)
    probs = softmax_np(logits.astype(np.float64))
    Y = label_binarize(y, classes=LABELS)

    acc = float(accuracy_score(y, p))
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, average="macro", labels=LABELS, zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y, p, average="weighted", labels=LABELS, zero_division=0)
    report = classification_report(y, p, labels=LABELS, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, p, labels=LABELS)
    errors = int((y != p).sum())

    ci = binomtest(n - errors, n).proportion_ci(confidence_level=0.95, method="exact")
    rng = np.random.default_rng(SEED)
    f1s = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        f1s.append(f1_score(y[idx], p[idx], average="macro", labels=LABELS, zero_division=0))
    f1_lo, f1_hi = np.percentile(f1s, [2.5, 97.5])

    auc_macro = float(roc_auc_score(Y, probs, average="macro"))
    auc_weighted = float(roc_auc_score(Y, probs, average="weighted"))
    auc_cls = [float(roc_auc_score(Y[:, k], probs[:, k])) for k in LABELS]
    nll, ece = nll_and_ece(logits, y)
    brier = float(((probs - Y) ** 2).sum(1).mean())
    conf = probs.max(1)
    correct = (y == p)

    dark = test_ds.df["dark"].to_numpy().astype(bool)
    lb = CLASS_NAMES.index("Late Blight")
    subsets = {}
    for name, m in [("dark (black background)", dark), ("not dark", ~dark),
                    ("Late Blight, dark", (y == lb) & dark), ("Late Blight, not dark", (y == lb) & ~dark)]:
        subsets[name] = {"n": int(m.sum()), "errors": int((~correct[m]).sum()),
                         "accuracy": float(correct[m].mean()) if m.any() else None}

    pc = pd.DataFrame({c: {"precision": report[c]["precision"], "recall": report[c]["recall"],
                           "f1": report[c]["f1-score"], "support": int(report[c]["support"]),
                           "roc_auc": auc_cls[k]} for k, c in enumerate(CLASS_NAMES)}).T

    out.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    pc.to_csv(out / "per_class_test.csv")
    np.savetxt(out / "confusion_matrix_test.csv", cm, fmt="%d", delimiter=",", header=",".join(CLASS_NAMES))
    plot_confusion(cm, out / "confusion_matrix_test.png", f"{tag}: test set ({n} images)")
    plot_reliability(conf, correct, out / "reliability_test.png")
    pred_df = test_ds.df[["rel_path", "class", "dark"]].copy()
    pred_df["true_idx"], pred_df["pred_idx"] = y, p
    pred_df["pred_class"] = [CLASS_NAMES[i] for i in p]
    pred_df["confidence"] = conf
    for j, c in enumerate(CLASS_NAMES):
        pred_df[f"logit_{c}"] = logits[:, j]
    pred_df.to_csv(out / "test_predictions.csv", index=False)
    wrong = pred_df[pred_df["true_idx"] != pred_df["pred_idx"]].sort_values("confidence", ascending=False)
    wrong[["rel_path", "class", "pred_class", "confidence", "dark"]].to_csv(out / "misclassified_test.csv", index=False)
    if a.role == "final":
        hist = json.loads((EXP / tag / "history.json").read_text())
        plot_curves(hist, FIG / "fig07_training_curves_final.png")
        plot_confusion(cm, FIG / "fig08_confusion_matrix_test.png", f"{tag}: test set ({n} images)")

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    info_path = PROJECT_ROOT / "models" / "model_info.json"
    info = json.loads(info_path.read_text()) if (Path(a.ckpt).name == "best_model.pth" and info_path.exists()) else {}
    result = {
        "experiment": tag, "role": a.role, "evaluated_on": "test", "n_images": n,
        "when": datetime.now().isoformat(timespec="seconds"), "git_commit": commit, "checkpoint_sha256": digest,
        "metrics": {"accuracy": acc, "errors": errors, "accuracy_ci95_exact": [ci.low, ci.high],
                    "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
                    "macro_f1_ci95_bootstrap": [float(f1_lo), float(f1_hi)],
                    "weighted_precision": float(wp), "weighted_recall": float(wr), "weighted_f1": float(wf)},
        "roc_auc_ovr": {"macro": auc_macro, "weighted": auc_weighted},
        "calibration": {"nll": nll, "ece_15bins": ece, "brier": brier,
                        "mean_confidence_correct": float(conf[correct].mean()),
                        "mean_confidence_wrong": float(conf[~correct].mean()) if errors else None},
        "subsets": subsets,
        "efficiency": {"parameters": sum(p_.numel() for p_ in model.parameters()),
                       "file_mb": round(Path(a.ckpt).stat().st_size / 1e6, 1),
                       "cpu_total_seconds_for_test_set": round(secs, 1),
                       "laptop_benchmark": info.get("cpu_benchmark_laptop")},
        "torch": torch.__version__,
    }
    (out / "test_metrics.json").write_text(json.dumps(result, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    with open(OUT_ROOT / "TEST_LOG.txt", "a") as f:
        f.write(f"{result['when']} | {tag} | {a.role} | sha {digest[:12]} | commit {commit[:10]} | force={a.force}\n")

    pd.set_option("display.width", 200)
    print(f"\n=== TEST RESULTS: {tag} ({a.role}), {n} images ===")
    print(f"Accuracy: {acc:.4f} ({errors} errors) | exact 95% CI [{ci.low:.4f}, {ci.high:.4f}]")
    print(f"Macro precision {mp:.4f} | macro recall {mr:.4f} | macro F1 {mf:.4f} (bootstrap 95% CI [{f1_lo:.4f}, {f1_hi:.4f}])")
    print(f"Weighted precision {wp:.4f} | weighted recall {wr:.4f} | weighted F1 {wf:.4f}")
    print(f"ROC-AUC (one-vs-rest): macro {auc_macro:.5f} | weighted {auc_weighted:.5f}")
    print(f"Calibration: NLL {nll:.4f} | ECE {ece:.4f} | Brier {brier:.4f}")
    print(f"Mean confidence: correct {conf[correct].mean():.4f} | wrong {conf[~correct].mean() if errors else float('nan'):.4f}")
    print("\nPer class:")
    print(pc.round(4).to_string())
    print("\nSubsets:")
    for k_, v in subsets.items():
        print(f"  {k_}: n={v['n']}, errors={v['errors']}, accuracy={v['accuracy']}")
    print(f"\nMisclassified images (highest confidence first), {errors} total:")
    print(wrong[["class", "pred_class", "confidence", "dark"]].head(30).round(4).to_string())
    print(f"\nTime for the whole test set on this CPU: {secs:.0f} s | checkpoint {result['efficiency']['file_mb']} MB")
    print(f"Saved in {out}")


if __name__ == "__main__":
    main()