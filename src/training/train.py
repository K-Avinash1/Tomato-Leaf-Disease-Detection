import argparse
import json
import os
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from torch.utils.data import DataLoader

from src.data.dataset import CLASS_NAMES, TomatoDataset
from src.models.factory import build_model, count_params
from src.preprocessing.transforms import (AUG_CONFIG, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE,
                                          get_eval_transform, get_train_transform)
from src.utils.seed import SEED, set_seed

K = len(CLASS_NAMES)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--model", default="custom_cnn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-batches", type=int, default=0, help="debug only: limit batches per epoch")
    return ap.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=0):
    training = optimizer is not None
    model.train(training)
    loss_sum, n = 0.0, 0
    preds, labels = [], []
    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
            preds.append(out.argmax(1).cpu())
            labels.append(y.cpu())
    return loss_sum / n, torch.cat(preds).numpy(), torch.cat(labels).numpy()


def summary_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))}


def save_checkpoint(path, payload):
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)          # a disconnect during saving cannot leave a broken file


def plot_curves(history, path):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validation")
    ax[0].set_title("Loss")
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="validation")
    ax[1].set_title("Accuracy")
    ax[2].plot(ep, [h["val_macro_f1"] for h in history], color="seagreen", label="validation macro F1")
    ax[2].set_title("Validation macro F1")
    for a_ in ax:
        a_.set_xlabel("epoch")
        a_.legend()
        a_.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_confusion(cm, path):
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, data, title, fmt in [(axes[0], cm, "Confusion matrix (counts)", "d"),
                                 (axes[1], cmn, "Row-normalized (recall per class)", ".2f")]:
        ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > data.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    a = parse_args()
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out_dir or f"runs/{a.exp_id}")
    out.mkdir(parents=True, exist_ok=True)
    workers = 2 if device == "cuda" else 0

    # Only train and validation are loaded here. The test split is never touched.
    train_ds = TomatoDataset("train", get_train_transform())
    val_ds = TomatoDataset("val", get_eval_transform())
    kw = dict(num_workers=workers, pin_memory=(device == "cuda"), persistent_workers=workers > 0)
    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, **kw)

    model = build_model(a.model, K, a.dropout).to(device)
    n_params = count_params(model)

    weight = None
    if a.class_weights:
        counts = np.bincount(train_ds.labels, minlength=K)
        weight = torch.tensor(len(train_ds.labels) / (K * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)

    hardware = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    config = {**vars(a), "seed": SEED, "device": device, "hardware": hardware, "img_size": IMG_SIZE,
              "aug_config": AUG_CONFIG, "params": n_params, "python": platform.python_version(),
              "torch": torch.__version__, "torchvision": torchvision.__version__,
              "train_images": len(train_ds), "val_images": len(val_ds)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))

    history, best_f1, best_epoch, start, train_seconds = [], -1.0, 0, 0, 0.0
    last_path, best_path = out / "last.pth", out / "best_model.pth"
    if a.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        history, best_f1, best_epoch = ck["history"], ck["best_f1"], ck["best_epoch"]
        start, train_seconds = ck["epoch"], ck["train_seconds"]
        print(f"Resumed from epoch {start}")

    for epoch in range(start + 1, a.epochs + 1):
        t0 = time.time()
        tr_loss, tr_p, tr_y = run_epoch(model, train_dl, criterion, device, optimizer, a.max_batches)
        scheduler.step()
        va_loss, va_p, va_y = run_epoch(model, val_dl, criterion, device, None, a.max_batches)
        secs = time.time() - t0
        train_seconds += secs
        m = summary_metrics(va_y, va_p)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": float((tr_p == tr_y).mean()),
                        "val_loss": va_loss, "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"],
                        "lr": optimizer.param_groups[0]["lr"], "seconds": secs})
        print(f"Epoch {epoch:3d}/{a.epochs} | train loss {tr_loss:.4f} acc {history[-1]['train_acc']:.4f} | "
              f"val loss {va_loss:.4f} acc {m['accuracy']:.4f} macro-F1 {m['macro_f1']:.4f} | {secs:.0f}s")

        if m["macro_f1"] > best_f1:
            best_f1, best_epoch = m["macro_f1"], epoch
            save_checkpoint(best_path, {
                "model_state": model.state_dict(), "model_name": a.model, "dropout": a.dropout,
                "class_names": CLASS_NAMES, "img_size": IMG_SIZE,
                "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                "val_macro_f1": best_f1, "epoch": epoch, "exp_id": a.exp_id, "seed": SEED})
        save_checkpoint(last_path, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "history": history,
            "best_f1": best_f1, "best_epoch": best_epoch, "train_seconds": train_seconds})
        (out / "history.json").write_text(json.dumps(history, indent=2))

    # Final validation evaluation with the best checkpoint
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    va_loss, va_p, va_y = run_epoch(model, val_dl, criterion, device, None, a.max_batches)
    m = summary_metrics(va_y, va_p)
    report = classification_report(va_y, va_p, labels=list(range(K)), target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(va_y, va_p, labels=list(range(K)))
    np.savetxt(out / "confusion_matrix_val.csv", cm, fmt="%d", delimiter=",", header=",".join(CLASS_NAMES))
    plot_confusion(cm, out / "confusion_matrix_val.png")
    plot_curves(history, out / "learning_curves.png")
    result = {"exp_id": a.exp_id, "evaluated_on": "validation", "best_epoch": best_epoch,
              "train_seconds": train_seconds, "params": n_params, "hardware": hardware,
              "metrics": m, "per_class": report}
    (out / "metrics_val.json").write_text(json.dumps(result, indent=2))

    print("\n=== Validation results (best checkpoint) ===")
    print(classification_report(va_y, va_p, labels=list(range(K)), target_names=CLASS_NAMES,
                                digits=4, zero_division=0))
    print("Summary:", {k: round(v, 4) for k, v in m.items()})
    print(f"Best epoch: {best_epoch} | parameters: {n_params:,} | training time: {train_seconds / 60:.1f} min")
    print(f"\n--- EXPERIMENT_LOG entry (copy into EXPERIMENT_LOG.md, then fill Observations/Conclusion) ---")
    print(f"""
### {a.exp_id}
- **Objective:** baseline custom CNN trained from scratch (reference for transfer learning)
- **Dataset version:** PlantVillage tomato, 15,997 clean images (clean_manifest.csv)
- **Data split:** 70/15/15, seed {SEED}, stratified, near-duplicate-group-aware (train {len(train_ds)}, val {len(val_ds)}); test not used
- **Preprocessing:** resize {IMG_SIZE}x{IMG_SIZE}, ImageNet mean/std normalization
- **Augmentation:** {AUG_CONFIG} (training only)
- **Model / architecture:** {a.model}, 5 conv stages + global average pooling, {n_params:,} parameters
- **Optimizer / LR:** AdamW, lr {a.lr}, weight decay {a.weight_decay}, cosine schedule
- **Batch size / epochs:** {a.batch_size} / {a.epochs}
- **Hyperparameters:** dropout {a.dropout}, class weights {'on' if a.class_weights else 'off'}
- **Hardware:** {hardware} (PyTorch {torch.__version__}, Python {platform.python_version()})
- **Training time:** {train_seconds / 60:.1f} min
- **Validation (best epoch {best_epoch}):** accuracy {m['accuracy']:.4f}, macro precision {m['macro_precision']:.4f}, macro recall {m['macro_recall']:.4f}, macro F1 {m['macro_f1']:.4f}, weighted F1 {m['weighted_f1']:.4f}
- **Observations:** <fill in>
- **Conclusion:** <fill in>
- **Retained / Rejected:** <fill in>
""")


if __name__ == "__main__":
    main()