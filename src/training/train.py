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
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from torch.utils.data import DataLoader

from src.data.dataset import CLASS_NAMES, TomatoDataset
from src.models.factory import (PARTIAL_DESC, build_model, count_params, freeze_frozen_bn,
                                set_trainable, trainable_params)
from src.preprocessing.transforms import (AUG_CONFIG, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE,
                                          get_eval_transform, get_train_transform)
from src.utils.seed import SEED, set_seed

K = len(CLASS_NAMES)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--model", default="custom_cnn")
    ap.add_argument("--strategy", default="auto", choices=["auto", "full", "two_stage"])
    ap.add_argument("--epochs", type=int, default=30, help="epochs for the 'full' strategy")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate for the 'full' strategy")
    ap.add_argument("--stage-a-epochs", type=int, default=5)
    ap.add_argument("--stage-b-epochs", type=int, default=15)
    ap.add_argument("--lr-a", type=float, default=1e-3)
    ap.add_argument("--lr-b", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-batches", type=int, default=0, help="debug only: limit batches per epoch")
    ap.add_argument("--stop-after", type=int, default=0, help="debug only: stop after this many epochs")
    return ap.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=0):
    training = optimizer is not None
    model.train(training)
    if training:
        freeze_frozen_bn(model)
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


@torch.no_grad()
def predict_all(model, loader, device, max_batches=0):
    model.eval()
    logits = []
    for i, (x, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        logits.append(model(x.to(device)).float().cpu())
    return torch.cat(logits).numpy()


def summary_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))}


def save_checkpoint(path, payload):
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)          # a disconnect during saving cannot leave a broken file


def plot_curves(history, boundary, path):
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
        if boundary:
            a_.axvline(boundary + 0.5, color="gray", linestyle="--", label="stage A | stage B")
        a_.set_xlabel("epoch")
        a_.legend()
        a_.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_confusion(cm, path):
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
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


def build_plan(a):
    strategy = a.strategy
    if strategy == "auto":
        strategy = "full" if a.model == "custom_cnn" else "two_stage"
    if strategy == "full":
        return strategy, [("full", "full", a.epochs, a.lr)]
    return strategy, [("A_head", "head", a.stage_a_epochs, a.lr_a),
                      ("B_partial", "partial", a.stage_b_epochs, a.lr_b)]


def main():
    a = parse_args()
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out_dir or f"runs/{a.exp_id}")
    out.mkdir(parents=True, exist_ok=True)
    workers = 2 if device == "cuda" else 0
    strategy, plan = build_plan(a)
    total_epochs = sum(p[2] for p in plan)

    def stage_index(ep):             # ep is the global epoch number, starting at 1
        left = ep
        for i, p in enumerate(plan):
            if left <= p[2]:
                return i
            left -= p[2]
        return len(plan) - 1

    # Only train and validation are loaded here. The test split is never touched.
    train_ds = TomatoDataset("train", get_train_transform())
    val_ds = TomatoDataset("val", get_eval_transform())
    kw = dict(num_workers=workers, pin_memory=(device == "cuda"), persistent_workers=workers > 0)
    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, **kw)

    pretrained = a.model != "custom_cnn"
    model = build_model(a.model, K, a.dropout, pretrained=pretrained).to(device)
    n_params = count_params(model)

    weight = None
    if a.class_weights:
        counts = np.bincount(train_ds.labels, minlength=K)
        weight = torch.tensor(len(train_ds.labels) / (K * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    hardware = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    config = {**vars(a), "strategy_used": strategy, "plan": plan, "total_epochs": total_epochs,
              "pretrained_weights": "torchvision IMAGENET1K_V1" if pretrained else "none (from scratch)",
              "seed": SEED, "device": device, "hardware": hardware, "img_size": IMG_SIZE,
              "aug_config": AUG_CONFIG, "params": n_params, "python": platform.python_version(),
              "torch": torch.__version__, "torchvision": torchvision.__version__,
              "train_images": len(train_ds), "val_images": len(val_ds)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))

    history, best_f1, best_epoch, done, train_seconds = [], -1.0, 0, 0, 0.0
    last_path, best_path = out / "last.pth", out / "best_model.pth"
    resumed = None
    if a.resume and last_path.exists():
        resumed = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(resumed["model"])
        history, best_f1, best_epoch = resumed["history"], resumed["best_f1"], resumed["best_epoch"]
        done, train_seconds = resumed["epoch"], resumed["train_seconds"]
        print(f"Resumed from epoch {done}")

    cur, optimizer, scheduler = None, None, None
    for epoch in range(done + 1, total_epochs + 1):
        idx = stage_index(epoch)
        if idx != cur:
            name, mode, n_ep, lr = plan[idx]
            set_trainable(model, a.model, mode)
            params = trainable_params(model)
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=a.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_ep)
            n_train_params = sum(p.numel() for p in params)
            print(f"\n--- Stage {name}: {n_ep} epochs, lr {lr}, trainable parameters "
                  f"{n_train_params:,} of {n_params:,} ---")
            if resumed is not None and resumed.get("stage_idx") == idx:
                optimizer.load_state_dict(resumed["optimizer"])
                scheduler.load_state_dict(resumed["scheduler"])
            resumed, cur = None, idx

        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_p, tr_y = run_epoch(model, train_dl, criterion, device, optimizer, a.max_batches)
        scheduler.step()
        va_loss, va_p, va_y = run_epoch(model, val_dl, criterion, device, None, a.max_batches)
        secs = time.time() - t0
        train_seconds += secs
        m = summary_metrics(va_y, va_p)
        history.append({"epoch": epoch, "stage": plan[idx][0], "train_loss": tr_loss,
                        "train_acc": float((tr_p == tr_y).mean()), "val_loss": va_loss,
                        "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"],
                        "lr": lr_now, "seconds": secs})
        print(f"Epoch {epoch:3d}/{total_epochs} [{plan[idx][0]}] | train loss {tr_loss:.4f} "
              f"acc {history[-1]['train_acc']:.4f} | val loss {va_loss:.4f} acc {m['accuracy']:.4f} "
              f"macro-F1 {m['macro_f1']:.4f} | {secs:.0f}s")

        if m["macro_f1"] > best_f1:
            best_f1, best_epoch = m["macro_f1"], epoch
            save_checkpoint(best_path, {
                "model_state": model.state_dict(), "model_name": a.model, "dropout": a.dropout,
                "num_classes": K, "class_names": CLASS_NAMES, "img_size": IMG_SIZE,
                "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "val_macro_f1": best_f1,
                "epoch": epoch, "stage": plan[idx][0], "exp_id": a.exp_id, "seed": SEED})
        save_checkpoint(last_path, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "stage_idx": idx,
            "history": history, "best_f1": best_f1, "best_epoch": best_epoch,
            "train_seconds": train_seconds})
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if a.stop_after and epoch >= a.stop_after:
            print(f"(debug) stopping after epoch {epoch}")
            break

    # ---- final validation evaluation with the best checkpoint ----
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    logits = predict_all(model, val_dl, device, a.max_batches)
    n = len(logits)
    va_y = np.array(val_ds.labels[:n])
    va_p = logits.argmax(1)
    m = summary_metrics(va_y, va_p)
    report = classification_report(va_y, va_p, labels=list(range(K)), target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(va_y, va_p, labels=list(range(K)))
    np.savetxt(out / "confusion_matrix_val.csv", cm, fmt="%d", delimiter=",", header=",".join(CLASS_NAMES))
    plot_confusion(cm, out / "confusion_matrix_val.png")
    boundary = plan[0][2] if len(plan) > 1 else 0
    plot_curves(history, boundary, out / "learning_curves.png")

    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pred_df = val_ds.df.iloc[:n][["rel_path", "class", "dark"]].copy()
    pred_df["true_idx"], pred_df["pred_idx"] = va_y, va_p
    pred_df["pred_class"] = [CLASS_NAMES[i] for i in va_p]
    pred_df["confidence"] = probs.max(1)
    for j, c in enumerate(CLASS_NAMES):
        pred_df[f"logit_{c}"] = logits[:, j]
    pred_df.to_csv(out / "val_predictions.csv", index=False)

    stage_summary = {}
    for p in plan:
        rows = [h for h in history if h["stage"] == p[0]]
        if rows:
            b = max(rows, key=lambda h: h["val_macro_f1"])
            stage_summary[p[0]] = {"best_epoch": b["epoch"], "best_val_macro_f1": b["val_macro_f1"],
                                   "last_epoch_val_macro_f1": rows[-1]["val_macro_f1"]}
    size_mb = best_path.stat().st_size / 1e6
    result = {"exp_id": a.exp_id, "evaluated_on": "validation", "best_epoch": best_epoch,
              "train_seconds": train_seconds, "params": n_params, "checkpoint_mb": size_mb,
              "hardware": hardware, "plan": plan, "stage_summary": stage_summary,
              "metrics": m, "per_class": report}
    (out / "metrics_val.json").write_text(json.dumps(result, indent=2))

    print("\n=== Validation results (best checkpoint) ===")
    print(classification_report(va_y, va_p, labels=list(range(K)), target_names=CLASS_NAMES,
                                digits=4, zero_division=0))
    print("Summary:", {k: round(v, 4) for k, v in m.items()})
    for k, v in stage_summary.items():
        print(f"Stage {k}: best val macro-F1 {v['best_val_macro_f1']:.4f} (epoch {v['best_epoch']}), "
              f"last epoch {v['last_epoch_val_macro_f1']:.4f}")
    print(f"Best epoch: {best_epoch} | parameters: {n_params:,} | checkpoint: {size_mb:.1f} MB | "
          f"training time: {train_seconds / 60:.1f} min")

    plan_text = "; ".join(f"{p[0]}: {p[2]} epochs, lr {p[3]}" for p in plan)
    unfrozen = ("all layers" if strategy == "full" else
                f"stage A head only; stage B {PARTIAL_DESC.get(a.model, 'head + deepest blocks')}")
    print("\n--- EXPERIMENT_LOG entry (copy into EXPERIMENT_LOG.md, then fill Observations/Conclusion) ---")
    print(f"""
### {a.exp_id}
- **Objective:** {'baseline custom CNN from scratch' if strategy == 'full' else 'transfer learning: ' + a.model + ', feature extraction then partial fine-tuning'}
- **Dataset version:** PlantVillage tomato, 15,997 clean images (clean_manifest.csv)
- **Data split:** 70/15/15, seed {SEED}, stratified, near-duplicate-group-aware (train {len(train_ds)}, val {len(val_ds)}); test not used
- **Preprocessing:** resize {IMG_SIZE}x{IMG_SIZE}, ImageNet mean/std normalization
- **Augmentation:** {AUG_CONFIG} (training only)
- **Model / architecture:** {a.model}, {config['pretrained_weights']}, {n_params:,} parameters
- **Trainable layers:** {unfrozen}
- **Optimizer / LR:** AdamW, weight decay {a.weight_decay}, cosine schedule per stage; {plan_text}
- **Batch size / epochs:** {a.batch_size} / {total_epochs}
- **Hyperparameters:** dropout {a.dropout}, class weights {'on' if a.class_weights else 'off'}
- **Hardware:** {hardware} (PyTorch {torch.__version__}, Python {platform.python_version()})
- **Training time:** {train_seconds / 60:.1f} min | checkpoint {size_mb:.1f} MB
- **Validation (best epoch {best_epoch}):** accuracy {m['accuracy']:.4f}, macro precision {m['macro_precision']:.4f}, macro recall {m['macro_recall']:.4f}, macro F1 {m['macro_f1']:.4f}, weighted F1 {m['weighted_f1']:.4f}
- **Stage results (best val macro F1):** {', '.join(f"{k} {v['best_val_macro_f1']:.4f}" for k, v in stage_summary.items())}
- **Observations:** <fill in>
- **Conclusion:** <fill in>
- **Retained / Rejected:** <fill in>
""")


if __name__ == "__main__":
    main()