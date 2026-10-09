import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.data.dataset import CLASS_NAMES, TomatoDataset
from src.evaluation.calibration import nll_and_ece
from src.models.factory import build_model
from src.preprocessing.transforms import get_eval_transform
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    assert list(ck["class_names"]) == CLASS_NAMES, "class order mismatch"
    model = build_model(ck["model_name"], len(CLASS_NAMES), ck["dropout"], pretrained=False).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    ds = TomatoDataset("val", get_eval_transform())      # validation only; test is not allowed here
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2 if device == "cuda" else 0)
    logits = []
    with torch.no_grad():
        for x, _ in dl:
            logits.append(model(x.to(device)).float().cpu())
    logits = torch.cat(logits).numpy()

    y, p = np.array(ds.labels), logits.argmax(1)
    nll, ece = nll_and_ece(logits, y)
    print(f"{ck.get('exp_id', '?')} ({ck['model_name']}), best epoch {ck.get('epoch', '?')}")
    print(f"accuracy {accuracy_score(y, p):.4f} | macro F1 {f1_score(y, p, average='macro'):.4f} | "
          f"errors {int((y != p).sum())} / {len(y)} | NLL {nll:.4f} | ECE {ece:.4f}")

    df = ds.df[["rel_path", "class", "dark"]].copy()
    df["true_idx"], df["pred_idx"] = y, p
    df["pred_class"] = [CLASS_NAMES[i] for i in p]
    probs = np.exp(logits - logits.max(1, keepdims=True))
    df["confidence"] = (probs / probs.sum(1, keepdims=True)).max(1)
    for j, c in enumerate(CLASS_NAMES):
        df[f"logit_{c}"] = logits[:, j]
    out = Path(a.out) if a.out else Path(a.ckpt).with_name("val_predictions.csv")
    df.to_csv(out, index=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()