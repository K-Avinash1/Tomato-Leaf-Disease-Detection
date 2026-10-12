import argparse
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.config import DATASET_ROOT, PROJECT_ROOT, TABLES_DIR
from src.data.dataset import CLASS_NAMES, TomatoDataset
from src.evaluation.calibration import softmax_np
from src.models.checkpoint import load_checkpoint
from src.preprocessing.transforms import get_eval_transform
from src.utils.seed import SEED

UNKNOWN_FOLDERS = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
                   "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
PER_FOLDER = 600
EXT = {".jpg", ".jpeg", ".png"}
EXP = PROJECT_ROOT / "reports" / "experiments"
LOGIT_COLS = [f"logit_{c}" for c in CLASS_NAMES]


class ImageList(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def build_unknown_list():
    rnd = random.Random(SEED)
    rows = []
    for folder in UNKNOWN_FOLDERS:
        d = DATASET_ROOT / folder
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in EXT)
        for p in rnd.sample(files, min(PER_FOLDER, len(files))):
            rows.append({"folder": folder, "path": str(p)})
    return pd.DataFrame(rows)


@torch.inference_mode()
def logits_for(model, ds):
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    return torch.cat([model(x).float() for x in dl]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    a = ap.parse_args()
    tf = get_eval_transform()
    unk = build_unknown_list()
    unk.to_csv(TABLES_DIR / "unknown_eval_list.csv", index=False)
    print("Unknown images per folder:", unk["folder"].value_counts().to_dict())

    val_ds = TomatoDataset("val", tf)
    summary, per_folder = [], []
    for path in a.ckpt:
        model, ck = load_checkpoint(path)
        exp = ck["exp_id"]
        vp = pd.read_csv(next(EXP.glob(exp.split("_")[0] + "_*")) / "val_predictions.csv")
        known = vp[LOGIT_COLS].to_numpy()

        x = torch.stack([val_ds[i][0] for i in range(64)])
        with torch.inference_mode():
            diff = float(np.abs(model(x).numpy() - known[:64]).max())
        print(f"\n=== {exp} | local CPU vs Colab logits, max abs difference on 64 val images: {diff:.5f}")

        unk_logits = logits_for(model, ImageList(unk["path"].tolist(), tf))
        k_ms, u_ms = softmax_np(known).max(1), softmax_np(unk_logits).max(1)
        k_ml, u_ml = known.max(1), unk_logits.max(1)
        y = np.r_[np.ones(len(k_ms)), np.zeros(len(u_ms))]
        row = {"exp": exp}
        for name, k, u in [("max_softmax", k_ms, u_ms), ("max_logit", k_ml, u_ml)]:
            thr = np.percentile(k, 5)                    # lets 95% of known tomato images through
            row[f"AUROC_{name}"] = round(float(roc_auc_score(y, np.r_[k, u])), 4)
            row[f"unknown_accepted_at_95pct_known_{name}"] = round(float((u >= thr).mean()), 4)
        row["median_conf_known"] = round(float(np.median(k_ms)), 4)
        row["median_conf_unknown"] = round(float(np.median(u_ms)), 4)
        row["known_conf_ge_0.9"] = round(float((k_ms >= 0.9).mean()), 4)
        row["unknown_conf_ge_0.9"] = round(float((u_ms >= 0.9).mean()), 4)
        row["unknown_conf_ge_0.99"] = round(float((u_ms >= 0.99).mean()), 4)
        summary.append(row)

        thr = np.percentile(k_ms, 5)
        pred = unk_logits.argmax(1)
        for folder in UNKNOWN_FOLDERS:
            m = (unk["folder"] == folder).to_numpy()
            per_folder.append({"exp": exp, "folder": folder, "n": int(m.sum()),
                               "accepted_at_95pct_known": round(float((u_ms[m] >= thr).mean()), 3),
                               "median_conf": round(float(np.median(u_ms[m])), 3),
                               "most_common_tomato_label": CLASS_NAMES[int(np.bincount(pred[m], minlength=10).argmax())]})

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    s, p = pd.DataFrame(summary), pd.DataFrame(per_folder)
    print("\n=== Summary (known = validation tomato images, unknown = Pepper/Potato) ===")
    print(s.to_string(index=False))
    print("\n=== Per unknown folder ===")
    print(p.to_string(index=False))
    s.to_csv(TABLES_DIR / "unknown_check_summary.csv", index=False)
    p.to_csv(TABLES_DIR / "unknown_check_per_folder.csv", index=False)


if __name__ == "__main__":
    main()