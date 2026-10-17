import argparse

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import DATASET_ROOT, PROJECT_ROOT, TABLES_DIR
from src.data.dataset import CLASS_NAMES, CLASS_TO_IDX
from src.reliability.features import extract, load_final_model, open_rgb
from src.reliability.scores import Scorer

OUT = PROJECT_ROOT / "reports" / "ood"
CACHE = OUT / "cache"
PHOTOS = PROJECT_ROOT / "data" / "external" / "my_photos"
GROUPS = ["tomato_leaf", "other_leaf", "non_leaf", "poor_quality"]
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
METHODS = ["max_softmax", "max_logit", "energy", "mahalanobis", "knn_cosine"]


def quality(path):
    img = open_rgb(path)
    w, h = img.size
    g = cv2.cvtColor(np.asarray(img.resize((256, 256))), cv2.COLOR_RGB2GRAY)
    return w, h, float(g.mean()), float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    model, ck = load_final_model()
    print("Model:", ck["exp_id"], "| best epoch", ck["epoch"])

    def get(name, paths):
        paths = [str(p) for p in paths]
        f = CACHE / f"{name}.npz"
        if f.exists() and not a.recompute:
            z = np.load(f)
            if list(z["paths"]) == paths:
                return z["logits"], z["feats"]
        print(f"Extracting {name} ({len(paths)} images) ...", flush=True)
        L, F = extract(model, paths, desc=name)
        np.savez(f, paths=np.array(paths), logits=L, feats=F)
        return L, F

    sm = pd.read_csv(TABLES_DIR / "split_manifest.csv")
    train = sm[sm["split"] == "train"].reset_index(drop=True)
    val = sm[sm["split"] == "val"].reset_index(drop=True)
    unk = pd.read_csv(TABLES_DIR / "unknown_eval_list.csv")
    photos = [(g, p) for g in GROUPS for p in sorted((PHOTOS / g).iterdir()) if p.suffix.lower() in EXT]

    Ltr, Ftr = get("train", [DATASET_ROOT / r for r in train["rel_path"]])
    Lva, Fva = get("val", [DATASET_ROOT / r for r in val["rel_path"]])
    Lun, Fun = get("unknown_pepper_potato", unk["path"].tolist())
    Lph, Fph = get("my_photos", [p for _, p in photos])

    ytr = train["class"].map(CLASS_TO_IDX).to_numpy()
    yva = val["class"].map(CLASS_TO_IDX).to_numpy()
    wrong = Lva.argmax(1) != yva
    print(f"Validation errors recomputed: {int(wrong.sum())} of {len(yva)} (should be 4)")

    print("Fitting the fingerprint scores on tomato training images ...", flush=True)
    scorer = Scorer(Ftr, ytr)
    sv, su, sp = (scorer.all_scores(L, F) for L, F in [(Lva, Fva), (Lun, Fun), (Lph, Fph)])

    # fixed dev/eval halves of the Pepper/Potato images (seed 42, per folder)
    unk["half"] = ""
    rng = np.random.default_rng(42)
    for _, idx in unk.groupby("folder").groups.items():
        idx = rng.permutation(np.array(list(idx)))
        unk.loc[idx[: len(idx) // 2], "half"] = "dev"
        unk.loc[idx[len(idx) // 2:], "half"] = "eval"
    unk[["path", "folder", "half"]].to_csv(OUT / "unknown_halves.csv", index=False)

    # ---- Table 1: all scores, thresholds from tomato VALIDATION only ----
    rows = []
    for m in METHODS:
        t95, t99 = np.percentile(sv[m], 5), np.percentile(sv[m], 1)
        row = {"method": m}
        for half in ("dev", "eval"):
            u = su[m][(unk["half"] == half).to_numpy()]
            y = np.r_[np.ones(len(sv[m])), np.zeros(len(u))]
            row[f"AUROC_{half}"] = round(float(roc_auc_score(y, np.r_[sv[m], u])), 4)
            row[f"falseAccept@95_{half}"] = round(float((u >= t95).mean()), 4)
            row[f"falseAccept@99_{half}"] = round(float((u >= t99).mean()), 4)
        rows.append(row)
    t1 = pd.DataFrame(rows)
    print("\n=== Pepper/Potato leaves as unknown (known = tomato validation). Choose by the DEV columns only. ===")
    print("falseAccept@95 = share of unknown images accepted when 95% of tomato validation images are accepted")
    print(t1.to_string(index=False))
    t1.to_csv(OUT / "ood_dev_summary.csv", index=False)

    # ---- Table 2: per unknown folder (all images, informational) ----
    rows = []
    for folder in sorted(unk["folder"].unique()):
        mask = (unk["folder"] == folder).to_numpy()
        r = {"folder": folder, "n": int(mask.sum())}
        for m in METHODS:
            r[m] = round(float((su[m][mask] >= np.percentile(sv[m], 5)).mean()), 3)
        rows.append(r)
    t2 = pd.DataFrame(rows)
    print("\n=== False acceptance at 95% known acceptance, per folder (dev + eval halves together) ===")
    print(t2.to_string(index=False))
    t2.to_csv(OUT / "ood_per_folder.csv", index=False)

    # ---- Table 3: 'uncertain' signal on validation, using max softmax ----
    rows = []
    msp = sv["max_softmax"]
    for q in (0.90, 0.95, 0.98, 0.99):
        tau = np.percentile(msp, 100 * (1 - q))
        flagged = msp < tau
        rows.append({"accept_target": q, "confidence_threshold": round(float(tau), 4),
                     "images_flagged_uncertain": int(flagged.sum()),
                     "validation_errors_caught": f"{int((flagged & wrong).sum())} of {int(wrong.sum())}"})
    t3 = pd.DataFrame(rows)
    print("\n=== 'Uncertain' signal on validation (max softmax; only 4 validation errors, so rough) ===")
    print(t3.to_string(index=False))
    t3.to_csv(OUT / "uncertain_val.csv", index=False)

    # ---- Table 4: your 57 photos ----
    q = [quality(p) for _, p in photos]
    pdf = pd.DataFrame({"group": [g for g, _ in photos], "file": [p.name for _, p in photos],
                        "width": [x[0] for x in q], "height": [x[1] for x in q],
                        "brightness": [round(x[2], 1) for x in q], "sharpness": [round(x[3], 1) for x in q],
                        "pred_class": [CLASS_NAMES[i] for i in Lph.argmax(1)],
                        "confidence": np.round(sp["max_softmax"], 3)})
    for m in METHODS:
        pdf[f"acc95_{m}"] = (sp[m] >= np.percentile(sv[m], 5)).astype(int)
    pdf.to_csv(OUT / "my_photos_scores.csv", index=False)
    t4 = pdf.groupby("group")[[f"acc95_{m}" for m in METHODS]].mean().round(2)
    t4.insert(0, "n", pdf.groupby("group").size())
    print("\n=== Your photos: share accepted at the 95% known-acceptance threshold (not used to choose anything) ===")
    print(t4.to_string())
    print("\n=== Tomato-leaf photos, one by one (1 = accepted at 95% threshold) ===")
    cols = ["file", "pred_class", "confidence", "acc95_max_logit", "acc95_energy", "acc95_mahalanobis", "acc95_knn_cosine"]
    show = pdf[pdf["group"] == "tomato_leaf"][cols].copy()
    show["file"] = show["file"].str[-22:]
    print(show.to_string(index=False))
    print("\n=== Poor-quality photos: size, brightness, sharpness (Phase 3 tomato images: brightness about 116, sharpness 39-20,000) ===")
    print(pdf[pdf["group"] == "poor_quality"][["file", "width", "height", "brightness", "sharpness"]].to_string(index=False))
    print("\nSaved in reports/ood/")


if __name__ == "__main__":
    main()