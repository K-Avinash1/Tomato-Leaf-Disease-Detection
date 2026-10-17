import argparse
import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, TABLES_DIR
from src.data.dataset import CLASS_NAMES, CLASS_TO_IDX
from src.reliability.decision import ReliabilityGate, softmax_max
from src.reliability.knn import KnnBank

OUT = PROJECT_ROOT / "reports" / "ood"
CACHE = OUT / "cache"
MODELS = PROJECT_ROOT / "models"
X_GRID = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
X_CONF = 0.98
STATES = ("reliable", "uncertain", "unsupported")


def load(name):
    z = np.load(CACHE / f"{name}.npz")
    return z["logits"], z["feats"]


def share(status, mask):
    s = status[mask]
    return {k: round(float((s == k).mean()), 3) for k in STATES}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.08)
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    Ltr, Ftr = load("train")
    Lva, Fva = load("val")
    Lun, Fun = load("unknown_pepper_potato")
    Lph, Fph = load("my_photos")
    sm = pd.read_csv(TABLES_DIR / "split_manifest.csv")
    yva = sm[sm["split"] == "val"]["class"].map(CLASS_TO_IDX).to_numpy()
    wrong = Lva.argmax(1) != yva
    unk = pd.read_csv(OUT / "unknown_halves.csv")
    photos = pd.read_csv(OUT / "my_photos_scores.csv")

    KnnBank.from_features(Ftr, k=10).save(MODELS / "knn_bank.npz")
    bank = KnnBank.load(MODELS / "knn_bank.npz")        # thresholds use exactly the stored bank
    sim_va, sim_un = bank.similarity(Fva), bank.similarity(Fun)
    conf_va, conf_un = softmax_max(Lva), softmax_max(Lun)
    tau_conf = float(np.percentile(conf_va, 100 * (1 - X_CONF)))
    print(f"Confidence threshold (accept {X_CONF:.0%} of tomato validation images): {tau_conf:.4f}")

    rows = []
    for X in X_GRID:
        tk = float(np.percentile(sim_va, 100 * (1 - X)))
        rej = (sim_va < tk) | (conf_va < tau_conf)
        r = {"knn_accept": X, "tau_knn": round(tk, 4), "val_false_rejection": round(float(rej.mean()), 4),
             "val_errors_still_accepted": int((wrong & ~rej).sum())}
        for half in ("dev", "eval"):
            m = (unk["half"] == half).to_numpy()
            r[f"unknown_accepted_{half}"] = round(float(((sim_un[m] >= tk) & (conf_un[m] >= tau_conf)).mean()), 4)
        rows.append(r)
    grid = pd.DataFrame(rows)
    print(f"\n=== Operating points (thresholds from tomato validation only; unknown = Pepper/Potato) ===")
    print(grid.to_string(index=False))
    grid.to_csv(OUT / "operating_points.csv", index=False)

    ok = [r for r in rows if r["val_false_rejection"] <= a.budget]
    chosen = min(ok, key=lambda r: r["knn_accept"]) if ok else rows[-1]
    if not ok:
        print("WARNING: no setting fits the budget; using the most permissive one")
    tau_knn = chosen["tau_knn"]
    print(f"\nRejection budget {a.budget:.0%} -> strictest kNN setting inside it: accept {chosen['knn_accept']:.0%} "
          f"of tomato validation images, kNN threshold {tau_knn}")

    gate = ReliabilityGate(bank, tau_knn, tau_conf)
    st_va, sim_v, conf_v = gate.decide(Lva, Fva)
    st_un, _, _ = gate.decide(Lun, Fun)
    st_ph, sim_p, conf_p = gate.decide(Lph, Fph)

    acc = st_va == "reliable"
    print("\n=== Tomato validation images with the chosen rule ===")
    print("Status shares:", {k: round(float((st_va == k).mean()), 4) for k in STATES})
    print(f"Accuracy on images the rule accepts: {1 - (wrong & acc).sum() / acc.sum():.4f} "
          f"({int((wrong & acc).sum())} errors among {int(acc.sum())} accepted)")
    print("The validation errors and what the rule does with them:")
    for i in np.where(wrong)[0]:
        print(f"  true {CLASS_NAMES[yva[i]]:<20} predicted {CLASS_NAMES[Lva[i].argmax()]:<20} "
              f"conf {conf_v[i]:.3f} kNN {sim_v[i]:.3f} -> {st_va[i]}")

    print("\n=== Pepper/Potato leaves with the chosen rule (share given each status; 'reliable' = wrongly accepted) ===")
    for half in ("dev", "eval"):
        print(f"{half}:", share(st_un, (unk["half"] == half).to_numpy()))
    per = [{"folder": f, **share(st_un, (unk["folder"] == f).to_numpy())} for f in sorted(unk["folder"].unique())]
    print(pd.DataFrame(per).to_string(index=False))

    print("\n=== Your photos (not used to choose anything) ===")
    photos["status"], photos["knn_similarity"], photos["gate_confidence"] = st_ph, np.round(sim_p, 3), np.round(conf_p, 3)
    print(photos.groupby("group")["status"].value_counts().unstack(fill_value=0).reindex(columns=list(STATES), fill_value=0).to_string())
    print("\nTomato-leaf photos that were NOT rejected as unsupported:")
    keep = photos[(photos["group"] == "tomato_leaf") & (photos["status"] != "unsupported")]
    print(keep[["file", "pred_class", "gate_confidence", "knn_similarity", "status"]].to_string(index=False) if len(keep) else "  none")
    photos.to_csv(OUT / "my_photos_gate.csv", index=False)

    info = json.loads((MODELS / "model_info.json").read_text())
    cfg = {"created": date.today().isoformat(), "model_sha256": info["sha256"],
           "knn_bank_file": "knn_bank.npz", "knn_bank_sha256": sha256(MODELS / "knn_bank.npz"), "k": bank.k,
           "tau_knn": tau_knn, "tau_conf": tau_conf,
           "knn_accept_on_validation": chosen["knn_accept"], "confidence_accept_on_validation": X_CONF,
           "rejection_budget_on_validation": a.budget,
           "rule": "unsupported if kNN similarity < tau_knn; else uncertain if max softmax < tau_conf; else reliable",
           "thresholds_fitted_on": "tomato validation images only; scorer built from tomato training images only"}
    (MODELS / "reliability_config.json").write_text(json.dumps(cfg, indent=2))
    print("\nSaved: models/reliability_config.json, models/knn_bank.npz, reports/ood/operating_points.csv")


if __name__ == "__main__":
    main()