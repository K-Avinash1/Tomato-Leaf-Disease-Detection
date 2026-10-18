import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.config import DATASET_ROOT, PROJECT_ROOT
from src.data.dataset import CLASS_NAMES
from src.reliability.decision import ReliabilityGate, softmax_max
from src.reliability.features import extract, load_final_model

MODELS = PROJECT_ROOT / "models"
OOD = PROJECT_ROOT / "reports" / "ood"
TAG = "E010_mnv3_deep_seed7"
P12 = PROJECT_ROOT / "reports" / "test_results" / TAG / "test_predictions.csv"
OUT = PROJECT_ROOT / "reports" / "test_results" / f"gate_{TAG}"
FIG = PROJECT_ROOT / "reports" / "figures"
STATES = ("reliable", "uncertain", "unsupported")
X_GRID = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shares(status, mask=None):
    s = status if mask is None else status[mask]
    return {k: round(float((s == k).mean()), 4) for k in STATES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if (OUT / "gate_test_metrics.json").exists() and not a.force:
        print("STOP: the gate was already evaluated on the test set. Use --force only if you will report the re-run.")
        sys.exit(1)

    cfg = json.loads((MODELS / "reliability_config.json").read_text())
    info = json.loads((MODELS / "model_info.json").read_text())
    if not (sha256(MODELS / "best_model.pth") == info["sha256"] == cfg["model_sha256"]):
        print("STOP: best_model.pth does not match the recorded SHA-256.")
        sys.exit(1)
    if sha256(MODELS / "knn_bank.npz") != cfg["knn_bank_sha256"]:
        print("STOP: knn_bank.npz does not match the recorded SHA-256.")
        sys.exit(1)
    print("Checks 1-2: model and kNN bank match their recorded SHA-256")
    print(f"Thresholds fixed beforehand on validation: kNN similarity {cfg['tau_knn']}, confidence {cfg['tau_conf']:.4f}")

    gate = ReliabilityGate.from_files(MODELS / "reliability_config.json", MODELS / "knn_bank.npz")
    model, _ = load_final_model()
    test = pd.read_csv(P12)
    print(f"Extracting {len(test)} test images ...", flush=True)
    L, F = extract(model, [DATASET_ROOT / r for r in test["rel_path"]], desc="test")
    mism = int((L.argmax(1) != test["pred_idx"].to_numpy()).sum())
    print(f"Check 3: predictions differ from the Phase 12 test predictions on {mism} of {len(test)} images")
    if mism > 2:
        print("STOP: the model does not reproduce its Phase 12 predictions.")
        sys.exit(1)

    y = test["true_idx"].to_numpy()
    wrong = L.argmax(1) != y
    status, sim, conf = gate.decide(L, F)
    n = len(y)
    acc = status == "reliable"
    n_acc, err_acc = int(acc.sum()), int((wrong & acc).sum())
    ci = binomtest(n_acc - err_acc, n_acc).proportion_ci(confidence_level=0.95, method="exact")
    correct_rej = int((~wrong & ~acc).sum())
    false_rej = correct_rej / int((~wrong).sum())

    # unknown images: cached features from Phase 14 (Pepper/Potato, own photos)
    zu = np.load(OOD / "cache" / "unknown_pepper_potato.npz")
    halves = pd.read_csv(OOD / "unknown_halves.csv")
    assert len(halves) == len(zu["logits"])
    ev = (halves["half"] == "eval").to_numpy()
    st_u, sim_u, conf_u = gate.decide(zu["logits"], zu["feats"])
    zp = np.load(OOD / "cache" / "my_photos.npz")
    photos = pd.read_csv(OOD / "my_photos_scores.csv")[["group", "file"]]
    assert len(photos) == len(zp["logits"])
    st_p, sim_p, conf_p = gate.decide(zp["logits"], zp["feats"])
    zv = np.load(OOD / "cache" / "val.npz")
    sim_va = gate.bank.similarity(zv["feats"])

    # per class and dark-background subsets
    rows = []
    for k, c in enumerate(CLASS_NAMES):
        m = y == k
        rows.append({"class": c, "n": int(m.sum()), **shares(status, m), "errors": int(wrong[m].sum()),
                     "errors_accepted": int((wrong & m & acc).sum())})
    per_class = pd.DataFrame(rows)
    dark = test["dark"].to_numpy().astype(bool)

    # threshold sensitivity (informational only: the operating point was fixed on validation)
    sens = []
    for X in X_GRID:
        tk = float(np.percentile(sim_va, 100 * (1 - X)))
        rej = (sim < tk) | (conf < cfg["tau_conf"])
        sens.append({"knn_accept_on_val": X, "tau_knn": round(tk, 4),
                     "test_false_rejection": round(float(rej[~wrong].mean()), 4),
                     "test_errors_still_accepted": int((wrong & ~rej).sum()),
                     "unknown_eval_half_accepted": round(float(((sim_u[ev] >= tk) & (conf_u[ev] >= cfg["tau_conf"])).mean()), 4),
                     "chosen": "<-- chosen on validation" if abs(X - cfg["knn_accept_on_validation"]) < 1e-9 else ""})
    sens = pd.DataFrame(sens)

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    dec = test[["rel_path", "class", "pred_class"]].copy()
    dec["confidence"], dec["knn_similarity"], dec["status"], dec["wrong"] = conf, sim, status, wrong
    dec.to_csv(OUT / "gate_test_decisions.csv", index=False)
    per_class.to_csv(OUT / "per_class_gate_test.csv", index=False)
    sens.to_csv(OUT / "operating_points_test.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for a_, known, unk, ph_k, ph_u, tau, title, bins in [
            (ax[0], sim, sim_u[ev], sim_p, None, cfg["tau_knn"], "kNN similarity to tomato training images", np.linspace(0.3, 1.0, 50)),
            (ax[1], conf, conf_u[ev], conf_p, None, cfg["tau_conf"], "Max softmax confidence", np.linspace(0.3, 1.0, 50))]:
        a_.hist(known, bins=bins, density=True, alpha=0.55, color="seagreen", label="tomato test images")
        a_.hist(unk, bins=bins, density=True, alpha=0.55, color="darkorange", label="Pepper/Potato (eval half)")
        top = a_.get_ylim()[1]
        is_t = (photos["group"] == "tomato_leaf").to_numpy()
        a_.plot(ph_k[is_t], np.full(is_t.sum(), 0.03 * top), "|", color="red", markersize=14, label="your tomato photos")
        a_.plot(ph_k[~is_t], np.full((~is_t).sum(), 0.08 * top), "|", color="navy", markersize=14, label="your non-tomato / poor photos")
        a_.axvline(tau, color="black", linestyle="--", label="threshold (fixed on validation)")
        a_.set_title(title)
        a_.set_xlabel("score")
        a_.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG / "fig09_gate_scores.png", dpi=130)
    plt.close()

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    result = {"experiment": TAG, "evaluated_on": "test", "n_images": n,
              "when": datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
              "model_sha256": info["sha256"], "config": cfg,
              "test": {"status_shares": shares(status), "accepted_images": n_acc, "errors_among_accepted": err_acc,
                       "accuracy_on_accepted": 1 - err_acc / n_acc, "accuracy_on_accepted_ci95": [ci.low, ci.high],
                       "test_errors_total": int(wrong.sum()),
                       "test_errors_by_status": {k: int((wrong & (status == k)).sum()) for k in STATES},
                       "correct_images_rejected": correct_rej, "false_rejection_rate": false_rej,
                       "dark_background_images": {"n": int(dark.sum()), **shares(status, dark)}},
              "unknown_pepper_potato_eval_half": {"n": int(ev.sum()), "status_shares": shares(st_u, ev)},
              "own_photos": {g: {"n": int((photos["group"] == g).sum()),
                                 **shares(st_p, (photos["group"] == g).to_numpy())} for g in photos["group"].unique()}}
    (OUT / "gate_test_metrics.json").write_text(json.dumps(result, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    with open(PROJECT_ROOT / "reports" / "test_results" / "TEST_LOG.txt", "a") as f:
        f.write(f"{result['when']} | gate on {TAG} | sha {info['sha256'][:12]} | commit {commit[:10]} | force={a.force}\n")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"\n=== GATE ON THE TEST SET: {n} tomato images (thresholds fixed on validation) ===")
    print("Status shares:", shares(status))
    print(f"Accepted ('reliable'): {n_acc} images | errors among them: {err_acc} | accuracy on accepted "
          f"{1 - err_acc / n_acc:.4f} (exact 95% CI [{ci.low:.4f}, {ci.high:.4f}])")
    print(f"The {int(wrong.sum())} test errors by status:", {k: int((wrong & (status == k)).sum()) for k in STATES})
    print(f"Correct predictions that were rejected: {correct_rej} of {int((~wrong).sum())} ({100 * false_rej:.2f}%)")
    print(f"Dark-background test images (n={int(dark.sum())}):", shares(status, dark))
    print("\nPer class (shares of each status on the test set):")
    print(per_class.to_string(index=False))
    print("\nPepper/Potato leaves, eval half only (reliable = wrongly accepted):", shares(st_u, ev), f"(n={int(ev.sum())})")
    print("Per folder, eval half:")
    for fol in sorted(halves["folder"].unique()):
        m = ev & (halves["folder"] == fol).to_numpy()
        print(f"  {fol:<32} n={int(m.sum()):<4}", shares(st_u, m))
    print("\nYour photos:")
    for g in photos["group"].unique():
        print(f"  {g:<14} n={int((photos['group'] == g).sum()):<3}", shares(st_p, (photos["group"] == g).to_numpy()))
    print("\nSensitivity (informational; the operating point was fixed on validation before this run):")
    print(sens.to_string(index=False))
    print(f"\nSaved in {OUT}")


if __name__ == "__main__":
    main()