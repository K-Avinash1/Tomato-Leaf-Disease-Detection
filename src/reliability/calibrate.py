import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy.optimize import minimize_scalar

from src.config import DATASET_ROOT, PROJECT_ROOT, TABLES_DIR
from src.data.dataset import CLASS_NAMES, CLASS_TO_IDX
from src.evaluation.calibration import nll_and_ece, softmax_np
from src.preprocessing.transforms import get_eval_transform
from src.reliability.features import load_final_model
from src.reliability.quality import open_image
from src.reliability.quality_experiment import N_PER_CLASS, forward, scale

TAG = "E010_mnv3_deep_seed7"
VAL_CSV = PROJECT_ROOT / "reports" / "experiments" / TAG / "val_predictions.csv"
TEST_CSV = PROJECT_ROOT / "reports" / "test_results" / TAG / "test_predictions.csv"
OUT = PROJECT_ROOT / "reports" / "calibration"
FIG = PROJECT_ROOT / "reports" / "figures"
MODELS = PROJECT_ROOT / "models"
LOGIT_COLS = [f"logit_{c}" for c in CLASS_NAMES]
ADOPT_IF_NLL_IMPROVES = 0.05
EDGES = [0.0, 0.6, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0001]


def metrics(logits, y, T=1.0):
    z = logits / T
    nll, ece = nll_and_ece(z, y)
    p = softmax_np(z.astype(np.float64))
    brier = float(((p - np.eye(10)[y]) ** 2).sum(1).mean())
    return {"NLL": round(nll, 4), "ECE": round(ece, 4), "Brier": round(brier, 4),
            "mean_conf": round(float(p.max(1).mean()), 4), "accuracy": round(float((p.argmax(1) == y).mean()), 4)}


def fit_T(logits, y):
    r = minimize_scalar(lambda lt: nll_and_ece(logits / np.exp(lt), y)[0],
                        bounds=(np.log(0.5), np.log(5.0)), method="bounded")
    return float(np.exp(r.x))


def bands(logits, y, T=1.0):
    p = softmax_np((logits / T).astype(np.float64))
    conf, ok = p.max(1), p.argmax(1) == y
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            rows.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()), "mean_conf": float(conf[m].mean()), "accuracy": float(ok[m].mean())})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    val, test = pd.read_csv(VAL_CSV), pd.read_csv(TEST_CSV)
    Lv, yv = val[LOGIT_COLS].to_numpy(), val["true_idx"].to_numpy()
    Lt, yt = test[LOGIT_COLS].to_numpy(), test["true_idx"].to_numpy()

    # 1. temperature on validation, with a 2-fold check
    T = fit_T(Lv, yv)
    perm = np.random.default_rng(42).permutation(len(yv))
    halves = [perm[: len(yv) // 2], perm[len(yv) // 2:]]
    raw_cv, cal_cv = [], []
    for a, b in [(0, 1), (1, 0)]:
        Tf = fit_T(Lv[halves[a]], yv[halves[a]])
        raw_cv.append(nll_and_ece(Lv[halves[b]], yv[halves[b]])[0])
        cal_cv.append(nll_and_ece(Lv[halves[b]] / Tf, yv[halves[b]])[0])
        print(f"  fold {a}->{b}: T fitted on half {a} = {Tf:.3f}; NLL on half {b}: raw {raw_cv[-1]:.4f}, calibrated {cal_cv[-1]:.4f}")
    gain = 1 - np.mean(cal_cv) / np.mean(raw_cv)
    adopted = gain >= ADOPT_IF_NLL_IMPROVES
    T_final = T if adopted else 1.0
    print(f"\nTemperature fitted on all validation images: T = {T:.3f}")
    print(f"Cross-checked NLL improvement: {100 * gain:.1f}% (adoption rule: at least {100 * ADOPT_IF_NLL_IMPROVES:.0f}%) -> "
          f"{'ADOPTED' if adopted else 'NOT adopted: T = 1, raw confidence is shown'}; T used = {T_final:.3f}")

    # 2. metrics
    rows = []
    for name, L, y in [("validation", Lv, yv), ("test", Lt, yt)]:
        rows.append({"set": name, "version": "raw", **metrics(L, y)})
        rows.append({"set": name, "version": f"temperature {T:.2f}", **metrics(L, y, T)})
    res = pd.DataFrame(rows)
    print("\n=== Calibration metrics (T fitted on validation only; test shown for reporting) ===")
    print(res.to_string(index=False))
    res.to_csv(OUT / "calibration_metrics.csv", index=False)
    pt = softmax_np(Lt.astype(np.float64))
    ok = pt.argmax(1) == yt
    print(f"Test mean confidence: correct {pt.max(1)[ok].mean():.4f} | wrong {pt.max(1)[~ok].mean():.4f}")

    Lall, yall = np.vstack([Lv, Lt]), np.concatenate([yv, yt])
    b_raw, b_cal = bands(Lall, yall), bands(Lall, yall, T)
    print("\n=== Accuracy by raw confidence band (validation + test, 4,800 images) ===")
    print(b_raw.round(4).to_string(index=False))
    b_raw.to_csv(OUT / "confidence_bands_raw.csv", index=False)

    # 3. calibration under shift
    sm = pd.read_csv(TABLES_DIR / "split_manifest.csv")
    vm = sm[sm["split"] == "val"].reset_index(drop=True)
    rng = np.random.default_rng(42)
    idx = np.concatenate([rng.choice(np.where(vm["class"] == c)[0], min(N_PER_CLASS, int((vm["class"] == c).sum())), replace=False)
                          for c in sorted(vm["class"].unique())])
    sub = vm.iloc[idx].reset_index(drop=True)
    originals = [open_image(DATASET_ROOT / r)[0] for r in sub["rel_path"]]
    ys = sub["class"].map(CLASS_TO_IDX).to_numpy()
    model, _ = load_final_model()
    tf = get_eval_transform()
    conds = [("clean", lambda im: im), ("blur sigma1", lambda im: im.filter(ImageFilter.GaussianBlur(1))),
             ("blur sigma2", lambda im: im.filter(ImageFilter.GaussianBlur(2))),
             ("size 128px", lambda im: im.resize((128, 128), Image.BICUBIC)),
             ("darken x0.5", lambda im: scale(im, 0.5)), ("brighten x2.0", lambda im: scale(im, 2.0))]
    rows = []
    for name, fn in conds:
        L, _ = forward(model, [fn(im) for im in originals], tf)
        r, c = metrics(L, ys), metrics(L, ys, T)
        rows.append({"condition": name, "accuracy": r["accuracy"], "mean_conf_raw": r["mean_conf"], "mean_conf_cal": c["mean_conf"],
                     "overconfidence_raw": round(r["mean_conf"] - r["accuracy"], 4),
                     "overconfidence_cal": round(c["mean_conf"] - c["accuracy"], 4),
                     "ECE_raw": r["ECE"], "ECE_cal": c["ECE"], "NLL_raw": r["NLL"], "NLL_cal": c["NLL"]})
    sh = pd.DataFrame(rows)
    print(f"\n=== Calibration under shift ({len(sub)} validation images; overconfidence = mean confidence minus accuracy) ===")
    print(sh.to_string(index=False))
    sh.to_csv(OUT / "calibration_under_shift.csv", index=False)

    # 4. figure
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot([0.5, 1], [0.5, 1], "--", color="gray", label="perfect calibration")
    for b, lab, col in [(b_raw, "raw", "seagreen"), (b_cal, f"temperature {T:.2f}", "darkorange")]:
        ax[0].plot(b["mean_conf"], b["accuracy"], "o-", color=col, label=lab)
    for _, r in b_raw.iterrows():
        ax[0].annotate(str(int(r["n"])), (r["mean_conf"], r["accuracy"]), textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")
    ax[0].set_xlabel("mean confidence in band")
    ax[0].set_ylabel("accuracy in band")
    ax[0].set_title("Reliability (validation + test, numbers = images in raw band)")
    ax[0].legend(fontsize=8)
    w = 0.27
    x = np.arange(len(sh))
    ax[1].bar(x - w, sh["accuracy"], w, color="seagreen", label="accuracy")
    ax[1].bar(x, sh["mean_conf_raw"], w, color="steelblue", label="mean confidence (raw)")
    ax[1].bar(x + w, sh["mean_conf_cal"], w, color="darkorange", label="mean confidence (calibrated)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(sh["condition"], rotation=30, ha="right")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Confidence versus accuracy on damaged images")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig11_calibration.png", dpi=130)
    plt.close()

    info = json.loads((MODELS / "model_info.json").read_text())
    cfg = {"created": datetime.now().date().isoformat(), "model_sha256": info["sha256"], "temperature_fitted_on_validation": T,
           "cross_checked_nll_improvement": float(gain), "adoption_rule": f"adopt if cross-checked NLL improves by at least {ADOPT_IF_NLL_IMPROVES:.0%}",
           "adopted": bool(adopted), "temperature_used_for_displayed_confidence": T_final,
           "note": "The reliability gate (decision.py) keeps using RAW confidence; only the displayed 'Model confidence' uses the temperature."}
    (MODELS / "calibration.json").write_text(json.dumps(cfg, indent=2))
    with open(PROJECT_ROOT / "reports" / "test_results" / "TEST_LOG.txt", "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} | calibration report from stored test logits (no new model pass) | {TAG}\n")
    print("\nSaved: models/calibration.json, reports/calibration/*.csv, reports/figures/fig11_calibration.png")


if __name__ == "__main__":
    main()