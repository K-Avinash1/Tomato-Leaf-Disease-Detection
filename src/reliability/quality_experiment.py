import json
import tempfile
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

from src.config import DATASET_ROOT, PROJECT_ROOT, TABLES_DIR
from src.data.dataset import CLASS_TO_IDX
from src.preprocessing.transforms import get_eval_transform
from src.reliability.decision import ReliabilityGate
from src.reliability.features import load_final_model
from src.reliability.quality import QualityChecker, measure, open_image

OUT = PROJECT_ROOT / "reports" / "quality"
MODELS = PROJECT_ROOT / "models"
FIG = PROJECT_ROOT / "reports" / "figures"
PHOTOS = PROJECT_ROOT / "data" / "external" / "my_photos"
GROUPS = ["tomato_leaf", "other_leaf", "non_leaf", "poor_quality"]
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZES, DARK, BRIGHT, BLUR = [192, 160, 128, 96, 64, 48], [0.7, 0.5, 0.35, 0.25, 0.15], [1.3, 1.6, 2.0, 2.5], [1, 2, 3, 5, 8]
ACC_TARGET, N_PER_CLASS, BUDGET_PCT, DARK_GUARD, MIN_SIDE_FLOOR = 0.97, 60, 0.5, 0.02, 64
NAN_ROW = {"min_side": 0, "p5": np.nan, "p95": np.nan, "std": np.nan, "sharpness": np.nan}


def measure_paths(paths, label):
    rows = []
    for i, p in enumerate(paths):
        img, _ = open_image(p)
        rows.append(measure(img) if img is not None else dict(NAN_ROW))
        if (i + 1) % 3000 == 0:
            print(f"  {label}: {i + 1}/{len(paths)}", flush=True)
    return pd.DataFrame(rows)


def scale(img, f):
    return Image.fromarray(np.clip(np.asarray(img, dtype=np.float32) * f, 0, 255).astype(np.uint8))


@torch.inference_mode()
def forward(model, imgs, tf, batch=64):
    L, F = [], []
    for s in range(0, len(imgs), batch):
        x = torch.stack([tf(im) for im in imgs[s:s + batch]])
        f = torch.flatten(model.avgpool(model.features(x)), 1)
        h = model.classifier[1](model.classifier[0](f))
        L.append(model.classifier[3](model.classifier[2](h)).float().numpy())
        F.append(h.float().numpy())
    return np.concatenate(L), np.concatenate(F)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    sm = pd.read_csv(TABLES_DIR / "split_manifest.csv")

    # 1. thresholds from tomato TRAINING images
    tr = sm[sm["split"] == "train"].reset_index(drop=True)
    cache = OUT / "train_metrics.csv"
    if cache.exists():
        tm = pd.read_csv(cache)
    else:
        print(f"Measuring {len(tr)} training images ...", flush=True)
        tm = measure_paths([DATASET_ROOT / r for r in tr["rel_path"]], "train")
        tm.to_csv(cache, index=False)
    tm["dark"] = tr["dark"].astype(bool).to_numpy()
    T = {"p95_dark": float(np.percentile(tm["p95"], BUDGET_PCT)),
         "p5_bright": float(np.percentile(tm["p5"], 100 - BUDGET_PCT)),
         "sharpness": float(np.percentile(tm["sharpness"], BUDGET_PCT)),
         "std": float(np.percentile(tm["std"], BUDGET_PCT))}
    dark_share = float((tm.loc[tm["dark"], "p95"] < T["p95_dark"]).mean())
    print(f"\nThresholds at a {BUDGET_PCT}% budget on tomato training images: " + ", ".join(f"{k} {v:.2f}" for k, v in T.items()))
    print(f"Black-background training images (n={int(tm['dark'].sum())}) that the dark threshold would reject: {100 * dark_share:.1f}%")
    if dark_share > DARK_GUARD:
        T["p95_dark"] = float(np.percentile(tm.loc[tm["dark"], "p95"], 100 * DARK_GUARD))
        print(f"Guard applied: dark threshold lowered to {T['p95_dark']:.2f} so that only {100 * DARK_GUARD:.0f}% of them are rejected")

    # 2. degradation experiment on a fixed validation subset
    val = sm[sm["split"] == "val"].reset_index(drop=True)
    rng = np.random.default_rng(42)
    idx = np.concatenate([rng.choice(np.where(val["class"] == c)[0], min(N_PER_CLASS, int((val["class"] == c).sum())), replace=False)
                          for c in sorted(val["class"].unique())])
    sub = val.iloc[idx].reset_index(drop=True)
    originals = [open_image(DATASET_ROOT / r)[0] for r in sub["rel_path"]]
    y = sub["class"].map(CLASS_TO_IDX).to_numpy()
    model, _ = load_final_model()
    gate = ReliabilityGate.from_files(MODELS / "reliability_config.json", MODELS / "knn_bank.npz")
    tf = get_eval_transform()
    conds = [("clean", "clean", lambda im: im)]
    conds += [("size", f"{s}px", lambda im, s=s: im.resize((s, s), Image.BICUBIC)) for s in SIZES]
    conds += [("darken", f"x{f}", lambda im, f=f: scale(im, f)) for f in DARK]
    conds += [("brighten", f"x{g}", lambda im, g=g: scale(im, g)) for g in BRIGHT]
    conds += [("blur", f"sigma{s}", lambda im, s=s: im.filter(ImageFilter.GaussianBlur(s))) for s in BLUR]
    store = []
    for kind, level, fn in conds:
        imgs = [fn(im) for im in originals]
        M = pd.DataFrame([measure(im) for im in imgs])
        L, F = forward(model, imgs, tf)
        status, _, _ = gate.decide(L, F)
        store.append({"kind": kind, "level": level, "M": M, "acc": float((L.argmax(1) == y).mean()),
                      "reliable": float((status == "reliable").mean())})
        print(f"  {kind:<9}{level:<8} accuracy {store[-1]['acc']:.3f}", flush=True)

    # 3. size limit from the experiment
    clean_acc = store[0]["acc"]
    size_acc = {256: clean_acc, **{int(s["level"][:-2]): s["acc"] for s in store if s["kind"] == "size"}}
    data_min = 256
    for s in sorted(size_acc, reverse=True):
        if size_acc[s] >= ACC_TARGET:
            data_min = s
        else:
            break
    T["min_side"] = int(max(data_min, MIN_SIDE_FLOOR))
    print(f"\nSmallest side with accuracy >= {ACC_TARGET:.0%} (all larger sizes also pass): {data_min}px; with the {MIN_SIDE_FLOOR}px floor: {T['min_side']}px")

    checker = QualityChecker({"thresholds": T})
    rows = []
    for s in store:
        M = s["M"].copy()
        if s["kind"] == "size":
            M["min_side"] = int(s["level"][:-2])
        flagged = np.array([bool(checker.reasons(r)) for r in M.to_dict("records")])
        rows.append({"kind": s["kind"], "level": s["level"], "mean_p5": round(M["p5"].mean(), 1), "mean_p95": round(M["p95"].mean(), 1),
                     "mean_sharpness": round(M["sharpness"].mean(), 1), "accuracy": round(s["acc"], 3),
                     "gate_reliable": round(s["reliable"], 3), "quality_flagged": round(float(flagged.mean()), 3)})
    deg = pd.DataFrame(rows)
    deg.to_csv(OUT / "degradation_curves.csv", index=False)
    print(f"\n=== Degradation experiment on {len(sub)} validation images (thresholds above) ===")
    print(deg.to_string(index=False))
    gaps = deg[(deg["accuracy"] < ACC_TARGET) & (deg["quality_flagged"] < 0.5)]
    print("\nLevels where accuracy fell below 97% but the quality check flags fewer than half of the images (covered only by the gate):")
    print(gaps[["kind", "level", "accuracy", "quality_flagged"]].to_string(index=False) if len(gaps) else "  none")

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, kind in zip(axes, ["size", "darken", "brighten", "blur"]):
        d = deg[(deg["kind"] == kind) | (deg["kind"] == "clean")]
        d = pd.concat([d[d["kind"] == "clean"], d[d["kind"] == kind]])
        ax.plot(d["level"], d["accuracy"], "o-", color="seagreen", label="model accuracy")
        ax.plot(d["level"], d["gate_reliable"], "s--", color="steelblue", label="gate: reliable share")
        ax.plot(d["level"], d["quality_flagged"], "^:", color="crimson", label="quality check: flagged share")
        ax.set_title(kind)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis="x", rotation=40)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG / "fig10_quality_degradation.png", dpi=130)
    plt.close()

    # 4. false rejection on all validation images
    vm = measure_paths([DATASET_ROOT / r for r in val["rel_path"]], "val")
    reasons = [checker.reasons(r) for r in vm.to_dict("records")]
    fl = np.array([bool(r) for r in reasons])
    vd = val["dark"].astype(bool).to_numpy()
    print(f"\n=== Valid tomato validation images flagged by the quality check: {int(fl.sum())} of {len(fl)} ({100 * fl.mean():.2f}%) ===")
    print("By reason:", pd.Series([x for r in reasons for x in r]).value_counts().to_dict())
    print(f"Black-background validation images flagged: {int(fl[vd].sum())} of {int(vd.sum())}")
    print("Flagged by class:", pd.Series(val["class"][fl]).value_counts().to_dict())

    # 5. your photos
    rows = []
    for g in GROUPS:
        for p in sorted((PHOTOS / g).iterdir()):
            if p.suffix.lower() in EXT:
                r, m = checker.check(p)
                rows.append({"group": g, "file": p.name[-30:], "min_side": m["min_side"], "p5": round(m["p5"], 1), "p95": round(m["p95"], 1),
                             "sharpness": round(m["sharpness"], 1), "reasons": ",".join(r) if r else "ok"})
    pdf = pd.DataFrame(rows)
    pdf.to_csv(OUT / "my_photos_quality.csv", index=False)
    print("\n=== Your photos: share flagged by the quality check, by group ===")
    print(pdf.assign(flagged=pdf["reasons"] != "ok").groupby("group")["flagged"].agg(["sum", "count"]).to_string())
    print("\nPoor-quality photos in detail:")
    print(pdf[pdf["group"] == "poor_quality"].drop(columns="group").to_string(index=False))
    print("\nTomato-leaf photos flagged:")
    tl = pdf[(pdf["group"] == "tomato_leaf") & (pdf["reasons"] != "ok")]
    print(tl.drop(columns="group").to_string(index=False) if len(tl) else "  none")

    # 6. file and content edge cases
    print("\n=== Edge cases ===")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "fake.jpg").write_text("this is not an image")
        (d / "empty.png").write_bytes(b"")
        Image.new("RGB", (40, 40), (60, 140, 60)).save(d / "tiny.png")
        Image.new("RGB", (400, 400), (128, 128, 128)).save(d / "blank_gray.png")
        Image.new("RGB", (400, 400), (0, 0, 0)).save(d / "black.png")
        Image.new("RGB", (400, 400), (255, 255, 255)).save(d / "white.png")
        for name in ["fake.jpg", "empty.png", "tiny.png", "blank_gray.png", "black.png", "white.png"]:
            print(f"  {name:<16}", checker.check(d / name)[0])

    cfg = {"created": date.today().isoformat(), "thresholds": T,
           "rule": "flag if min side < min_side; std < std; p95 < p95_dark; p5 > p5_bright; sharpness < sharpness (all on a 256x256 gray version, as in Phase 3)",
           "chosen_by": {"budget_pct_on_tomato_training_images": BUDGET_PCT, "black_background_guard": DARK_GUARD,
                         "min_side": f"smallest side keeping accuracy >= {ACC_TARGET:.0%} on degraded validation images, floor {MIN_SIDE_FLOOR}px"}}
    (MODELS / "quality_config.json").write_text(json.dumps(cfg, indent=2))
    print("\nSaved: models/quality_config.json, reports/quality/*.csv, reports/figures/fig10_quality_degradation.png")


if __name__ == "__main__":
    main()