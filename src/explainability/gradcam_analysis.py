import json
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.config import DATASET_ROOT, PROJECT_ROOT
from src.data.dataset import CLASS_NAMES
from src.evaluation.calibration import softmax_np
from src.preprocessing.transforms import get_eval_transform
from src.reliability.decision import ReliabilityGate
from src.reliability.features import load_final_model
from src.reliability.quality import open_image

TAG = "E010_mnv3_deep_seed7"
REP = PROJECT_ROOT / "reports"
VAL_CSV = REP / "experiments" / TAG / "val_predictions.csv"
TEST_CSV = REP / "test_results" / TAG / "test_predictions.csv"
OOD, OUT, FIG = REP / "ood", REP / "gradcam", REP / "figures"
MODELS = PROJECT_ROOT / "models"
LB = CLASS_NAMES.index("Late Blight")
N_PER_CLASS, BORDER = 60, 22


def load_arr(path):
    img, _ = open_image(path)
    return np.asarray(img.resize((224, 224), Image.BILINEAR))


def leaf_mask(arr):
    s = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[:, :, 1]
    _, m = cv2.threshold(s, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
    return out


def focus(cam, m):
    m = m.astype(np.float32)
    band = np.ones((224, 224), np.float32)
    band[BORDER:-BORDER, BORDER:-BORDER] = 0
    tot = float(cam.sum()) + 1e-9
    bg_share, bg_area = float((cam * (1 - m)).sum() / tot), float(1 - m.mean())
    return {"bg_share": bg_share, "bg_area": bg_area, "relative_bg": bg_share / max(bg_area, 1e-6),
            "border_share": float((cam * band).sum() / tot)}


def predict_arrs(model, tf, arrs, batch=64):
    out = []
    with torch.no_grad():
        for s in range(0, len(arrs), batch):
            out.append(model(torch.stack([tf(Image.fromarray(a)) for a in arrs[s:s + batch]])).float().numpy())
    return np.concatenate(out)


def run_cams(cam, tf, arrs, classes, batch=16):
    out = []
    for s in range(0, len(arrs), batch):
        x = torch.stack([tf(Image.fromarray(a)) for a in arrs[s:s + batch]])
        out.append(cam(input_tensor=x, targets=[ClassifierOutputTarget(int(c)) for c in classes[s:s + batch]]))
    return np.concatenate(out)


def overlay(arr, c):
    return show_cam_on_image(arr.astype(np.float32) / 255.0, c, use_rgb=True)


def show(ax, img, title, fs=7):
    ax.imshow(img)
    ax.set_title(title, fontsize=fs)
    ax.axis("off")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    tf = get_eval_transform()
    model, _ = load_final_model()
    for m in model.modules():                      # in-place activations can break gradient hooks; outputs are unchanged
        if hasattr(m, "inplace"):
            m.inplace = False
    cam = GradCAM(model=model, target_layers=[model.features[-1]])

    # ---- validation sample ----
    val = pd.read_csv(VAL_CSV)
    rng = np.random.default_rng(42)
    idx = np.concatenate([rng.choice(np.where(val["class"] == c)[0], min(N_PER_CLASS, int((val["class"] == c).sum())), replace=False)
                          for c in CLASS_NAMES])
    sub = val.iloc[idx].reset_index(drop=True)
    arrs = [load_arr(DATASET_ROOT / r) for r in sub["rel_path"]]
    masks = [leaf_mask(a) for a in arrs]
    L = predict_arrs(model, tf, arrs)
    pred, y = L.argmax(1), sub["true_idx"].to_numpy()
    conf = softmax_np(L.astype(np.float64)).max(1)
    cams = run_cams(cam, tf, arrs, pred)
    F = pd.DataFrame([focus(c, m) for c, m in zip(cams, masks)])
    F["class"], F["correct"] = sub["class"].to_numpy(), pred == y
    valid = ((F["bg_area"] >= 0.1) & (F["bg_area"] <= 0.9)).to_numpy()
    V = F[valid]
    print(f"Leaf mask: median share of the image covered by the leaf {np.median([m.mean() for m in masks]):.2f}; "
          f"images with a usable mask (background 10-90% of the image): {int(valid.sum())} of {len(F)}")
    print(f"\n=== Grad-CAM focus, validation sample (usable-mask images, n={len(V)}) ===")
    print(f"Share of heat-map weight on the background: {V['bg_share'].mean():.3f} (the background covers {V['bg_area'].mean():.3f} of the image)")
    rel = float(V["relative_bg"].mean())
    print(f"Relative background focus (share / area): {rel:.3f}   [below 0.5 = looks at the leaf, 0.5-1.0 mixed, above 1.0 background-biased]")
    print(f"Share of heat-map weight in the outer 10% border band: {V['border_share'].mean():.3f} (the band covers 0.354 of the image)")
    bycls = V.groupby("class").agg(n=("bg_share", "size"), bg_share=("bg_share", "mean"), bg_area=("bg_area", "mean"),
                                   relative_bg=("relative_bg", "mean"), border_share=("border_share", "mean")).round(3)
    print(bycls.to_string())
    bycls.to_csv(OUT / "focus_by_class.csv")
    if (~V["correct"]).any():
        print(f"Relative background focus, correct predictions {V.loc[V['correct'], 'relative_bg'].mean():.3f} vs wrong {V.loc[~V['correct'], 'relative_bg'].mean():.3f} (n wrong = {int((~V['correct']).sum())})")

    # ---- black-background images (dark Late Blight, validation) ----
    dk = val[val["dark"].astype(bool) & (val["class"] == "Late Blight")].reset_index(drop=True)
    arr_d = [load_arr(DATASET_ROOT / r) for r in dk["rel_path"]]
    mask_d = [leaf_mask(a) for a in arr_d]
    Ld = predict_arrs(model, tf, arr_d)
    pred_d = Ld.argmax(1)
    cams_d = run_cams(cam, tf, arr_d, pred_d)
    Fd = pd.DataFrame([focus(c, m) for c, m in zip(cams_d, mask_d)])
    print(f"\n=== Dark (black-background) Late Blight validation images, n={len(dk)} ===")
    print(f"Predicted Late Blight: {int((pred_d == LB).sum())} of {len(dk)}")
    print(f"Background covers {Fd['bg_area'].mean():.3f} of the image; heat-map weight on it {Fd['bg_share'].mean():.3f}; relative focus {Fd['relative_bg'].mean():.3f}")

    # ---- experiment (a): turn the background black on other classes ----
    keep = [i for i in range(len(sub)) if y[i] != LB and valid[i]]
    black = [(arrs[i] * masks[i][..., None]).astype(np.uint8) for i in keep]
    pb = predict_arrs(model, tf, black).argmax(1)
    acc0, acc1 = float((pred[keep] == y[keep]).mean()), float((pb == y[keep]).mean())
    lb0, lb1 = float((pred[keep] == LB).mean()), float((pb == LB).mean())
    print(f"\n=== Experiment (a): background turned black on {len(keep)} validation images of the other 9 classes ===")
    print(f"Accuracy: original {acc0:.3f} -> black background {acc1:.3f} | predictions changed: {float((pb != pred[keep]).mean()):.3f}")
    print(f"Predicted as Late Blight: original {lb0:.3f} -> black background {lb1:.3f} (shortcut suspected if the rise is above 0.05)")

    # ---- experiment (b): replace the black with grey on dark Late Blight ----
    bgs = [arrs[i][masks[i] == 0] for i in range(len(sub)) if not bool(sub["dark"].astype(bool)[i]) and masks[i].mean() < 0.95]
    bg_color = np.median(np.concatenate(bgs[:100]), axis=0).astype(np.uint8)
    gray = []
    for a, m in zip(arr_d, mask_d):
        b = a.copy()
        b[m == 0] = bg_color
        gray.append(b)
    Lg = predict_arrs(model, tf, gray)
    pg, cg = Lg.argmax(1), softmax_np(Lg.astype(np.float64)).max(1)
    acc_d0, acc_d1 = float((pred_d == LB).mean()), float((pg == LB).mean())
    print(f"\n=== Experiment (b): black background replaced by the usual grey {tuple(int(v) for v in bg_color)} on the dark Late Blight images ===")
    print(f"Predicted Late Blight: original {acc_d0:.3f} -> grey background {acc_d1:.3f} (shortcut suspected if below 0.90)")
    print("What the others became:", pd.Series([CLASS_NAMES[i] for i in pg[pg != LB]]).value_counts().to_dict() or "none")

    # ---- the 15 test errors (explanation only) ----
    te = pd.read_csv(TEST_CSV)
    err = te[te["true_idx"] != te["pred_idx"]].sort_values("confidence", ascending=False).reset_index(drop=True)
    arr_e = [load_arr(DATASET_ROOT / r) for r in err["rel_path"]]
    pe, ye = err["pred_idx"].to_numpy(), err["true_idx"].to_numpy()
    cams_ep, cams_et = run_cams(cam, tf, arr_e, pe), run_cams(cam, tf, arr_e, ye)
    Fe = pd.DataFrame([focus(c, leaf_mask(a)) for c, a in zip(cams_ep, arr_e)])
    print(f"\n=== The {len(err)} test errors: relative background focus of the heat map for the predicted class ===")
    print(pd.DataFrame({"true": [CLASS_NAMES[i] for i in ye], "predicted": [CLASS_NAMES[i] for i in pe],
                        "confidence": err["confidence"].round(3), "relative_bg": Fe["relative_bg"].round(2),
                        "border_share": Fe["border_share"].round(2)}).to_string(index=False))

    # ---- potato late blight accepted by the gate ----
    gate = ReliabilityGate.from_files(MODELS / "reliability_config.json", MODELS / "knn_bank.npz")
    z = np.load(OOD / "cache" / "unknown_pepper_potato.npz")
    halves = pd.read_csv(OOD / "unknown_halves.csv")
    status, _, _ = gate.decide(z["logits"], z["feats"])
    sel = halves[(halves["folder"] == "Potato___Late_blight") & (halves["half"] == "eval") & (status == "reliable")].head(10)
    arr_u = [load_arr(p) for p in sel["path"]]
    Lu = predict_arrs(model, tf, arr_u)
    cams_u = run_cams(cam, tf, arr_u, Lu.argmax(1))
    Fu = pd.DataFrame([focus(c, leaf_mask(a)) for c, a in zip(cams_u, arr_u)])
    print(f"\nPotato late blight leaves accepted by the gate (first {len(sel)} of the eval half): relative background focus {Fu['relative_bg'].mean():.2f}; "
          f"predicted: {pd.Series([CLASS_NAMES[i] for i in Lu.argmax(1)]).value_counts().to_dict()}")

    # ---- figures ----
    fig, ax = plt.subplots(2, 10, figsize=(25, 5.6))
    for c, name in enumerate(CLASS_NAMES):
        i = int(np.where((y == c) & (pred == c))[0][0])
        show(ax[0][c], arrs[i], f"{name}\nconfidence {conf[i]:.3f}")
        show(ax[1][c], overlay(arrs[i], cams[i]), "Grad-CAM (predicted class)")
    plt.tight_layout()
    plt.savefig(FIG / "fig12_gradcam_correct.png", dpi=90)
    plt.close()

    def error_fig(rows, path):
        f, a = plt.subplots(len(rows), 3, figsize=(9, 3.1 * len(rows)))
        for r, i in enumerate(rows):
            show(a[r][0], arr_e[i], f"true: {CLASS_NAMES[ye[i]]}")
            show(a[r][1], overlay(arr_e[i], cams_ep[i]), f"predicted: {CLASS_NAMES[pe[i]]} ({err['confidence'][i]:.2f})")
            show(a[r][2], overlay(arr_e[i], cams_et[i]), "evidence for the TRUE class")
        plt.tight_layout()
        plt.savefig(path, dpi=100)
        plt.close()

    error_fig(list(range(min(8, len(err)))), FIG / "fig13a_gradcam_test_errors.png")
    if len(err) > 8:
        error_fig(list(range(8, len(err))), FIG / "fig13b_gradcam_test_errors.png")

    fig, ax = plt.subplots(3, 6, figsize=(18, 9.5))
    for a_ in ax.ravel():
        a_.axis("off")
    for j in range(min(6, len(dk))):
        show(ax[0][j], overlay(arr_d[j], cams_d[j]), f"dark Late Blight: Grad-CAM, predicted {CLASS_NAMES[pred_d[j]]}")
        show(ax[1][j], gray[j], f"grey background: predicted {CLASS_NAMES[pg[j]]} ({cg[j]:.2f})")
    for j in range(min(6, len(keep))):
        k = keep[j]
        show(ax[2][j], black[j], f"black background: true {CLASS_NAMES[y[k]]} -> predicted {CLASS_NAMES[pb[j]]}")
    plt.tight_layout()
    plt.savefig(FIG / "fig14_gradcam_dark_experiments.png", dpi=90)
    plt.close()

    if len(sel):
        fig, ax = plt.subplots(2, len(sel), figsize=(2.6 * len(sel), 5.4), squeeze=False)
        for j in range(len(sel)):
            show(ax[0][j], arr_u[j], f"potato late blight\npredicted {CLASS_NAMES[Lu[j].argmax()]}")
            show(ax[1][j], overlay(arr_u[j], cams_u[j]), "Grad-CAM")
        plt.tight_layout()
        plt.savefig(FIG / "fig15_gradcam_unknown_accepted.png", dpi=90)
        plt.close()

    pick = np.random.default_rng(7).choice(len(sub), 12, replace=False)
    fig, ax = plt.subplots(2, 6, figsize=(18, 6.4))
    for a_, i in zip(ax.ravel(), pick):
        vis = arrs[i].copy()
        cnts, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 0, 0), 2)
        show(a_, vis, f"{sub['class'][i]}: leaf covers {masks[i].mean():.2f}")
    plt.tight_layout()
    plt.savefig(FIG / "fig16_leaf_mask_check.png", dpi=90)
    plt.close()

    verdict = {"relative_bg": rel, "focus": "the model mainly looks at the leaf" if rel < 0.5 else ("mixed focus" if rel <= 1.0 else "background-biased"),
               "experiment_a_late_blight_rise": lb1 - lb0, "experiment_a_shortcut": bool(lb1 - lb0 > 0.05),
               "experiment_b_late_blight_kept": acc_d1, "experiment_b_shortcut": bool(acc_d1 < 0.90)}
    (OUT / "gradcam_summary.json").write_text(json.dumps(verdict, indent=2))
    with open(REP / "test_results" / "TEST_LOG.txt", "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} | Grad-CAM explanation of the {len(err)} test errors (explanation only, no selection) | {TAG}\n")
    print("\n=== Verdicts (thresholds fixed beforehand) ===")
    print(json.dumps(verdict, indent=2))
    print("Saved: reports/gradcam/, figures fig12-fig16")


if __name__ == "__main__":
    main()