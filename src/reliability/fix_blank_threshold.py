import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.config import DATASET_ROOT, PROJECT_ROOT, TABLES_DIR
from src.reliability.quality import QualityChecker, measure, open_image

OUT = PROJECT_ROOT / "reports" / "quality"
CFG = PROJECT_ROOT / "models" / "quality_config.json"
PHOTOS = PROJECT_ROOT / "data" / "external" / "my_photos"
GROUPS = ["tomato_leaf", "other_leaf", "non_leaf", "poor_quality"]
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    s = pd.read_csv(OUT / "train_metrics.csv")["std"].dropna()
    print(f"Contrast (std) of valid tomato training images: min {s.min():.2f} | 0.1th pct {np.percentile(s, 0.1):.2f} | "
          f"0.5th pct {np.percentile(s, 0.5):.2f} | median {s.median():.2f}")
    cfg = json.loads(CFG.read_text())
    old = cfg["thresholds"]["std"]
    new = float(0.5 * s.min())
    cfg["thresholds"]["std"] = new
    cfg["history"] = {"std_threshold_before": old,
                      "reason": "the 0.5% budget rule flagged 14 valid validation leaves as 'blank'; replaced by half the lowest contrast seen in any valid training image"}
    cfg["chosen_by"]["std"] = "half of the lowest contrast (std) of any valid tomato training image"
    checker = QualityChecker(cfg)
    print(f"Blank threshold: {old:.2f} -> {new:.2f}")

    val = pd.read_csv(TABLES_DIR / "split_manifest.csv").query("split == 'val'").reset_index(drop=True)
    metrics = [measure(open_image(DATASET_ROOT / r)[0]) for r in val["rel_path"]]
    reasons = [checker.reasons(m) for m in metrics]
    fl = np.array([bool(r) for r in reasons])
    vd = val["dark"].astype(bool).to_numpy()
    print(f"\nValid tomato validation images flagged: {int(fl.sum())} of {len(fl)} ({100 * fl.mean():.2f}%)")
    print("By reason:", pd.Series([x for r in reasons for x in r]).value_counts().to_dict())
    print(f"Black-background images flagged: {int(fl[vd].sum())} of {int(vd.sum())}")
    print("Flagged by class:", pd.Series(val["class"][fl]).value_counts().to_dict())

    rows = []
    for g in GROUPS:
        for p in sorted((PHOTOS / g).iterdir()):
            if p.suffix.lower() in EXT:
                r, _ = checker.check(p)
                rows.append({"group": g, "reasons": ",".join(r) if r else "ok"})
    pdf = pd.DataFrame(rows)
    print("\nYour photos flagged, by group:")
    print(pdf.assign(flagged=pdf["reasons"] != "ok").groupby("group")["flagged"].agg(["sum", "count"]).to_string())

    print("\nEdge cases:")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "fake.jpg").write_text("this is not an image")
        Image.new("RGB", (40, 40), (60, 140, 60)).save(d / "tiny.png")
        Image.new("RGB", (400, 400), (128, 128, 128)).save(d / "blank_gray.png")
        Image.new("RGB", (400, 400), (0, 0, 0)).save(d / "black.png")
        Image.new("RGB", (400, 400), (255, 255, 255)).save(d / "white.png")
        for name in ["fake.jpg", "tiny.png", "blank_gray.png", "black.png", "white.png"]:
            print(f"  {name:<16}", checker.check(d / name)[0])
    CFG.write_text(json.dumps(cfg, indent=2))
    print("\nSaved: models/quality_config.json")


if __name__ == "__main__":
    main()