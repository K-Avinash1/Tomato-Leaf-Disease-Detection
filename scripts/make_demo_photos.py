"""Collects one demo photo per class for trying the web app.

For each class it takes test-split photos the model got right, runs them through the real app pipeline,
and keeps the first one that the app also accepts as 'reliable'. So every file in data/demo_photos/
shows a full result (diagnosis, heat map and treatment).

Run from the repo root:   python scripts\\make_demo_photos.py

Note for the report: these are demo photos chosen because the app accepts them; they are not evidence of accuracy
(that is the Phase 12 test result and the Phase 15 gate result).
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATASET_ROOT                      # noqa: E402
from src.data.dataset import CLASS_NAMES                 # noqa: E402
from src.inference.loader import load_engine            # noqa: E402
from src.inference.pipeline import analyze               # noqa: E402

CSV = ROOT / "reports" / "test_results" / "E010_mnv3_deep_seed7" / "test_predictions.csv"
OUT = ROOT / "data" / "demo_photos"
TRY_PER_CLASS = 40


def main():
    df = pd.read_csv(CSV)
    need = {"rel_path", "true_idx", "pred_idx"}
    if not need <= set(df.columns):
        sys.exit(f"{CSV.name} does not have the columns {sorted(need)}; it has {list(df.columns)}")

    engine = load_engine()
    OUT.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(CLASS_NAMES):
        rows = df[(df["true_idx"] == idx) & (df["pred_idx"] == idx)].sample(frac=1, random_state=42)
        chosen = None
        for rel in rows["rel_path"].head(TRY_PER_CLASS):
            path = DATASET_ROOT / rel
            res = analyze(engine, path.read_bytes(), "en")
            if res["status"] == "reliable" and res["prediction"]["class_name_en"] == name:
                chosen = path
                break
        if chosen is None:
            print(f"{name}: no photo accepted as reliable among the first {TRY_PER_CLASS} correct test photos")
            continue
        target = OUT / f"{idx + 1:02d}_{name.replace(' ', '_')}{chosen.suffix}"
        shutil.copy(chosen, target)
        print(f"{name}: {target.name}")
    print(f"\nDemo photos are in {OUT}")


if __name__ == "__main__":
    main()