"""Run the full pipeline on photos from the command line (no web server needed).

    python -m src.inference.smoke_test path\\to\\leaf1.jpg path\\to\\leaf2.jpg --lang hi

Prints an ASCII summary per photo; the full result (JSON, UTF-8) and the Grad-CAM picture go to reports/app_smoke/.
"""
import argparse
import base64
import json
from pathlib import Path

from src.config import PROJECT_ROOT
from src.inference.loader import load_engine
from src.inference.pipeline import analyze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--lang", default="en", choices=["en", "hi"])
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / "reports" / "app_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = load_engine()
    print("Engine loaded: model and kNN bank match their recorded SHA-256, forward self-check passed.\n")

    for p in args.images:
        res = analyze(engine, Path(p).read_bytes(), args.lang)
        stem = Path(p).stem
        png = res.pop("gradcam_png_base64")
        if png:
            (out_dir / f"{stem}_gradcam.png").write_bytes(base64.b64decode(png))
        (out_dir / f"{stem}_{args.lang}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        pred, gate = res["prediction"], res["gate"]
        line = f"{Path(p).name}: {res['status']}"
        if pred:
            line += f" | {pred['class_name_en']} {pred['confidence_text']}"
        if gate:
            line += f" | knn {gate['knn_similarity']:.3f} (min {gate['tau_knn']}) conf {gate['confidence']:.3f} (min {gate['tau_conf']:.4f})"
        if res["quality"]["reasons"]:
            line += f" | quality: {res['quality']['reasons']}"
        print(f"{line} | {res['timing_ms']} ms | treatment shown: {res['treatment'] is not None}")
    print(f"\nFull results saved in {out_dir}")


if __name__ == "__main__":
    main()