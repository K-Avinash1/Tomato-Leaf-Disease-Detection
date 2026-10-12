import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import DATASET_ROOT, TABLES_DIR
from src.models.checkpoint import load_checkpoint
from src.preprocessing.transforms import IMG_SIZE, get_eval_transform


def timeit(fn, n, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    a = ap.parse_args()
    man = pd.read_csv(TABLES_DIR / "split_manifest.csv")
    img_path = DATASET_ROOT / man[man["split"] == "val"]["rel_path"].iloc[0]
    tf = get_eval_transform()
    print(f"CPU threads: {torch.get_num_threads()} | torch {torch.__version__}")

    rows = []
    for path in a.ckpt:
        model, ck = load_checkpoint(path)
        x1 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        x16 = torch.randn(16, 3, IMG_SIZE, IMG_SIZE)

        def end_to_end():
            t = tf(Image.open(img_path).convert("RGB")).unsqueeze(0)
            model(t)

        with torch.inference_mode():
            b1, b1_p95 = timeit(lambda: model(x1), n=60, warmup=10)
            b16, _ = timeit(lambda: model(x16), n=10, warmup=2)
            e2e, e2e_p95 = timeit(end_to_end, n=40, warmup=5)
        rows.append({"exp": ck["exp_id"], "model": ck["model_name"],
                     "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                     "file_MB": round(os.path.getsize(path) / 1e6, 1),
                     "ms_per_image_batch1_median": round(b1, 1), "batch1_p95": round(b1_p95, 1),
                     "ms_per_image_batch16": round(b16 / 16, 1),
                     "ms_end_to_end_median": round(e2e, 1), "end_to_end_p95": round(e2e_p95, 1)})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    df.to_csv(TABLES_DIR / "cpu_benchmark.csv", index=False)
    print("Saved: reports/tables/cpu_benchmark.csv")


if __name__ == "__main__":
    main()