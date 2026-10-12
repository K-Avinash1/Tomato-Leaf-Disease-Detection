import hashlib
import json
import platform
import shutil
from datetime import date

import pandas as pd
import torch
import torchvision

from src.config import PROJECT_ROOT, TABLES_DIR
from src.models.checkpoint import load_checkpoint

SRC = PROJECT_ROOT / "models" / "E010_best_model.pth"
DST = PROJECT_ROOT / "models" / "best_model.pth"
EXP = PROJECT_ROOT / "reports" / "experiments" / "E010_mnv3_deep_seed7"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    shutil.copy2(SRC, DST)
    model, ck = load_checkpoint(DST)
    cfg = json.loads((EXP / "config.json").read_text())
    met = json.loads((EXP / "metrics_val.json").read_text())
    bench = pd.read_csv(TABLES_DIR / "cpu_benchmark.csv")
    row = bench[bench["exp"] == ck["exp_id"]].iloc[0].to_dict()
    info = {
        "model_file": "best_model.pth",
        "sha256": sha256(DST),
        "source_experiment": ck["exp_id"],
        "selected_epoch": ck["epoch"],
        "selection": "Phase 11 (decisions 31-32); test set not used for selection",
        "architecture": {"name": ck["model_name"], "dropout": ck["dropout"],
                         "num_classes": ck["num_classes"],
                         "parameters": sum(p.numel() for p in model.parameters())},
        "classes_in_output_order": list(ck["class_names"]),
        "input": {"size": ck["img_size"], "color": "RGB", "resize": "whole image to size x size, no crop",
                  "mean": list(ck["mean"]), "std": list(ck["std"])},
        "training": {k: cfg[k] for k in ["strategy_used", "plan", "unfreeze", "aug", "aug_config", "seed",
                                         "batch_size", "weight_decay", "hardware", "torch", "torchvision",
                                         "pretrained_weights"]},
        "validation": {"metrics": met["metrics"], "calibration": met["calibration"]},
        "cpu_benchmark_laptop": row,
        "created": date.today().isoformat(),
        "local_environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "torchvision": torchvision.__version__},
        "note": "Raw softmax confidence is not a reliability guarantee; see the reliability phases (14-18).",
    }
    (PROJECT_ROOT / "models" / "model_info.json").write_text(json.dumps(info, indent=2, default=str))
    print("Copied:", SRC.name, "->", DST.name)
    print("SHA-256:", info["sha256"])
    print("Saved: models/model_info.json")


if __name__ == "__main__":
    main()