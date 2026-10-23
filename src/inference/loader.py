"""Loads everything the app needs ONCE at startup, and refuses to start if a file is not the one that was tested."""
import copy
import hashlib
import json
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.config import PROJECT_ROOT
from src.data.dataset import CLASS_NAMES
from src.knowledge_base.loader import CLASS_TO_ID, KnowledgeBase
from src.preprocessing.transforms import get_eval_transform
from src.reliability.decision import ReliabilityGate
from src.reliability.features import load_final_model
from src.reliability.quality import QualityChecker

MODELS = PROJECT_ROOT / "models"
SKIP_ENV = "TOMATO_SKIP_HASH_CHECK"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _verify(path, expected_by_source, label):
    """expected_by_source: {where the expected hash was recorded: hash}. All must equal the file's hash."""
    if os.environ.get(SKIP_ENV) == "1":
        print(f"[warning] {SKIP_ENV}=1, skipping the {label} hash check")
        return
    actual = sha256_file(path)
    for where, expected in expected_by_source.items():
        if actual != expected:
            raise RuntimeError(
                f"{label} does not match the tested version.\n  file: {path}\n  sha256 of file: {actual}\n"
                f"  expected ({where}): {expected}\nRestore the right file from the Drive backup. "
                f"(Set {SKIP_ENV}=1 only if you know why the hash differs.)")


def _build_gradcam(model):
    from pytorch_grad_cam import GradCAM          # same package and layer as Phase 19
    cam_model = copy.deepcopy(model).eval()       # separate copy: Grad-CAM hooks must not touch the serving model
    for m in cam_model.modules():                 # in-place activations break gradient hooks; outputs are unchanged
        if hasattr(m, "inplace"):
            m.inplace = False
    return GradCAM(model=cam_model, target_layers=[cam_model.features[-1]])


@dataclass
class Engine:
    model: Any
    cam: Any
    gate: ReliabilityGate
    quality: QualityChecker
    transform: Any
    kb: KnowledgeBase
    classes: list
    temperature: float          # used for the DISPLAYED confidence only (1.0 = raw, Phase 18)
    model_sha256: str
    lock: Any

    def forward(self, x):
        """Logits (1 x 10) and the 1,280-number fingerprint, exactly as in reliability/features.py."""
        with torch.inference_mode():
            f = torch.flatten(self.model.avgpool(self.model.features(x)), 1)
            h = self.model.classifier[1](self.model.classifier[0](f))
            logits = self.model.classifier[3](self.model.classifier[2](h))
        return logits.float().numpy(), h.float().numpy()


def load_engine():
    cfg = json.load(open(MODELS / "reliability_config.json"))
    info = json.load(open(MODELS / "model_info.json"))
    calib = json.load(open(MODELS / "calibration.json"))

    ckpt_path = MODELS / "best_model.pth"
    _verify(ckpt_path, {"reliability_config.json": cfg["model_sha256"],
                        "calibration.json": calib["model_sha256"],
                        "model_info.json": info["sha256"]}, "model")
    _verify(MODELS / cfg["knn_bank_file"], {"reliability_config.json": cfg["knn_bank_sha256"]}, "kNN bank")

    model, _ = load_final_model()
    classes = list(CLASS_NAMES)
    if classes != info["classes_in_output_order"]:
        raise RuntimeError("class order in the code differs from models/model_info.json")
    unknown = [c for c in classes if c not in CLASS_TO_ID]
    if unknown:
        raise RuntimeError(f"classes without a knowledge-base id: {unknown}")

    tf = get_eval_transform()
    # self-check: the manual forward used for the fingerprint must reproduce model()
    torch.manual_seed(0)
    probe = torch.randn(1, 3, 224, 224)
    engine = Engine(model=model, cam=_build_gradcam(model),
                    gate=ReliabilityGate.from_files(MODELS / "reliability_config.json", MODELS / cfg["knn_bank_file"]),
                    quality=QualityChecker.from_file(MODELS / "quality_config.json"),
                    transform=tf, kb=KnowledgeBase.load(), classes=classes,
                    temperature=float(calib["temperature_used_for_displayed_confidence"]),
                    model_sha256=cfg["model_sha256"], lock=threading.Lock())
    logits, _ = engine.forward(probe)
    with torch.inference_mode():
        ref = model(probe).float().numpy()
    if np.abs(ref - logits).max() > 1e-4:
        raise RuntimeError("manual forward does not match model(); the fingerprint would be wrong")
    return engine