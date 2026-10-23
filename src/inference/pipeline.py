"""One photo in, one result out: quality -> prediction -> reliability gate -> Grad-CAM -> treatment text.

Result rules (Phases 14-18):
  quality_failed : a quality check failed; no prediction is made
  unsupported    : the photo is far from tomato training images; no disease name, no treatment
  uncertain      : top guess shown with the caveat; treatment hidden (a wrong treatment is the worst outcome)
  reliable       : disease, confidence, Grad-CAM and treatment text
"""
import base64
import io
import time

import numpy as np
from PIL import Image

from src.inference import display
from src.knowledge_base.loader import CLASS_TO_ID
from src.reliability.quality import measure, open_image


def _softmax(z):
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def _png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _gradcam_overlay(engine, x, img, class_idx):
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    cam = engine.cam(input_tensor=x, targets=[ClassifierOutputTarget(int(class_idx))])[0]
    base = np.asarray(img.resize((224, 224), Image.BILINEAR)).astype(np.float32) / 255.0
    return show_cam_on_image(base, cam, use_rgb=True)


def _prediction(engine, lang, idx, p):
    name_en = engine.classes[idx]
    disease_id = CLASS_TO_ID[name_en]
    return {"class_id": disease_id, "class_name_en": name_en,
            "class_name": engine.kb.display_name(disease_id, lang),
            "confidence": round(float(p), 4), "confidence_text": display.confidence_text(p)}


def _quality_block(reasons, metrics, ui):
    return {"passed": not reasons, "reasons": list(reasons),
            "messages": [ui[display.QUALITY_UI_KEYS[r]] for r in reasons],
            "metrics": None if metrics is None else {k: round(float(v), 2) for k, v in metrics.items()}}


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000, 1)


def analyze(engine, data, lang="en"):
    """data: raw bytes of the uploaded file. Returns a dict matching app.schemas.PredictResponse."""
    t0 = time.perf_counter()
    ui = engine.kb.ui(lang)                      # ValueError for an unsupported language

    img, err = open_image(io.BytesIO(data))
    metrics = None if err else measure(img)
    reasons = [err] if err else engine.quality.reasons(metrics)
    quality = _quality_block(reasons, metrics, ui)
    if reasons:
        return {"status": "quality_failed", "lang": lang, "headline": quality["messages"][0],
                "messages": {"headline": quality["messages"][0], "scope": ui["scope_message"]},
                "quality": quality, "gate": None, "prediction": None, "alternatives": [],
                "gradcam_png_base64": None, "treatment": None, "timing_ms": _ms(t0)}

    x = engine.transform(img).unsqueeze(0)
    with engine.lock:                            # Grad-CAM hooks keep state, so one photo at a time
        logits, feats = engine.forward(x)
        status_arr, sim, conf_gate = engine.gate.decide(logits, feats)
        status = str(status_arr[0])
        top = int(logits[0].argmax())
        overlay = _gradcam_overlay(engine, x, img, top) if status != "unsupported" else None

    gate = {"status": status, "knn_similarity": round(float(sim[0]), 4), "tau_knn": engine.gate.tau_knn,
            "confidence": round(float(conf_gate[0]), 4), "tau_conf": engine.gate.tau_conf}
    headline = ui[display.STATUS_UI_KEYS[status]]
    out = {"status": status, "lang": lang, "headline": headline, "quality": quality, "gate": gate,
           "prediction": None, "alternatives": [], "gradcam_png_base64": None, "treatment": None,
           "timing_ms": None}

    if status == "unsupported":
        out["messages"] = {"headline": headline, "scope": ui["scope_message"], "disclaimer": ui["disclaimer"]}
    else:
        probs = _softmax(logits[0].astype(np.float64) / engine.temperature)   # displayed confidence (T = 1.0)
        out["prediction"] = _prediction(engine, lang, top, probs[top])
        out["gradcam_png_base64"] = _png_b64(overlay)
        if status == "uncertain":
            order = np.argsort(-probs)[1:3]
            out["alternatives"] = [_prediction(engine, lang, int(i), probs[i]) for i in order]
            out["messages"] = {"headline": headline, "scope": ui["scope_message"],
                               "caveat": ui["confidence_caveat"], "disclaimer": ui["disclaimer"]}
        else:
            entry = engine.kb.entry(CLASS_TO_ID[engine.classes[top]], lang)
            out["treatment"] = entry
            out["messages"] = {"headline": headline, "caveat": ui["confidence_caveat"],
                               "disclaimer": ui["disclaimer"]}
            if entry["expert_review_flag"]:
                out["messages"]["expert_review"] = ui["expert_review_note"]
    out["timing_ms"] = _ms(t0)
    return out