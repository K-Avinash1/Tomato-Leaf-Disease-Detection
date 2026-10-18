import json

import cv2
import numpy as np
from PIL import Image, ImageOps

MESSAGES = {
    "invalid_file": "The file could not be read as an image.",
    "too_small": "The image is too small. Please upload a larger photo.",
    "blank": "The image has almost no detail. Please photograph a tomato leaf.",
    "too_dark": "The photo is too dark. Please take it in good light.",
    "too_bright": "The photo is overexposed. Please avoid strong glare or direct sun.",
    "too_blurry": "The photo is too blurry. Please hold the camera steady and focus on the leaf.",
}


def open_image(source):
    """Returns (RGB PIL image, None), or (None, 'invalid_file')."""
    try:
        img = Image.open(source)
        img.load()
        return ImageOps.exif_transpose(img).convert("RGB"), None
    except Exception:
        return None, "invalid_file"


def measure(img):
    """Quality numbers, computed on a 256x256 grayscale version (as in Phase 3)."""
    w, h = img.size
    g = cv2.cvtColor(np.asarray(img.resize((256, 256))), cv2.COLOR_RGB2GRAY)
    return {"min_side": min(w, h), "p5": float(np.percentile(g, 5)), "p95": float(np.percentile(g, 95)),
            "std": float(g.std()), "sharpness": float(cv2.Laplacian(g, cv2.CV_64F).var())}


class QualityChecker:
    def __init__(self, cfg):
        self.t = cfg["thresholds"]

    @classmethod
    def from_file(cls, path):
        return cls(json.load(open(path)))

    def reasons(self, m):
        t, r = self.t, []
        if m["min_side"] < t["min_side"]:
            r.append("too_small")
        if m["std"] < t["std"]:
            r.append("blank")
        if m["p95"] < t["p95_dark"]:
            r.append("too_dark")
        if m["p5"] > t["p5_bright"]:
            r.append("too_bright")
        if m["sharpness"] < t["sharpness"]:
            r.append("too_blurry")
        return r

    def check(self, source):
        img, err = open_image(source)
        if err:
            return [err], None
        m = measure(img)
        return self.reasons(m), m