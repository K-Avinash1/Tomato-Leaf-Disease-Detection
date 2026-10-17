import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from src.config import PROJECT_ROOT
from src.models.checkpoint import load_checkpoint
from src.preprocessing.transforms import get_eval_transform

FINAL_CKPT = PROJECT_ROOT / "models" / "best_model.pth"


def load_final_model():
    model, ck = load_checkpoint(FINAL_CKPT)
    assert ck["model_name"] == "mobilenet_v3_large", "feature extraction is written for MobileNetV3-Large"
    return model, ck


def open_rgb(path):
    img = ImageOps.exif_transpose(Image.open(path))      # honour phone rotation
    return img.convert("RGB")


class PathDataset(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = [str(p) for p in paths], tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(open_rgb(self.paths[i]))


@torch.inference_mode()
def extract(model, paths, batch_size=64, desc=""):
    """Logits (N x 10) and the 1,280-number 'fingerprint' the final layer sees (N x 1280)."""
    dl = DataLoader(PathDataset(paths, get_eval_transform()), batch_size=batch_size, shuffle=False, num_workers=0)
    L, F = [], []
    for i, x in enumerate(dl):
        f = torch.flatten(model.avgpool(model.features(x)), 1)
        h = model.classifier[1](model.classifier[0](f))
        logits = model.classifier[3](model.classifier[2](h))
        if i == 0:
            ref = model(x).float().numpy()
            assert np.abs(ref - logits.float().numpy()).max() < 1e-4, "manual forward does not match model()"
        L.append(logits.float().numpy())
        F.append(h.float().numpy())
        if (i + 1) % 20 == 0:
            print(f"  {desc}: {min((i + 1) * batch_size, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(L), np.concatenate(F)