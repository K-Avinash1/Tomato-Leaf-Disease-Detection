import torch

from src.data.dataset import CLASS_NAMES
from src.models.factory import build_model


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    assert list(ck["class_names"]) == CLASS_NAMES, "class order mismatch"
    model = build_model(ck["model_name"], len(CLASS_NAMES), ck["dropout"], pretrained=False)
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval(), ck