import random

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as T

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Every augmentation setting lives here, so experiments can log and tune it.
AUG_CONFIG = {
    "crop_scale": (0.75, 1.0),
    "crop_ratio": (0.9, 1.1),
    "hflip_p": 0.5,
    "rot90": True,          # random 0/90/180/270 degree rotation (no black corners)
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.1,
    "hue": 0.0,             # hue shifts could change symptom color, so none
}

# Named overrides on top of AUG_CONFIG (used for tuning experiments)
AUG_PRESETS = {
    "default": {},
    "strong": {"crop_scale": (0.5, 1.0), "brightness": 0.3, "contrast": 0.3, "saturation": 0.2},
}


class RandomRightAngleRotation:
    """Rotate by 0, 90, 180 or 270 degrees. On a square image this adds no borders."""

    def __call__(self, img):
        k = random.choice([0, 90, 180, 270])
        return TF.rotate(img, k) if k else img


def get_eval_transform(img_size: int = IMG_SIZE):
    """Deterministic pipeline used for validation, test and the web app."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_train_transform(img_size: int = IMG_SIZE, cfg: dict = None):
    """Random augmentation pipeline. Used for the TRAINING split only."""
    c = {**AUG_CONFIG, **(cfg or {})}
    steps = [
        T.RandomResizedCrop(img_size, scale=c["crop_scale"], ratio=c["crop_ratio"]),
        T.RandomHorizontalFlip(c["hflip_p"]),
    ]
    if c["rot90"]:
        steps.append(RandomRightAngleRotation())
    steps += [
        T.ColorJitter(brightness=c["brightness"], contrast=c["contrast"],
                      saturation=c["saturation"], hue=c["hue"]),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return T.Compose(steps)


def denormalize(tensor):
    """Undo normalization so a tensor can be displayed as a picture."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)