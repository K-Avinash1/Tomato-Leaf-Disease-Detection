import torch.nn as nn
from torchvision import models

from src.models.custom_cnn import TomatoCNN

UNFREEZE_DESC = {
    "partial": {
        "resnet18": "layer4 + fc",
        "efficientnet_b0": "features[6:] (last two MBConv stages + final conv) + classifier",
        "mobilenet_v3_large": "features[13:] (last 3 blocks + final conv) + classifier",
        "mobilenet_v3_small": "features[9:] (last 3 blocks + final conv) + classifier",
    },
    "deep": {
        "resnet18": "layer3 + layer4 + fc",
        "efficientnet_b0": "features[4:] (last four MBConv stages + final conv) + classifier",
        "mobilenet_v3_large": "features[7:] (blocks 7-15 + final conv) + classifier",
        "mobilenet_v3_small": "features[6:] (blocks 6-11 + final conv) + classifier",
    },
}


def build_model(name: str, num_classes: int = 10, dropout: float = 0.3, pretrained: bool = True):
    if name == "custom_cnn":
        return TomatoCNN(num_classes, dropout)
    if name == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
        return m
    if name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, num_classes))
        return m
    if name in ("mobilenet_v3_large", "mobilenet_v3_small"):
        if name == "mobilenet_v3_large":
            fn, w = models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        else:
            fn, w = models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        m = fn(weights=w if pretrained else None)
        m.classifier[2] = nn.Dropout(dropout)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m
    raise ValueError(f"Unknown model: {name}")


def head_module(model, name):
    return model.fc if name == "resnet18" else model.classifier


def partial_blocks(model, name, mode="partial"):
    if mode == "deep":
        if name == "resnet18":
            return [model.layer3, model.layer4]
        if name == "efficientnet_b0":
            return list(model.features)[4:]
        if name == "mobilenet_v3_large":
            return list(model.features)[7:]
        if name == "mobilenet_v3_small":
            return list(model.features)[6:]
    if name == "resnet18":
        return [model.layer4]
    if name == "efficientnet_b0":
        return list(model.features)[6:]
    if name == "mobilenet_v3_large":
        return list(model.features)[13:]
    if name == "mobilenet_v3_small":
        return list(model.features)[9:]
    return []


def set_trainable(model, name, mode):
    """mode: 'full' (all layers), 'head' (new head only), 'partial' or 'deep' (head + deepest blocks)."""
    if mode == "full" or name == "custom_cnn":
        for p in model.parameters():
            p.requires_grad = True
        return
    for p in model.parameters():
        p.requires_grad = False
    modules = [head_module(model, name)]
    if mode in ("partial", "deep"):
        modules += partial_blocks(model, name, mode)
    for mod in modules:
        for p in mod.parameters():
            p.requires_grad = True


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def freeze_frozen_bn(model):
    """Keep BatchNorm layers whose parameters are all frozen in eval mode (use pretrained statistics)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if all(not p.requires_grad for p in m.parameters()):
                m.eval()


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())