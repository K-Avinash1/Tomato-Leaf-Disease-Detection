from torchvision import transforms as T

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_eval_transform(img_size: int = IMG_SIZE):
    """Deterministic pipeline used for validation, test and the web app."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),                              # HWC uint8 [0,255] -> CHW float [0,1]
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(tensor):
    """Undo normalization so a tensor can be displayed as a picture."""
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)