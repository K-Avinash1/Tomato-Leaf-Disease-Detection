import platform, sys
import torch, torchvision, sklearn, cv2, PIL, numpy, pandas

print("Python      :", sys.version.split()[0])
print("OS          :", platform.platform())
print("PyTorch     :", torch.__version__)
print("torchvision :", torchvision.__version__)
print("NumPy       :", numpy.__version__)
print("pandas      :", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("OpenCV      :", cv2.__version__)
print("Pillow      :", PIL.__version__)
print("CUDA avail. :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))

# Smoke test: can pretrained architectures be built?
from torchvision import models
for name, fn in [("ResNet-18", models.resnet18),
                 ("EfficientNet-B0", models.efficientnet_b0),
                 ("MobileNetV3-Small", models.mobilenet_v3_small)]:
    n = sum(p.numel() for p in fn(weights=None).parameters())
    print(f"{name:18s} builds OK  ({n:,} params)")

from src.utils.seed import set_seed
set_seed()
print("Seed utility OK")