import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.config import DATASET_ROOT, TABLES_DIR

# Locked class order from the project spec. Index = the model's output position.
CLASS_NAMES = [
    "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Septoria Leaf Spot",
    "Spider Mites", "Target Spot", "Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Healthy",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class TomatoDataset(Dataset):
    """Reads images listed in split_manifest.csv for one split: 'train', 'val' or 'test'."""

    def __init__(self, split: str, transform=None):
        df = pd.read_csv(TABLES_DIR / "split_manifest.csv")
        if "rel_path" not in df.columns:
            raise RuntimeError("split_manifest.csv has no rel_path column: run src.data.make_portable first")
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.paths = [str(DATASET_ROOT / r) for r in self.df["rel_path"]]
        self.labels = [CLASS_TO_IDX[c] for c in self.df["class"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]