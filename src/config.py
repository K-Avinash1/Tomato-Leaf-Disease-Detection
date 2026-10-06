import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Local default is the E drive. On Colab we set TOMATO_DATA_ROOT to the unzipped data folder.
DATASET_ROOT = Path(os.environ.get("TOMATO_DATA_ROOT", r"E:\PlantVillage"))
INNER_ROOT = DATASET_ROOT / "PlantVillage"   # nested folder, checked for duplicates
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

# Dataset folder name -> locked class name
CLASS_FOLDERS = {
    "Tomato_Bacterial_spot": "Bacterial Spot",
    "Tomato_Early_blight": "Early Blight",
    "Tomato_Late_blight": "Late Blight",
    "Tomato_Leaf_Mold": "Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spider Mites",
    "Tomato__Target_Spot": "Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato_healthy": "Healthy",
}