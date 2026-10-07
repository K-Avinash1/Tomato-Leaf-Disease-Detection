# Project Context

**Project:** Tomato Leaf Disease Detection and Treatment Recommendation Using Deep Learning
**Current phase:** Phase 6 (Data augmentation)
**Completed phases:** Phases 1 to 5
**Random seed:** 42

## Environment
- Local development in VS Code on Windows, project on E drive
- Python virtual environment: `venv`
- PyTorch (CPU build), no NVIDIA GPU
- Laptop: AMD Ryzen 5 7520U, 8 GB RAM
Colab: Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu130. Local: CPU, Python 3.11.9, PyTorch 2.14.1+cpu. Data verified byte-identical on both (0 missing, 0 hash mismatches).

## Dataset
## Dataset
- PlantVillage Tomato subset (Kaggle: emmarex/plantdisease), at `E:\PlantVillage` (outer tomato folders only)
- Valid images: 16,011 (all 256x256, RGB, JPEG). Largest class 3,208 (Yellow Leaf Curl Virus), smallest 373 (Tomato Mosaic Virus), imbalance ratio 8.60
- 1 corrupted file and 14 exact duplicate copies to be excluded
- Per-class counts: reports/tables/dataset_summary.csv

## Locked classes (10)
Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy

## Models planned
Custom CNN, ResNet-18, EfficientNet-B0, MobileNetV3

## Key findings
Shortcut risk: 144 of 148 very dark (black-background) images are Late Blight (7.6% of that class). To be checked via per-subset evaluation (Phase 12/13) and Grad-CAM (Phase 19)

Colab T4 benchmark at 224, batch 32, all layers (compute only): ResNet-18 about 0.6, EfficientNet-B0 about 0.8, MobileNetV3-Large about 0.5, MobileNetV3-Small about 0.2 min/epoch.

## Next phase
Phase 2: Dataset acquisition and verification