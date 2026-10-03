# Project Context

**Project:** Tomato Leaf Disease Detection and Treatment Recommendation Using Deep Learning
**Current phase:** Phase 4 (Dataset splitting)
**Completed phases:** Phases 1, 2 and 3 (EDA done, near-duplicate grouping to be applied in Phase 4)
**Random seed:** 42

## Environment
- Local development in VS Code on Windows, project on E drive
- Python virtual environment: `venv`
- PyTorch (CPU build), no NVIDIA GPU
- Laptop: AMD Ryzen 5 7520U, 8 GB RAM

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
None yet.

## Next phase
Phase 2: Dataset acquisition and verification