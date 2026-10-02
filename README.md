# Tomato Leaf Disease Detection and Treatment Recommendation Using Deep Learning

A deep-learning system that takes a tomato leaf image, checks whether it is usable and supported, classifies it into one of 10 conditions, explains the prediction with Grad-CAM, and shows management information from a controlled knowledge base. Unsupported or uncertain images are rejected instead of being forced into a disease class.

## Classes (10)
1. Bacterial Spot
2. Early Blight
3. Late Blight
4. Leaf Mold
5. Septoria Leaf Spot
6. Spider Mites
7. Target Spot
8. Yellow Leaf Curl Virus
9. Tomato Mosaic Virus
10. Healthy

## Dataset
PlantVillage Tomato subset (Kaggle: emmarex/plantdisease). Not stored in this repository.

## Status
Phase 1 (environment and setup) complete. Further sections will be added phase by phase.

## Setup
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    python check_environment.py