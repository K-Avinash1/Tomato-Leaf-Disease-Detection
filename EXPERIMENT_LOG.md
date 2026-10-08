# Experiment Log

No experiments have been run yet. The first entry will be added in Phase 7 (baseline CNN).

## Template for each experiment

### E001_custom_cnn
- **Objective:** baseline custom CNN trained from scratch (reference for transfer learning)
- **Dataset version:** PlantVillage tomato, 15,997 clean images (clean_manifest.csv)
- **Data split:** 70/15/15, seed 42, stratified, near-duplicate-group-aware (train 11197, val 2400); test not used
- **Preprocessing:** resize 224x224, ImageNet mean/std normalization
- **Augmentation:** {'crop_scale': (0.75, 1.0), 'crop_ratio': (0.9, 1.1), 'hflip_p': 0.5, 'rot90': True, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.1, 'hue': 0.0} (training only)
- **Model / architecture:** custom_cnn, 5 conv stages + global average pooling, 2,347,178 parameters
- **Optimizer / LR:** AdamW, lr 0.001, weight decay 0.0001, cosine schedule
- **Batch size / epochs:** 64 / 30
- **Hyperparameters:** dropout 0.3, class weights off
- **Hardware:** Tesla T4 (PyTorch 2.11.0+cu130, Python 3.13.15)
- **Training time:** 40.9 min
- **Validation (best epoch 26):** accuracy 0.9942, macro precision 0.9925, macro recall 0.9948, macro F1 0.9937, weighted F1 0.9942
- **Observations:** <fill in>
- **Conclusion:** <fill in>
- **Retained / Rejected:** <fill in>

- **Experiment ID:**
- **Objective:**
- **Dataset version:**
- **Data split:**
- **Preprocessing:**
- **Augmentation:**
- **Model:**
- **Architecture:**
- **Optimizer:**
- **Learning rate:**
- **Batch size:**
- **Epochs:**
- **Hyperparameters:**
- **Hardware:**
- **Training time:**
- **Accuracy:**
- **Precision:**
- **Recall:**
- **F1:**
- **Macro F1:**
- - **Observations:** Validation accuracy 0.9942 and macro F1 0.9937 (14 errors out of 2,400). No overfitting; validation tracks training. Noisy epochs 7-8, stable after epoch 20. Most errors are between related classes (Early/Late Blight) or Yellow Leaf Curl predicted as Bacterial Spot (3 images). About 81 s per epoch on a T4.

- **Conclusion:** A from-scratch CNN is already very strong on PlantVillage, which fits its controlled imaging conditions. Transfer-learning gains will be small on this dataset. Background shortcuts are not ruled out and need Grad-CAM and subset checks.

- **Retained / Rejected:** Retained as the baseline.