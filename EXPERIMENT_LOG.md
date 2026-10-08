# Experiment Log


Each entry below is produced by the training script (settings and numbers are printed automatically). Observations, Conclusion and Retained/Rejected are filled in after reviewing the results.


## Experiments


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
- **Observations:** Validation accuracy 0.9942 and macro F1 0.9937 (14 errors out of 2,400). No overfitting; validation tracks training. Noisy epochs 7-8, stable after epoch 20. Most errors are between related classes (Early/Late Blight) or Yellow Leaf Curl predicted as Bacterial Spot (3 images). About 81 s per epoch on a T4.
- **Conclusion:** A from-scratch CNN is already very strong on PlantVillage, which fits its controlled imaging conditions. Background shortcuts are not ruled out and need Grad-CAM and subset checks.
- **Retained / Rejected:** Retained as the baseline.


### E002_resnet18
- **Objective:** transfer learning: resnet18, feature extraction then partial fine-tuning
- **Dataset version:** PlantVillage tomato, 15,997 clean images (clean_manifest.csv)
- **Data split:** 70/15/15, seed 42, stratified, near-duplicate-group-aware (train 11197, val 2400); test not used
- **Preprocessing:** resize 224x224, ImageNet mean/std normalization
- **Augmentation:** {'crop_scale': (0.75, 1.0), 'crop_ratio': (0.9, 1.1), 'hflip_p': 0.5, 'rot90': True, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.1, 'hue': 0.0} (training only)
- **Model / architecture:** resnet18, torchvision IMAGENET1K_V1, 11,181,642 parameters
- **Trainable layers:** stage A head only; stage B layer4 + fc
- **Optimizer / LR:** AdamW, weight decay 0.0001, cosine schedule per stage; A_head: 5 epochs, lr 0.001; B_partial: 15 epochs, lr 0.0001
- **Batch size / epochs:** 64 / 20
- **Hyperparameters:** dropout 0.3, class weights off
- **Hardware:** Tesla T4 (PyTorch 2.11.0+cu130, Python 3.13.15)
- **Training time:** 22.0 min | checkpoint 44.8 MB
- **Validation (best epoch 18):** accuracy 0.9883, macro precision 0.9858, macro recall 0.9878, macro F1 0.9868, weighted F1 0.9884
- **Stage results (best val macro F1):** A_head 0.8351, B_partial 0.9868
- **Observations:** Stage A (head only) reached best validation macro F1 0.8351 (epoch 4); Stage B (layer4 + fc unfrozen) jumped to 0.9436 in its first epoch and reached 0.9868 at epoch 18. Final: accuracy 0.9883 (28 errors out of 2,400), macro F1 0.9868. Weakest class is Early Blight (F1 0.9474): 8 Late Blight images were predicted as Early Blight and 3 Early Blight as Late Blight; 4 Healthy images were also misclassified. No overfitting. About 66 s per epoch on a T4.
- **Conclusion:** Unfreezing the deepest block was essential (+0.152 macro F1 over Stage A). ResNet-18 is the weakest of the four models on validation and the largest (11.18M parameters, 44.8 MB), and did not beat the from-scratch baseline. Its gap to the others (28 vs 14-17 errors) has not been tested for significance.
- **Retained / Rejected:** Retained for the Phase 11 comparison (required architecture). Currently ranked last on validation.


### E003_efficientnet_b0
- **Objective:** transfer learning: efficientnet_b0, feature extraction then partial fine-tuning
- **Dataset version:** PlantVillage tomato, 15,997 clean images (clean_manifest.csv)
- **Data split:** 70/15/15, seed 42, stratified, near-duplicate-group-aware (train 11197, val 2400); test not used
- **Preprocessing:** resize 224x224, ImageNet mean/std normalization
- **Augmentation:** {'crop_scale': (0.75, 1.0), 'crop_ratio': (0.9, 1.1), 'hflip_p': 0.5, 'rot90': True, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.1, 'hue': 0.0} (training only)
- **Model / architecture:** efficientnet_b0, torchvision IMAGENET1K_V1, 4,020,358 parameters
- **Trainable layers:** stage A head only; stage B features[6:] (last two MBConv stages + final conv) + classifier
- **Optimizer / LR:** AdamW, weight decay 0.0001, cosine schedule per stage; A_head: 5 epochs, lr 0.001; B_partial: 15 epochs, lr 0.0001
- **Batch size / epochs:** 64 / 20
- **Hyperparameters:** dropout 0.3, class weights off
- **Hardware:** Tesla T4 (PyTorch 2.11.0+cu130, Python 3.13.15)
- **Training time:** 24.2 min | checkpoint 16.4 MB
- **Validation (best epoch 17):** accuracy 0.9929, macro precision 0.9919, macro recall 0.9925, macro F1 0.9922, weighted F1 0.9929
- **Stage results (best val macro F1):** A_head 0.8906, B_partial 0.9922
- **Observations:** Stage A best validation macro F1 0.8906 (epoch 5); Stage B best 0.9922 (epoch 17). Late-epoch macro F1 fluctuates between 0.9893 and 0.9922, which is within noise at this validation size. Accuracy 0.9929 (17 errors out of 2,400). Errors are spread over many classes; Early Blight has the lowest F1 (0.9669), with Early/Late Blight confusions again the largest group. About 72 s per epoch on a T4.
- **Conclusion:** Competitive with the baseline (17 vs 14 errors, not a meaningful difference at this validation size) with 4.02M parameters and a 16.4 MB checkpoint. Partial fine-tuning added +0.102 macro F1 over Stage A.
- **Retained / Rejected:** Retained for the Phase 11 comparison.


### E004_mobilenet_v3_large
- **Objective:** transfer learning: mobilenet_v3_large, feature extraction then partial fine-tuning
- **Dataset version:** PlantVillage tomato, 15,997 clean images (clean_manifest.csv)
- **Data split:** 70/15/15, seed 42, stratified, near-duplicate-group-aware (train 11197, val 2400); test not used
- **Preprocessing:** resize 224x224, ImageNet mean/std normalization
- **Augmentation:** {'crop_scale': (0.75, 1.0), 'crop_ratio': (0.9, 1.1), 'hflip_p': 0.5, 'rot90': True, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.1, 'hue': 0.0} (training only)
- **Model / architecture:** mobilenet_v3_large, torchvision IMAGENET1K_V1, 4,214,842 parameters
- **Trainable layers:** stage A head only; stage B features[13:] (last 3 blocks + final conv) + classifier
- **Optimizer / LR:** AdamW, weight decay 0.0001, cosine schedule per stage; A_head: 5 epochs, lr 0.001; B_partial: 15 epochs, lr 0.0001
- **Batch size / epochs:** 64 / 20
- **Hyperparameters:** dropout 0.3, class weights off
- **Hardware:** Tesla T4 (PyTorch 2.11.0+cu130, Python 3.13.15)
- **Training time:** 23.4 min | checkpoint 17.1 MB
- **Validation (best epoch 16):** accuracy 0.9942, macro precision 0.9941, macro recall 0.9930, macro F1 0.9935, weighted F1 0.9942
- **Stage results (best val macro F1):** A_head 0.9614, B_partial 0.9935
- **Observations:** Stage A already strong (best macro F1 0.9614, epoch 5), but its trainable head includes a pretrained 960-to-1280 layer (1.24M trainable parameters), so it is not comparable with the other models' head-only Stage A. Stage B reached 0.9935 at epoch 16. Train loss rose briefly at the start of Stage B (0.112 to 0.187) and recovered. Accuracy 0.9942 (14 errors out of 2,400). Early Blight has the lowest F1 (0.9730; 6 of 150 misclassified). About 70 s per epoch on a T4.
- **Conclusion:** Best of the pretrained models and tied with the from-scratch baseline on error count (14 vs 14; macro F1 0.9935 vs 0.9937 is not a meaningful difference), at 4.21M parameters and 17.1 MB.
- **Retained / Rejected:** Retained for the Phase 11 comparison.

