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


### E005_mnv3_seed7
- **Objective:** noise check: E004 settings, different training seed
- **Dataset version:** PlantVillage tomato, 15,997 clean images (clean_manifest.csv)
- **Data split:** 70/15/15, split seed 42, stratified, near-duplicate-group-aware (train 11197, val 2400); test not used
- **Preprocessing:** resize 224x224, ImageNet mean/std normalization
- **Augmentation:** preset 'default': {'crop_scale': (0.75, 1.0), 'crop_ratio': (0.9, 1.1), 'hflip_p': 0.5, 'rot90': True, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.1, 'hue': 0.0} (training only)
- **Model / architecture:** mobilenet_v3_large, torchvision IMAGENET1K_V1, 4,214,842 parameters
- **Trainable layers:** stage A head only; stage B features[13:] (last 3 blocks + final conv) + classifier
- **Optimizer / LR:** AdamW, weight decay 0.0001, cosine schedule per stage; A_head: 5 epochs, lr 0.001; B_partial: 15 epochs, lr 0.0001
- **Batch size / epochs:** 64 / 20
- **Hyperparameters:** dropout 0.3, class weights off, training seed 7
- **Hardware:** Tesla T4 (PyTorch 2.11.0+cu130, Python 3.13.15)
- **Training time:** 22.1 min | checkpoint 17.1 MB
- **Validation (best epoch 18):** accuracy 0.9942, macro precision 0.9939, macro recall 0.9935, macro F1 0.9937, weighted F1 0.9942 (14 errors)
- **Calibration (validation, best epoch):** NLL 0.0173, ECE 0.0031
- **Stage results (best val macro F1):** A_head 0.9656, B_partial 0.9937
- **Observations:** Same settings as E004 with training seed 7. Accuracy 0.9942, macro F1 0.9937 (14 errors), within 0.0002 of E004 (0.9935, 14 errors). Validation loss differs more: 0.0171 at epoch 20 versus 0.0228 for E004, with no change in settings. NLL 0.0173, ECE 0.0031. Early Blight is again the weakest class (recall 0.9667).
- **Conclusion:** Macro F1 and error count were stable across these two seeds, but validation loss can swing by about a quarter, so the 10%-lower-loss retention criterion was too loose and was dropped (decision 25). One pair of runs is only a rough noise estimate.
- **Retained / Rejected:** Retained as the seed-noise reference (not a candidate model).


### E006_mnv3_deep
- **Objective:** tuning: deeper unfreezing in Stage B vs E004
- **Dataset version:** PlantVillage tomato, 15,997 clean images (clean_manifest.csv)
- **Data split:** 70/15/15, split seed 42, stratified, near-duplicate-group-aware (train 11197, val 2400); test not used
- **Preprocessing:** resize 224x224, ImageNet mean/std normalization
- **Augmentation:** preset 'default': {'crop_scale': (0.75, 1.0), 'crop_ratio': (0.9, 1.1), 'hflip_p': 0.5, 'rot90': True, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.1, 'hue': 0.0} (training only)
- **Model / architecture:** mobilenet_v3_large, torchvision IMAGENET1K_V1, 4,214,842 parameters
- **Trainable layers:** stage A head only; stage B features[7:] (blocks 7-15 + final conv) + classifier
- **Optimizer / LR:** AdamW, weight decay 0.0001, cosine schedule per stage; A_head: 5 epochs, lr 0.001; B_deep: 15 epochs, lr 0.0001
- **Batch size / epochs:** 64 / 20
- **Hyperparameters:** dropout 0.3, class weights off, training seed 42
- **Hardware:** Tesla T4 (PyTorch 2.11.0+cu130, Python 3.13.15)
- **Training time:** 22.0 min | checkpoint 17.1 MB
- **Validation (best epoch 14):** accuracy 0.9983, macro precision 0.9982, macro recall 0.9986, macro F1 0.9984, weighted F1 0.9983 (4 errors)
- **Calibration (validation, best epoch):** NLL 0.0106, ECE 0.0037
- **Stage results (best val macro F1):** A_head 0.9614, B_deep 0.9984
- **Observations:** Unfreezing blocks 7-15 in Stage B (4.15M of 4.21M parameters trainable) gave the best result so far: accuracy 0.9983, macro F1 0.9984 (4 errors at epoch 14); at the last epoch macro F1 0.9975 (about 6 errors), still above E004's last epoch (0.9930). Lowest validation loss 0.0088, NLL 0.0106, ECE 0.0037. Early Blight recall 1.00 (E004: 0.96). Remaining errors: Late Blight to Early Blight 1, Septoria to Yellow Leaf Curl 1, Yellow Leaf Curl to Bacterial Spot 1, Healthy to Early Blight 1. Same size and about the same time as E004.
- **Conclusion:** Passes the retention rule (macro F1 +0.0049 over E004, 10 fewer errors), and the gain is far larger than the E004/E005 seed difference. It is the best of four tried changes and the best epoch was chosen on validation, so a replicate (E010) is needed before relying on it.
- **Retained / Rejected:** Retained.


### E007_mnv3_lrb3e4
- **Objective:** tuning: Stage B learning rate 3e-4 vs E004
- **Dataset version:** PlantVillage tomato, 15,997 clean images (clean_manifest.csv)
- **Data split:** 70/15/15, split seed 42, stratified, near-duplicate-group-aware (train 11197, val 2400); test not used
- **Preprocessing:** resize 224x224, ImageNet mean/std normalization
- **Augmentation:** preset 'default': {'crop_scale': (0.75, 1.0), 'crop_ratio': (0.9, 1.1), 'hflip_p': 0.5, 'rot90': True, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.1, 'hue': 0.0} (training only)
- **Model / architecture:** mobilenet_v3_large, torchvision IMAGENET1K_V1, 4,214,842 parameters
- **Trainable layers:** stage A head only; stage B features[13:] (last 3 blocks + final conv) + classifier
- **Optimizer / LR:** AdamW, weight decay 0.0001, cosine schedule per stage; A_head: 5 epochs, lr 0.001; B_partial: 15 epochs, lr 0.0003
- **Batch size / epochs:** 64 / 20
- **Hyperparameters:** dropout 0.3, class weights off, training seed 42
- **Hardware:** Tesla T4 (PyTorch 2.11.0+cu130, Python 3.13.15)
- **Training time:** 21.8 min | checkpoint 17.1 MB
- **Validation (best epoch 17):** accuracy 0.9967, macro precision 0.9970, macro recall 0.9955, macro F1 0.9962, weighted F1 0.9967 (8 errors)
- **Calibration (validation, best epoch):** NLL 0.0139, ECE 0.0016
- **Stage results (best val macro F1):** A_head 0.9614, B_partial 0.9962
- **Observations:** Stage B learning rate 3e-4: accuracy 0.9967, macro F1 0.9962 (8 errors), +0.0027 over E004; last epoch 0.9959. Noisier early Stage B (validation loss up to 0.0526 at epoch 10) but settled. NLL 0.0139, ECE 0.0016 (lowest so far).
- **Conclusion:** The improvement is below the 0.004 retention threshold, so it is not retained by the rule. The direction is positive and it might combine with deeper unfreezing; inconclusive.
- **Retained / Rejected:** Rejected as a standalone change (inconclusive).


### E008_mnv3_classw
- **Objective:** tuning: class weights on vs E004
- **Dataset version:** PlantVillage tomato, 15,997 clean images (clean_manifest.csv)
- **Data split:** 70/15/15, split seed 42, stratified, near-duplicate-group-aware (train 11197, val 2400); test not used
- **Preprocessing:** resize 224x224, ImageNet mean/std normalization
- **Augmentation:** preset 'default': {'crop_scale': (0.75, 1.0), 'crop_ratio': (0.9, 1.1), 'hflip_p': 0.5, 'rot90': True, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.1, 'hue': 0.0} (training only)
- **Model / architecture:** mobilenet_v3_large, torchvision IMAGENET1K_V1, 4,214,842 parameters
- **Trainable layers:** stage A head only; stage B features[13:] (last 3 blocks + final conv) + classifier
- **Optimizer / LR:** AdamW, weight decay 0.0001, cosine schedule per stage; A_head: 5 epochs, lr 0.001; B_partial: 15 epochs, lr 0.0001
- **Batch size / epochs:** 64 / 20
- **Hyperparameters:** dropout 0.3, class weights on, training seed 42
- **Hardware:** Tesla T4 (PyTorch 2.11.0+cu130, Python 3.13.15)
- **Training time:** 21.5 min | checkpoint 17.1 MB
- **Validation (best epoch 14):** accuracy 0.9929, macro precision 0.9925, macro recall 0.9923, macro F1 0.9924, weighted F1 0.9929 (17 errors)
- **Calibration (validation, best epoch):** NLL 0.0261, ECE 0.0043
- **Stage results (best val macro F1):** A_head 0.9656, B_partial 0.9924
- **Observations:** Class weights on: accuracy 0.9929, macro F1 0.9924 (17 errors) versus E004's 14. Early Blight did not improve (recall 0.96, F1 0.9697 versus 0.9730). NLL 0.0261, ECE 0.0043, the worst of the round.
- **Conclusion:** No benefit. The class imbalance (up to 8.6:1) is not what limits performance here; the smallest class (Tomato Mosaic Virus) is already classified perfectly.
- **Retained / Rejected:** Rejected.


### E009_mnv3_strongaug
- **Objective:** tuning: stronger augmentation vs E004
- **Dataset version:** PlantVillage tomato, 15,997 clean images (clean_manifest.csv)
- **Data split:** 70/15/15, split seed 42, stratified, near-duplicate-group-aware (train 11197, val 2400); test not used
- **Preprocessing:** resize 224x224, ImageNet mean/std normalization
- **Augmentation:** preset 'strong': {'crop_scale': (0.5, 1.0), 'crop_ratio': (0.9, 1.1), 'hflip_p': 0.5, 'rot90': True, 'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.2, 'hue': 0.0} (training only)
- **Model / architecture:** mobilenet_v3_large, torchvision IMAGENET1K_V1, 4,214,842 parameters
- **Trainable layers:** stage A head only; stage B features[13:] (last 3 blocks + final conv) + classifier
- **Optimizer / LR:** AdamW, weight decay 0.0001, cosine schedule per stage; A_head: 5 epochs, lr 0.001; B_partial: 15 epochs, lr 0.0001
- **Batch size / epochs:** 64 / 20
- **Hyperparameters:** dropout 0.3, class weights off, training seed 42
- **Hardware:** Tesla T4 (PyTorch 2.11.0+cu130, Python 3.13.15)
- **Training time:** 21.1 min | checkpoint 17.1 MB
- **Validation (best epoch 16):** accuracy 0.9929, macro precision 0.9919, macro recall 0.9924, macro F1 0.9921, weighted F1 0.9929 (17 errors)
- **Calibration (validation, best epoch):** NLL 0.0246, ECE 0.0035
- **Stage results (best val macro F1):** A_head 0.9563, B_partial 0.9921
- **Observations:** Stronger augmentation (crop 0.5-1.0, brightness/contrast 0.3, saturation 0.2): accuracy 0.9929, macro F1 0.9921 (17 errors), NLL 0.0246, ECE 0.0035. Training accuracy at epoch 20 was lower than E004's (0.9952 versus 0.9972), with no validation gain.
- **Conclusion:** No benefit on clean validation images. It may still help robustness to field photographs, which this validation set cannot measure, so it will be revisited in the robustness tests rather than here.
- **Retained / Rejected:** Rejected for now.


### E010_mnv3_deep_seed7
- **Objective:** replicate E006 with a different training seed
- **Dataset version:** PlantVillage tomato, 15,997 clean images (clean_manifest.csv)
- **Data split:** 70/15/15, split seed 42, stratified, near-duplicate-group-aware (train 11197, val 2400); test not used
- **Preprocessing:** resize 224x224, ImageNet mean/std normalization
- **Augmentation:** preset 'default': {'crop_scale': (0.75, 1.0), 'crop_ratio': (0.9, 1.1), 'hflip_p': 0.5, 'rot90': True, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.1, 'hue': 0.0} (training only)
- **Model / architecture:** mobilenet_v3_large, torchvision IMAGENET1K_V1, 4,214,842 parameters
- **Trainable layers:** stage A head only; stage B features[7:] (blocks 7-15 + final conv) + classifier
- **Optimizer / LR:** AdamW, weight decay 0.0001, cosine schedule per stage; A_head: 5 epochs, lr 0.001; B_deep: 15 epochs, lr 0.0001
- **Batch size / epochs:** 64 / 20
- **Hyperparameters:** dropout 0.3, class weights off, training seed 7
- **Hardware:** Tesla T4 (PyTorch 2.11.0+cu130, Python 3.13.15)
- **Training time:** 22.1 min | checkpoint 17.1 MB
- **Validation (best epoch 17):** accuracy 0.9983, macro precision 0.9982, macro recall 0.9980, macro F1 0.9981, weighted F1 0.9983 (4 errors)
- **Calibration (validation, best epoch):** NLL 0.0090, ECE 0.0020
- **Stage results (best val macro F1):** A_head 0.9656, B_deep 0.9981
- **Observations:** Replicate of E006 with training seed 7: accuracy 0.9983, macro F1 0.9981 (4 errors at epoch 17); last epoch 0.9976 (5 errors). NLL 0.0090, ECE 0.0020. Errors: Early Blight to Late Blight 1, Early Blight to Target Spot 1, Yellow Leaf Curl to Bacterial Spot 1, Healthy to Early Blight 1. Against the seed-matched partial run E005: 4 vs 14 errors, macro F1 +0.0044.
- **Conclusion:** Deep unfreezing is reproduced across two seeds (4 errors at the best epoch both times), so the E006 gain is not a seed accident. MobileNetV3-Large with deep unfreezing is the strongest configuration so far.
- **Retained / Rejected:** Retained (confirms E006).


### E011_effnet_deep
- **Objective:** tuning: deeper unfreezing applied to EfficientNet-B0 vs E003
- **Dataset version:** PlantVillage tomato, 15,997 clean images (clean_manifest.csv)
- **Data split:** 70/15/15, split seed 42, stratified, near-duplicate-group-aware (train 11197, val 2400); test not used
- **Preprocessing:** resize 224x224, ImageNet mean/std normalization
- **Augmentation:** preset 'default': {'crop_scale': (0.75, 1.0), 'crop_ratio': (0.9, 1.1), 'hflip_p': 0.5, 'rot90': True, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.1, 'hue': 0.0} (training only)
- **Model / architecture:** efficientnet_b0, torchvision IMAGENET1K_V1, 4,020,358 parameters
- **Trainable layers:** stage A head only; stage B features[4:] (last four MBConv stages + final conv) + classifier
- **Optimizer / LR:** AdamW, weight decay 0.0001, cosine schedule per stage; A_head: 5 epochs, lr 0.001; B_deep: 15 epochs, lr 0.0001
- **Batch size / epochs:** 64 / 20
- **Hyperparameters:** dropout 0.3, class weights off, training seed 42
- **Hardware:** Tesla T4 (PyTorch 2.11.0+cu130, Python 3.13.15)
- **Training time:** 23.6 min | checkpoint 16.4 MB
- **Validation (best epoch 16):** accuracy 0.9979, macro precision 0.9976, macro recall 0.9976, macro F1 0.9976, weighted F1 0.9979 (5 errors)
- **Calibration (validation, best epoch):** NLL 0.0090, ECE 0.0031
- **Stage results (best val macro F1):** A_head 0.8906, B_deep 0.9976
- **Observations:** Deep unfreezing on EfficientNet-B0 (features[4:], 3.95M of 4.02M parameters trainable in Stage B): accuracy 0.9979, macro F1 0.9976 (5 errors at epoch 16) versus E003's 0.9922 (17 errors). NLL 0.0090, ECE 0.0031. The last epoch dropped to macro F1 0.9935 (16 errors; validation loss 0.0212 versus 0.0088 at epoch 19), so the best-epoch result is optimistic by several images. The cause was not investigated.
- **Conclusion:** Passes the retention rule (+0.0054 macro F1 over E003) and still holds at the last epoch (+0.0042 over E003's last epoch). Less stable at the end of training than the MobileNetV3-Large deep runs.
- **Retained / Rejected:** Retained.