# Project Context

**Project:** Tomato Leaf Disease Detection and Treatment Recommendation Using Deep Learning
**Current phase:** Phase 17 (Supported-scope validation)
**Completed phases:** Phases 1 to 16
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


E001 baseline custom CNN: val macro F1 0.9937, accuracy 0.9942, 2.35M params, 40.9 min on T4. Test set untouched.


Validation macro F1: E001 custom CNN 0.9937 (14 errors), E002 ResNet-18 0.9868 (28), E003 EfficientNet-B0 0.9922 (17), E004 MobileNetV3-Large 0.9935 (14). Stage B fine-tuning gave +0.03 to +0.15 macro F1 over head-only. Validation is saturated, so differences among E001/E003/E004 are within noise. Early Blight is the hardest class for all models. Test set untouched

Tuning result: deep unfreezing of Stage B is the retained change (MobileNetV3-L E006/E010: 4 errors, val macro F1 0.998; EfficientNet-B0 E011: 5 errors, 0.9976). Replicated across seeds. E004 and E005 (same settings, different seeds) share 11 of 14 errors, so remaining errors are mostly a fixed set of hard images. Best-epoch scores are optimistic; test set untouched

Phase 11 so far: finalists are the deep-unfreezing models (E006, E010 MobileNetV3-Large; E011 EfficientNet-B0). 14 validation images are misclassified by at least 3 of 10 runs (hard-image set). E010 chosen over E006 as the checkpoint by the preset lower-NLL rule, pending the CPU speed and unknown-image checks.

Final model: E010 (MobileNetV3-Large, deep unfreezing, seed 7): validation 4 errors/2,400, macro F1 0.9981, NLL 0.0090, ECE 0.0020, 17.1 MB, 25.7 ms per image on laptop CPU (batch 1). E011 (EfficientNet-B0) is 44.0 ms. Saved as models/best_model.pth with model_info.json; backups on Drive.


Phase 12 test (E010, once): accuracy 0.9938 (15/2,400 errors; 95% CI 0.9897-0.9965), macro F1 0.9928 (CI 0.9885-0.9964), weighted F1 0.9937, ROC-AUC 0.99998, NLL 0.0155, ECE 0.0032, Brier 0.0093; mean confidence correct 0.997 vs wrong 0.791. Test is lower than validation (4 errors), consistent with selection optimism. Weakest class: Early Blight (recall 0.973). All 21 dark test images (Late Blight) correct. Not evidence of field performance.


Phase 13: all 15 test errors have confidence below 0.95 (44/2,400 test images are below 0.95); 8 of 15 errors show mild or no visible symptoms, 4 are large dead-tissue lesions confusable across Early/Late Blight, 1 many small spots, 2 unusual framing/shape. Early Blight is the hardest class (2.67% error); all Target Spot errors go to Spider Mites. Errors lean toward softer images (median sharpness percentile 23), weak evidence. Confidence catches in-domain errors but not Pepper/Potato leaves.

Own evaluation photos: data/external/my_photos (tomato_leaf 16, other_leaf 16, non_leaf 15, poor_quality 10; mostly web/WhatsApp images; git-ignored; evaluation only).

Phase 14 (so far): kNN cosine selected for 'unsupported' (dev false acceptance 19.7% at 95% known acceptance; max softmax 28.8%, max logit 32.6%, energy 33.2%, Mahalanobis 44.7%). Worst unknown: Potato Late Blight (52% accepted by kNN). Own photos: kNN rejected 41/41 non-tomato or poor-quality images but also 15/16 tomato photos from field-style images (domain gap).


Quality checks: min side 160 px, sharpness 29.25, too bright p5 > 117, too dark p95 < 72.12 (black-background guard), blank = half the lowest training contrast. Valid validation images flagged about 1.7% before the blank fix. Degradation experiment: model accuracy 99.7% clean, 94.6% at mild blur (sigma 1), 56% at sigma 2; 96.5% at 128 px; 88.6% at x2 overexposure. The model is fragile to blur; mild damage is covered by the gate, severe by the quality check.