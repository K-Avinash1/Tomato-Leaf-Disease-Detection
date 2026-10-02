# Decision Log

| # | Decision | Alternatives | Selected | Reason | Evidence | Consequences | Date |
|---|----------|--------------|----------|--------|----------|--------------|------|
| 1 | Deep-learning framework | TensorFlow/Keras | PyTorch | Strong torchvision pretrained models, easy Grad-CAM | Project spec | Code is PyTorch-based | 2026-10-02 |
| 2 | Development environment | Google Colab | Local VS Code + venv | Reproducible, per project spec | Project spec, Phase 1 | Training runs on CPU | 2026-10-02 |
| 3 | Class taxonomy | Add Fusarium Wilt or extra classes | 10 locked classes | Locked by project spec | Project spec | No new classes without approval | 2026-10-02 |
| 4 | Dataset location | Copy into project folder | Keep at `E:\PlantVillage` | Saves disk space, avoids duplicates | Local setup | Code points to this path via config | 2026-10-02 |
   | 5 | Which folders to use | Outer folders, nested `PlantVillage` copy, or both | Outer folders only | Nested copy has identical file names and counts in all 10 classes, so it looks like a duplicate | Phase 2 verification output; hash check pending | Prevents double counting and train/test leakage | 2026-10-02 |