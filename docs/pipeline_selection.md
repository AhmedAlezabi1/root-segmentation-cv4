# Pipeline Selection

## Comparison table

All 5 team members' Block B pipelines were compared on the following criteria:

| Metric | 236918 | 242824 | 245226 | 245769 | 244968 |
|--------|--------|--------|--------|--------|--------|
| F1-score | 0.8201 | 0.8308 | 0.85 | 0.88 | N/A |
| sMAPE | 5,773 | 6,877 | 9,359 | 5,747 | 14,564 |
| Kaggle # | 6 | 12 | 20 | 5 | 39 |
| Library | TensorFlow | TensorFlow | TensorFlow | TensorFlow | N/A |
| Model type | binary U-Net | multiclass U-Net | binary U-Net | binary U-Net | N/A |
| Model size | 355 MB | 31 MB | 22.4 MB | 104 MB | N/A |
| Weights size | 118 MB | 1 KB* | 7.40 MB | 33 MB | N/A |
| Code quality | .py files | one big .py file | .ipynb | .ipynb | N/A |

**Notes:**
- 244968's pipeline was incomplete and could not be evaluated on most criteria.
- 242824 uses a multiclass U-Net (root, shoot, seed, background) rather than binary classification. While not a dealbreaker, this adds complexity — retraining requires multi-class labeled data and additional evaluation metrics per class. Combined with its lower F1 score (0.83 vs 0.88), it was not selected.
- 245769's sMAPE score (5,747) is from an earlier version of the model. The original model was lost during server maintenance and retrained. The F1 score of 0.88 is from the retrained model. The retrained model was not resubmitted to the Kaggle competition.

## Decision

**Selected pipeline: 245769 (Abdelrahman Samara)**

**Reasoning:**
- Highest F1 score (0.88) across all candidates
- Kaggle rank 5th (best in team)
- Reasonable model size (104 MB model, 33 MB weights) — lighter than 236918 (355 MB)
- All team members use TensorFlow, so no framework conflict regardless of choice
- Full end-to-end pipeline from raw image to root tip coordinates with mm conversion
- While 236918 had the advantage of already being in .py files, refactoring notebooks into modules is part of the ILO 9.3 work and distributes coding effort across the team

**Selected model details:**
- Architecture: Attention U-Net (binary classification)
- Framework: TensorFlow/Keras
- Input: 256×256 grayscale patches (single channel)
- Parameters: ~8.9M
- Weights file: abdelrahman_245769_unet_model_256px.h5
- Custom metric: F1 score