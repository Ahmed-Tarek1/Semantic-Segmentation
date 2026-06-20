# U-Net Skip Connection Ablation — Football Semantic Segmentation

Ablation study on a 3-level U-Net, testing how each skip connection level affects segmentation quality on broadcast football footage (COCO-format dataset).

## Setup

- **Dataset:** [Football Semantic Segmentation](https://www.kaggle.com/datasets/sadhliroomyprime/football-semantic-segmentation) (COCO format), 100 images, 512×512
- **Split:** 80 train / 10 val / 10 test
- **Model:** 3-level U-Net (64→128→256ch), skip connections toggleable per level
- **Metrics:** IoU (Jaccard), Dice coefficient
- **Training:** 50 epochs, Adam, CosineAnnealingLR, CrossEntropyLoss

## Experiments

| Exp | Skip connections active |
|-----|--------------------------|
| 1   | Level 1 only (shallowest — 64ch, 512×512) |
| 2   | Level 2 only (middle — 128ch, 256×256) |
| 3   | Level 3 only (deepest — 256ch, 64×64) |
| 4   | All levels (standard U-Net) |

## Results

| Experiment | L1 | L2 | L3 | Test IoU | Test Dice |
|------------|:--:|:--:|:--:|:--------:|:---------:|
| Exp 1 — L1 only   | ✔ | ✘ | ✘ | 0.428 | 0.501 |
| **Exp 2 — L2 only** | ✘ | ✔ | ✘ | **0.460** | **0.523** |
| Exp 3 — L3 only   | ✘ | ✘ | ✔ | 0.355 | 0.396 |
| Exp 4 — All skips | ✔ | ✔ | ✔ | 0.441 | 0.508 |

![Test IoU & Dice](bar_chart.png)
![Validation curves](training_curves.png)
![Qualitative predictions](predictions.png)

## Conclusion

Results don't match standard U-Net theory: **Exp 2 (L2-only) outperforms the full model** on both IoU and Dice, while Exp 4 (all skips) comes second. Exp 1 (L1-only) is close behind Exp 4. Exp 3 (L3-only) is worst across the board.

Likely cause: very small test set (10 images) and a single run per config — differences may partly reflect noise rather than a true effect of skip connections. Future work should average over multiple seeds and a larger held-out set before drawing firm conclusions.

For this dataset, **Exp 2 would be the better choice to deploy**, not the standard full-skip U-Net.

## Files

- `semantic-segmentation.ipynb` — full notebook (data loading, model, training, evaluation)
- `ablation_results.csv` — results table
- `training_curves.png`, `bar_chart.png`, `predictions.png` — output figures
