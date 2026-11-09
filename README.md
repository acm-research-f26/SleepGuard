# SleepGuard linear logistic regression and small random forest

Only the original [model_data.parquet](https://drive.google.com/file/d/1AE1LE32x_k0TUFJElFSHIYiqMPRtOw46/view) from the shared Drive folder is read as data. The run uses the 99 eligible nights, including 31 hypoglycemic nights, from 10 patients. Outcome labels and pre-sleep features are used as provided.

Each model compares the original pre-sleep glucose mean and slope against the same features plus RMSSD, LF/HF, or both. Linear logistic regression tunes C over 0.01, 0.1, 1, and 10. Small forests use 32 trees and tune depth (2 or 3) and minimum leaf size (4 or 8). Both classifiers use balanced class weights.

For each outer patient, three patient-grouped inner splits choose settings. For the augmentation arm, the same inner splits also select between roughly one synthetic row per four eligible anchors and one per eligible anchor. Synthetic rows interpolate features between two distinct nights from the same training patient with the same outcome. Imputation and any scaling are fitted on the training split. The outer augmented score averages five fixed seeds, and the random forest's own seed is held constant. Every scored row is a real night from a held-out patient.

The summary reports pooled AUROC, average precision, the difference against the same classifier and training mode with glucose alone, and the difference against the matching real-only model. Intervals use 1,000 paired patient-block resamples of held-out predictions. Rankings among model configurations are exploratory on this cohort. The source archive did not independently verify HRV timestamp alignment and quality.

## Run

Unzip this package, open its folder in VS Code, and put the original `model_data.parquet` beside `augment_drive_only.py`.

```bash
/usr/local/bin/python3 -m pip install numpy pandas scikit-learn pyarrow
/usr/local/bin/python3 augment_drive_only.py --drive-dir . --compare-linear-forest
```

See `results/linear_forest_summary.csv` for scores and paired intervals, `linear_forest_choices.csv` for selected settings and row counts, `linear_forest_synthetic_rows.csv` for the generated outer-training rows, and `linear_forest_seed_auroc.csv` for seed sensitivity. Synthetic row IDs record parent nights, budgets, and outer held-out patients. Omitting the comparison flag runs the prior Drive-only logistic experiment.

## Main results

| Classifier and training data | Glucose only AUROC | Glucose plus RMSSD AUROC |
| --- | ---: | ---: |
| Linear logistic regression, real only | 0.6176 | 0.5859 |
| Linear logistic regression, augmented | 0.6281 | 0.5982 |
| Small forest, real only | 0.5927 | 0.5674 |
| Small forest, augmented | 0.5674 | 0.5769 |

The highest pooled AUROC among these 16 runs was augmented glucose-only logistic regression (0.6281). Its gain over real-only training was +0.0104 (paired 95% interval −0.0244 to +0.0551). The highest HRV score was 0.5982 from the augmented logistic model with RMSSD. Against augmented glucose-only logistic regression, the difference was −0.0299 (−0.0883 to +0.0397). The augmented forest gained +0.0095 from adding RMSSD (−0.0581 to +0.0852).
