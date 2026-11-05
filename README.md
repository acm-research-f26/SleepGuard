# SleepGuard

ACM Research — SleepGuard project.

## LOPO evaluation

Open `notebooks/03_lopo_evaluation.ipynb`, restart the Python kernel, and run all
cells. The notebook calls `scripts/lopo_evaluation.py`, which can also be run:

```bash
python3 scripts/lopo_evaluation.py
```

Dependencies: NumPy, pandas, scikit-learn, PyArrow, and NeuroKit2 (for raw ECG
extraction). Running the notebook also needs Jupyter/IPython. Exact versions and
input hashes are recorded with each run. Tests use Python's unittest:

```bash
python3 -m unittest discover -s tests -v
```

The prediction cutoff is **22:00 in each recording's local clock time**. Glucose
mean and slope use **20:00–22:00**. Baseline A also includes an exact prior-calendar-
night low indicator, daytime low during **10:00–22:00**, and the patient's low
rate over earlier known labeled nights. B1/B2/B3 add within-patient HRV z-scores
computed with strictly earlier nights. First-night and insufficient histories
are missing and imputed inside each fold using only training patients.

The original table's glucose means matched **22:00–00:00 for all 99 rows**. Those
values are replaced. The old nightly HRV columns also lack verified timestamp
filters and are never used as a fallback. The new extraction uses raw ECG before
22:00, with source timestamps and file hashes saved for inspection.

**Remaining limitation:** the original 99-night selection, eligibility flags,
and outcome labels are retained. The missing upstream aggregation/label notebook
still needs an audit. A shuffled-label result near chance cannot certify that an
entire research dataset is free of leakage. Results are exploratory, especially
with ten patients and fewer nights after requiring usable pre-cutoff ECG.

### Rebuild raw inputs and HRV

Raw recordings are ignored by Git. For another checkout, retrieve the CGM CSVs
and required ECG/quality recordings from the public [PhysioCGM dataset](https://doi.org/10.6084/m9.figshare.28136294):

```bash
python3 scripts/fetch_cgm.py \
  --index data/physiocgm/processed/physiocgm_download_index.json \
  --output data/physiocgm/raw
python3 scripts/build_presleep_hrv.py \
  --index data/physiocgm/processed/physiocgm_download_index.json
```

These scripts require network access and `curl`. They request individual ZIP
members by HTTP byte range and verify ZIP CRCs. Existing local raw files are
preserved. Downloaded ECG can take several GB; full ZIP archives are not needed.
Patient extraction caches under the ignored raw directory are reused when the
extraction code and requested nights match.

HRV extraction uses 250 Hz ECG with HRConfidence=100 and ECGNoise<0.001. Complete,
continuous five-minute windows are processed independently; no post-cutoff ECG
or quality sample enters any operation. Windows with more than 5% RR artifacts
are rejected. At most 5% flagged RR intervals are interpolated within that same
window. A night requires six accepted non-overlapping windows (30 minutes); the
nightly RMSSD and LF/HF are medians. Fixed rules and NeuroKit versions are saved
in `presleep_hrv_sources_20_22.json`. Outputs are:

- `data/physiocgm/processed/presleep_hrv_20_22.parquet`
- `data/physiocgm/processed/hrv_windows_20_22.parquet`
- `data/physiocgm/processed/presleep_hrv_sources_20_22.json`

The original `model_data.parquet` is read only. All variants share the same
evaluation cohort. Historical labels from excluded predictor rows may still be
used on later nights when those outcomes were known.

### Results

Each completed run creates `results/lopo/run_<UTC timestamp>/` containing:

- `REPORT.md`: results, definitions, and limitations.
- `pooled_metrics_ci.csv`: AUROC and PR-AUC with paired patient-bootstrap CIs.
- `paired_auc_differences.csv`: paired ΔAUROC and ΔPR-AUC versus A.
- `patient_metrics.csv` and `patient_macro_metrics.csv`: per-patient and macro
  metrics; single-class patients are NA and remain in pooled metrics.
- `permutation_summary.csv` and `permutation_scores.csv`: 199 global and 199
  within-patient shuffles, with full refits of primary logistic A and B3 and
  reconstruction of outcome-derived histories after every shuffle.
- OOF predictions, cohort exclusions, feature/timing audits, fold audit,
  bootstrap draws, extraction provenance, and configuration hashes.

PR-AUC means [non-interpolated average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
The 1,000 bootstrap draws resample whole patients with replacement, preserving
multiplicity and pairing across models. These fixed-prediction intervals do not
include full refitting uncertainty. Global shuffling is an IID negative control;
within-patient shuffling preserves patient burden. Neither preserves serial
dependence, and pooled LOPO null AUROC need not center exactly on 0.5.

The primary comparison is logistic B3−A. Other variants and the fixed shallow
random forest are exploratory. New runs do not produce a 0.5-threshold confusion
counts table. Previous run folders are preserved for comparison.
