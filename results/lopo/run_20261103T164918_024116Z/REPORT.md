# SleepGuard LOPO rerun

**Exploratory: glucose and HRV were rebuilt before 22:00. Stored outcome construction and the original cohort selection still need an upstream audit; a permutation test cannot certify absence of leakage.**

Evaluated 62 nights from 10 patients, including 22 low nights (prevalence 0.355).

99/99 stored glucose means matched 22:00–00:00 CGM. They were replaced with glucose mean and OLS slope from 20:00–22:00, before the user-confirmed 22:00 cutoff. HRV was re-extracted from raw ECG in the same pre-cutoff window. A total of 37 nights were excluded consistently across variants for missing eligible predictors. Existing outcome labels and original eligibility flags were retained; their construction must be audited before using this as a validated 22:00 prediction experiment.

Rebuilt HRV uses 250 Hz ECG, HRConfidence=100 and ECGNoise<0.001, continuous five-minute windows, NeuroKit cleaning/R-peak detection, and Welch LF/HF. Windows with more than 5% implausible/deviant RR intervals are rejected; at most 5% are interpolated within that pre-cutoff window. A night requires six accepted non-overlapping windows (30 minutes); nightly values are their medians. These fixed quality rules are recorded in the extraction manifest. No legacy HRV value is used as a fallback.

A uses corrected pre-cutoff glucose mean/slope, previous-calendar-night low, 10:00–22:00 daytime low, and the patient's low rate over earlier known labeled nights in the supplied table. B1 adds within-patient RMSSD z-score; B2 adds LF/HF z-score; B3 adds both. History is shifted before computing features. No future/current label enters its own predictors. First-night history and HRV histories with fewer than two prior measurements or zero SD are missing, then imputed using training patients only. This evaluates sequential use of an unseen patient's accumulating history, not a completely history-free new patient.

The daytime feature requires at least 80% occupied five-minute CGM bins and indicates any reading below 70 mg/dL; it is not a duration-defined episode. Missing daytime coverage is not coded as absence of a low. Ambiguous duplicate CGM timestamps are excluded. The five incomplete outcome windows in the original input are positive; an observed positive can be known despite missing later measurements.

| Model | Variant | Pooled AUROC (95% CI) | PR-AUC / AP (95% CI) |
|---|---|---|---|
| logistic | A | 0.495 (0.337, 0.642) | 0.337 (0.228, 0.522) |
| logistic | B1 | 0.474 (0.292, 0.640) | 0.325 (0.216, 0.521) |
| logistic | B2 | 0.478 (0.314, 0.641) | 0.329 (0.220, 0.525) |
| logistic | B3 | 0.457 (0.281, 0.616) | 0.319 (0.212, 0.510) |
| random_forest | A | 0.458 (0.348, 0.604) | 0.334 (0.193, 0.547) |
| random_forest | B1 | 0.443 (0.327, 0.592) | 0.321 (0.184, 0.527) |
| random_forest | B2 | 0.450 (0.334, 0.585) | 0.328 (0.192, 0.536) |
| random_forest | B3 | 0.416 (0.305, 0.563) | 0.317 (0.183, 0.511) |

Primary comparison: logistic B3−A. Other variants and the shallow random forest are exploratory. Settings were held fixed from the previous notebook.

| Model | Comparison | ΔAUROC (95% CI) | ΔPR-AUC / AP (95% CI) |
|---|---|---|---|
| logistic | B1-A | -0.022 (-0.097, +0.023) | -0.011 (-0.044, +0.017) |
| logistic | B2-A | -0.017 (-0.052, +0.015) | -0.007 (-0.024, +0.019) |
| logistic | B3-A | -0.039 (-0.112, +0.007) | -0.017 (-0.052, +0.009) |
| random_forest | B1-A | -0.015 (-0.064, +0.019) | -0.013 (-0.054, +0.008) |
| random_forest | B2-A | -0.008 (-0.043, +0.020) | -0.005 (-0.040, +0.017) |
| random_forest | B3-A | -0.042 (-0.094, -0.001) | -0.016 (-0.079, +0.017) |

Intervals use 1,000 paired patient-cluster bootstrap replicates of fixed out-of-fold predictions. Every selected patient's nights are included with multiplicity, with identical draws for every model. They do not capture full refitting/model-selection uncertainty; ten clusters give imprecise inference.

Per-patient values for every model are in `patient_metrics.csv`; single-class patients receive NA for both ranking metrics and remain in pooled results. Patient macro averages are in `patient_macro_metrics.csv`.

| Patient | Nights | Low nights | Logistic A AUROC | Logistic B3 AUROC | ΔAUROC | A PR-AUC / AP | B3 PR-AUC / AP |
|---|---|---|---|---|---|---|---|
| c1s01 | 4 | 2 | 0.000 | 0.000 | 0.000 | 0.417 | 0.417 |
| c1s02 | 9 | 0 | NA | NA | NA | NA | NA |
| c1s03 | 7 | 1 | 0.833 | 0.833 | 0.000 | 0.500 | 0.500 |
| c1s04 | 5 | 1 | 0.250 | 0.000 | -0.250 | 0.250 | 0.200 |
| c1s05 | 9 | 4 | 0.300 | 0.350 | 0.050 | 0.430 | 0.443 |
| c2s01 | 11 | 5 | 0.133 | 0.067 | -0.067 | 0.345 | 0.328 |
| c2s02 | 3 | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| c2s03 | 5 | 1 | 0.250 | 0.000 | -0.250 | 0.250 | 0.200 |
| c2s04 | 6 | 3 | 0.667 | 0.556 | -0.111 | 0.833 | 0.722 |
| c2s05 | 3 | 3 | NA | NA | NA | NA | NA |

Permutation diagnostics refit primary logistic A and B3 for each of 199 global and 199 within-patient label shuffles. Outcome-derived history is rebuilt after every shuffle. Global shuffling is an IID negative control; within-patient shuffling preserves patient low counts. Neither preserves serial dependence. Their null means need not be exactly 0.5, especially for pooled LOPO scores with fold-dependent probabilities and burden-preserving shuffles. A chance-like null cannot rule out upstream post-cutoff feature leakage.

| Shuffle | Metric | Observed | Null mean | Null central 95% | Upper-tail p |
|---|---|---|---|---|---|
| global | pooled_auroc_A | 0.495 | 0.454 | (0.258, 0.646) | 0.315 |
| global | pooled_auroc_B3 | 0.457 | 0.465 | (0.220, 0.681) | 0.560 |
| global | pooled_delta_auroc | -0.039 | 0.011 | (-0.091, 0.173) | 0.780 |
| global | macro_auroc_A | 0.429 | 0.520 | (0.304, 0.765) | 0.775 |
| global | macro_auroc_B3 | 0.351 | 0.517 | (0.255, 0.774) | 0.910 |
| global | macro_delta_auroc | -0.078 | -0.003 | (-0.141, 0.204) | 0.820 |
| within_patient | pooled_auroc_A | 0.495 | 0.501 | (0.293, 0.696) | 0.550 |
| within_patient | pooled_auroc_B3 | 0.457 | 0.493 | (0.304, 0.690) | 0.625 |
| within_patient | pooled_delta_auroc | -0.039 | -0.009 | (-0.091, 0.120) | 0.690 |
| within_patient | macro_auroc_A | 0.429 | 0.457 | (0.177, 0.758) | 0.600 |
| within_patient | macro_auroc_B3 | 0.351 | 0.461 | (0.160, 0.756) | 0.795 |
| within_patient | macro_delta_auroc | -0.078 | 0.004 | (-0.123, 0.204) | 0.850 |

PR-AUC here is non-interpolated average precision, with prevalence provided as its reference level. [Scikit-learn metric definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). Raw CGM comes from the [public PhysioCGM dataset](https://doi.org/10.6084/m9.figshare.28136294); source hashes are recorded in `run_config.json`.

Audits: `glucose_timing_audit.csv`, `hrv_timing_audit.csv`, `daytime_feature_audit.csv`, `feature_audit.csv`, `fold_audit.csv`, and `excluded_nights.csv`. There is no 0.5-threshold confusion/count table in this run. Historical run folders were preserved.
