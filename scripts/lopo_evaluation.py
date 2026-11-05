"""Chronological features and reproducible leave-one-patient-out evaluation.

The held-out patient's *earlier* outcomes and HRV are available as history;
no held-out row is ever used to fit a classifier, imputer, or scaler.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RAW_FEATURES = ["mean_glucose_presleep", "glucose_slope_presleep",
                "rmssd_presleep", "lfhf_presleep"]
GLUCOSE_FEATURES = RAW_FEATURES[:2] + ["prior_night_low", "daytime_low", "earlier_night_low_rate"]
FEATURE_SETS = {
    "A": GLUCOSE_FEATURES,
    "B1": GLUCOSE_FEATURES + ["rmssd_presleep_z"],
    "B2": GLUCOSE_FEATURES + ["lfhf_presleep_z"],
    "B3": GLUCOSE_FEATURES + ["rmssd_presleep_z", "lfhf_presleep_z"],
}
RENAME = {"participant_id": "subject", "night": "night_id",
          "hypo": "hypo_night", "lf_hf_presleep": "lfhf_presleep"}


@dataclass(frozen=True)
class Config:
    prediction_hour: int = 22
    daytime_start_hour: int = 10
    presleep_glucose_hours: int = 2
    low_mg_dl: float = 70.0
    cgm_interval_minutes: int = 5
    min_daytime_coverage: float = 0.80
    min_hrv_history: int = 2
    n_bootstraps: int = 1000
    bootstrap_seed: int = 20261026
    n_permutations: int = 199
    permutation_seed: int = 20261102


def classifiers():
    # Prespecified settings from the previous run; no tuning on test patients.
    return {
        "logistic": LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=3, min_samples_leaf=10,
            random_state=42, n_jobs=1),
    }


def read_model_table(path):
    raw = pd.read_parquet(path).rename(columns=RENAME)
    required = ["subject", "night_id", "hypo_night", "Model_Eligible", "outcome_complete"] + RAW_FEATURES
    missing = set(required) - set(raw.columns)
    if missing or raw.columns.duplicated().any():
        raise ValueError(f"Missing or duplicated input columns: {sorted(missing)}")
    if raw[["subject", "night_id"]].isna().any().any():
        raise ValueError("Missing patient/night key")
    raw["subject"] = raw.subject.astype(str).str.strip()
    raw["night_id"] = pd.to_datetime(raw.night_id, errors="raise")
    if raw.night_id.dt.tz is not None or not raw.night_id.eq(raw.night_id.dt.normalize()).all():
        raise ValueError("night_id must be a timezone-naive local calendar date")
    if raw.subject.eq("").any() or raw.duplicated(["subject", "night_id"]).any():
        raise ValueError("Empty patient or duplicate patient/night key")
    for column in ("Model_Eligible", "outcome_complete"):
        if not raw[column].dropna().isin([True, False]).all():
            raise ValueError(f"{column} must be boolean")
    raw["hypo_night"] = pd.to_numeric(raw.hypo_night, errors="raise").astype(float)
    if not raw.hypo_night.dropna().isin([0, 1]).all():
        raise ValueError("Outcome must be binary")
    eligible = raw.Model_Eligible.eq(True).fillna(False)
    if raw.loc[eligible, "hypo_night"].isna().any():
        raise ValueError("An eligible night has no label")
    # A detected positive is known even if later outcome samples are missing;
    # a negative needs the full outcome window. This explains the five
    # incomplete positive nights in the supplied table without changing labels.
    label_known = raw.outcome_complete.eq(True).fillna(False) | raw.hypo_night.eq(1)
    if not label_known.loc[eligible].all():
        raise ValueError("An eligible negative night has an incomplete outcome")
    raw["history_label_known"] = label_known
    for col in RAW_FEATURES:
        raw[col] = pd.to_numeric(raw[col], errors="raise").replace([np.inf, -np.inf], np.nan)
    return raw.sort_values(["subject", "night_id"]).reset_index(drop=True)


def read_cgm(path):
    """Read Dexcom EGV measurements only; alerts/calibrations are not readings."""
    data = pd.read_csv(path, low_memory=False)
    time_col = "Timestamp (YYYY-MM-DDThh:mm:ss)"
    glucose_col = "Glucose Value (mg/dL)"
    if not {time_col, glucose_col, "Event Type"}.issubset(data.columns):
        raise ValueError(f"Unrecognized CGM schema: {path}")
    data = data.loc[data["Event Type"].eq("EGV"), [time_col, glucose_col]].copy()
    # Dexcom Low/High are censored readings, not missing data. Their limits
    # retain the correct side of the fixed 70 mg/dL feature threshold.
    data[glucose_col] = data[glucose_col].replace({"Low": 39, "LOW": 39, "High": 401, "HIGH": 401})
    result = pd.DataFrame({
        "timestamp": pd.to_datetime(data[time_col], format="mixed", errors="raise"),
        "glucose": pd.to_numeric(data[glucose_col], errors="raise"),
    }).dropna()
    if result.timestamp.dt.tz is not None:
        raise ValueError("Expected local wall-clock CGM times; do not silently convert timezones")
    if result.glucose.le(0).any():
        raise ValueError("Nonpositive CGM readings")
    conflicts = result.groupby("timestamp").glucose.nunique().gt(1)
    # Ambiguous readings (including repeated wall-clock times around DST) are
    # excluded, not averaged or resolved using any later glucose information.
    result = result.loc[~result.timestamp.isin(conflicts.index[conflicts])]
    result = result.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    result.attrs["conflicting_timestamps_excluded"] = int(conflicts.sum())
    return result


def daytime_features(raw, cgm_by_patient, config=Config()):
    """All windows are [10:00, 22:00); missing coverage never means no low."""
    if not 0 <= config.daytime_start_hour < config.prediction_hour < 24:
        raise ValueError("Invalid daytime/prediction window")
    if not 0 < config.min_daytime_coverage <= 1:
        raise ValueError("Invalid minimum CGM coverage")
    rows = []
    for row in raw.itertuples():
        cutoff = row.night_id + pd.Timedelta(config.prediction_hour, unit="h")
        start = row.night_id + pd.Timedelta(config.daytime_start_hour, unit="h")
        cgm = cgm_by_patient.get(row.subject)
        coverage, low, latest, n = 0.0, np.nan, pd.NaT, 0
        status = "missing_cgm_file"
        if cgm is not None:
            window = cgm.loc[cgm.timestamp.ge(start) & cgm.timestamp.lt(cutoff)]
            n = len(window)
            # Count occupied five-minute bins, so duplicate/dense readings
            # cannot inflate coverage. Do not interpolate across missing gaps.
            bins = ((window.timestamp - start).dt.total_seconds()
                    // (config.cgm_interval_minutes * 60)).nunique()
            expected = (cutoff - start).total_seconds() / (config.cgm_interval_minutes * 60)
            coverage = bins / expected
            latest = window.timestamp.max()
            status = "insufficient_coverage"
            if coverage >= config.min_daytime_coverage:
                low = float(window.glucose.lt(config.low_mg_dl).any())
                status = "ok"
        rows.append({"subject": row.subject, "night_id": row.night_id,
                     "prediction_time": cutoff, "daytime_start": start,
                     "daytime_low": low, "daytime_coverage": coverage,
                     "daytime_readings": n, "daytime_last_reading": latest,
                     "daytime_status": status})
    return pd.DataFrame(rows)


def rebuild_presleep_glucose(raw, cgm_by_patient, config=Config()):
    """Replace stored post-cutoff glucose features with [20:00,22:00) CGM.

    Keep an explicit audit of the original values. Source-label definitions and
    HRV timestamps are unavailable in the supplied merged table; this does not
    reconstruct or certify those upstream inputs.
    """
    result = raw.copy()
    result[RAW_FEATURES[:2]] = result[RAW_FEATURES[:2]].astype(float)
    rows = []
    for idx, row in raw.iterrows():
        cutoff = row.night_id + pd.Timedelta(config.prediction_hour, unit="h")
        start = cutoff - pd.Timedelta(config.presleep_glucose_hours, unit="h")
        cgm = cgm_by_patient[row.subject]
        pre = cgm.loc[cgm.timestamp.ge(start) & cgm.timestamp.lt(cutoff)]
        post = cgm.loc[cgm.timestamp.ge(cutoff) & cgm.timestamp.lt(
            cutoff + pd.Timedelta(config.presleep_glucose_hours, unit="h"))]
        bins = ((pre.timestamp - start).dt.total_seconds()
                // (config.cgm_interval_minutes * 60)).nunique()
        coverage = bins / (config.presleep_glucose_hours * 60 / config.cgm_interval_minutes)
        mean, slope = np.nan, np.nan
        if coverage >= config.min_daytime_coverage and len(pre) >= 2:
            mean = pre.glucose.mean()
            hours = (pre.timestamp - start).dt.total_seconds().to_numpy() / 3600
            slope = np.polyfit(hours, pre.glucose.to_numpy(), 1)[0]
        result.loc[idx, "mean_glucose_presleep"] = mean
        result.loc[idx, "glucose_slope_presleep"] = slope
        rows.append({"subject": row.subject, "night_id": row.night_id,
                     "prediction_time": cutoff, "glucose_window_start": start,
                     "glucose_last_reading": pre.timestamp.max(), "glucose_coverage": coverage,
                     "stored_mean_glucose": row.mean_glucose_presleep,
                     "stored_glucose_slope": row.glucose_slope_presleep,
                     "postcutoff_mean_glucose": post.glucose.mean(),
                     "stored_mean_matches_postcutoff": bool(np.isclose(
                         row.mean_glucose_presleep, post.glucose.mean(), rtol=0, atol=1e-6)),
                     "mean_glucose_presleep": mean, "glucose_slope_presleep": slope,
                     "status": "ok" if np.isfinite(mean) else "insufficient_presleep_coverage"})
    return result, pd.DataFrame(rows)


def load_rebuilt_hrv(raw, path, config=Config()):
    """Require timestamp-audited ECG features instead of unverified old HRV."""
    hrv = pd.read_parquet(path)
    keys = ["subject", "night_id"]
    required = keys + RAW_FEATURES[2:] + ["hrv_source_start", "hrv_source_end", "hrv_last_sample",
                                          "hrv_last_quality", "pre_cutoff_hrv_windows", "hrv_status"]
    if set(required) - set(hrv.columns):
        raise ValueError("Rebuilt HRV table lacks timestamp/quality provenance")
    if hrv.duplicated(keys).any():
        raise ValueError("Duplicate rebuilt HRV patient/night")
    known = hrv[RAW_FEATURES[2:]].notna().any(axis=1)
    cutoff = hrv.night_id + pd.Timedelta(config.prediction_hour, unit="h")
    start = cutoff - pd.Timedelta(config.presleep_glucose_hours, unit="h")
    valid = (hrv.hrv_source_start.ge(start) & hrv.hrv_source_end.le(cutoff)
             & hrv.hrv_last_sample.lt(cutoff) & hrv.hrv_last_quality.lt(cutoff)
             & hrv.pre_cutoff_hrv_windows.ge(6) & hrv.hrv_status.eq("ok"))
    if not valid.loc[known].all():
        raise ValueError("HRV features fail pre-cutoff timestamp or coverage checks")
    # All missing nights remain explicit; no fallback to the legacy HRV values.
    result = raw.drop(columns=RAW_FEATURES[2:]).merge(hrv, on=keys, how="left", validate="one_to_one")
    if result.hrv_status.isna().any():
        raise ValueError("Rebuilt HRV file does not cover all supplied patient/night keys")
    return result, hrv


def add_history(raw, daytime, config=Config()):
    """Rebuilt for EVERY label permutation, including the held-out history."""
    if config.min_hrv_history < 2:
        raise ValueError("A sample standard deviation needs at least two earlier nights")
    df = raw.sort_values(["subject", "night_id"]).reset_index(drop=True).copy()
    if df.duplicated(["subject", "night_id"]).any():
        raise ValueError("Duplicate patient/night key")
    df = df.merge(daytime, on=["subject", "night_id"], validate="one_to_one", how="left")
    if df.prediction_time.isna().any():
        raise ValueError("Daytime audit missing a patient/night")
    valid_labels = df.hypo_night.where(df.history_label_known.eq(True))
    df["history_label"] = valid_labels
    groups = df.groupby("subject", sort=False)
    df["earlier_night_count"] = groups.history_label.transform(
        lambda s: s.expanding().count().shift(1, fill_value=0)).astype(int)
    df["earlier_night_low_rate"] = groups.history_label.transform(
        lambda s: s.expanding().mean().shift(1))
    # Exact preceding calendar night: shift alone would incorrectly bridge gaps.
    lookup = df.set_index(["subject", "night_id"])["history_label"]
    keys = pd.MultiIndex.from_arrays([df.subject, df.night_id - pd.Timedelta(1, unit="D")])
    df["prior_night_low"] = lookup.reindex(keys).to_numpy()
    prior_date = df.night_id - pd.Timedelta(1, unit="D")
    df["prior_night_source"] = prior_date.where(df.prior_night_low.notna())
    df["latest_label_history_night"] = groups.night_id.transform(
        lambda s: s.where(valid_labels.loc[s.index].notna()).ffill().shift(1))
    for feature in RAW_FEATURES[2:]:
        history = groups[feature]
        count = history.transform(lambda s: s.expanding().count().shift(1, fill_value=0))
        mean = history.transform(lambda s: s.expanding().mean().shift(1))
        std = history.transform(lambda s: s.expanding().std(ddof=1).shift(1))
        usable = count.ge(config.min_hrv_history) & std.gt(0)
        df[f"{feature}_history_n"] = count.astype(int)
        df[f"{feature}_history_mean"] = mean
        df[f"{feature}_history_std"] = std
        df[f"{feature}_z"] = ((df[feature] - mean) / std).where(usable)
        df[f"{feature}_history_last_night"] = groups.night_id.transform(
            lambda s: s.where(df.loc[s.index, feature].notna()).ffill().shift(1))
    return df.drop(columns="history_label")


def prepare_cohort(feature_table):
    # Keep the SAME observed raw-feature cohort across all variants. New
    # history missingness is expected and handled inside each training fold.
    df = feature_table.copy()
    eligible = df.Model_Eligible.eq(True).fillna(False)
    observed = df[RAW_FEATURES].notna().all(axis=1)
    keep = eligible & observed
    excluded = df.loc[~keep, ["subject", "night_id", "Model_Eligible"]].copy()
    excluded["reason"] = [
        "upstream_ineligible" if not eligible.loc[idx] else ";".join(
            reason for features, reason in ((RAW_FEATURES[:2], "insufficient_presleep_glucose"),
                                            (RAW_FEATURES[2:], "insufficient_presleep_hrv"))
            if df.loc[idx, features].isna().any())
        for idx in excluded.index]
    excluded["missing_features"] = [",".join(df.loc[idx, RAW_FEATURES].index[
        df.loc[idx, RAW_FEATURES].isna()]) for idx in excluded.index]
    cohort = df.loc[keep].reset_index(drop=True)
    cohort["hypo_night"] = cohort.hypo_night.astype(int)
    if cohort.subject.nunique() < 3 or cohort.hypo_night.nunique() != 2:
        raise ValueError("Insufficient patients or classes")
    counts = df.groupby("subject").agg(input_nights=("night_id", "size"))
    retained = cohort.groupby("subject").agg(included_nights=("night_id", "size"),
                                             low_nights=("hypo_night", "sum"))
    counts = counts.join(retained).fillna(0).astype(int).reset_index()
    return cohort, excluded, counts


def generate_oof(df, feature_sets=None, estimators=None):
    feature_sets = FEATURE_SETS if feature_sets is None else feature_sets
    estimators = classifiers() if estimators is None else estimators
    y = df.hypo_night.to_numpy()
    oof = df[["subject", "night_id", "hypo_night"]].copy()
    columns = [(name, variant, f"{name}__{variant}")
               for name in estimators for variant in feature_sets]
    for _, _, col in columns:
        oof[col] = np.nan
    audits, visits = [], np.zeros(len(df), dtype=int)
    for train, test in LeaveOneGroupOut().split(df, y, df.subject):
        held_out = df.iloc[test].subject.iloc[0]
        if held_out in set(df.iloc[train].subject):
            raise AssertionError("Patient overlap")
        if np.unique(y[train]).size != 2:
            raise ValueError(f"Single-class training fold for {held_out}")
        for name, variant, col in columns:
            features = feature_sets[variant]
            # Imputer/scaler statistics come ONLY from training patients.
            # An entirely missing training feature receives neutral zero.
            model = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                StandardScaler(), clone(estimators[name]))
            model.fit(df.iloc[train][features], y[train])
            positive = int(np.flatnonzero(model.classes_ == 1)[0])
            oof.loc[test, col] = model.predict_proba(df.iloc[test][features])[:, positive]
        visits[test] += 1
        audits.append({"held_out_subject": held_out,
                       "train_subjects": ",".join(sorted(df.iloc[train].subject.unique())),
                       "train_nights": len(train), "test_nights": len(test),
                       "train_low_nights": int(y[train].sum()), "test_low_nights": int(y[test].sum())})
    scores = oof[[col for _, _, col in columns]].to_numpy()
    if not np.all(visits == 1) or not (np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()):
        raise AssertionError("Invalid OOF predictions")
    return oof, pd.DataFrame(audits), columns


def metric_pair(y, scores):
    # PR-AUC is non-interpolated average precision throughout this project.
    if len(np.unique(y)) < 2:
        return np.array([np.nan, np.nan])
    return np.array([roc_auc_score(y, scores), average_precision_score(y, scores)])


def paired_patient_bootstrap(oof, columns, n_bootstraps=1000, seed=42):
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be positive")
    y = oof.hypo_night.to_numpy()
    scores = oof[[col for _, _, col in columns]].to_numpy()
    subjects = oof.subject.to_numpy()
    unique = np.unique(subjects)
    indices = {p: np.flatnonzero(subjects == p) for p in unique}
    point = np.array([metric_pair(y, scores[:, j]) for j in range(len(columns))])
    rng = np.random.default_rng(seed)
    rows, draws, skipped, attempts = [], [], 0, 0
    while len(rows) < n_bootstraps and attempts < n_bootstraps * 20:
        attempts += 1
        sample = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([indices[p] for p in sample])  # retain repeated clusters
        if np.unique(y[idx]).size < 2:
            skipped += 1
            continue
        rows.append([metric_pair(y[idx], scores[idx, j]) for j in range(len(columns))])
        draws.append({"replicate": len(rows), "sampled_subjects": ",".join(sample), "sampled_nights": len(idx)})
    if len(rows) != n_bootstraps:
        raise ValueError("Too few valid bootstrap samples")
    boot = np.asarray(rows)
    bounds = np.quantile(boot, [0.025, 0.975], axis=0)
    positions = {(name, variant): j for j, (name, variant, _) in enumerate(columns)}
    summary, differences, boot_rows = [], [], []
    for j, (name, variant, col) in enumerate(columns):
        row = {"algorithm": name, "feature_set": variant, "n_patients": len(unique),
               "n_nights": len(y), "n_low_nights": int(y.sum()), "prevalence": y.mean(),
               "valid_bootstraps": len(rows), "skipped_single_class": skipped}
        for k, metric in enumerate(("auroc", "pr_auc")):
            row[f"pooled_{metric}"] = point[j, k]
            row[f"{metric}_ci_lower"] = bounds[0, j, k]
            row[f"{metric}_ci_upper"] = bounds[1, j, k]
        summary.append(row)
        if variant != "A":
            a = positions[(name, "A")]
            delta = {"algorithm": name, "comparison": f"{variant}-A", "valid_bootstraps": len(rows)}
            for k, metric in enumerate(("auroc", "pr_auc")):
                lo, hi = np.quantile(boot[:, j, k] - boot[:, a, k], [0.025, 0.975])
                delta.update({f"delta_{metric}": point[j, k] - point[a, k],
                              f"{metric}_ci_lower": lo, f"{metric}_ci_upper": hi})
            differences.append(delta)
        for b, values in enumerate(boot[:, j, :], 1):
            boot_rows.append({"replicate": b, "algorithm": name, "feature_set": variant,
                              "auroc": values[0], "pr_auc": values[1]})
    return pd.DataFrame(summary), pd.DataFrame(differences), pd.DataFrame(boot_rows), pd.DataFrame(draws)


def per_patient_metrics(oof, columns):
    rows = []
    for subject, patient in oof.groupby("subject", sort=True):
        y = patient.hypo_night.to_numpy()
        values = {(name, variant): metric_pair(y, patient[col]) for name, variant, col in columns}
        for name, variant, _ in columns:
            auroc, pr_auc = values[(name, variant)]
            baseline = values[(name, "A")]
            rows.append({"subject": subject, "algorithm": name, "feature_set": variant,
                         "n_nights": len(y), "n_low_nights": int(y.sum()), "prevalence": y.mean(),
                         "status": "ok" if np.unique(y).size == 2 else "undefined_single_class",
                         "auroc": auroc, "pr_auc": pr_auc,
                         "delta_auroc_vs_A": auroc - baseline[0], "delta_pr_auc_vs_A": pr_auc - baseline[1]})
    return pd.DataFrame(rows)


def permutation_test(raw, daytime, observed_oof, config=Config(), progress=print):
    """Refit primary logistic A/B3 after each shuffle; rebuild label history.

    Global shuffling is an IID negative control. Within-patient shuffling is a
    second, restricted null preserving each patient's burden. Neither preserves
    serial dependence, and neither is proof against upstream feature leakage.
    """
    if config.n_permutations < 1:
        raise ValueError("At least one permutation is required")
    variants = {v: FEATURE_SETS[v] for v in ("A", "B3")}
    estimator = {"logistic": classifiers()["logistic"]}
    columns = [("logistic", v, f"logistic__{v}") for v in variants]

    def summarize(oof):
        patient = per_patient_metrics(oof, columns)
        pooled = {v: metric_pair(oof.hypo_night, oof[f"logistic__{v}"])[0] for v in variants}
        macro = patient.groupby("feature_set").auroc.mean().to_dict()
        return {"pooled_auroc_A": pooled["A"], "pooled_auroc_B3": pooled["B3"],
                "pooled_delta_auroc": pooled["B3"] - pooled["A"],
                "macro_auroc_A": macro["A"], "macro_auroc_B3": macro["B3"],
                "macro_delta_auroc": macro["B3"] - macro["A"]}

    observed = summarize(observed_oof)
    labeled = np.flatnonzero(raw.hypo_night.notna().to_numpy())
    patient_indices = [np.array(list(idx), dtype=int) for idx in
                       raw.loc[raw.hypo_night.notna()].groupby("subject").groups.values()]
    rows, summaries = [], []
    seed_sequences = np.random.SeedSequence(config.permutation_seed).spawn(2)
    for mode, seed in zip(("global", "within_patient"), seed_sequences):
        rng = np.random.default_rng(seed)
        for repeat in range(1, config.n_permutations + 1):
            shuffled = raw.copy()
            for idx in ([labeled] if mode == "global" else patient_indices):
                shuffled.loc[idx, "hypo_night"] = rng.permutation(raw.loc[idx, "hypo_night"].to_numpy())
            features = add_history(shuffled, daytime, config)
            cohort, _, _ = prepare_cohort(features)
            oof, _, _ = generate_oof(cohort, variants, estimator)
            if not oof[["subject", "night_id"]].equals(observed_oof[["subject", "night_id"]]):
                raise AssertionError("Permutation changed the evaluation cohort")
            rows.append({"shuffle": mode, "replicate": repeat, **summarize(oof)})
            if progress and (repeat == 1 or repeat % 25 == 0 or repeat == config.n_permutations):
                progress(f"Permutation {mode}: {repeat}/{config.n_permutations}")
        null = pd.DataFrame([row for row in rows if row["shuffle"] == mode])
        for metric, actual in observed.items():
            lo, hi = np.quantile(null[metric], [0.025, 0.975])
            summaries.append({"shuffle": mode, "algorithm": "logistic", "metric": metric,
                              "observed": actual, "null_mean": null[metric].mean(),
                              "null_sd": null[metric].std(), "null_2_5_percentile": lo,
                              "null_97_5_percentile": hi,
                              "p_greater_equal": (1 + null[metric].ge(actual).sum()) / (config.n_permutations + 1),
                              "n_permutations": config.n_permutations})
    return pd.DataFrame(summaries), pd.DataFrame(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_report(output_dir, outputs, config):
    pooled = outputs["pooled_metrics_ci.csv"]
    delta = outputs["paired_auc_differences.csv"]
    patient = outputs["patient_metrics.csv"]
    permutation = outputs["permutation_summary.csv"]
    cohort = pooled.iloc[0]
    glucose_audit = outputs["glucose_timing_audit.csv"]
    exclusions = outputs["excluded_nights.csv"]
    matched = int(glucose_audit.stored_mean_matches_postcutoff.sum())
    lines = [
        "# SleepGuard LOPO rerun",
        "",
        "**Exploratory: glucose and HRV were rebuilt before 22:00. Stored outcome construction and the original cohort selection still need an upstream audit; a permutation test cannot certify absence of leakage.**",
        "",
        f"Evaluated {int(cohort.n_nights)} nights from {int(cohort.n_patients)} patients, including {int(cohort.n_low_nights)} low nights (prevalence {cohort.prevalence:.3f}).",
        "",
        f"{matched}/{len(glucose_audit)} stored glucose means matched 22:00–00:00 CGM. They were replaced with glucose mean and OLS slope from 20:00–22:00, before the user-confirmed 22:00 cutoff. HRV was re-extracted from raw ECG in the same pre-cutoff window. A total of {len(exclusions)} nights were excluded consistently across variants for missing eligible predictors. Existing outcome labels and original eligibility flags were retained; their construction must be audited before using this as a validated 22:00 prediction experiment.",
        "",
        "Rebuilt HRV uses 250 Hz ECG, HRConfidence=100 and ECGNoise<0.001, continuous five-minute windows, NeuroKit cleaning/R-peak detection, and Welch LF/HF. Windows with more than 5% implausible/deviant RR intervals are rejected; at most 5% are interpolated within that pre-cutoff window. A night requires six accepted non-overlapping windows (30 minutes); nightly values are their medians. These fixed quality rules are recorded in the extraction manifest. No legacy HRV value is used as a fallback.",
        "",
        "A uses corrected pre-cutoff glucose mean/slope, previous-calendar-night low, 10:00–22:00 daytime low, and the patient's low rate over earlier known labeled nights in the supplied table. B1 adds within-patient RMSSD z-score; B2 adds LF/HF z-score; B3 adds both. History is shifted before computing features. No future/current label enters its own predictors. First-night history and HRV histories with fewer than two prior measurements or zero SD are missing, then imputed using training patients only. This evaluates sequential use of an unseen patient's accumulating history, not a completely history-free new patient.",
        "",
        "The daytime feature requires at least 80% occupied five-minute CGM bins and indicates any reading below 70 mg/dL; it is not a duration-defined episode. Missing daytime coverage is not coded as absence of a low. Ambiguous duplicate CGM timestamps are excluded. The five incomplete outcome windows in the original input are positive; an observed positive can be known despite missing later measurements.",
        "",
        "| Model | Variant | Pooled AUROC (95% CI) | PR-AUC / AP (95% CI) |",
        "|---|---|---|---|",
    ]
    for r in pooled.itertuples():
        lines.append(f"| {r.algorithm} | {r.feature_set} | {r.pooled_auroc:.3f} ({r.auroc_ci_lower:.3f}, {r.auroc_ci_upper:.3f}) | {r.pooled_pr_auc:.3f} ({r.pr_auc_ci_lower:.3f}, {r.pr_auc_ci_upper:.3f}) |")
    lines += ["", "Primary comparison: logistic B3−A. Other variants and the shallow random forest are exploratory. Settings were held fixed from the previous notebook.",
              "", "| Model | Comparison | ΔAUROC (95% CI) | ΔPR-AUC / AP (95% CI) |", "|---|---|---|---|"]
    for r in delta.itertuples():
        lines.append(f"| {r.algorithm} | {r.comparison} | {r.delta_auroc:+.3f} ({r.auroc_ci_lower:+.3f}, {r.auroc_ci_upper:+.3f}) | {r.delta_pr_auc:+.3f} ({r.pr_auc_ci_lower:+.3f}, {r.pr_auc_ci_upper:+.3f}) |")
    lines += ["", f"Intervals use {config.n_bootstraps:,} paired patient-cluster bootstrap replicates of fixed out-of-fold predictions. Every selected patient's nights are included with multiplicity, with identical draws for every model. They do not capture full refitting/model-selection uncertainty; ten clusters give imprecise inference.",
              "", "Per-patient values for every model are in `patient_metrics.csv`; single-class patients receive NA for both ranking metrics and remain in pooled results. Patient macro averages are in `patient_macro_metrics.csv`.",
              "", "| Patient | Nights | Low nights | Logistic A AUROC | Logistic B3 AUROC | ΔAUROC | A PR-AUC / AP | B3 PR-AUC / AP |", "|---|---|---|---|---|---|---|---|"]
    def value(x):
        return "NA" if pd.isna(x) else f"{x:.3f}"
    for subject, p in patient.loc[patient.algorithm.eq("logistic")].groupby("subject"):
        a, b = p.loc[p.feature_set.eq("A")].iloc[0], p.loc[p.feature_set.eq("B3")].iloc[0]
        lines.append(f"| {subject} | {a.n_nights} | {a.n_low_nights} | {value(a.auroc)} | {value(b.auroc)} | {value(b.delta_auroc_vs_A)} | {value(a.pr_auc)} | {value(b.pr_auc)} |")
    lines += ["", f"Permutation diagnostics refit primary logistic A and B3 for each of {config.n_permutations} global and {config.n_permutations} within-patient label shuffles. Outcome-derived history is rebuilt after every shuffle. Global shuffling is an IID negative control; within-patient shuffling preserves patient low counts. Neither preserves serial dependence. Their null means need not be exactly 0.5, especially for pooled LOPO scores with fold-dependent probabilities and burden-preserving shuffles. A chance-like null cannot rule out upstream post-cutoff feature leakage.",
              "", "| Shuffle | Metric | Observed | Null mean | Null central 95% | Upper-tail p |", "|---|---|---|---|---|---|"]
    for r in permutation.itertuples():
        lines.append(f"| {r.shuffle} | {r.metric} | {r.observed:.3f} | {r.null_mean:.3f} | ({r.null_2_5_percentile:.3f}, {r.null_97_5_percentile:.3f}) | {r.p_greater_equal:.3f} |")
    lines += ["", "PR-AUC here is non-interpolated average precision, with prevalence provided as its reference level. [Scikit-learn metric definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). Raw CGM comes from the [public PhysioCGM dataset](https://doi.org/10.6084/m9.figshare.28136294); source hashes are recorded in `run_config.json`.",
              "", "Audits: `glucose_timing_audit.csv`, `hrv_timing_audit.csv`, `daytime_feature_audit.csv`, `feature_audit.csv`, `fold_audit.csv`, and `excluded_nights.csv`. There is no 0.5-threshold confusion/count table in this run. Historical run folders were preserved."]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def run_evaluation(repo_root, config=Config(), raw_root=None, progress=print):
    repo_root = Path(repo_root).resolve()
    input_file = repo_root / "data/physiocgm/processed/model_data.parquet"
    hrv_sources = repo_root / "data/physiocgm/processed/presleep_hrv_sources_20_22.json"
    if not hrv_sources.is_file():
        raise FileNotFoundError("Missing HRV extraction provenance; run scripts/build_presleep_hrv.py first")
    raw_root = Path(raw_root) if raw_root else repo_root / "data/physiocgm/raw"
    if progress:
        progress("Using rebuilt pre-22:00 glucose/HRV. Stored outcome/cohort provenance remains exploratory.")
    raw = read_model_table(input_file)
    files = {s: raw_root / s / "cgm.csv" for s in raw.subject.unique()}
    missing = [s for s, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Raw CGM required for daytime low; missing {missing} under {raw_root}")
    cgm_by_patient = {s: read_cgm(path) for s, path in files.items()}
    daytime = daytime_features(raw, cgm_by_patient, config)
    raw, glucose_audit = rebuild_presleep_glucose(raw, cgm_by_patient, config)
    hrv_file = repo_root / "data/physiocgm/processed/presleep_hrv_20_22.parquet"
    raw, hrv_audit = load_rebuilt_hrv(raw, hrv_file, config)
    available = daytime.groupby("subject").daytime_low.count()
    if available.eq(0).any():
        raise ValueError(f"No usable daytime CGM for: {available[available.eq(0)].index.tolist()}")
    feature_table = add_history(raw, daytime, config)
    cohort, excluded, counts = prepare_cohort(feature_table)
    if progress:
        progress(f"Cohort: {len(cohort)} nights, {cohort.subject.nunique()} patients, {cohort.hypo_night.sum()} low nights")
    oof, audit, columns = generate_oof(cohort)
    if progress:
        progress("LOPO fits complete; computing paired patient bootstrap")
    pooled, delta, boot, draws = paired_patient_bootstrap(oof, columns, config.n_bootstraps, config.bootstrap_seed)
    patient = per_patient_metrics(oof, columns)
    macro = patient.groupby(["algorithm", "feature_set"]).agg(
        mean_patient_auroc=("auroc", "mean"), mean_patient_pr_auc=("pr_auc", "mean"),
        mean_patient_delta_auroc=("delta_auroc_vs_A", "mean"),
        evaluable_patients=("auroc", "count"), total_patients=("subject", "size")).reset_index()
    perm_summary, perm_values = permutation_test(raw, daytime, oof, config, progress)
    outputs = {"oof_predictions.csv": oof, "pooled_metrics_ci.csv": pooled,
               "paired_auc_differences.csv": delta, "patient_metrics.csv": patient,
               "patient_macro_metrics.csv": macro, "cohort_counts.csv": counts,
               "excluded_nights.csv": excluded, "fold_audit.csv": audit,
               "bootstrap_metrics.csv": boot, "bootstrap_patient_draws.csv": draws,
               "feature_audit.csv": feature_table, "daytime_feature_audit.csv": daytime,
               "glucose_timing_audit.csv": glucose_audit,
               "hrv_timing_audit.csv": hrv_audit,
               "permutation_summary.csv": perm_summary, "permutation_scores.csv": perm_values}
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output_dir = repo_root / "results/lopo" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, na_rep="NA")
    metadata = {
        "input_file": str(input_file), "input_sha256": sha256(input_file),
        "validation_status": "glucose/HRV inputs rebuilt pre-22:00; exploratory because stored outcome construction and original cohort selection remain unverified",
        "rebuilt_hrv_file": str(hrv_file), "rebuilt_hrv_sha256": sha256(hrv_file),
        "code_sha256": sha256(__file__), "config": asdict(config),
        "cgm_inputs": {s: {"file": str(p), "sha256": sha256(p),
                            "conflicting_timestamps_excluded": cgm_by_patient[s].attrs["conflicting_timestamps_excluded"]}
                       for s, p in files.items()},
        "feature_sets": FEATURE_SETS,
        "classifiers": {name: model.get_params() for name, model in classifiers().items()},
        "primary_comparison": "logistic B3-A",
        "timing": "Recording-local naive times: daytime [10:00,22:00), prediction 22:00, previous night ends 10:00. Calendar-day keys are not measured sleep onset.",
        "glucose_correction": "Mean and OLS slope (mg/dL/hour) rebuilt from [20:00,22:00), with >=80% 5-minute-bin coverage. Original values and post-cutoff matches preserved in glucose_timing_audit.csv. No outcome labels redefined.",
        "history": "Earlier known labels in the supplied model table only (complete outcomes or observed positives). Label availability fixed before permutations. Prior night is exact preceding calendar day; gaps missing. Held-out history is available sequentially, but never used to fit models.",
        "hrv_normalization": "Within-patient expanding mean/sample SD shifted by one night. At least two finite prior measurements and positive SD; otherwise missing.",
        "daytime_low": "Any EGV <70 mg/dL in [10:00,22:00), requiring >=80% occupied 5-minute bins. This is a low-reading indicator, not a duration-defined episode.",
        "missing_policy": "Complete-case on four original predictors; history missingness retained and median-imputed with indicators using training patients only. All-empty training features filled with 0.",
        "pr_auc_definition": "Non-interpolated average precision (sklearn.metrics.average_precision_score). Single-class patients NA for both ranking metrics, retained in pooled estimates.",
        "bootstrap": "Paired patient-cluster percentile intervals of fixed OOF scores; multiplicity preserved; does not include refitting uncertainty.",
        "permutation": "Primary logistic A and B3 refit for each global and within-patient shuffle. All outcome-derived histories rebuilt from shuffled labels; raw CGM and HRV unchanged. Upper-tail +1 p-values; global IID negative control and restricted burden-preserving null are exploratory, not evidence that upstream features are leakage-free.",
        "hrv_extraction": "ECG exclusively [20:00,22:00); original HRV replaced. Six accepted five-minute windows required, median aggregation. Full extraction settings, timestamps, software version and raw-file hashes in hrv_extraction_sources.json and hrv_timing_audit.csv.",
        "upstream_limitations": "Original Model_Eligible/cohort selection and stored outcome labels retained, not reconstructed for the 22:00 cutoff. Historical labels and HRV references limited to selected model-table nights; no claim of leakage-free validation.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))
    source_manifest = raw_root / "cgm_download_sources.json"
    if source_manifest.is_file():
        (output_dir / "cgm_download_sources.json").write_text(source_manifest.read_text())
    (output_dir / "hrv_extraction_sources.json").write_text(hrv_sources.read_text())
    write_report(output_dir, outputs, config)
    if progress:
        progress(f"Saved results to: {output_dir}")
    return output_dir, outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()
    run_evaluation(args.repo_root, Config(n_permutations=args.permutations, n_bootstraps=args.bootstraps),
                   args.raw_root, progress=lambda message: print(message, flush=True))
