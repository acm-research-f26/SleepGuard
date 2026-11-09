"""Compare CGM features and models on the same held-out patients, then test augmentation seeds."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_cgm_features import RECENT


CORE = ["mean_2h", "slope_2h", "min_2h"]
HISTORY = ["prev_night_hypo", "prior_hypo_rate", "day_low_events", "day_min_glucose"]
FEATURE_SETS = {
    "A0": CORE,
    "A": CORE + HISTORY,
    "A0_recent": CORE + RECENT,
    "A_recent": CORE + HISTORY + RECENT,
}
SEEDS = (42, *range(19))
LR_C = (1.0, 0.1, 0.01, 10.0)
RF_GRID = tuple(product((2, 3), (4, 8)))


def load_nights(path):
    df = pd.read_csv(path, parse_dates=["night_of"]).sort_values(["subject", "night_of"]).reset_index(drop=True)
    if df.duplicated(["subject", "night_of"]).any() or not df.hypo_night.isin([0, 1]).all():
        raise ValueError("Expected one binary outcome per subject and night")
    prev = df.groupby("subject").night_of.shift()
    label = df.groupby("subject").hypo_night
    df["prev_night_hypo"] = label.shift().where(df.night_of.sub(prev).eq(pd.Timedelta(days=1)))
    df["prior_hypo_rate"] = label.transform(lambda s: s.shift().expanding().mean())
    return df


def estimator(kind, settings):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if kind == "logistic":
        return make_pipeline(imputer, StandardScaler(),
                             LogisticRegression(C=settings, class_weight="balanced", max_iter=1000))
    depth, leaf = settings
    return make_pipeline(imputer, RandomForestClassifier(
        n_estimators=80, max_depth=depth, min_samples_leaf=leaf,
        class_weight="balanced", random_state=42, n_jobs=1,
    ))


def folds(y, g):
    return LeaveOneGroupOut().split(np.empty(len(y)), y, g)


def predict(kind, settings, x_train, y_train, x_test):
    return estimator(kind, settings).fit(x_train, y_train).predict_proba(x_test)[:, 1]


def tune(kind, candidates, X, y, g):
    splits = list(GroupKFold(n_splits=3).split(X, y, g))
    scores = []
    for settings in candidates:
        pred = np.empty(len(y))
        for train, val in splits:
            pred[val] = predict(kind, settings, X[train], y[train], X[val])
        scores.append(roc_auc_score(y, pred))
    best = int(np.argmax(scores))
    return candidates[best], scores[best]


def compare_models(df, output):
    y, g = df.hypo_night.to_numpy(), df.subject.to_numpy()
    predictions = df[["subject", "night_of", "hypo_night"]].copy()
    choices = []
    # Fixed pipelines reproduce the original baselines. Tuned candidates get their own inner CV.
    for features, cols in FEATURE_SETS.items():
        X = df[cols].to_numpy(float)
        variants = [("logistic_tuned", "logistic", LR_C), ("forest_tuned", "forest", RF_GRID)]
        if features in ("A0", "A"):
            variants.insert(0, ("logistic_original", "logistic", (1.0,)))
        for variant, kind, candidates in variants:
            model = f"{features}_{variant}"
            pred = np.empty(len(y))
            for train, test in folds(y, g):
                settings, score = (candidates[0], np.nan) if len(candidates) == 1 else tune(
                    kind, candidates, X[train], y[train], g[train]
                )
                choices.append({"model": model, "held_out_subject": g[test[0]],
                                "settings": str(settings), "inner_auroc": score})
                pred[test] = predict(kind, settings, X[train], y[train], X[test])
            predictions[model] = pred
            print(model, f"{roc_auc_score(y, pred):.4f}", flush=True)
    choice_table = pd.DataFrame(choices)
    tuned = choice_table.dropna(subset=["inner_auroc"])
    selected = tuned.loc[tuned.groupby("held_out_subject").inner_auroc.idxmax()]
    for row in selected.itertuples(index=False):
        mask = predictions.subject.eq(row.held_out_subject)
        predictions.loc[mask, "nested_selected"] = predictions.loc[mask, row.model]
    predictions.to_csv(output / "held_out_predictions.csv", index=False)
    choice_table.to_csv(output / "inner_cv_choices.csv", index=False)
    selected.to_csv(output / "nested_selected_choices.csv", index=False)
    model_metrics(predictions, output)


def model_metrics(predictions, output):
    y = predictions.hypo_night.to_numpy()
    patients = predictions.groupby("subject").indices
    models = predictions.columns[3:]
    pairs = []
    for model in models:
        if model.endswith("_original"):
            continue
        if model == "nested_selected":
            pairs.extend((model, f"{base}_logistic_original") for base in ("A0", "A"))
            continue
        features, kind, tuning = model.rsplit("_", 2)
        variant = f"{kind}_{tuning}"
        base = features.removesuffix("_recent")
        pairs.append((model, f"{base}_logistic_original"))
        if features.endswith("_recent"):
            pairs.append((model, f"{base}_{variant}"))
    arrays = {model: predictions[model].to_numpy() for model in models}
    aucs = {model: roc_auc_score(y, pred) for model, pred in arrays.items()}
    rng = np.random.default_rng(42)
    draws = []
    blocks = list(patients.values())
    for _ in range(1000):
        rows = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        if len(np.unique(y[rows])) == 2:
            draws.append({model: roc_auc_score(y[rows], pred[rows]) for model, pred in arrays.items()})
    boot = pd.DataFrame(draws)
    contrasts = []
    for model, ref in pairs:
        lo, hi = np.percentile(boot[model] - boot[ref], [2.5, 97.5])
        contrasts.append({"model": model, "reference": ref, "delta_auroc": aucs[model] - aucs[ref],
                          "delta_ci_low": lo, "delta_ci_high": hi})
    pd.DataFrame(contrasts).to_csv(output / "paired_auroc_comparisons.csv", index=False)
    patients_out, summary = [], []
    for model, pred in arrays.items():
        reliable_aucs = []
        for subject, rows in patients.items():
            positive = y[rows].sum()
            negative = len(rows) - positive
            score = roc_auc_score(y[rows], pred[rows]) if min(positive, negative) >= 1 else np.nan
            reliable = min(positive, negative) >= 3
            if reliable:
                reliable_aucs.append(score)
            patients_out.append({"model": model, "subject": subject, "hypo_nights": positive,
                                 "non_hypo_nights": negative, "auroc": score, "reliable": reliable})
        lo, hi = np.percentile(boot[model], [2.5, 97.5])
        summary.append({"model": model, "nights": len(y), "patients": len(patients), "auroc": aucs[model],
                        "auroc_ci_low": lo, "auroc_ci_high": hi, "pr_auc_ap": average_precision_score(y, pred),
                        "mean_patient_auroc": np.mean(reliable_aucs) if reliable_aucs else np.nan,
                        "patients_in_mean": len(reliable_aucs)})
    pd.DataFrame(summary).to_csv(output / "model_summary.csv", index=False)
    pd.DataFrame(patients_out).to_csv(output / "per_patient_auroc.csv", index=False)


def augmented_fold(train_x, train_y, train_g, rng, cap_rng, count):
    rows, targets = [], []
    # Same order and random draws as the prior experiment.
    groups = pd.DataFrame({"subject": train_g, "label": train_y}).groupby(["subject", "label"], sort=False)
    for _, nights in groups:
        positions = nights.index.to_numpy()
        n = len(positions)
        if n < 2:
            continue
        anchor = train_x[positions].copy()
        partner = (np.arange(n) + rng.integers(1, n, size=n)) % n
        weight = rng.uniform(0.25, 0.75, size=n)[:, None]
        anchor[:, :len(CORE)] += weight * (anchor[partner, :len(CORE)] - anchor[:, :len(CORE)])
        rows.append(anchor)
        targets.append(train_y[positions])
    synthetic, labels = np.concatenate(rows), np.concatenate(targets)
    if count is not None:
        keep = np.sort(cap_rng.choice(len(synthetic), size=count, replace=False))
        synthetic, labels = synthetic[keep], labels[keep]
    return np.vstack((train_x, synthetic)), np.r_[train_y, labels], len(labels)


def seed_experiments(full, subset, output):
    results = []
    for scenario, df, cap in (("129_real_all_synthetic", full, None), ("129_real_99_synthetic", full, 99),
                              ("99_real_all_synthetic", subset, None)):
        y, g = df.hypo_night.to_numpy(), df.subject.to_numpy()
        X = df[FEATURE_SETS["A"]].to_numpy(float)
        splits = list(folds(y, g))
        baseline = {}
        for features, n_cols in (("A0", len(CORE)), ("A", X.shape[1])):
            base = np.empty(len(y))
            for train, test in splits:
                base[test] = predict("logistic", 1.0, X[train, :n_cols], y[train], X[test, :n_cols])
            baseline[features] = roc_auc_score(y, base), average_precision_score(y, base)

        for seed in SEEDS:
            rng, cap_rng = np.random.default_rng(seed), np.random.default_rng(seed)
            preds = {name: np.empty(len(y)) for name in baseline}
            rows_added = 0
            for i, (train, test) in enumerate(splits):
                count = None
                if cap is not None:
                    per_fold, extra = divmod(cap, len(splits))
                    count = per_fold + int(i < extra)
                train_x, train_y, added = augmented_fold(X[train], y[train], g[train], rng, cap_rng, count)
                rows_added += added
                for name, n_cols in (("A0", len(CORE)), ("A", X.shape[1])):
                    preds[name][test] = predict("logistic", 1.0, train_x[:, :n_cols], train_y, X[test, :n_cols])
            for name, pred in preds.items():
                auc, ap = roc_auc_score(y, pred), average_precision_score(y, pred)
                base_auc, base_ap = baseline[name]
                results.append({"scenario": scenario, "model": name, "seed": seed, "real_nights": len(y),
                                "synthetic_rows_across_folds": rows_added, "baseline_auroc": base_auc,
                                "augmented_auroc": auc, "delta_auroc": auc - base_auc,
                                "baseline_pr_auc_ap": base_ap, "augmented_pr_auc_ap": ap})
    trials = pd.DataFrame(results)
    trials.to_csv(output / "synthetic_seed_trials.csv", index=False)
    summary = trials.groupby(["scenario", "model"], sort=False).agg(
        seeds=("seed", "size"), baseline_auroc=("baseline_auroc", "first"),
        mean_augmented_auroc=("augmented_auroc", "mean"), mean_delta=("delta_auroc", "mean"),
        min_delta=("delta_auroc", "min"), max_delta=("delta_auroc", "max"),
        seeds_with_gain=("delta_auroc", lambda s: int(s.gt(0).sum())),
    )
    summary.to_csv(output / "synthetic_seed_summary.csv")
    print(summary.round(4).to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    full = load_nights(args.data / "cgm_features.csv")
    subset = load_nights(args.data / "real_99_nights.csv")
    compare_models(full, args.output)
    seed_experiments(full, subset, args.output)
