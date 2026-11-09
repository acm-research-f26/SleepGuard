"""Compare real and augmented training using only Drive model_data.parquet."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from run_sleepguard import LR_C, folds, predict, tune


KEYS = ["subject", "night_of"]
GLUCOSE = ["mean_glucose_presleep", "glucose_slope_presleep"]
INPUTS = GLUCOSE + ["lf_hf"]
SEEDS = (42, 0, 1, 2, 3)


def generate(train, seed, columns=INPUTS):
    rng = np.random.default_rng(seed)
    records = []
    grouped = train.groupby([train.subject, train.hypo_night, train.lf_hf.notna()], sort=False)
    for (subject, label, measured), group in grouped:
        n = len(group)
        if n < 2:
            continue
        X = group[columns].to_numpy(float)
        partners = (np.arange(n) + rng.integers(1, n, size=n)) % n
        alpha = rng.uniform(0.25, 0.75, size=n)
        result = X + alpha[:, None] * (X[partners] - X)
        rows = pd.DataFrame(result, columns=columns)
        rows["subject"] = subject
        rows["hypo_night"] = label
        rows["has_measured_hrv_parents"] = measured
        rows["anchor_night"] = group.night_of.to_numpy()
        rows["partner_night"] = group.night_of.to_numpy()[partners]
        rows["fraction_from_partner"] = alpha
        records.append(rows)
    if not records:
        return pd.DataFrame(columns=columns + ["hypo_night"])
    return pd.concat(records, ignore_index=True)


def evaluate(data, name, out):
    data = data.reset_index(drop=True)
    g, y = data.subject.to_numpy(), data.hypo_night.astype(int).to_numpy()
    inputs = {"glucose": GLUCOSE, "glucose_lf_hf": INPUTS}
    baseline = {model: np.empty(len(data)) for model in inputs}
    augmented = {model: np.empty((len(SEEDS), len(data))) for model in inputs}
    audit, counts = [], []
    for train, test in folds(y, g):
        train_data = data.iloc[train]
        variants = [generate(train_data, seed) for seed in SEEDS]
        fold_id = g[test[0]]
        audit.append(variants[0].assign(held_out_subject=fold_id))
        counts.append({"held_out_subject": fold_id, "real_train_nights": len(train),
                       "synthetic_rows_per_seed": len(variants[0]),
                       "real_test_nights": len(test)})
        for model, cols in inputs.items():
            x = data[cols].to_numpy(float)
            c, _ = tune("logistic", LR_C, x[train], y[train], g[train])
            baseline[model][test] = predict("logistic", c, x[train], y[train], x[test])
            for i, synthetic in enumerate(variants):
                features = np.vstack((x[train], synthetic[cols].to_numpy(float)))
                labels = np.r_[y[train], synthetic.hypo_night.to_numpy(int)]
                # Adjust C to keep the loss-to-penalty scale comparable across budgets.
                fit_c = c * len(train) / len(labels)
                augmented[model][i, test] = predict("logistic", fit_c, features, labels, x[test])

    pd.concat(audit, ignore_index=True).to_csv(out / f"{name}_synthetic_rows_seed42.csv", index=False)
    pd.DataFrame(counts).to_csv(out / f"{name}_fold_sizes.csv", index=False)
    predictions = data[KEYS + ["hypo_night"]].assign(has_measured_hrv=data.lf_hf.notna())
    for model in inputs:
        predictions[model] = baseline[model]
        predictions[f"{model}_augmented"] = augmented[model].mean(axis=0)
    predictions.to_csv(out / f"{name}_predictions.csv", index=False)

    groups = list(data.groupby("subject").indices.values())
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(1000):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if len(np.unique(y[idx])) == 2:
            draws.append({col: roc_auc_score(y[idx], predictions[col].iloc[idx])
                          for col in predictions.columns[4:]})
    bootstrap = pd.DataFrame(draws)
    summary, seed_rows = [], []
    for model in inputs:
        ref = roc_auc_score(y, predictions[model])
        aug_col = f"{model}_augmented"
        aug = roc_auc_score(y, predictions[aug_col])
        lo, hi = np.percentile(bootstrap[aug_col] - bootstrap[model], [2.5, 97.5])
        summary.append({"model": model, "nights": len(y), "hypo_nights": y.sum(),
                        "nights_with_measured_hrv": data.lf_hf.notna().sum(),
                        "real_only_auroc": ref, "augmented_auroc": aug, "delta": aug - ref,
                        "delta_ci_low": lo, "delta_ci_high": hi,
                        "real_only_average_precision": average_precision_score(y, predictions[model]),
                        "augmented_average_precision": average_precision_score(y, predictions[aug_col])})
        for seed, p in zip(SEEDS, augmented[model]):
            seed_rows.append({"model": model, "seed": seed, "auroc": roc_auc_score(y, p)})
    pd.DataFrame(seed_rows).to_csv(out / f"{name}_seed_auroc.csv", index=False)
    result = pd.DataFrame(summary)
    result.to_csv(out / f"{name}_summary.csv", index=False)
    print(name, result.round(4).to_string(index=False), sep="\n", flush=True)


def load_drive_data(path, include_rmssd=False):
    source = pd.read_parquet(path)
    cols = {
        "participant_id": "subject", "night": "night_of", "hypo": "hypo_night",
        "lf_hf_presleep": "lf_hf", **{col: col for col in GLUCOSE},
    }
    if include_rmssd:
        cols["rmssd_presleep"] = "rmssd"
    missing = set(cols).union({"Model_Eligible"}) - set(source.columns)
    if missing:
        raise ValueError(f"Drive table missing columns: {sorted(missing)}")
    data = source.loc[source.Model_Eligible.eq(True), list(cols)].rename(columns=cols)
    if data.empty or data[KEYS].isna().any().any() or data.duplicated(KEYS).any():
        raise ValueError("Expected nonempty eligible rows with unique patient-night keys")
    if not data.hypo_night.isin([0, 1]).all():
        raise ValueError("Expected real binary labels from the Drive table")
    return data.sort_values(KEYS).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-dir", type=Path, required=True,
                        help="folder containing the original model_data.parquet")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--compare-linear-forest", action="store_true",
                        help="compare linear logistic regression and shallow random forests")
    args = parser.parse_args()
    data = load_drive_data(args.drive_dir / "model_data.parquet", include_rmssd=args.compare_linear_forest)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.compare_linear_forest:
        from compare_linear_forest import evaluate_comparison
        evaluate_comparison(data, args.output)
    else:
        evaluate(data, "drive_only", args.output)
