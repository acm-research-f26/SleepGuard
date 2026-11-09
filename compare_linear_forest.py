"""Compare linear logistic regression and small forests with patient-held-out validation."""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from augment_drive_only import GLUCOSE, INPUTS, KEYS, SEEDS, generate


FEATURES = {
    "glucose": GLUCOSE,
    "glucose_rmssd": GLUCOSE + ["rmssd"],
    "glucose_lf_hf": INPUTS,
    "glucose_both": INPUTS + ["rmssd"],
}
FAMILIES = {
    "linear_logistic": ((0.01,), (0.1,), (1.0,), (10.0,)),
    "small_forest": tuple(product((2, 3), (4, 8))),
}
MODES = {"real": (0.0,), "augmented": (0.25, 1.0)}


def synthetic_rows(train, seed, fraction):
    synthetic = generate(train, seed, FEATURES["glucose_both"])
    if fraction == 1.0 or synthetic.empty:
        return synthetic
    return synthetic.sample(n=max(1, round(fraction * len(synthetic))), random_state=seed)


def fit_predict(train, validation, cols, family, setting, synthetic):
    real_count = len(train)
    if synthetic is not None:
        train = pd.concat([train, synthetic], ignore_index=True)
    if family == "linear_logistic":
        # Hold nominal regularization per observation constant when adding rows.
        c = setting[0] * real_count / len(train)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=c, class_weight="balanced", max_iter=1000))
    else:
        depth, leaf = setting
        model = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=32, max_depth=depth, min_samples_leaf=leaf,
            class_weight="balanced", random_state=42, n_jobs=1,
        ))
    return model.fit(train[cols], train.hypo_night).predict_proba(validation[cols])[:, 1]


def inner_search(data, splits, cols, family, mode, synthetic_cache):
    scores = []
    for setting, fraction in product(FAMILIES[family], MODES[mode]):
        p = np.empty(len(data))
        for i, (train, val) in enumerate(splits):
            p[val] = fit_predict(data.iloc[train], data.iloc[val], cols, family, setting,
                                 synthetic_cache.get((i, fraction)))
        scores.append((roc_auc_score(data.hypo_night, p), setting, fraction))
    return max(scores, key=lambda item: item[0])


def evaluate_comparison(data, out):
    y, groups = data.hypo_night.astype(int).to_numpy(), data.subject.to_numpy()
    specs = list(product(FAMILIES, MODES, FEATURES))
    preds = {"__".join(spec): np.empty(len(data)) for spec in specs}
    choices, audit, seed_predictions = [], [], []
    for outer_train, outer_test in LeaveOneGroupOut().split(data, y, groups):
        train, test = data.iloc[outer_train], data.iloc[outer_test]
        subject = test.subject.iloc[0]
        splits = list(GroupKFold(n_splits=3).split(train, train.hypo_night, train.subject))
        cache = {(i, fraction): synthetic_rows(train.iloc[fit], 42, fraction)
                 for i, (fit, _) in enumerate(splits) for fraction in MODES["augmented"]}
        outer_synthetic = {(fraction, seed): synthetic_rows(train, seed, fraction)
                           for fraction in MODES["augmented"] for seed in SEEDS}
        for (fraction, seed), rows in outer_synthetic.items():
            audit.append(rows.assign(held_out_subject=subject, fraction=fraction, seed=seed))
        for family, mode, name in specs:
            cols = FEATURES[name]
            score, setting, fraction = inner_search(train, splits, cols, family, mode, cache)
            model = "__".join((family, mode, name))
            seeds = SEEDS if fraction else (42,)
            p = np.array([fit_predict(train, test, cols, family, setting,
                                      outer_synthetic.get((fraction, seed))) for seed in seeds])
            preds[model][outer_test] = p.mean(axis=0)
            choices.append({"model": model, "held_out_subject": subject, "setting": str(setting),
                            "inner_auroc": score, "synthetic_fraction": fraction,
                            "synthetic_rows_per_fit": len(outer_synthetic[(fraction, 42)]) if fraction else 0,
                            "real_training_nights": len(train)})
            for seed, values in zip(seeds, p):
                seed_predictions.append(test[KEYS + ["hypo_night"]].assign(
                    model=model, seed=seed, score=values))
        print(f"Finished held-out patient {subject}", flush=True)
    predictions = pd.DataFrame(preds)
    data[KEYS + ["hypo_night"]].join(predictions).to_csv(out / "linear_forest_predictions.csv", index=False)
    pd.DataFrame(choices).to_csv(out / "linear_forest_choices.csv", index=False)
    pd.concat(audit, ignore_index=True).to_csv(out / "linear_forest_synthetic_rows.csv", index=False)
    seed_data = pd.concat(seed_predictions, ignore_index=True)
    seeds = [{"model": model, "seed": seed, "auroc": roc_auc_score(rows.hypo_night, rows.score)}
             for (model, seed), rows in seed_data.groupby(["model", "seed"])]
    pd.DataFrame(seeds).to_csv(out / "linear_forest_seed_auroc.csv", index=False)
    summarize(data, predictions, out)


def summarize(data, predictions, out):
    y = data.hypo_night.to_numpy()
    groups = list(data.groupby("subject").indices.values())
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(1000):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if len(np.unique(y[idx])) == 2:
            draws.append({model: roc_auc_score(y[idx], p.iloc[idx]) for model, p in predictions.items()})
    boot = pd.DataFrame(draws)
    auc = {model: roc_auc_score(y, p) for model, p in predictions.items()}
    rows = []
    for model, p in predictions.items():
        family, mode, features = model.split("__")
        ref = "__".join((family, mode, "glucose"))
        real = "__".join((family, "real", features))
        lo, hi = np.percentile(boot[model] - boot[ref], [2.5, 97.5])
        alo, ahi = np.percentile(boot[model] - boot[real], [2.5, 97.5])
        rows.append({"model": model, "auroc": auc[model], "average_precision": average_precision_score(y, p),
                     "glucose_baseline": auc[ref], "delta_vs_glucose": auc[model] - auc[ref],
                     "delta_ci_low": lo, "delta_ci_high": hi,
                     "delta_vs_real_training": auc[model] - auc[real],
                     "augmentation_ci_low": alo, "augmentation_ci_high": ahi})
    result = pd.DataFrame(rows)
    result.to_csv(out / "linear_forest_summary.csv", index=False)
    print(result.sort_values("auroc", ascending=False).round(4).to_string(index=False), flush=True)
