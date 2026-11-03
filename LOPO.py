"""Run: python3 LOPO_updated.py (or --input /path/to/features.csv).
A/B1/B2/B3 are feature sets. Each runs logistic regression and a simple forest.
"""
from pathlib import Path
from datetime import datetime
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DOWNLOADS = Path.home() / 'Downloads'
GROUP, LABEL, NIGHT = 'subject', 'hypo_night', 'night_of'
GLUCOSE = ['mean_1h', 'slope_1h', 'min_1h']
FEATURE_SETS = {
    'A': GLUCOSE,
    'B1': GLUCOSE + ['rmssd'],
    'B2': GLUCOSE + ['lf_hf'],
    'B3': GLUCOSE + ['rmssd', 'lf_hf'],
}
THRESHOLD = 0.50  # Fixed before evaluation; not tuned on test predictions.
CLASSIFIERS = {
    'logistic': make_pipeline(StandardScaler(), LogisticRegression(
        class_weight='balanced', max_iter=1000)),
    # Fixed, small forest: no grid search or nested tuning.
    'random_forest': RandomForestClassifier(
        n_estimators=100, max_depth=3, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1),
}


def metrics(y, probabilities):
    tn, fp, fn, tp = confusion_matrix(y, probabilities >= THRESHOLD, labels=[0, 1]).ravel()
    return {
        'auroc': roc_auc_score(y, probabilities) if len(np.unique(y)) == 2 else np.nan,
        'precision': tp / (tp + fp) if tp + fp else np.nan,
        'recall': tp / (tp + fn) if tp + fn else np.nan,
        'caught_lows': int(tp), 'missed_lows': int(fn),
        'false_alarms': int(fp), 'correct_nonlow': int(tn),
    }


def run(input_file, output_root, n_bootstraps=1000):
    output = []

    def show(value=''):
        output.append(str(value))
        print(value, flush=True)

    df = pd.read_csv(input_file, dtype={GROUP: str})
    show(f'Using table: {input_file}')
    show(f'Columns found: {list(df.columns)}')
    required = [GROUP, LABEL, NIGHT]
    if not set(required).issubset(df):
        raise ValueError(f'Missing required columns: {sorted(set(required) - set(df))}')
    runnable = {}
    for name, columns in FEATURE_SETS.items():
        missing = sorted(set(columns) - set(df))
        if missing:
            show(f'Skipping feature set {name}: missing columns {missing}')
        else:
            runnable[name] = columns
    if not runnable:
        raise ValueError('No feature set has the required columns.')
    features = sorted({c for columns in runnable.values() for c in columns})
    df[features] = df[features].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
    df[LABEL] = pd.to_numeric(df[LABEL], errors='raise')
    if not df[LABEL].dropna().isin([0, 1]).all():
        raise ValueError('Known hypo_night labels must be 0 or 1.')
    before = len(df)
    df = df.dropna(subset=features + required).copy()
    df[GROUP] = df[GROUP].str.strip()
    if df[GROUP].eq('').any():
        raise ValueError('Empty subject ID.')
    df[NIGHT] = pd.to_datetime(df[NIGHT], errors='raise')
    if df[NIGHT].isna().any() or df.duplicated([GROUP, NIGHT]).any():
        raise ValueError('Invalid or duplicate patient/night keys.')
    df = df.sort_values([GROUP, NIGHT]).reset_index(drop=True)
    if df[GROUP].nunique() < 3 or df[LABEL].nunique() != 2:
        raise ValueError('Need at least three retained patients and both outcome classes.')
    show(f'Nights: {len(df)} / {before}; patients: {df[GROUP].nunique()}; low nights: {int(df[LABEL].sum())}')
    show('Retained means complete predictors/keys, not independently verified outcome coverage or timing.')
    y, groups = df[LABEL].to_numpy(dtype=int), df[GROUP].to_numpy()
    scores, predictions, folds = {}, [], []
    for algorithm, estimator in CLASSIFIERS.items():
        for name, columns in runnable.items():
            probabilities = np.full(len(df), np.nan)
            coverage = np.zeros(len(df), dtype=int)
            for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(df, y, groups), 1):
                held_out = groups[te][0]
                assert set(groups[tr]).isdisjoint(set(groups[te]))
                if len(np.unique(y[tr])) < 2:
                    raise ValueError(f'{algorithm}/{name}/{held_out}: training fold has only one class.')
                model = clone(estimator)
                model.fit(df.iloc[tr][columns], y[tr])
                positive = list(model.classes_).index(1)
                probabilities[te] = model.predict_proba(df.iloc[te][columns])[:, positive]
                coverage[te] += 1
                folds.append(dict(algorithm=algorithm, model=name, fold=fold,
                    held_out_subject=held_out, training_patients=len(set(groups[tr])),
                    training_nights=len(tr), test_nights=len(te), test_hypo_nights=int(y[te].sum()),
                    patient_overlap=0))
            assert (coverage == 1).all() and np.isfinite(probabilities).all()
            scores[(algorithm, name)] = probabilities
            predictions.append(pd.DataFrame(dict(algorithm=algorithm, model=name, subject=groups,
                night_of=df[NIGHT], y_true=y, y_prob=probabilities,
                y_pred=(probabilities >= THRESHOLD).astype(int), threshold=THRESHOLD)))
            show(f'Completed {algorithm} / {name}: {len(probabilities)} held-out predictions.')
    # Resample patients, not individual nights. Repeated patients retain all copies.
    subjects = np.unique(groups)
    patient_indices = {s: np.flatnonzero(groups == s) for s in subjects}
    rng = np.random.default_rng(42)
    boot = {key: {'auroc': [], 'precision': [], 'recall': []} for key in scores}
    draws = []
    for repetition in range(n_bootstraps):
        sampled = rng.choice(subjects, len(subjects), replace=True)
        idx = np.concatenate([patient_indices[s] for s in sampled])
        draws.append(dict(replicate=repetition, sampled_subjects=','.join(sampled), sampled_nights=len(idx)))
        for key, probabilities in scores.items():
            result = metrics(y[idx], probabilities[idx])
            for metric in boot[key]:
                boot[key][metric].append(result[metric])
    rows = []
    for (algorithm, name), probabilities in scores.items():
        row = dict(algorithm=algorithm, model=name, threshold=THRESHOLD,
                   n_nights=len(df), n_low_nights=int(y.sum()), n_patients=len(subjects),
                   **metrics(y, probabilities))
        for metric, values in boot[(algorithm, name)].items():
            valid = np.array(values)[np.isfinite(values)]
            bounds = np.quantile(valid, [.025, .975]) if len(valid) else [np.nan, np.nan]
            row.update({metric + '_ci_lower': bounds[0], metric + '_ci_upper': bounds[1],
                        metric + '_valid_bootstraps': len(valid)})
        rows.append(row)
    summary = pd.DataFrame(rows)
    patient_rows = []
    for (algorithm, name), probabilities in scores.items():
        for subject, idx in patient_indices.items():
            patient_rows.append(dict(algorithm=algorithm, model=name, subject=subject,
                n_nights=len(idx), n_low_nights=int(y[idx].sum()), **metrics(y[idx], probabilities[idx])))
    pred = pd.concat(predictions, ignore_index=True)
    assert not pred.duplicated(['algorithm', 'model', 'subject', 'night_of']).any()
    show('\nRESULTS AT THRESHOLD 0.50:')
    show(summary.round(4).to_string(index=False, na_rep='N/A'))
    show('\nALL LOPO PREDICTIONS:')
    show(pred.to_string(index=False, float_format=lambda value: f'{value:.6f}'))
    out = Path(output_root) / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    for filename, frame in {
        'lopo_predictions.csv': pred, 'lopo_folds.csv': pd.DataFrame(folds),
        'metrics_confidence_intervals.csv': summary, 'patient_metrics.csv': pd.DataFrame(patient_rows),
        'bootstrap_patient_draws.csv': pd.DataFrame(draws),
    }.items():
        frame.to_csv(out / filename, index=False, na_rep='NA')
    config = dict(input=str(Path(input_file).resolve()), feature_sets=runnable,
        classifiers={name: str(model) for name, model in CLASSIFIERS.items()}, threshold=THRESHOLD,
        bootstrap_repetitions=n_bootstraps, seed=42,
        method='Patient-cluster percentile bootstrap of fixed LOPO predictions; no refitting.',
        limitation='Exploratory intervals; do not capture full refitting uncertainty. Unknown metrics stay NA.',
        eligibility='Complete required values only; upstream timing/quality/labels not independently verified.')
    (out / 'run_config.json').write_text(json.dumps(config, indent=2))
    show(f'\nSaved results to: {out.resolve()}')
    (out / 'lopo_full_output.txt').write_text('\n'.join(output) + '\n')
    return summary, pred, out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, default=DOWNLOADS / 'SleepGuard_LOPO_results')
    parser.add_argument('--bootstraps', type=int, default=1000)
    args = parser.parse_args()
    table = args.input
    if table is None:
        merged = DOWNLOADS / 'merged_features.csv'
        table = merged if merged.exists() else DOWNLOADS / 'model_a_glucose_features.csv'
    if not table.is_file():
        parser.error(f'Input not found: {table}. Use --input with your CSV path.')
    if args.bootstraps < 1:
        parser.error('--bootstraps must be positive.')
    run(table, args.output, args.bootstraps)