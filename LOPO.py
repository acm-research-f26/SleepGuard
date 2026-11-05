"""Run: python enhanced_lopo.py --data-dir ~/Downloads

For an existing per-night feature table, pass --features /path/to/features.csv.

Inputs: glucose_readings.csv (subject,timestamp,glucose in mg/dL),
night_labels.csv (subject,sleep_start,hypo_night; optional night_of).
All timestamps must have explicit UTC offsets. For naive local times use
--timezone America/Chicago. sleep_start must be the actual prediction cutoff,
and hypo_night must describe a fully observed outcome AFTER that cutoff.
Optional HRV: merged_features_windows.csv, keyed by subject and night_of,
with rmssd_{window}h and lf_hf_{window}h. These must already be quality checked
and computed strictly before the same cutoff. HRV provenance is not inferred.
No actual AUROC improvement is guaranteed.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WINDOWS = (0.5, 1, 2)
C_GRID = (0.1, 1.0, 10.0)
BASE = ('mean', 'min', 'slope')
EXTRA = ('last', 'std', 'delta', 'below70_fraction')
SEED = 42

def parse_times(values, timezone):
    parsed = []
    for value in values:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            raise ValueError('Missing timestamp')
        if ts.tzinfo is None:
            if timezone is None:
                raise ValueError('Naive timestamps require --timezone')
            ts = ts.tz_localize(timezone, ambiguous='raise', nonexistent='raise')
        parsed.append(ts)
    return parsed

def build_features(glucose, nights, timezone=None):
    glucose, nights = glucose.copy(), nights.copy()
    for frame, cols in ((glucose, ['subject','timestamp','glucose']),
                        (nights, ['subject','sleep_start','hypo_night'])):
        if not set(cols).issubset(frame):
            raise ValueError(f'Missing columns: {set(cols)-set(frame)}')
        if frame['subject'].isna().any():
            raise ValueError('Missing subject ID')
        frame['subject'] = frame['subject'].astype(str).str.strip()
        if frame['subject'].eq('').any():
            raise ValueError('Empty subject ID')
    cutoffs = parse_times(nights['sleep_start'], timezone)
    if 'night_of' not in nights:
        # Derive the local evening date BEFORE UTC conversion.
        nights['night_of'] = [str(t.date()) for t in cutoffs]
    if nights['night_of'].isna().any():
        raise ValueError('Missing night_of')
    nights['night_of'] = pd.to_datetime(nights['night_of'], errors='raise').dt.strftime('%Y-%m-%d')
    nights['sleep_start'] = pd.to_datetime(cutoffs, utc=True)
    glucose['timestamp'] = pd.to_datetime(parse_times(glucose['timestamp'], timezone), utc=True)
    nights['hypo_night'] = pd.to_numeric(nights['hypo_night'], errors='raise')
    if not nights['hypo_night'].isin([0,1]).all():
        raise ValueError('Labels must be 0 or 1')
    if nights.duplicated(['subject','night_of']).any():
        raise ValueError('Duplicate subject/night_of')
    glucose['glucose'] = pd.to_numeric(glucose['glucose'], errors='coerce')
    glucose = glucose[np.isfinite(glucose['glucose']) & (glucose['glucose'] > 0)]
    glucose = glucose.groupby(['subject','timestamp'], as_index=False)['glucose'].mean()
    by_subject = {s: d.sort_values('timestamp').set_index('timestamp')['glucose']
                  for s,d in glucose.groupby('subject')}
    rows = []
    for n in nights.itertuples(index=False):
        row = dict(subject=n.subject, night_of=n.night_of,
                   sleep_start=n.sleep_start, hypo_night=int(n.hypo_night))
        series = by_subject.get(n.subject)
        for hours in WINDOWS:
            suffix = f'{hours:g}h'
            for feature in BASE + EXTRA:
                row[f'{feature}_{suffix}'] = np.nan
            row[f'quality_ok_{suffix}'] = False
            row[f'n_readings_{suffix}'] = 0
            if series is None:
                continue
            start = n.sleep_start - pd.Timedelta(hours=hours)
            w = series[(series.index >= start) & (series.index < n.sleep_start)]
            row[f'n_readings_{suffix}'] = len(w)
            if len(w) < 3:
                continue
            # Fixed quality rules for nominal 5-minute CGM; do not tune on AUROC.
            gaps = w.index.to_series().diff().dt.total_seconds().dropna()/60
            first_gap = (w.index[0]-start).total_seconds()/60
            last_gap = (n.sleep_start-w.index[-1]).total_seconds()/60
            coverage = min(1., len(w)/(hours*12))
            row[f'coverage_{suffix}'] = coverage
            row[f'last_age_min_{suffix}'] = last_gap
            if coverage < .8 or first_gap > 10 or last_gap > 10 or gaps.max() > 15:
                continue
            values = w.to_numpy(float)
            elapsed = (w.index-w.index[0]).total_seconds().to_numpy()/3600
            # Duration-weighted fraction: each sample represents up to five
            # minutes following it, clipped at cutoff; gaps are not filled.
            ends = list(w.index[1:]) + [n.sleep_start]
            durations = np.minimum(np.array([(b-a).total_seconds()/60
                                             for a,b in zip(w.index, ends)]), 5.)
            features = dict(mean=values.mean(), min=values.min(),
                slope=np.polyfit(elapsed, values, 1)[0], last=values[-1],
                std=values.std(ddof=1), delta=values[-1]-values[0],
                below70_fraction=np.average(values < 70, weights=durations))
            row.update({f'{k}_{suffix}':v for k,v in features.items()})
            row[f'quality_ok_{suffix}'] = True
        rows.append(row)
    if not rows:
        raise ValueError('No nights')
    return pd.DataFrame(rows)

def estimator(algorithm, c):
    if algorithm == 'logistic':
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight='balanced', max_iter=2000, random_state=SEED))
    return RandomForestClassifier(n_estimators=100, max_depth=3,
        min_samples_leaf=5, class_weight='balanced', random_state=SEED, n_jobs=-1)

def auc(y,p):
    return roc_auc_score(y,p) if len(np.unique(y)) == 2 else np.nan

def load_precomputed_features(path):
    """Load per-night feature rows such as model_a_glucose_features.csv."""
    frame = pd.read_csv(path, dtype={'subject': str})
    required = {'subject', 'night_of', 'hypo_night'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required feature-table columns: {sorted(missing)}')
    if frame['subject'].isna().any():
        raise ValueError('Missing subject ID')
    frame['subject'] = frame['subject'].str.strip()
    if frame['subject'].eq('').any():
        raise ValueError('Empty subject ID')
    frame['hypo_night'] = pd.to_numeric(frame['hypo_night'], errors='raise')
    if frame['hypo_night'].isna().any() or not frame['hypo_night'].isin([0, 1]).all():
        raise ValueError('Labels must be present and contain only 0 or 1')
    if frame['night_of'].isna().any():
        raise ValueError('Missing night_of')
    frame['night_of'] = pd.to_datetime(frame['night_of'], errors='raise').dt.strftime('%Y-%m-%d')
    if frame.duplicated(['subject', 'night_of']).any():
        raise ValueError('Duplicate subject/night_of')
    return frame

def evaluate(frame, out, n_boot=1000, include_hrv=False):
    # Keep A as the original three-feature baseline; A_plus tests new features.
    groups = {'A': BASE, 'A_plus': BASE+EXTRA}
    if include_hrv:
        groups.update(B1=BASE+EXTRA+('rmssd',),
                      B2=BASE+EXTRA+('lf_hf',),
                      B3=BASE+EXTRA+('rmssd','lf_hf'))
    configs = {}
    for model, features in groups.items():
        candidates = []
        for hours in WINDOWS:
            cols = [f'{f}_{hours:g}h' for f in features]
            if set(cols).issubset(frame):
                for algorithm in ('logistic','forest'):
                    for c in (C_GRID if algorithm == 'logistic' else (1.,)):
                        candidates.append((hours,algorithm,c,cols))
        if candidates:
            configs[model] = candidates
    if not configs:
        raise ValueError(
            'No usable feature windows found. Expected columns such as '
            'mean_1h, min_1h, and slope_1h.'
        )
    needed = sorted({col for cs in configs.values() for *_,cols in cs for col in cols})
    frame[needed] = frame[needed].apply(pd.to_numeric, errors='raise').replace([np.inf,-np.inf],np.nan)
    before = len(frame)
    df = frame.dropna(subset=needed).sort_values(['subject','night_of']).reset_index(drop=True)
    print(f'Comparable nights: {len(df)}/{before}; patients: {df.subject.nunique()}', flush=True)
    if df.subject.nunique() < 4 or df.hypo_night.nunique() != 2:
        raise ValueError('Need >=4 retained patients and both classes')
    y, g = df.hypo_night.to_numpy(int), df.subject.to_numpy()
    predictions, choices = [], []
    for model, candidates in configs.items():
        for patient in np.unique(g):
            tr, te = np.flatnonzero(g != patient), np.flatnonzero(g == patient)
            best = None
            for hours,algorithm,c,cols in candidates:
                X = df[cols].to_numpy(float)
                inner = np.full(len(tr), np.nan)
                for held in np.unique(g[tr]):
                    train, test = tr[g[tr] != held], g[tr] == held
                    if len(np.unique(y[train])) < 2:
                        break
                    clf = estimator(algorithm,c).fit(X[train],y[train])
                    inner[test] = clf.predict_proba(X[tr[test]])[:,1]
                # Never score a candidate on a partial set of inner patients.
                if not np.isfinite(inner).all():
                    continue
                score = auc(y[tr],inner)
                if np.isfinite(score) and (best is None or score > best[0]):
                    best = (score,hours,algorithm,c,cols)
            if best is None:
                raise ValueError(f'{model}/{patient}: insufficient inner-fold classes')
            score,hours,algorithm,c,cols = best
            X = df[cols].to_numpy(float)
            clf = estimator(algorithm,c).fit(X[tr],y[tr])
            prob = clf.predict_proba(X[te])[:,1]
            predictions.append(pd.DataFrame(dict(model=model,subject=g[te],
                night_of=df.loc[te,'night_of'].to_numpy(),y_true=y[te],y_prob=prob)))
            choices.append(dict(model=model,held_out=patient,window_h=hours,
                algorithm=algorithm,C=c,inner_auc=score))
        print(f'Completed {model}', flush=True)
    pred = pd.concat(predictions,ignore_index=True)
    if pred.duplicated(['model','subject','night_of']).any():
        raise AssertionError('Duplicate held-out predictions')
    wide = pred.pivot(index=['subject','night_of','y_true'],columns='model',values='y_prob').reset_index()
    models = list(configs)
    if len(wide) != len(df) or wide[models].isna().any().any():
        raise AssertionError('Incomplete outer-fold predictions')
    by_patient = [d.index.to_numpy() for _,d in wide.groupby('subject')]
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(n_boot):
        ix = np.concatenate([by_patient[i] for i in rng.integers(len(by_patient),size=len(by_patient))])
        yy = wide.loc[ix,'y_true']
        if yy.nunique() == 2:
            boot.append([auc(yy,wide.loc[ix,m]) for m in models])
    if not boot:
        raise ValueError('No valid bootstrap samples')
    boot = np.asarray(boot)
    point = {m:auc(wide.y_true,wide[m]) for m in models}
    summary, differences = [], []
    for j,m in enumerate(models):
        lo,hi = np.quantile(boot[:,j],[.025,.975])
        summary.append(dict(model=m,auroc=point[m],ci_low=lo,ci_high=hi,
                            nights=len(wide),patients=len(by_patient),valid_bootstraps=len(boot)))
        # HRV additions must beat the enhanced glucose baseline, too.
        references = ['A'] + (['A_plus'] if m not in ('A','A_plus') else [])
        for ref in references:
            if m == ref:
                continue
            delta = boot[:,j]-boot[:,models.index(ref)]
            lo,hi = np.quantile(delta,[.025,.975])
            differences.append(dict(model=m,reference=ref,delta_auroc=point[m]-point[ref],
                                    ci_low=lo,ci_high=hi))
    out.mkdir(parents=True,exist_ok=True)
    summary, differences = pd.DataFrame(summary), pd.DataFrame(differences)
    patient_metrics = [dict(subject=s,model=m,nights=len(d),auroc=auc(d.y_true,d[m]))
                       for s,d in wide.groupby('subject') for m in models]
    for name,table in {'enhanced_features':frame,'nested_predictions':pred,
        'chosen_per_fold':pd.DataFrame(choices),'auroc':summary,'paired_improvement':differences,
        'patient_auroc':pd.DataFrame(patient_metrics)}.items():
        table.to_csv(out/f'{name}.csv',index=False)
    (out/'run_config.json').write_text(json.dumps(dict(windows=WINDOWS,C_grid=C_GRID,
        seed=SEED,bootstrap_repetitions=n_boot,input_nights=before,retained_nights=len(df),
        method='Nested patient-held-out selection; paired patient bootstrap of fixed outer predictions',
        limitations='Exploratory intervals do not include model-refitting uncertainty. Outcome completeness and HRV provenance require upstream verification.'),indent=2))
    print(summary.to_string(index=False))
    print(differences.to_string(index=False))
    return summary,pred,differences

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path.home()/'Downloads')
    parser.add_argument('--features',type=Path,
        help='Use a precomputed per-night feature CSV instead of raw CGM and label files')
    parser.add_argument('--timezone')
    parser.add_argument('--bootstraps',type=int,default=1000)
    parser.add_argument('--include-hrv',action='store_true',help='Use only after confirming HRV timing/quality')
    parser.add_argument('--output-dir',type=Path,
        help='Directory for results (defaults to a SleepGuard_enhanced_LOPO folder beside the input)')
    args = parser.parse_args()
    if args.bootstraps < 1:
        parser.error('bootstraps must be positive')
    if args.features:
        features = load_precomputed_features(args.features)
        if args.include_hrv and not any(
            c.startswith(('rmssd_', 'lf_hf_')) for c in features.columns
        ):
            parser.error('--include-hrv requires window-specific HRV columns in the feature CSV')
        input_dir = args.features.parent
    else:
        features = build_features(pd.read_csv(args.data_dir/'glucose_readings.csv',dtype={'subject':str}),
            pd.read_csv(args.data_dir/'night_labels.csv',dtype={'subject':str}),args.timezone)
        input_dir = args.data_dir
    if args.include_hrv and not args.features:
        hrv = pd.read_csv(args.data_dir/'merged_features_windows.csv',dtype={'subject':str})
        hrv['night_of'] = pd.to_datetime(hrv['night_of'],errors='raise').dt.strftime('%Y-%m-%d')
        cols = ['subject','night_of']+[c for c in hrv if c.startswith(('rmssd_','lf_hf_'))]
        if len(cols) == 2:
            raise ValueError('No window-specific HRV columns found')
        features = features.merge(hrv[cols],on=['subject','night_of'],how='left',validate='one_to_one')
    output_root = args.output_dir or input_dir/'SleepGuard_enhanced_LOPO'
    out = output_root/pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S_%f')
    evaluate(features,out,args.bootstraps,args.include_hrv)
    print('Saved results:',out)

if __name__ == '__main__':
    main()
