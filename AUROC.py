"""
Nocturnal hypoglycemia forecasting: glucose baseline vs glucose + HRV
Leave-one-patient-out (LOPO) evaluation with patient-level bootstrap,
per-patient AUROC, permutation test, and two robustness checks.
 
Input : one row per participant-night
        required columns: subject, night_of, hypo_night,
                          mean_{w}h, slope_{w}h, min_{w}h   (w = 1 and 2)
        optional columns: rmssd_{w}h, lf_hf_{w}h            (HRV, for Models B1-B3)
                          ecg_good (1 = good-quality ECG night)
Tables searched in Downloads: merged_features_windows.csv, then model_a_glucose_features.csv
Run   : python hypo_forecast_full.py
"""
import os
import warnings
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
 
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
warnings.filterwarnings("ignore")
 
# =====================================================================
# 1. SETTINGS
# =====================================================================
DOWNLOADS = Path(os.environ.get("HYPO_DIR", Path.home() / "Downloads"))
MERGED = DOWNLOADS / "merged_features_windows.csv"
GLUCOSE_ONLY = DOWNLOADS / "model_a_glucose_features.csv"
 
GROUP, LABEL, NIGHT = "subject", "hypo_night", "night_of"
W, W_ALT = "2", "1"                 # primary pre-sleep window (hours) and the alternative
HISTORY = ["prev_night_hypo", "prior_hypo_rate", "day_low_events", "day_min_glucose"]
 
THRESH, MIN_EVENT_MIN = 70, 15      # only used for the optional daytime-low features
MAX_GAP = pd.Timedelta(minutes=15)
DAY_START_HOUR, BEDTIME_HOUR = 10, 22
MIN_PRIOR_NIGHTS_Z = 3              # earlier nights needed before an HRV z-score exists
MIN_PER_CLASS = 3                   # nights per class for a patient's AUROC to count as reliable
N_BOOT, N_PERM, SEED = 1000, 200, 42
 
 
def header(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
 
 
# =====================================================================
# 2. LOAD TABLE
# =====================================================================
TABLE = MERGED if MERGED.exists() else GLUCOSE_ONLY
if not TABLE.exists():
    raise FileNotFoundError(
        f"Neither {MERGED.name} nor {GLUCOSE_ONLY.name} was found in {DOWNLOADS}.\n"
        "Run the Model A feature script first so model_a_glucose_features.csv exists.")
 
df = pd.read_csv(TABLE)
for col in (GROUP, LABEL, NIGHT):
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' is not in {TABLE.name}. Columns: {list(df.columns)}")
df[NIGHT] = pd.to_datetime(df[NIGHT])
df = df.sort_values([GROUP, NIGHT]).reset_index(drop=True)
print(f"Using table: {TABLE}  ({len(df)} nights)")
 
 
# =====================================================================
# 3. STRONGER BASELINE: history features available at bedtime
# =====================================================================
def load_cgm(zip_path):
    with ZipFile(zip_path) as z:
        files = [n for n in z.namelist() if n.lower().endswith("cgm.csv")]
        if len(files) != 1:
            return None
        cgm = pd.read_csv(BytesIO(z.read(files[0])), low_memory=False)
    cgm = cgm[cgm["Event Type"] == "EGV"].copy()
    cgm["time"] = pd.to_datetime(cgm["Timestamp (YYYY-MM-DDThh:mm:ss)"])
    cgm["glucose"] = pd.to_numeric(
        cgm["Glucose Value (mg/dL)"].replace({"Low": 39, "High": 401}), errors="coerce")
    return cgm.dropna(subset=["glucose"]).sort_values("time").reset_index(drop=True)[["time", "glucose"]]
 
 
def low_event_starts(cgm):
    """Start times of all >= 15-minute stretches below 70 mg/dL (any time of day)."""
    low = cgm["glucose"] < THRESH
    run = ((low != low.shift()) | (cgm["time"].diff() > MAX_GAP)).cumsum()
    lows = cgm[low].assign(run=run[low])
    if lows.empty:
        return pd.Series([], dtype="datetime64[ns]")
    ev = lows.groupby("run").agg(start=("time", "min"), end=("time", "max"))
    dur = (ev["end"] - ev["start"]).dt.total_seconds() / 60 + 5
    return ev.loc[dur >= MIN_EVENT_MIN, "start"].reset_index(drop=True)
 
 
# 3a. Daytime lows (10am to 10pm bedtime on the same date), from the PhysioCGM zips if present
if not {"day_low_events", "day_min_glucose"}.issubset(df.columns):
    day_rows = []
    for zp in sorted(DOWNLOADS.glob("c*s*_raw.zip")):
        subj = zp.name.removesuffix("_raw.zip")
        cgm = load_cgm(zp)
        if cgm is None:
            continue
        starts = low_event_starts(cgm)
        for d in df.loc[df[GROUP] == subj, NIGHT]:
            t0 = d + pd.Timedelta(hours=DAY_START_HOUR)
            t1 = d + pd.Timedelta(hours=BEDTIME_HOUR)
            day = cgm[(cgm["time"] >= t0) & (cgm["time"] < t1)]
            day_rows.append({GROUP: subj, NIGHT: d,
                             "day_low_events": int(((starts >= t0) & (starts < t1)).sum()),
                             "day_min_glucose": day["glucose"].min() if len(day) else np.nan})
    if day_rows:
        df = df.merge(pd.DataFrame(day_rows), on=[GROUP, NIGHT], how="left")
        print("Daytime-low features added from the PhysioCGM zips.")
    else:
        print("No PhysioCGM zips found: daytime-low features skipped (other history features still used).")
 
# 3b. Prior-night low, and the patient's own hypo rate from EARLIER nights only
grp_label = df.groupby(GROUP)[LABEL]
prev_date = df.groupby(GROUP)[NIGHT].shift(1)
df["prev_night_hypo"] = grp_label.shift(1).where((df[NIGHT] - prev_date) == pd.Timedelta(days=1))
df["prior_hypo_rate"] = grp_label.transform(lambda s: s.shift(1).expanding().mean())
 
# =====================================================================
# 4. PER-PATIENT HRV NORMALIZATION (z-score vs that patient's EARLIER nights only)
# =====================================================================
for w in (W, W_ALT):
    for name in ("rmssd", "lf_hf"):
        col = f"{name}_{w}h"
        if col in df.columns:
            grp = df.groupby(GROUP)[col]
            mu = grp.transform(lambda s: s.shift(1).expanding(min_periods=MIN_PRIOR_NIGHTS_Z).mean())
            sd = grp.transform(lambda s: s.shift(1).expanding(min_periods=MIN_PRIOR_NIGHTS_Z).std())
            df[f"{col}_z"] = (df[col] - mu) / sd.replace(0, np.nan)
 
 
# =====================================================================
# 5. MODELS
#    A0 = original glucose-only baseline (mean, slope, min)
#    A  = strengthened baseline (A0 + history features)
#    B1 / B2 / B3 = A + patient-normalized RMSSD / LF-HF / both
# =====================================================================
def model_defs(w):
    glucose = [f"mean_{w}h", f"slope_{w}h", f"min_{w}h"]
    base = glucose + [c for c in HISTORY if c in df.columns]
    defs = {"A0": glucose, "A": base}
    r, l = f"rmssd_{w}h_z", f"lf_hf_{w}h_z"
    if r in df.columns:
        defs["B1"] = base + [r]
    if l in df.columns:
        defs["B2"] = base + [l]
    if r in df.columns and l in df.columns:
        defs["B3"] = base + [r, l]
    return {m: c for m, c in defs.items() if all(x in df.columns for x in c)}
 
 
models = model_defs(W)
core = [f"{k}_{w}h" for w in (W, W_ALT) for k in ("mean", "slope", "min") if f"{k}_{w}h" in df.columns]
if f"mean_{W}h" not in df.columns:
    raise KeyError(f"Column mean_{W}h not found. Rerun the Model A feature script with PRE_HOURS = [1, 2].")
df = df.dropna(subset=core + [LABEL]).reset_index(drop=True)
 
y = df[LABEL].astype(int).to_numpy()
groups = df[GROUP].to_numpy()
patients = sorted(np.unique(groups))
idx = {s: np.where(groups == s)[0] for s in patients}
B_MODELS = [m for m in models if m.startswith("B")]
 
 
# =====================================================================
# 6. LEAVE-ONE-PATIENT-OUT
# =====================================================================
def make_clf():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000))
 
 
def lopo(X, yy, gg):
    """Held-out probability for every night; the scaler/imputer see training patients only."""
    pred = np.full(len(yy), np.nan)
    for p in np.unique(gg):
        te = gg == p
        if len(np.unique(yy[~te])) < 2:
            continue
        pred[te] = make_clf().fit(X[~te], yy[~te]).predict_proba(X[te])[:, 1]
    return pred
 
 
def auc(yy, p):
    return roc_auc_score(yy, p) if len(np.unique(yy)) == 2 else np.nan
 
 
P = {m: lopo(df[c].to_numpy(float), y, groups) for m, c in models.items()}
for m, p in P.items():
    if np.isnan(p).any():
        raise RuntimeError(f"Model {m}: a fold had a single-class training set")
pooled = {m: auc(y, p) for m, p in P.items()}
 
pd.DataFrame({
    "model": [m for m in P for _ in range(len(y))],
    "subject": np.tile(groups, len(P)),
    "night_of": np.tile(df[NIGHT].dt.date.to_numpy(), len(P)),
    "y_true": np.tile(y, len(P)),
    "y_prob": np.concatenate(list(P.values())),
}).to_csv(DOWNLOADS / "lopo_predictions_v2.csv", index=False)
 
# =====================================================================
# 7. REPORT
# =====================================================================
header("1. DATASET AND MODELS")
print(f"Nights: {len(df)}   Patients: {len(patients)}   Hypo nights: {int(y.sum())} ({100 * y.mean():.0f}%)")
print("\nFeature sets:")
desc = {"A0": "glucose only (original baseline)", "A": "glucose + history (stronger baseline)",
        "B1": "A + RMSSD", "B2": "A + LF/HF", "B3": "A + RMSSD + LF/HF"}
for m, cols in models.items():
    print(f"  {m:<3} {desc[m]:<38} {cols}")
if not B_MODELS:
    print("\nNo HRV columns found (rmssd_2h, lf_hf_2h, ...), so only the glucose baselines run.")
 
# ---- patient-level bootstrap (resample whole patients: nights within a patient are correlated)
rng = np.random.default_rng(SEED)
boot = {m: {"auc": [], "ap": []} for m in models}
for _ in range(N_BOOT):
    rows = np.concatenate([idx[s] for s in rng.choice(patients, len(patients), replace=True)])
    if len(np.unique(y[rows])) < 2:
        continue
    for m in models:
        boot[m]["auc"].append(roc_auc_score(y[rows], P[m][rows]))
        boot[m]["ap"].append(average_precision_score(y[rows], P[m][rows]))
n_used = len(boot["A"]["auc"])
ci = lambda a: np.percentile(a, [2.5, 97.5])
 
header(f"2. OVERALL PERFORMANCE  (95% CI from {n_used} patient-bootstrap resamples; "
       f"chance: AUROC 0.50, PR-AUC {y.mean():.2f})")
overall = pd.DataFrame(index=models)
for m in models:
    a, p = ci(boot[m]["auc"]), ci(boot[m]["ap"])
    overall.loc[m, "AUROC (95% CI)"] = f"{pooled[m]:.3f}  ({a[0]:.3f} to {a[1]:.3f})"
    overall.loc[m, "PR-AUC (95% CI)"] = f"{average_precision_score(y, P[m]):.3f}  ({p[0]:.3f} to {p[1]:.3f})"
print(overall.to_string())
 
pairs = [("A", "A0")] + [(b, "A") for b in B_MODELS]
header("3. DOES THE EXTRA INFORMATION HELP?  (change in AUROC, paired patient bootstrap)")
rows, verdicts = [], {}
for m, ref in pairs:
    d = np.array(boot[m]["auc"]) - np.array(boot[ref]["auc"])
    dp = np.array(boot[m]["ap"]) - np.array(boot[ref]["ap"])
    lo, hi = ci(d)
    verdicts[m] = (lo, hi)
    rows.append({"Comparison": f"{m} vs {ref}",
                 "AUROC change": f"{pooled[m] - pooled[ref]:+.3f}",
                 "95% CI": f"{lo:+.3f} to {hi:+.3f}",
                 "PR-AUC change": f"{average_precision_score(y, P[m]) - average_precision_score(y, P[ref]):+.3f}",
                 "PR 95% CI": f"{ci(dp)[0]:+.3f} to {ci(dp)[1]:+.3f}",
                 "Verdict": "clear gain" if lo > 0 else "clearly worse" if hi < 0 else "no clear difference"})
print(pd.DataFrame(rows).set_index("Comparison").to_string())
print("\n'A vs A0' shows whether the history features strengthened the baseline.")
 
# ---- per patient
header(f"4. PER HELD-OUT PATIENT  (AUROC; reliable = at least {MIN_PER_CLASS} nights in each class)")
prow = []
for s in patients:
    i = idx[s]
    n_pos, n_neg = int(y[i].sum()), int((y[i] == 0).sum())
    row = {"Patient": s, "Nights": len(i), "Hypo": n_pos, "No hypo": n_neg}
    for m in models:
        row[m] = round(auc(y[i], P[m][i]), 3) if n_pos and n_neg else np.nan
    for m in B_MODELS:
        row[f"{m}-A"] = round(row[m] - row["A"], 3) if n_pos and n_neg else np.nan
    row["Reliable?"] = ("yes" if min(n_pos, n_neg) >= MIN_PER_CLASS
                        else "no: one class" if not (n_pos and n_neg) else "no: few in one class")
    prow.append(row)
per = pd.DataFrame(prow).set_index("Patient").sort_values("A", ascending=False)
print(per.to_string())
 
stable = per[per["Reliable?"] == "yes"]
header("5. KEY NUMBERS (reliable patients only)")
print(f"Reliable patients: {len(stable)} of {len(per)}")
for m in models:
    print(f"  Model {m:<3} median AUROC {stable[m].median():.3f}   above 0.5 in "
          f"{int((stable[m] > 0.5).sum())} of {len(stable)} patients")
for m in B_MODELS:
    d = stable[f"{m}-A"]
    print(f"  Model {m:<3} vs A: better in {int((d > 0).sum())}, worse in {int((d < 0).sum())} of {len(d)}"
          f"   non-responders: {list(d[d <= 0].index)}")
 
# ---- permutation test (leakage check)
header(f"6. PERMUTATION TEST  ({N_PERM} label shuffles, full LOPO pipeline re-run each time)")
rng_p = np.random.default_rng(SEED)
prow = []
for m in ["A"] + (["B3"] if "B3" in models else []):
    X = df[models[m]].to_numpy(float)
    for within, label in [(False, "all nights shuffled"), (True, "shuffled within each patient")]:
        null = []
        for _ in range(N_PERM):
            yp = y.copy()
            if within:
                for s in patients:
                    yp[idx[s]] = rng_p.permutation(y[idx[s]])
            else:
                yp = rng_p.permutation(y)
            null.append(auc(yp, lopo(X, yp, groups)))
        null = np.array(null)
        prow.append({"Model": m, "Shuffle": label, "Observed": round(pooled[m], 3),
                     "Null mean": round(null.mean(), 3), "Null 95th pct": round(np.percentile(null, 95), 3),
                     "p-value": round((1 + (null >= pooled[m]).sum()) / (1 + N_PERM), 3)})
print(pd.DataFrame(prow).set_index(["Model", "Shuffle"]).to_string())
print("\nAll-nights shuffle should give a null mean near 0.50; if it doesn't, look for leakage.\n"
      "The within-patient shuffle keeps each person's hypo rate, so its null can sit above 0.50\n"
      "(pooled AUROC also rewards telling patients apart). A result well above it is within-person signal.")
 
# ---- robustness 1: different pre-sleep window
header(f"7. ROBUSTNESS CHECK 1: DIFFERENT PRE-SLEEP WINDOW ({W_ALT} h vs {W} h)")
models_alt = model_defs(W_ALT)
P_alt = {m: lopo(df[c].to_numpy(float), y, groups) for m, c in models_alt.items()}
alt = pd.DataFrame({f"AUROC {W} h": pd.Series(pooled), f"AUROC {W_ALT} h": pd.Series({m: auc(y, p) for m, p in P_alt.items()})})
for b in B_MODELS:
    if b in P_alt:
        alt.loc[b, f"change vs A ({W} h)"] = pooled[b] - pooled["A"]
        alt.loc[b, f"change vs A ({W_ALT} h)"] = auc(y, P_alt[b]) - auc(y, P_alt["A"])
print(alt.round(3).fillna("").to_string())
print("If the sign of the B-vs-A change flips between windows, treat any HRV effect as fragile.")
 
# ---- robustness 2: good-quality ECG only
header("8. ROBUSTNESS CHECK 2: GOOD-QUALITY ECG NIGHTS ONLY")
if "ecg_good" in df.columns:
    keep = (df["ecg_good"] == 1).to_numpy()
    yk, gk = y[keep], groups[keep]
    print(f"Nights kept: {keep.sum()} of {len(df)}   hypo nights: {int(yk.sum())}")
    Pk = {m: lopo(df.loc[keep, models[m]].to_numpy(float), yk, gk) for m in models}
    good = pd.DataFrame({"AUROC": pd.Series({m: auc(yk, p) for m, p in Pk.items()})})
    for b in B_MODELS:
        good.loc[b, "change vs A"] = auc(yk, Pk[b]) - auc(yk, Pk["A"])
    print(good.round(3).fillna("").to_string())
else:
    print("No 'ecg_good' column in the table, so this check was skipped.\n"
          "Add one (1 = good night) from your R-peak pipeline. Fix the quality cutoff BEFORE looking\n"
          "at results, or the check becomes another choice made after seeing the data.")
 
# ---- conclusion
header("9. CONCLUSION")
print(f"Glucose baseline (A0): AUROC {pooled['A0']:.3f}.  Strengthened baseline (A): AUROC {pooled['A']:.3f}.")
if not B_MODELS:
    print("HRV models have not been run yet, so there is no HRV conclusion.")
for b in B_MODELS:
    lo, hi = verdicts[b]
    d = stable[f"{b}-A"]
    if lo > 0:
        msg = "HRV adds value: the interval for the AUROC gain excludes zero."
    elif hi < 0:
        msg = "HRV lowers performance: the interval is entirely below zero."
    else:
        msg = ("No detectable benefit in this sample, which was too small to rule out a modest effect "
               "(the interval spans zero).")
    print(f"Model {b} vs A: {pooled[b] - pooled['A']:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}). {msg}")
    if (d > 0).any() and (d < 0).any():
        print(f"   Per-patient results are mixed: better in {int((d > 0).sum())}, worse in "
              f"{int((d < 0).sum())} of {len(d)}. Non-responders: {list(d[d <= 0].index)}")
 
# ---- save
per.to_csv(DOWNLOADS / "per_patient_auroc_v2.csv")
overall.to_csv(DOWNLOADS / "overall_performance_v2.csv")
try:
    with pd.ExcelWriter(DOWNLOADS / "results_summary_v2.xlsx") as xw:
        overall.to_excel(xw, sheet_name="overall")
        pd.DataFrame(rows).to_excel(xw, sheet_name="comparisons", index=False)
        per.to_excel(xw, sheet_name="per_patient")
    print("\nSaved: lopo_predictions_v2.csv, per_patient_auroc_v2.csv, overall_performance_v2.csv, "
          "results_summary_v2.xlsx in", DOWNLOADS)
except ImportError:
    print("\nSaved: lopo_predictions_v2.csv, per_patient_auroc_v2.csv, overall_performance_v2.csv in", DOWNLOADS,
          "\n(pip install openpyxl to also get an Excel summary)")
 
 
# =====================================================================
# ADDED: HRV vs GLUCOSE COMPARISON
# Paste this at the very end of the script. Nothing above it is changed.
# Needs rmssd_{W}h and lf_hf_{W}h in merged_features_windows.csv; if they are
# missing it prints a message and skips, so the rest of the script is unaffected.
# =====================================================================
def hrv_glucose_comparison(min_per_class=3):
    gl = [f"mean_{W}h", f"slope_{W}h", f"min_{W}h"]              # glucose features
    hv = [f"rmssd_{W}h", f"lf_hf_{W}h"]                           # HRV features (raw)
    hz = [f"{c}_z" for c in hv]                                   # HRV, patient-normalized (used by models)
    nice = {gl[0]: "Glucose mean", gl[1]: "Glucose slope", gl[2]: "Glucose minimum",
            hv[0]: "HRV: RMSSD", hv[1]: "HRV: LF/HF"}
 
    header("ADDED: HRV vs GLUCOSE COMPARISON")
    missing = [c for c in gl + hv + hz if c not in df.columns]
    if missing:
        print(f"Skipped: columns not found: {missing}\n"
              f"HRV needs rmssd_{W}h and lf_hf_{W}h in merged_features_windows.csv.")
        return
 
    feats = gl + hv
    sub = df.dropna(subset=feats).reset_index(drop=True)          # nights with every raw feature present
    ys = sub[LABEL].astype(int).to_numpy()
    gs = sub[GROUP].to_numpy()
    idx_s = {s: np.where(gs == s)[0] for s in sorted(np.unique(gs))}
    eligible = [s for s, i in idx_s.items()
                if min(int(ys[i].sum()), int((ys[i] == 0).sum())) >= min_per_class]
    print(f"Window: {W} h before bedtime   Nights with every feature: {len(sub)} of {len(df)}   "
          f"Hypo nights: {int(ys.sum())}")
    print(f"Patients with at least {min_per_class} hypo and {min_per_class} non-hypo nights "
          f"(used for within-patient numbers): {len(eligible)} -> {eligible}")
    rng_h = np.random.default_rng(SEED)
 
    # ---- ADDED 1: features on hypo vs non-hypo nights, side by side ----
    def within_effects(col):
        out = []
        for s in eligible:
            v, lab = sub[col].to_numpy()[idx_s[s]], ys[idx_s[s]]
            sd = v.std(ddof=1)
            if sd > 0:
                out.append((v[lab == 1].mean() - v[lab == 0].mean()) / sd)
        return np.array(out)
 
    header("ADDED 1. HYPO vs NON-HYPO NIGHTS: GLUCOSE AND HRV SIDE BY SIDE")
    rows = []
    for col in feats:
        eff = within_effects(col)
        bs = [rng_h.choice(eff, len(eff), replace=True).mean() for _ in range(N_BOOT)] if len(eff) else [np.nan]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        hy, nh = sub.loc[ys == 1, col], sub.loc[ys == 0, col]
        rows.append({"Feature": nice[col], "No hypo (mean)": nh.mean(), "Hypo (mean)": hy.mean(),
                     "Difference": hy.mean() - nh.mean(),
                     "Within-patient effect (SD units)": eff.mean() if len(eff) else np.nan,
                     "95% CI": f"{lo:+.2f} to {hi:+.2f}", "Clear?": "yes" if (lo > 0 or hi < 0) else "no"})
    t1 = pd.DataFrame(rows).set_index("Feature")
    print(t1.round(2).to_string())
    print("\nWithin-patient effect: how many of that patient's own standard deviations the feature moves on hypo "
          "nights (negative = lower before hypo nights).\n'Clear?' = the 95% CI, resampling patients, excludes 0. "
          "Pooled differences mix people together; the within-patient column removes who the person is.")
 
    # ---- ADDED 2: each feature on its own (no model fitting, so no leakage) ----
    header("ADDED 2. HOW WELL EACH FEATURE ALONE SEPARATES HYPO NIGHTS (AUROC; 0.50 = no information)")
    rows = []
    for col in feats:
        x = sub[col].to_numpy()
        per = [roc_auc_score(ys[idx_s[s]], x[idx_s[s]]) for s in eligible]
        med = np.median(per) if per else np.nan
        up = med > 0.5
        rows.append({"Feature": nice[col], "Pooled AUROC": roc_auc_score(ys, x),
                     "Within-patient median AUROC": med,
                     "Within-patient direction": "higher -> hypo" if up else "lower -> hypo",
                     "Patients in that direction": f"{sum((a > 0.5) == up for a in per)} of {len(per)}",
                     "Strength": abs(med - 0.5)})
    t2 = pd.DataFrame(rows).set_index("Feature").sort_values("Strength", ascending=False)
    print(t2.round(3).to_string())
    print("\nStrength = distance of the within-patient median AUROC from 0.5. If the pooled and within-patient "
          "AUROCs sit on opposite sides of 0.5,\nthe pooled number is mostly differences between people, so "
          "trust the within-patient column.")
 
    # ---- ADDED 3: do HRV and glucose move together? ----
    def corr_table(frame):
        return pd.DataFrame({nice[c]: [frame[h].corr(frame[c], method="spearman") for h in hv] for c in gl},
                            index=[nice[h] for h in hv])
 
    header("ADDED 3. HOW HRV RELATES TO GLUCOSE (Spearman correlation; rows = HRV, columns = glucose)")
    centered = sub[feats] - sub.groupby(GROUP)[feats].transform("mean")
    print("All nights pooled:\n" + corr_table(sub).round(2).to_string())
    print("\nWithin patients (each patient's average removed):\n" + corr_table(centered).round(2).to_string())
    print("\nNear 0 = HRV carries different information from glucose, so it has room to add something. "
          "Near +/-1 = mostly redundant.")
 
    # ---- ADDED 4: glucose only vs HRV only vs combined (leave-one-patient-out) ----
    M = {"G": gl, "H1": [hz[0]], "H2": [hz[1]], "H3": hz,
         "G+H1": gl + [hz[0]], "G+H2": gl + [hz[1]], "G+H3": gl + hz}
    meaning = {"G": "glucose only (mean, slope, min)", "H1": "HRV only: RMSSD", "H2": "HRV only: LF/HF",
               "H3": "HRV only: RMSSD + LF/HF", "G+H1": "glucose + RMSSD", "G+H2": "glucose + LF/HF",
               "G+H3": "glucose + RMSSD + LF/HF"}
    Ph = {m: lopo(df[c].to_numpy(float), y, groups) for m, c in M.items()}
    for m, p in Ph.items():
        if np.isnan(p).any():
            raise RuntimeError(f"Model {m}: some folds had a single-class training set")
    auc_h = {m: auc(y, p) for m, p in Ph.items()}
    idx_a = {s: np.where(groups == s)[0] for s in patients}
    bt = {m: {"auc": [], "ap": []} for m in Ph}
    for _ in range(N_BOOT):
        rows_ = np.concatenate([idx_a[s] for s in rng_h.choice(patients, len(patients), replace=True)])
        if len(np.unique(y[rows_])) < 2:
            continue
        for m in Ph:
            bt[m]["auc"].append(roc_auc_score(y[rows_], Ph[m][rows_]))
            bt[m]["ap"].append(average_precision_score(y[rows_], Ph[m][rows_]))
    ci_h = lambda a: np.percentile(a, [2.5, 97.5])
 
    header(f"ADDED 4. MODEL COMPARISON, HELD-OUT PATIENTS (95% CI from {len(bt['G']['auc'])} patient-bootstrap "
           f"resamples; chance AUROC 0.50, PR-AUC {y.mean():.2f})")
    rows = []
    for m in M:
        a, p = ci_h(bt[m]["auc"]), ci_h(bt[m]["ap"])
        rows.append({"Model": m, "Features": meaning[m],
                     "AUROC (95% CI)": f"{auc_h[m]:.3f}  ({a[0]:.3f} to {a[1]:.3f})",
                     "PR-AUC (95% CI)": f"{average_precision_score(y, Ph[m]):.3f}  ({p[0]:.3f} to {p[1]:.3f})"})
    t4 = pd.DataFrame(rows).set_index("Model")
    print(t4.to_string())
 
    header("ADDED 5. HEAD-TO-HEAD (change in AUROC, paired patient bootstrap)")
    q = {"H1": "Does RMSSD alone match glucose?", "H2": "Does LF/HF alone match glucose?",
         "H3": "Does HRV alone match glucose?", "G+H1": "Does RMSSD add to glucose?",
         "G+H2": "Does LF/HF add to glucose?", "G+H3": "Does HRV add to glucose?"}
    pairs = [("H1", "G"), ("H2", "G"), ("H3", "G"), ("G+H1", "G"), ("G+H2", "G"), ("G+H3", "G"), ("G+H3", "H3")]
    rows, verdict = [], {}
    for m, ref in pairs:
        d = np.array(bt[m]["auc"]) - np.array(bt[ref]["auc"])
        lo, hi = ci_h(d)
        v = "clearly better" if lo > 0 else "clearly worse" if hi < 0 else "no clear difference"
        verdict[(m, ref)] = (auc_h[m] - auc_h[ref], lo, hi, v)
        rows.append({"Comparison": f"{m} vs {ref}", "Question": q[m] if ref == "G" else "Does glucose add to HRV?",
                     "AUROC change": f"{auc_h[m] - auc_h[ref]:+.3f}", "95% CI": f"{lo:+.3f} to {hi:+.3f}",
                     "Verdict": v})
    t5 = pd.DataFrame(rows).set_index("Comparison")
    print(t5.to_string())
 
    header(f"ADDED 6. PER HELD-OUT PATIENT AUROC: GLUCOSE vs HRV vs COMBINED "
           f"(reliable = at least {min_per_class} nights per class)")
    rows = []
    for s in patients:
        i = idx_a[s]
        n_pos, n_neg = int(y[i].sum()), int((y[i] == 0).sum())
        ok = n_pos > 0 and n_neg > 0
        row = {"Patient": s, "Nights": len(i), "Hypo": n_pos, "No hypo": n_neg}
        for m in ("G", "H3", "G+H3"):
            row[m] = round(auc(y[i], Ph[m][i]), 3) if ok else np.nan
        row["H3 - G"] = round(row["H3"] - row["G"], 3) if ok else np.nan
        row["G+H3 - G"] = round(row["G+H3"] - row["G"], 3) if ok else np.nan
        row["Reliable?"] = ("yes" if min(n_pos, n_neg) >= min_per_class
                            else "no: one class" if not ok else "no: few in one class")
        rows.append(row)
    t6 = pd.DataFrame(rows).set_index("Patient").sort_values("G", ascending=False)
    print(t6.to_string())
    st = t6[t6["Reliable?"] == "yes"]
    print(f"\nReliable patients: {len(st)} of {len(t6)}")
    for m in ("G", "H3", "G+H3"):
        print(f"  {m:<5} median AUROC {st[m].median():.3f}   above 0.5 in {int((st[m] > 0.5).sum())} of {len(st)}")
    for c, lab in [("H3 - G", "HRV alone vs glucose alone"), ("G+H3 - G", "glucose + HRV vs glucose alone")]:
        dd = st[c]
        print(f"  {lab}: better in {int((dd > 0).sum())}, worse in {int((dd < 0).sum())} of {len(dd)}")
 
    header("ADDED SUMMARY")
    clear = t1[t1["Clear?"] == "yes"]
    print("Features with a clear within-patient difference between hypo and non-hypo nights:",
          list(clear.index) if len(clear) else "none")
    for key, txt in [(("H3", "G"), "HRV alone vs glucose alone"), (("G+H3", "G"), "Adding HRV to glucose")]:
        chg, lo, hi, v = verdict[key]
        print(f"{txt}: AUROC change {chg:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}) -> {v}.")
    print("\nWith this few patients and hypo nights, 'no clear difference' means the sample cannot separate the "
          "models, not that they are equal.")
 
    t1.round(3).to_csv(DOWNLOADS / "compare_feature_differences.csv")
    t2.round(3).to_csv(DOWNLOADS / "compare_single_feature_auroc.csv")
    t6.to_csv(DOWNLOADS / "compare_per_patient_auroc.csv")
    try:
        with pd.ExcelWriter(DOWNLOADS / "hrv_vs_glucose.xlsx") as xw:
            t1.round(3).to_excel(xw, sheet_name="hypo_vs_nonhypo")
            t2.round(3).to_excel(xw, sheet_name="single_feature_auroc")
            corr_table(sub).round(3).to_excel(xw, sheet_name="corr_pooled")
            corr_table(centered).round(3).to_excel(xw, sheet_name="corr_within_patient")
            t4.to_excel(xw, sheet_name="models")
            t5.to_excel(xw, sheet_name="head_to_head")
            t6.to_excel(xw, sheet_name="per_patient")
        print("\nSaved: hrv_vs_glucose.xlsx and three compare_*.csv files in", DOWNLOADS)
    except ImportError:
        print("\nSaved three compare_*.csv files in", DOWNLOADS, "(pip install openpyxl to also get an Excel file)")
 
 
hrv_glucose_comparison()
 