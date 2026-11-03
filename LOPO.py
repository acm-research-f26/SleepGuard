from pathlib import Path


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler




# =====================================================================
# 1. SETTINGS
# =====================================================================
DOWNLOADS = Path.home() / "Downloads"


MERGED = DOWNLOADS / "merged_features.csv"
GLUCOSE_ONLY = DOWNLOADS / "model_a_glucose_features.csv"
TABLE = MERGED if MERGED.exists() else GLUCOSE_ONLY


GROUP = "subject"
LABEL = "hypo_night"
NIGHT = "night_of"


GLUCOSE = ["mean_1h", "slope_1h", "min_1h"]


MODELS = {
    "A": GLUCOSE,
    "B1": GLUCOSE + ["rmssd"],
    "B2": GLUCOSE + ["lf_hf"],
    "B3": GLUCOSE + ["rmssd", "lf_hf"],
}


output = []




def show(text=""):
    """Print output and retain it for the full output text file."""
    text = str(text)
    print(text)
    output.append(text)




# =====================================================================
# 2. LOAD AND VALIDATE
# =====================================================================
if not TABLE.exists():
    raise FileNotFoundError(
        f"Put {GLUCOSE_ONLY.name} or {MERGED.name} in {DOWNLOADS}"
    )


df = pd.read_csv(TABLE)


show(f"Using table: {TABLE}")
show(f"Columns found: {list(df.columns)}")


for col in [GROUP, LABEL, NIGHT]:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")


runnable = {}


for name, cols in MODELS.items():
    missing = [col for col in cols if col not in df.columns]


    if missing:
        show(f"Skipping model {name}: missing columns {missing}")
    else:
        runnable[name] = cols


if not runnable:
    raise ValueError("No model has all required feature columns.")


feature_cols = sorted({
    col
    for cols in runnable.values()
    for col in cols
})


df[feature_cols] = df[feature_cols].apply(
    pd.to_numeric, errors="raise"
)
df[LABEL] = pd.to_numeric(df[LABEL], errors="raise")


df[feature_cols] = df[feature_cols].replace(
    [np.inf, -np.inf], np.nan
)


# Use the same eligible nights for every runnable model.
before = len(df)


df = df.dropna(
    subset=feature_cols + [GROUP, LABEL, NIGHT]
).reset_index(drop=True)


if not df[LABEL].isin([0, 1]).all():
    raise ValueError("hypo_night must contain only 0 and 1.")


if df.duplicated([GROUP, NIGHT]).any():
    raise ValueError("Input contains duplicate subject/night rows.")


if df[GROUP].nunique() < 2:
    raise ValueError("LOPO requires at least two patients.")


show(f"\nModels: {list(runnable)}")
show(f"Nights in input: {before}")
show(f"Eligible nights: {len(df)}")
show(f"Dropped nights: {before - len(df)}")
show(f"Patients: {df[GROUP].nunique()}")
show(
    f"Hypoglycemic nights: {int(df[LABEL].sum())} "
    f"({df[LABEL].mean():.1%})"
)




# =====================================================================
# 3. LEAVE-ONE-PATIENT-OUT VALIDATION
# =====================================================================
y = df[LABEL].astype(int).to_numpy()
groups = df[GROUP].to_numpy()


logo = LeaveOneGroupOut()


prediction_tables = []
fold_records = []


for model_name, cols in runnable.items():
    X = df[cols].to_numpy(dtype=float)


    show(f"\nRunning model {model_name}: {cols}")


    for fold_number, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups),
        start=1,
    ):
        held_out = groups[test_idx][0]


        training_patients = set(groups[train_idx])
        testing_patients = set(groups[test_idx])


        # The held-out patient must never appear in training.
        assert training_patients.isdisjoint(testing_patients)
        assert len(testing_patients) == 1


        if len(np.unique(y[train_idx])) < 2:
            raise ValueError(
                f"Model {model_name}, patient {held_out}: "
                "training data contains only one class."
            )


        # Fit both scaling and regression using training patients only.
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
            ),
        )


        clf.fit(X[train_idx], y[train_idx])


        probabilities = clf.predict_proba(X[test_idx])[:, 1]


        prediction_tables.append(
            pd.DataFrame({
                "model": model_name,
                "subject": groups[test_idx],
                "night_of": df.loc[test_idx, NIGHT].to_numpy(),
                "y_true": y[test_idx],
                "y_prob": probabilities,
            })
        )


        fold_records.append({
            "model": model_name,
            "fold": fold_number,
            "held_out_subject": held_out,
            "training_patients": len(training_patients),
            "training_nights": len(train_idx),
            "test_nights": len(test_idx),
            "test_hypo_nights": int(y[test_idx].sum()),
            "patient_overlap": 0,
        })


        show(
            f"Fold {fold_number}: held out {held_out}; "
            f"trained on {len(training_patients)} patients "
            f"({len(train_idx)} nights); "
            f"predicted {len(test_idx)} nights"
        )


preds = pd.concat(prediction_tables, ignore_index=True)
folds = pd.DataFrame(fold_records)




# =====================================================================
# 4. VERIFY LOPO COVERAGE
# =====================================================================
duplicates = int(
    preds.duplicated(["model", "subject", "night_of"]).sum()
)


counts = preds.groupby("model").size()


assert duplicates == 0, "Duplicate predictions detected."
assert (counts == len(df)).all(), "Some nights lack predictions."


show("\nLOPO FOLD DETAILS:")
show(folds.to_string(index=False))


show("\nPREDICTIONS PER MODEL:")
show(counts.to_string())


show(f"\nDuplicate predictions: {duplicates}")
show("Every eligible night was predicted exactly once per model.")


show("\nMEAN PREDICTED PROBABILITY BY TRUE LABEL:")
show(
    preds.groupby(["model", "y_true"])["y_prob"]
    .mean()
    .unstack()
    .round(4)
    .to_string()
)




# =====================================================================
# 5. PRINT ALL PREDICTIONS
# =====================================================================
show("\nALL LOPO PREDICTIONS:")
show(
    preds.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)




# =====================================================================
# 6. SAVE COMPLETE RESULTS
# =====================================================================
predictions_path = DOWNLOADS / "lopo_predictions.csv"
folds_path = DOWNLOADS / "lopo_folds.csv"
text_path = DOWNLOADS / "lopo_full_output.txt"


preds.to_csv(predictions_path, index=False)
folds.to_csv(folds_path, index=False)


show(f"\nSaved predictions: {predictions_path}")
show(f"Saved LOPO folds: {folds_path}")
show(f"Saved full output: {text_path}")


text_path.write_text("\n".join(output) + "\n", encoding="utf-8")
