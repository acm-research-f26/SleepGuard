"""Build additional pre-bedtime features from the participant CGM ZIP files."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


KEYS = ["subject", "night_of"]
RECENT = ["bedtime_glucose", "slope_30min", "change_60min"]
MINUTE = pd.Timedelta(minutes=1)


def load_cgm(path):
    with ZipFile(path) as archive:
        name, = [name for name in archive.namelist() if name.lower().endswith("cgm.csv")]
        with archive.open(name) as raw:
            readings = pd.read_csv(raw, low_memory=False)
    readings = readings.loc[readings["Event Type"].eq("EGV")].copy()
    readings["time"] = pd.to_datetime(readings["Timestamp (YYYY-MM-DDThh:mm:ss)"])
    readings["glucose"] = pd.to_numeric(
        readings["Glucose Value (mg/dL)"].replace({"Low": 39, "High": 401}), errors="coerce"
    )
    return readings.dropna(subset=["time", "glucose"]).sort_values("time")


def night_features(cgm, night):
    cutoff = night + pd.Timedelta(hours=22)
    raw = cgm.loc[cgm.time.ge(cutoff - 2 * 60 * MINUTE) & cgm.time.lt(cutoff)]
    # Count each timestamp once for the new features.
    pre = raw.groupby("time")["glucose"].median().sort_index()
    bedtime, slope, change = np.nan, np.nan, np.nan
    if not pre.empty and cutoff - pre.index[-1] <= 10 * MINUTE:
        bedtime = pre.iloc[-1]
        recent = pre.loc[pre.index >= cutoff - 30 * MINUTE]
        minutes = (recent.index - recent.index[0]) / MINUTE
        if len(recent) >= 3 and minutes[-1] >= 15 and np.diff(minutes).max() <= 15:
            slope = np.polyfit(minutes / 60, recent.to_numpy(), 1)[0]
        anchor = cutoff - 60 * MINUTE
        hour_start = pre.loc[(pre.index >= anchor - 10 * MINUTE) & (pre.index <= anchor)]
        if not hour_start.empty:
            change = bedtime - hour_start.iloc[-1]
    return dict(zip(RECENT, (bedtime, slope, change))) | {
        "raw_mean_2h": raw.glucose.mean(), "raw_min_2h": raw.glucose.min()
    }


def build(nights, raw_dir):
    records = []
    for subject, rows in nights.groupby("subject", sort=True):
        cgm = load_cgm(raw_dir / f"{subject}_raw.zip")
        for night in rows.night_of:
            records.append({"subject": subject, "night_of": night, **night_features(cgm, night)})
    result = nights.merge(pd.DataFrame(records), on=KEYS, validate="one_to_one")
    for stat in ("mean", "min"):
        matches = np.isclose(result[f"{stat}_2h"], result[f"raw_{stat}_2h"], atol=1e-8, rtol=0)
        print(f"Original 2h {stat} matches raw cutoff for {matches.sum()}/{len(result)} nights")
    return result.drop(columns=["raw_mean_2h", "raw_min_2h"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nights", type=Path)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build(pd.read_csv(args.nights, parse_dates=["night_of"]), args.raw_dir)
    result.to_csv(args.output, index=False)
    print(result[RECENT].notna().sum().to_string())
