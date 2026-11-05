"""Rebuild ECG HRV from strictly pre-cutoff [20:00,22:00) raw samples.

Uses existing local recordings or fetches only required members of the official
PhysioCGM ZIPs. Each complete five-minute continuous, quality-approved ECG
window is cleaned/detected independently. No post-cutoff sample enters filters,
peak detection, RR correction, quality matching, or HRV computation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sleepguard-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sleepguard-cache")

import neurokit2 as nk
import numpy as np
import pandas as pd

try:
    from .fetch_cgm import RemoteZip
except ImportError:
    from fetch_cgm import RemoteZip

FS = 250
TIME_FORMAT = "%d/%m/%Y %H:%M:%S.%f"
SETTINGS = {
    "start_hour": 20, "end_hour": 22, "sampling_rate": FS,
    "window_seconds": 300, "hr_confidence": 100, "max_ecg_noise": 0.001,
    "rr_min_ms": 400, "rr_max_ms": 2000, "rr_local_deviation": 0.20,
    "max_corrected_rr_fraction": 0.05, "minimum_windows_per_night": 6,
    "aggregation": "median of accepted non-overlapping five-minute windows",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_presleep(frame, nights, time_column="Time"):
    dates = frame[time_column].dt.normalize()
    hours = frame[time_column].dt.hour
    return frame.loc[dates.isin(nights) & hours.ge(20) & hours.lt(22)].copy()


def read_summary(path):
    data = pd.read_csv(path, usecols=["Time", "HRConfidence", "ECGNoise"])
    data["Time"] = pd.to_datetime(data.Time, format=TIME_FORMAT, errors="raise")
    return data.sort_values("Time").drop_duplicates("Time", keep=False)


def calculate_windows(ecg, summary, subject, recording):
    """Return accepted/rejected windows with source timestamps and RR audit."""
    if ecg.empty:
        return []
    ecg = ecg.sort_values("Time").drop_duplicates("Time", keep=False)
    # Both inputs must already be cropped. Check again at the public boundary.
    for frame in (ecg, summary):
        if not (frame.Time.dt.hour.ge(20) & frame.Time.dt.hour.lt(22)).all():
            raise ValueError("Post-cutoff or out-of-window data reached HRV extraction")
    checked = pd.merge_asof(ecg, summary.rename(columns={"Time": "quality_time"}),
                            left_on="Time", right_on="quality_time", direction="nearest",
                            tolerance=pd.Timedelta(1, unit="s"))
    good = (checked.quality_time.notna() & checked.HRConfidence.eq(100)
            & checked.ECGNoise.lt(0.001) & np.isfinite(checked.EcgWaveform))
    # Do not bridge dropped samples, quality failures, sessions, or day gaps.
    new_run = (~good.shift(fill_value=False) | checked.Time.diff().ne(pd.Timedelta(4, unit="ms")))
    checked["run"] = new_run.cumsum()
    rows = []
    for _, run in checked.loc[good].groupby("run", sort=False):
        for offset in range(0, len(run) - FS * 300 + 1, FS * 300):
            window = run.iloc[offset:offset + FS * 300]
            start = window.Time.iloc[0]
            row = {"subject": subject, "night_id": start.normalize(), "recording": recording,
                   "window_start": start, "window_end": window.Time.iloc[-1] + pd.Timedelta(4, unit="ms"),
                   "last_ecg_sample": window.Time.iloc[-1], "last_quality_sample": window.quality_time.max(),
                   "n_samples": len(window), "n_peaks": 0, "bad_rr_fraction": np.nan,
                   "rmssd": np.nan, "lfhf": np.nan, "status": "insufficient_peaks"}
            waveform = window.EcgWaveform.to_numpy(dtype=float)
            cleaned = nk.ecg_clean(waveform, sampling_rate=FS, method="neurokit")
            _, info = nk.ecg_peaks(cleaned, sampling_rate=FS, method="neurokit", correct_artifacts=False)
            peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
            row["n_peaks"] = len(peaks)
            if len(peaks) >= 150:
                rr = pd.Series(np.diff(peaks) * 1000 / FS)
                local_median = rr.rolling(11, center=True, min_periods=3).median()
                bad = rr.lt(400) | rr.gt(2000) | ((rr - local_median).abs() / local_median).gt(0.20)
                row["bad_rr_fraction"] = float(bad.mean())
                row["status"] = "excessive_rr_artifacts"
                if bad.mean() <= 0.05:
                    # Corrections use only neighbors within this pre-cutoff window.
                    corrected = rr.mask(bad).interpolate(limit_direction="both").to_numpy()
                    intervals = {"RRI": corrected, "RRI_Time": peaks[1:] / FS}
                    row["rmssd"] = float(np.sqrt(np.mean(np.diff(corrected) ** 2)))
                    frequency = nk.hrv_frequency(intervals, sampling_rate=FS, psd_method="welch", show=False)
                    row["lfhf"] = float(frequency["HRV_LFHF"].iloc[0])
                    row["status"] = "ok" if np.isfinite([row["rmssd"], row["lfhf"]]).all() else "nonfinite_hrv"
            rows.append(row)
    return rows


def read_ecg_subset(path, nights):
    pieces = []
    # Filter date/hour strings before parsing to avoid parsing entire recordings.
    dates = {pd.Timestamp(n).strftime("%d/%m/%Y") for n in nights}
    for block in pd.read_csv(path, usecols=["Time", "EcgWaveform"], chunksize=500_000):
        strings = block.Time.str.strip()
        choose = strings.str[:10].isin(dates) & strings.str[11:13].isin(["20", "21"])
        if choose.any():
            selected = block.loc[choose].copy()
            selected["Time"] = pd.to_datetime(selected.Time, format=TIME_FORMAT, errors="raise")
            pieces.append(selected)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["Time", "EcgWaveform"])


def process_patient(subject, nights, raw_root, entry, code_hash):
    folder = raw_root / subject
    cache = folder / "presleep_hrv_windows_20_22.parquet"
    metadata_file = folder / "presleep_hrv_sources_20_22.json"
    if cache.exists() and metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        if metadata.get("code_sha256") == code_hash and metadata.get("nights") == [str(n) for n in nights]:
            print(f"{subject}: reused verified pre-cutoff extraction cache", flush=True)
            return pd.read_parquet(cache), metadata
    local_summaries = sorted((folder / "zephyr").rglob("*_SummaryEnhanced.csv"))
    sources, window_rows = [], []
    archive = None
    if subject == "c1s01" and local_summaries:
        pairs = [(str(p), p) for p in local_summaries]
    else:
        archive = zipfile.ZipFile(RemoteZip(entry["download_url"], entry["size"], max_range=250_000_000))
        members = [m for m in archive.infolist() if m.filename.endswith("_SummaryEnhanced.csv")
                   and not m.filename.startswith("__MACOSX/")]
        # Different ZIP paths for identical recordings must not duplicate windows.
        unique = {}
        for m in members:
            key = (Path(m.filename).parent.name, Path(m.filename).name)
            if key in unique and (unique[key].CRC, unique[key].file_size) != (m.CRC, m.file_size):
                raise ValueError(f"Conflicting duplicate summary {key}")
            unique[key] = m
        pairs = []
        for member in unique.values():
            destination = folder / "zephyr" / Path(member.filename).parent.name / Path(member.filename).name
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
            pairs.append((member.filename, destination))
        print(f"{subject}: read {len(pairs)} recording summaries", flush=True)
    seen_recordings = set()
    for member_name, summary_path in pairs:
        summary = select_presleep(read_summary(summary_path), nights)
        if summary.empty:
            continue
        usable = summary.HRConfidence.eq(100) & summary.ECGNoise.lt(0.001)
        if usable.sum() < 300:
            continue
        ecg_path = summary_path.with_name(summary_path.name.replace("_SummaryEnhanced.csv", "_ECG.csv"))
        if not ecg_path.exists():
            ecg_name = member_name.replace("_SummaryEnhanced.csv", "_ECG.csv")
            info = archive.getinfo(ecg_name)
            print(f"{subject}: fetching {ecg_path.parent.name} ECG ({info.compress_size / 1e6:.1f} MB compressed)", flush=True)
            ecg_path.write_bytes(archive.read(info))
        signature = digest(ecg_path)
        if signature in seen_recordings:
            continue
        seen_recordings.add(signature)
        ecg = read_ecg_subset(ecg_path, nights)
        window_rows.extend(calculate_windows(ecg, summary, subject, ecg_path.parent.name))
        sources.append({"recording": ecg_path.parent.name, "ecg_sha256": signature,
                        "summary_sha256": digest(summary_path), "ecg_file": str(ecg_path),
                        "source_url": entry["download_url"]})
        print(f"{subject}: processed {ecg_path.parent.name}; {len(window_rows)} candidate windows so far", flush=True)
    if archive:
        archive.close()
    columns = ["subject", "night_id", "recording", "window_start", "window_end", "last_ecg_sample",
               "last_quality_sample", "n_samples", "n_peaks", "bad_rr_fraction", "rmssd", "lfhf", "status"]
    windows = pd.DataFrame(window_rows, columns=columns)
    metadata = {"subject": subject, "code_sha256": code_hash, "settings": SETTINGS,
                "nights": [str(n) for n in nights], "sources": sources, "neurokit2": nk.__version__}
    windows.to_parquet(cache, index=False)
    metadata_file.write_text(json.dumps(metadata, indent=2))
    print(f"{subject}: finished {len(windows)} windows", flush=True)
    return windows, metadata


def enforce_window_bounds(windows):
    windows = windows.copy()
    cutoff = windows.night_id + pd.Timedelta(22, unit="h")
    start = windows.night_id + pd.Timedelta(20, unit="h")
    if (windows.last_ecg_sample.ge(cutoff) | windows.last_quality_sample.ge(cutoff)
            | windows.window_start.lt(start)).any():
        raise AssertionError("Source ECG/quality samples are outside the prediction window")
    # The ECG grid can be offset by 1–3 ms from the clock cutoff. A 75,000-
    # sample window may then have its nominal end just after 22:00 although
    # its final sample is before 22:00. Reject it to require the full interval.
    windows.loc[windows.window_end.gt(cutoff), "status"] = "window_extends_past_cutoff"
    return windows


def aggregate_nights(raw_keys, windows):
    windows = enforce_window_bounds(windows)
    accepted = windows.loc[windows.status.eq("ok")].copy()
    # Greedily retain chronologically non-overlapping windows across recordings.
    selected = []
    for _, group in accepted.sort_values("window_start").groupby(["subject", "night_id"]):
        end = None
        for idx, row in group.iterrows():
            if end is None or row.window_start >= end:
                selected.append(idx)
                end = row.window_end
    accepted = accepted.loc[selected]
    grouped = accepted.groupby(["subject", "night_id"]).agg(
        rmssd_presleep=("rmssd", "median"), lfhf_presleep=("lfhf", "median"),
        pre_cutoff_hrv_windows=("rmssd", "size"), hrv_source_start=("window_start", "min"),
        hrv_source_end=("window_end", "max"), hrv_last_sample=("last_ecg_sample", "max"),
        hrv_last_quality=("last_quality_sample", "max"), max_bad_rr_fraction=("bad_rr_fraction", "max"))
    nightly = raw_keys.merge(grouped, on=["subject", "night_id"], how="left", validate="one_to_one")
    nightly["pre_cutoff_hrv_windows"] = nightly.pre_cutoff_hrv_windows.fillna(0).astype(int)
    enough = nightly.pre_cutoff_hrv_windows.ge(SETTINGS["minimum_windows_per_night"])
    nightly.loc[~enough, ["rmssd_presleep", "lfhf_presleep"]] = np.nan
    nightly["hrv_status"] = np.where(enough, "ok", "fewer_than_6_usable_5min_windows")
    cutoff = nightly.night_id + pd.Timedelta(22, unit="h")
    if (nightly.hrv_source_end.dropna() > cutoff[nightly.hrv_source_end.notna()]).any():
        raise AssertionError("HRV source exceeds prediction cutoff")
    if (nightly.hrv_last_quality.dropna() >= cutoff[nightly.hrv_last_quality.notna()]).any():
        raise AssertionError("Quality source is at/after cutoff")
    return nightly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Validate and aggregate existing extraction caches; no network or new signal processing")
    args = parser.parse_args()
    root = args.repo_root
    raw = pd.read_parquet(root / "data/physiocgm/processed/model_data.parquet")
    keys = raw[["participant_id", "night"]].rename(columns={"participant_id": "subject", "night": "night_id"})
    if args.subjects:
        keys = keys.loc[keys.subject.isin(args.subjects)]
    entries = {f["name"].split("_")[0]: f for f in json.loads(args.index.read_text())["files"]
               if f["name"].endswith("_raw.zip")}
    windows, sources = [], []
    if args.aggregate_only:
        for subject, group in keys.groupby("subject"):
            folder = root / "data/physiocgm/raw" / subject
            source = json.loads((folder / "presleep_hrv_sources_20_22.json").read_text())
            if source["settings"] != SETTINGS or source["nights"] != [str(n) for n in sorted(group.night_id.unique())]:
                raise ValueError(f"Cache settings/night mismatch for {subject}")
            windows.append(pd.read_parquet(folder / "presleep_hrv_windows_20_22.parquet"))
            sources.append(source)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_patient, subject, sorted(group.night_id.unique()),
                                   root / "data/physiocgm/raw", entries[subject], digest(__file__))
                       for subject, group in keys.groupby("subject")]
            for future in as_completed(futures):
                result, source = future.result()
                windows.append(result)
                sources.append(source)
    windows = pd.concat(windows, ignore_index=True)
    windows = enforce_window_bounds(windows)
    for source in sources:
        source["aggregation_code_sha256"] = digest(__file__)
        source["aggregation_boundary_policy"] = "Reject nominal window ends after 22:00 even when all recorded samples precede the cutoff"
    nightly = aggregate_nights(keys, windows)
    output = root / "data/physiocgm/processed"
    suffix = "_subset" if args.subjects else ""
    windows.to_parquet(output / f"hrv_windows_20_22{suffix}.parquet", index=False)
    nightly.to_parquet(output / f"presleep_hrv_20_22{suffix}.parquet", index=False)
    (output / f"presleep_hrv_sources_20_22{suffix}.json").write_text(json.dumps(sources, indent=2))
    print(nightly.groupby(["subject", "hrv_status"]).size().to_string(), flush=True)


if __name__ == "__main__":
    main()
