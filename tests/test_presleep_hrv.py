"""HRV extraction must never accept ECG or quality data at/after cutoff."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import build_presleep_hrv as hrv
from scripts import lopo_evaluation as ev


class HrvTimingTest(unittest.TestCase):
    def test_cutoff_is_enforced_before_filtering_or_peak_detection(self):
        time = pd.date_range("2022-06-01 21:55", periods=75000, freq="4ms")
        ecg = pd.DataFrame({"Time": time, "EcgWaveform": 1.0})
        summary = pd.DataFrame({"Time": pd.date_range("2022-06-01 21:55", periods=300, freq="s"),
                                "HRConfidence": 100, "ECGNoise": .0001})
        with patch.object(hrv.nk, "ecg_clean", side_effect=lambda wave, **kw: wave):
            with patch.object(hrv.nk, "ecg_peaks", return_value=(None, {"ECG_R_Peaks": np.arange(250, 75000, 250)})):
                with patch.object(hrv.nk, "hrv_frequency", return_value=pd.DataFrame({"HRV_LFHF": [1.5]})):
                    rows = hrv.calculate_windows(ecg, summary, "p1", "r1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["window_end"], pd.Timestamp("2022-06-01 22:00"))
        self.assertLess(rows[0]["last_ecg_sample"], rows[0]["window_end"])
        self.assertLess(rows[0]["last_quality_sample"], rows[0]["window_end"])
        with self.assertRaisesRegex(ValueError, "Post-cutoff"):
            hrv.calculate_windows(pd.concat([ecg, pd.DataFrame({"Time": [pd.Timestamp("2022-06-01 22:00")], "EcgWaveform": [1.]})]), summary, "p1", "r1")
        bad_summary = summary.copy()
        bad_summary.loc[bad_summary.index[-1], "Time"] = pd.Timestamp("2022-06-01 22:00")
        with self.assertRaisesRegex(ValueError, "Post-cutoff"):
            hrv.calculate_windows(ecg, bad_summary, "p1", "r1")

    def test_aggregation_deduplicates_overlaps_and_requires_six_windows(self):
        rows = []
        for k in range(6):
            start = pd.Timestamp("2022-06-01 20:00") + pd.Timedelta(k * 5, unit="m")
            end = start + pd.Timedelta(5, unit="m")
            rows.append({"subject": "p1", "night_id": start.normalize(), "window_start": start,
                         "window_end": end, "last_ecg_sample": end - pd.Timedelta(4, unit="ms"),
                         "last_quality_sample": end - pd.Timedelta(1, unit="s"),
                         "status": "ok", "rmssd": 10 + k, "lfhf": 1 + k, "bad_rr_fraction": 0})
        windows = pd.DataFrame(rows + [rows[0]])
        keys = pd.DataFrame({"subject": ["p1", "p2"], "night_id": pd.to_datetime(["2022-06-01"] * 2)})
        nights = hrv.aggregate_nights(keys, windows)
        self.assertEqual(nights.pre_cutoff_hrv_windows.iloc[0], 6)
        self.assertEqual(nights.rmssd_presleep.iloc[0], 12.5)
        self.assertTrue(np.isnan(nights.rmssd_presleep.iloc[1]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hrv.parquet"
            nights.to_parquet(path)
            raw = keys.assign(rmssd_presleep=999, lfhf_presleep=999)
            merged, _ = ev.load_rebuilt_hrv(raw, path)
            self.assertTrue(np.isnan(merged.rmssd_presleep.iloc[1]))  # no legacy fallback
            nights.loc[0, "hrv_last_sample"] = pd.Timestamp("2022-06-01 22:00")
            nights.to_parquet(path)
            with self.assertRaisesRegex(ValueError, "pre-cutoff"):
                ev.load_rebuilt_hrv(raw, path)

    def test_subsample_offset_past_cutoff_is_rejected(self):
        windows = pd.DataFrame({"night_id": [pd.Timestamp("2022-06-01")],
                                "window_start": [pd.Timestamp("2022-06-01 21:55:00.002")],
                                "window_end": [pd.Timestamp("2022-06-01 22:00:00.002")],
                                "last_ecg_sample": [pd.Timestamp("2022-06-01 21:59:59.998")],
                                "last_quality_sample": [pd.Timestamp("2022-06-01 21:59:59")],
                                "status": ["ok"]})
        checked = hrv.enforce_window_bounds(windows)
        self.assertEqual(checked.status.iloc[0], "window_extends_past_cutoff")


if __name__ == "__main__":
    unittest.main()
