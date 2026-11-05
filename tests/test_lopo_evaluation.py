"""Behavioral checks for temporal leakage, held-out fitting, and inference."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from scripts import lopo_evaluation as ev


def sample_raw():
    rows = []
    for number, subject in enumerate(["p1", "p2", "p3"]):
        for j, day in enumerate([1, 2, 3, 5, 6, 7]):
            rows.append({"subject": subject, "night_id": pd.Timestamp(2022, 6, day),
                         "hypo_night": (j + number) % 2, "Model_Eligible": True,
                         "outcome_complete": True, "history_label_known": True,
                         "mean_glucose_presleep": 100 + j * 3 + number,
                         "glucose_slope_presleep": j - 3,
                         "rmssd_presleep": [10, 20, 40, 15, 30, 25][j] + number * 100,
                         "lfhf_presleep": [1, 3, 2, 5, 4, 6][j] + number})
    return pd.DataFrame(rows)


def sample_daytime(raw):
    result = ev.daytime_features(raw, {})
    result["daytime_low"] = np.arange(len(result)) % 3 == 0
    return result


class TemporalFeaturesTest(unittest.TestCase):
    def test_history_is_strictly_earlier_with_gaps_and_patient_isolation(self):
        raw = sample_raw()
        got = ev.add_history(raw.sample(frac=1, random_state=2), sample_daytime(raw))
        p1 = got.loc[got.subject.eq("p1")].reset_index(drop=True)
        self.assertTrue(np.isnan(p1.prior_night_low.iloc[0]))
        self.assertEqual(p1.prior_night_low.iloc[1], 0)
        self.assertTrue(np.isnan(p1.prior_night_low.iloc[3]))  # no June 4 night
        self.assertEqual(p1.earlier_night_low_rate.iloc[3], 1 / 3)
        self.assertEqual(p1.earlier_night_count.iloc[3], 3)
        self.assertTrue(p1.rmssd_presleep_z.iloc[:2].isna().all())
        self.assertAlmostEqual(p1.rmssd_presleep_z.iloc[2], (40 - 15) / np.std([10, 20], ddof=1))
        changed = raw.copy()
        changed.loc[changed.subject.eq("p2"), ev.RAW_FEATURES + ["hypo_night"]] = 10000
        other = ev.add_history(changed, sample_daytime(raw))
        assert_frame_equal(got[got.subject.eq("p1")], other[other.subject.eq("p1")])

    def test_current_and_future_labels_cannot_change_current_predictors(self):
        raw = sample_raw()
        daytime = sample_daytime(raw)
        before = ev.add_history(raw, daytime)
        mask = raw.subject.eq("p1") & raw.night_id.ge(pd.Timestamp("2022-06-03"))
        raw.loc[mask, "hypo_night"] = 1 - raw.loc[mask, "hypo_night"]
        raw.loc[mask & raw.night_id.gt(pd.Timestamp("2022-06-03")), ev.RAW_FEATURES[2:]] = 999999
        after = ev.add_history(raw, daytime)
        check = before.subject.eq("p1") & before.night_id.le(pd.Timestamp("2022-06-03"))
        features = ev.FEATURE_SETS["B3"]
        assert_frame_equal(before.loc[check, features], after.loc[check, features])

    def test_constant_hrv_history_stays_missing_not_infinite(self):
        raw = sample_raw()
        raw["rmssd_presleep"] = 10.0
        got = ev.add_history(raw, sample_daytime(raw))
        self.assertTrue(got.rmssd_presleep_z.isna().all())

    def test_daytime_excludes_cutoff_and_missing_is_not_no_low(self):
        raw = sample_raw().iloc[:1]
        times = pd.date_range("2022-06-01 10:00", "2022-06-01 22:00", freq="5min")
        cgm = pd.DataFrame({"timestamp": times, "glucose": 100.0})
        cgm.loc[cgm.index[-1], "glucose"] = 40  # forbidden boundary reading
        day = ev.daytime_features(raw, {"p1": cgm})
        self.assertEqual(day.daytime_low.iloc[0], 0)
        self.assertEqual(day.daytime_coverage.iloc[0], 1)
        cgm.loc[cgm.index[-2], "glucose"] = 60
        self.assertEqual(ev.daytime_features(raw, {"p1": cgm}).daytime_low.iloc[0], 1)
        sparse = ev.daytime_features(raw, {"p1": cgm.iloc[:2]})
        self.assertTrue(np.isnan(sparse.daytime_low.iloc[0]))
        self.assertTrue(np.isnan(ev.daytime_features(raw, {}).daytime_low.iloc[0]))

    def test_cgm_ignores_alerts_and_ambiguous_duplicates(self):
        data = pd.DataFrame({
            "Timestamp (YYYY-MM-DDThh:mm:ss)": ["2022-06-01 10:00"] * 2 + ["2022-06-01 11:00", "2022-06-01 12:00"],
            "Glucose Value (mg/dL)": [50, 100, 40, "Low"],
            "Event Type": ["EGV", "EGV", "Alert", "EGV"],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cgm.csv"
            data.to_csv(path, index=False)
            got = ev.read_cgm(path)
        self.assertEqual(len(got), 1)
        self.assertLess(got.glucose.iloc[0], 70)
        self.assertEqual(got.attrs["conflicting_timestamps_excluded"], 1)

    def test_presleep_glucose_is_rebuilt_and_immune_to_postcutoff_data(self):
        raw = sample_raw().iloc[:1]
        times = pd.date_range("2022-06-01 20:00", "2022-06-02 00:00", freq="5min")
        cgm = pd.DataFrame({"timestamp": times, "glucose": 100 + np.arange(len(times))})
        before, audit = ev.rebuild_presleep_glucose(raw, {"p1": cgm})
        self.assertEqual(before.mean_glucose_presleep.iloc[0], 111.5)
        self.assertAlmostEqual(before.glucose_slope_presleep.iloc[0], 12)
        self.assertLess(audit.glucose_last_reading.iloc[0], audit.prediction_time.iloc[0])
        cgm.loc[cgm.timestamp.ge("2022-06-01 22:00"), "glucose"] = 10000
        after, _ = ev.rebuild_presleep_glucose(raw, {"p1": cgm})
        assert_frame_equal(before, after)


class EvaluationTest(unittest.TestCase):
    def test_heldout_labels_never_fit_model(self):
        raw = sample_raw()
        cohort, _, _ = ev.prepare_cohort(ev.add_history(raw, sample_daytime(raw)))
        models = {"logistic": ev.classifiers()["logistic"]}
        original, audit, columns = ev.generate_oof(cohort, estimators=models)
        mask = cohort.subject.eq("p1")
        cohort.loc[mask, "hypo_night"] = 1 - cohort.loc[mask, "hypo_night"]
        changed, _, _ = ev.generate_oof(cohort, estimators=models)
        scores = [col for _, _, col in columns]
        assert_frame_equal(original.loc[mask, scores], changed.loc[mask, scores])
        self.assertEqual(len(audit), 3)
        self.assertTrue(all(r.held_out_subject not in r.train_subjects.split(",") for r in audit.itertuples()))

    def test_bootstrap_is_paired_and_retains_patient_multiplicity(self):
        oof = pd.DataFrame({"subject": ["a", "a", "b", "b", "b", "c", "c"],
                            "hypo_night": [0, 1, 0, 1, 0, 0, 1],
                            "m__A": [.1, .2, .3, .4, .5, .6, .7]})
        oof["m__B3"] = oof["m__A"]
        cols = [("m", "A", "m__A"), ("m", "B3", "m__B3")]
        pooled, delta, boot, draws = ev.paired_patient_bootstrap(oof, cols, 25, 42)
        self.assertEqual(delta.delta_auroc.iloc[0], 0)
        self.assertEqual(delta.auroc_ci_lower.iloc[0], 0)
        self.assertEqual(delta.auroc_ci_upper.iloc[0], 0)
        self.assertEqual(delta.pr_auc_ci_upper.iloc[0], 0)
        repeated = False
        for row in draws.itertuples():
            sample = row.sampled_subjects.split(",")
            repeated |= len(set(sample)) < len(sample)
            expanded = pd.concat([oof[oof.subject.eq(s)] for s in sample])
            self.assertEqual(row.sampled_nights, len(expanded))
            expected = ev.metric_pair(expanded.hypo_night, expanded["m__A"])
            actual = boot[(boot.replicate == row.replicate) & (boot.feature_set == "A")].iloc[0]
            np.testing.assert_allclose([actual.auroc, actual.pr_auc], expected)
        self.assertTrue(repeated)

    def test_single_class_patients_are_na_and_kept_pooled(self):
        oof = pd.DataFrame({"subject": ["a", "a", "b", "b"], "hypo_night": [0, 0, 1, 0],
                            "m__A": [.1, .2, .9, .3]})
        cols = [("m", "A", "m__A")]
        patients = ev.per_patient_metrics(oof, cols)
        self.assertTrue(patients.loc[patients.subject.eq("a"), ["auroc", "pr_auc"]].isna().all().all())
        pooled, _, _, _ = ev.paired_patient_bootstrap(oof, cols, 10, 5)
        self.assertEqual(pooled.n_nights.iloc[0], 4)
        self.assertEqual(pooled.pooled_auroc.iloc[0], 1)

    def test_permutation_refits_and_rebuilds_outcome_histories(self):
        raw = sample_raw()
        daytime = sample_daytime(raw)
        features = ev.add_history(raw, daytime)
        cohort, _, _ = ev.prepare_cohort(features)
        models = {"logistic": ev.classifiers()["logistic"]}
        oof, _, _ = ev.generate_oof(cohort, estimators=models)
        with patch.object(ev, "add_history", wraps=ev.add_history) as build:
            with patch.object(ev, "generate_oof", wraps=ev.generate_oof) as fit:
                summary, draws = ev.permutation_test(raw, daytime, oof, ev.Config(n_permutations=2), progress=None)
        self.assertEqual(build.call_count, 4)
        self.assertEqual(fit.call_count, 4)
        self.assertEqual(len(draws), 4)
        self.assertTrue(summary.p_greater_equal.between(1 / 3, 1).all())
        for call in fit.call_args_list:
            rebuilt = call.args[0]
            for _, patient in rebuilt.groupby("subject"):
                expected = patient.hypo_night.expanding().mean().shift(1)
                np.testing.assert_allclose(patient.earlier_night_low_rate, expected, equal_nan=True)
        for call in build.call_args_list[-2:]:
            pd.testing.assert_series_equal(call.args[0].groupby("subject").hypo_night.sum(),
                                           raw.groupby("subject").hypo_night.sum())


if __name__ == "__main__":
    unittest.main()
