"""Synthetic metadata fixtures only; no models or study rows in these tests."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_mucars_reconstruction as checker


def synthetic_audits():
    sha = "1" * 64
    features = ["make", "year"]
    action_features = ["before_prediction_log"] + features
    policy_features = ["action_id"] + action_features
    return {
        "cohort": {"doi": "synthetic-only", "source_sha256": sha,
                   "output_hashes": {"splits/" + n + ".csv": sha for n in ("train", "validation", "calibration", "test_features")}
                   | {"sealed_labels/test_labels.csv": sha},
                   "split_counts": {"train": 10, "validation": 5, "calibration": 3, "test": 4}},
        "pairs": {"inputs": {n + "_sha256": sha for n in ("train", "validation", "calibration", "evaluation_features")},
                  "outputs": {phase: {name: sha for name in ("pairs", "features", "labels")} for phase in ("development", "calibration", "evaluation")},
                  "development_action_pair_count": 45, "development_listing_count": 15,
                  "calibration_action_pair_count": 9, "calibration_listing_count": 3,
                  "evaluation_action_pair_count": 12, "evaluation_listing_count": 4,
                  "seeds": [13, 42], "folds": 3, "model_parameters": {"iterations": 200, "depth": 6, "learning_rate": 0.08, "l2_leaf_reg": 10},
                  "feature_columns": features, "categorical_columns": ["make"], "action_fields": ["field_a"]},
        "development": {"inputs": {n: {"sha256": sha} for n in ("prediction_pairs", "pre_action_features", "labels")},
                        "listing_action_count": 45, "listing_count": 15, "output_sha256": sha, "feature_allowlist": action_features},
        "calibration": {"inputs": {n: {"sha256": sha} for n in ("prediction_pairs", "pre_action_features", "labels")},
                        "listing_action_count": 9, "listing_count": 3, "output_sha256": sha, "feature_allowlist": action_features},
        "value": {"inputs": {n + "_sha256": sha for n in ("train_actions", "calibration_actions", "evaluation_features")},
                  "feature_columns": policy_features, "categorical_columns": ["action_id", "make"], "seeds": [13, 42],
                  "evaluation_action_scores_sha256": sha,
                  "objectives": {"absolute_price_error": {"calibrator": {"alpha": 0.1, "minimum_group_size": 100}}}},
        "risk": {"inputs": {n + "_sha256": sha for n in ("train_actions", "evaluation_features")},
                 "feature_columns": policy_features, "categorical_columns": ["action_id", "make"], "seeds": [13, 42],
                 "evaluation_risk_scores_sha256": sha, "objective": "absolute_price_error"},
        "assembly": {"inputs": {n: {"sha256": sha} for n in ("value_scores", "risk_scores", "development_actions", "evaluation_features")},
                     "output": {"sha256": sha}, "objective": "absolute_price_error"},
        "join": {"inputs": {n + "_sha256": sha for n in ("frozen_policy_scores", "prediction_pairs", "sealed_labels")},
                 "output_sha256": sha, "freeze_id": "synthetic", "listing_count": 4, "listing_action_count": 12, "labels_opened": True},
        "evaluation": {"evaluation_outcomes_sha256": sha, "frozen_policy_scores_sha256": sha,
                       "objective": "absolute_price_error", "reference_policy": "score_uncertainty_only", "freeze_id": "synthetic",
                       "budgets": [0.1], "bootstrap_repetitions": 2, "bootstrap_seed": 2026, "oracle_diagnostic_included": True},
    }


class MetadataReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mucars-metadata-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.audits = synthetic_audits()

    def build_fixture(self):
        for name, relative in checker.AUDITS.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.audits[name]), encoding="utf-8")
        lines = []
        for relative in sorted(checker.FROZEN_CODE_AND_METADATA):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic code/document bytes", encoding="utf-8")
            lines.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + relative)
        # Intentionally do not create any CSVs or model files.
        lines += ["1" * 64 + "  " + relative for relative in sorted(checker.FROZEN_ROW_PATHS)]
        (self.root / checker.FREEZE).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_recorded_graph_44_checks(self):
        self.assertEqual(len(checker.recorded_graph_checks(self.audits)), 44)

    def test_hash_mismatch_rejected(self):
        self.audits["risk"]["inputs"]["train_actions_sha256"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "development_targets_to_risk"):
            checker.recorded_graph_checks(self.audits)

    def test_malformed_digest_rejected(self):
        self.audits["pairs"]["inputs"]["train_sha256"] = None
        with self.assertRaisesRegex(ValueError, "Invalid recorded SHA256"):
            checker.recorded_graph_checks(self.audits)

    def test_feature_order_mismatch_rejected(self):
        self.audits["value"]["feature_columns"] = list(reversed(self.audits["value"]["feature_columns"]))
        with self.assertRaisesRegex(ValueError, "value_feature_allowlist"):
            checker.recorded_graph_checks(self.audits)

    def test_wrong_objective_rejected(self):
        self.audits["risk"]["objective"] = "squared_log_error"
        with self.assertRaisesRegex(ValueError, "risk_primary_objective"):
            checker.recorded_graph_checks(self.audits)

    def test_count_mismatch_rejected(self):
        self.audits["join"]["listing_count"] = 5
        with self.assertRaisesRegex(ValueError, "evaluation_join_listing_count"):
            checker.recorded_graph_checks(self.audits)

    def test_opened_test_cannot_be_relabelled(self):
        for value in (False, "true", 1):
            self.audits["join"]["labels_opened"] = value
            with self.assertRaisesRegex(ValueError, "unopened"):
                checker.recorded_graph_checks(self.audits)

    def test_metadata_only_mode_needs_no_rows_or_models(self):
        self.build_fixture()
        with patch.object(checker, "inspect_models", side_effect=AssertionError("No models in default mode")):
            result = checker.reconstruct(self.root)
        self.assertEqual(len(result["row_entries_deliberately_not_opened"]), 4)
        self.assertFalse(result["saved_model_parameters_inspected"])
        self.assertFalse(result["listing_rows_read"])
        self.assertFalse(result["full_empirical_retraining_verified"])

    def test_explicit_opt_in_dispatches_model_inspection(self):
        self.build_fixture()
        with patch.object(checker, "inspect_models", return_value={"synthetic": True}) as mock:
            result = checker.reconstruct(self.root, True)
        mock.assert_called_once()
        self.assertTrue(result["saved_model_parameters_inspected"])

    def test_live_source_drift_reported_not_hidden(self):
        self.build_fixture()
        path = self.root / "experiments/fit_bvival_value_policy.py"
        path.write_text("changed source", encoding="utf-8")
        result = checker.reconstruct(self.root)
        self.assertEqual(sum(not item["matches"] for item in result["live_code_vs_pretest_freeze"]), 1)

    def test_incomplete_freeze_rejected(self):
        self.build_fixture()
        path = self.root / checker.FREEZE
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Incomplete recorded row"):
            checker.reconstruct(self.root)

    def test_duplicate_freeze_rejected(self):
        self.build_fixture()
        path = self.root / checker.FREEZE
        path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate freeze"):
            checker.reconstruct(self.root)

    def test_unclassified_freeze_path_rejected(self):
        self.build_fixture()
        path = self.root / checker.FREEZE
        path.write_text(path.read_text() + "1" * 64 + "  experiments/data/private.csv\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unclassified"):
            checker.reconstruct(self.root)

    def test_path_escape_and_symlink_rejected(self):
        self.build_fixture()
        for relative in ("../outside.json", "/tmp/outside.json", "experiments//bad.json"):
            with self.assertRaises(ValueError):
                checker.safe_path(self.root, relative)
        alias = self.root / "alias.json"
        alias.symlink_to(self.root / checker.AUDITS["cohort"])
        with self.assertRaisesRegex(ValueError, "Symlink"):
            checker.safe_path(self.root, "alias.json")


if __name__ == "__main__":
    unittest.main()
