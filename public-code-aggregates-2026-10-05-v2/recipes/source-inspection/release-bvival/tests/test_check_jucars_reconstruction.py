"""Synthetic audit, freeze and header fixtures only; no study rows/models."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_jucars_reconstruction as checker
from test_check_mucars_reconstruction import synthetic_audits


def toy_audits():
    old = synthetic_audits()
    a = {k: copy.deepcopy(old["development"] if k.startswith("development_") else old[k]) for k in checker.AUDITS}
    a["cohort"]["outputs"] = a["cohort"].pop("output_hashes")
    a["cohort"].update({"dataset_doi": "synthetic-only", "source_file": "toy-source.csv",
                         "source_file_sha256": "1" * 64, "data_dictionary_file": "toy-dictionary.csv",
                         "data_dictionary_sha256": "2" * 64})
    for k in ("development_value", "development_risk", "calibration"):
        a[k]["all_predictions_oof"] = True
    a["development_risk"]["output_sha256"] = "2" * 64
    a["risk"]["inputs"]["train_actions_sha256"] = "2" * 64
    a["assembly"]["inputs"]["development_actions"]["sha256"] = "2" * 64
    a["join"]["inputs"]["additional_frozen_policy_scores_sha256"] = "3" * 64
    for k in ("pairs", "risk", "value"):
        a[k]["seeds"] = checker.SEEDS
    a["evaluation"]["outputs"] = {"toy-reference.csv": "4" * 64}
    return a


class JUCarsRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jucars-recovery-toy-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.a = toy_audits()

    def write(self, relative, data):
        p = self.root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data)
        return p

    def fixture(self):
        for role, relative in checker.AUDITS.items():
            self.write(relative, json.dumps(self.a[role]))
        lines = []
        for relative in sorted(checker.FROZEN_CODE_AND_METADATA):
            p = self.write(relative, "# synthetic never-executed source\n")
            lines.append(checker.sha256(p) + "  " + relative)
        for relative in sorted(checker.FROZEN_ROWS):
            digest = "3" * 64 if "squared_log_error.csv" in relative else "1" * 64
            lines.append(digest + "  " + relative)
        self.write(checker.FREEZE, "\n".join(lines) + "\n")
        for name in ("build_jucars_bvival_manifest.py", "generate_bvival_prediction_pairs.py", "build_bvival_action_dataset.py",
                     "fit_bvival_value_policy.py", "fit_bvival_risk_baseline.py", "assemble_bvival_policy_scores.py"):
            self.write("experiments/" + name, "raise RuntimeError('Toy source must not execute')\n")

    def test_recorded_graph_has_55_checks_with_two_real_target_branches(self):
        checks = checker.recorded_graph_checks(self.a)
        self.assertEqual(len(checks), 55)
        self.assertIn("development_v1_to_value", checks)
        self.assertIn("development_v2_to_risk", checks)

    def test_swapping_risk_branch_to_value_hash_rejected(self):
        self.a["risk"]["inputs"]["train_actions_sha256"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "development_v2_to_risk"):
            checker.recorded_graph_checks(self.a)

    def test_assembly_must_bind_risk_target_branch(self):
        self.a["assembly"]["inputs"]["development_actions"]["sha256"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "development_v2_to_assembly"):
            checker.recorded_graph_checks(self.a)

    def test_separate_branches_must_match_their_recorded_upstream_inputs(self):
        self.a["development_risk"]["inputs"]["prediction_pairs"]["sha256"] = "3" * 64
        with self.assertRaisesRegex(ValueError, "pairs_development_pairs_to_risk"):
            checker.recorded_graph_checks(self.a)

    def test_categorical_or_feature_order_mismatch_rejected(self):
        for field in ("categorical_columns", "feature_columns"):
            a = copy.deepcopy(self.a)
            a["risk"][field].reverse()
            with self.assertRaises(ValueError):
                checker.recorded_graph_checks(a)

    def test_missing_original_seed_rejected(self):
        self.a["value"]["seeds"] = [13, 42]
        with self.assertRaisesRegex(ValueError, "value_seeds"):
            checker.recorded_graph_checks(self.a)

    def test_opened_test_status_cannot_be_numeric_or_unopened(self):
        for value in (False, 1, "true"):
            self.a["join"]["labels_opened"] = value
            with self.assertRaisesRegex(ValueError, "unopened"):
                checker.recorded_graph_checks(self.a)

    def test_oof_flag_must_be_real_boolean(self):
        self.a["development_value"]["all_predictions_oof"] = 1
        with self.assertRaisesRegex(ValueError, "out-of-sample"):
            checker.recorded_graph_checks(self.a)

    def test_default_reads_no_target_csv_or_model(self):
        self.fixture()
        with patch.object(checker, "inspect_models", side_effect=AssertionError("No model access")), patch.object(checker, "inspect_target_schemas", side_effect=AssertionError("No CSV access")):
            report = checker.reconstruct(self.root)
        self.assertFalse(report["empirical_replay_completed"])
        self.assertFalse(report["data_rows_parsed"])
        self.assertTrue(report["target_branches"]["upstream_inputs_equal_but_target_table_bytes_different"])
        self.assertEqual(len(report["pretest_freeze"]["row_entries_not_opened"]), 5)

    def test_opt_ins_are_explicit_and_separate(self):
        self.fixture()
        with patch.object(checker, "inspect_models", return_value={"toy": True}) as models, patch.object(checker, "inspect_target_schemas", return_value={"toy": True}) as headers:
            report = checker.reconstruct(self.root, include_schemas=True)
            models.assert_not_called()
            headers.assert_called_once()
        self.assertFalse(report["saved_final_models_inspected"])

    def test_freeze_duplicate_and_incomplete_inventory_rejected(self):
        self.fixture()
        p = self.root / checker.FREEZE
        original = p.read_text()
        for text in (original + original.splitlines()[0] + "\n", "\n".join(original.splitlines()[:-1])):
            p.write_text(text)
            with self.assertRaises(ValueError):
                checker.freeze_check(self.root, self.a)

    def test_extra_unclassified_freeze_entry_rejected(self):
        self.fixture()
        p = self.root / checker.FREEZE
        p.write_text(p.read_text() + "1" * 64 + "  private.csv\n")
        with self.assertRaisesRegex(ValueError, "Unclassified"):
            checker.freeze_check(self.root, self.a)

    def test_recorded_secondary_score_binding_kept_without_opening_it(self):
        self.fixture()
        self.a["join"]["inputs"]["additional_frozen_policy_scores_sha256"] = "4" * 64
        with self.assertRaisesRegex(ValueError, "recorded graph"):
            checker.freeze_check(self.root, self.a)

    def test_header_probe_hashes_bytes_but_parses_no_data_rows(self):
        old = ["listing_id", "action_id", "before_prediction_log", "make", "year", "value_absolute_price_error"]
        for role in ("development_value", "development_risk", "calibration"):
            cols = old + (["risk_target_absolute_price_error_before"] if role == "development_risk" else [])
            relative = checker.AUDITS[role].rsplit("/", 1)[0] + "/bvival_action_dataset.csv"
            p = self.write(relative, ",".join(cols) + "\nNOT_A_VALID_DATA_ROW_AND_MUST_NOT_BE_PARSED\n")
            self.a[role]["output_sha256"] = checker.sha256(p)
        report = checker.inspect_target_schemas(self.root, self.a)
        self.assertFalse(report["data_rows_parsed"])
        self.assertTrue(report["row_artifact_opaque_bytes_hashed"])
        self.assertEqual(report["added_in_risk_v2"], ["risk_target_absolute_price_error_before"])
        self.assertTrue(report["common_column_order_preserved"])

    def test_header_duplicate_and_hash_drift_rejected(self):
        for defect in ("duplicate", "hash"):
            a = copy.deepcopy(self.a)
            for role in ("development_value", "development_risk", "calibration"):
                relative = checker.AUDITS[role].rsplit("/", 1)[0] + "/bvival_action_dataset.csv"
                p = self.write(relative, "before_prediction_log,make,year" + (",make" if defect == "duplicate" else "") + "\n")
                a[role]["output_sha256"] = checker.sha256(p) if defect == "duplicate" else "4" * 64
            with self.assertRaises(ValueError):
                checker.inspect_target_schemas(self.root, a)


if __name__ == "__main__":
    unittest.main()
