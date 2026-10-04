"""Toy aggregates/templates only. No study labels, fitting or evaluation."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_mucars_outcome_replay as runner


class OutcomeReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="outcome-replay-toy-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rules = {"price_mae_absolute_tolerance_MAD": 0.05, "rmsle_absolute_tolerance": 1e-6,
                      "relative_gain_and_ci_endpoint_tolerance_percentage_points": 0.01,
                      "fraction_metrics_absolute_tolerance": 1e-6}
        self.row = {"n": "20", "relative_mae_gain_percent": "1.5", "ci_lower_percent": "1.0",
                    "ci_upper_percent": "1.9", "policy": "score_mean_value",
                    "reference_policy": "score_uncertainty_only", "budget": "0.10"}
        self.name = "bvival_reference_comparisons.csv"

    def write(self, filename, rows):
        path = self.root / filename
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def compare(self, changes=None, replay_rows=None):
        reference = self.write("reference.csv", [self.row])
        replay = self.write("replay.csv", replay_rows or [{**self.row, **(changes or {})}])
        return runner.compare_table(self.name, reference, replay, self.rules)

    def test_identical_tables_pass(self):
        report = self.compare()
        self.assertTrue(report["passes_fixed_rules"])
        self.assertTrue(report["byte_hash_matches"])

    def test_numeric_tolerances_are_percentage_points(self):
        self.assertTrue(self.compare({"relative_mae_gain_percent": "1.509"})["passes_fixed_rules"])
        report = self.compare({"relative_mae_gain_percent": "1.511"})
        self.assertFalse(report["passes_fixed_rules"])
        self.assertEqual(report["mismatches"][0]["tolerance"], 0.01)

    def test_count_difference_fails_exact_rule(self):
        report = self.compare({"n": "21"})
        self.assertFalse(report["passes_fixed_rules"])
        self.assertEqual(report["mismatches"][0]["tolerance"], 0)

    def test_missing_extra_keys_are_not_silently_aligned_away(self):
        report = self.compare({"policy": "score_other"})
        self.assertFalse(report["key_sets_match"])
        self.assertEqual({x["kind"] for x in report["mismatches"]}, {"missing_key", "extra_key"})

    def test_budget_key_numeric_format_does_not_create_false_mismatch(self):
        self.assertTrue(self.compare({"budget": "0.1"})["passes_fixed_rules"])

    def test_duplicate_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.compare(replay_rows=[self.row, self.row])

    def test_unknown_column_cannot_escape_comparison(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            self.compare({"new_metric": "1.0"})

    def test_nonfinite_or_noninteger_metric_rejected(self):
        for changes in ({"relative_mae_gain_percent": "nan"}, {"n": "20.0"}, {"ci_lower_percent": "inf"}):
            with self.assertRaises(ValueError):
                self.compare(changes)

    def test_zero_boundary_changed_even_inside_tolerance(self):
        self.row["ci_lower_percent"] = "0.001"
        report = self.compare({"ci_lower_percent": "-0.001"})
        self.assertFalse(report["mismatches"])
        self.assertFalse(report["passes_fixed_rules"])
        self.assertEqual(report["changed_interval_boundaries"][0]["threshold_percent"], 0)

    def test_two_percent_boundary_changed_even_inside_tolerance(self):
        self.row["ci_upper_percent"] = "1.999"
        report = self.compare({"ci_upper_percent": "2.001"})
        self.assertFalse(report["mismatches"])
        self.assertFalse(report["passes_fixed_rules"])
        self.assertEqual(report["changed_interval_boundaries"][0]["threshold_percent"], 2)

    def test_interval_order_must_be_valid(self):
        with self.assertRaisesRegex(ValueError, "interval"):
            self.compare({"ci_lower_percent": "2.0"})

    def test_blank_harm_rate_only_for_zero_actions(self):
        row = {"listing_count": "20", "action_count": "0", "action_rate": "0", "before_rmsle": "0.5",
               "post_rmsle": "0.5", "relative_rmsle_improvement_percent": "0", "before_mae_price": "100",
               "post_mae_price": "100", "harmful_action_rate": "", "policy": "score_no_acquisition",
               "budget": "0.1", "positive_gain_capture": "0"}
        a, b = self.write("a.csv", [row]), self.write("b.csv", [row])
        report = runner.compare_table("bvival_error_budget_curve.csv", a, b, self.rules)
        self.assertTrue(report["passes_fixed_rules"])
        self.assertEqual(report["matched_not_applicable_cells"], 1)
        row["action_count"] = "1"
        a, b = self.write("a.csv", [row]), self.write("b.csv", [row])
        with self.assertRaisesRegex(ValueError, "missing metric"):
            runner.compare_table("bvival_error_budget_curve.csv", a, b, self.rules)

    def test_price_and_fraction_tolerances_are_not_swapped(self):
        row = {"policy": "toy", "error_budget_auc": "100", "auc_metric": "post_mae_price", "mean_harmful_action_rate": "0.1"}
        a = self.write("a.csv", [row])
        b = self.write("b.csv", [{**row, "error_budget_auc": "100.04"}])
        self.assertTrue(runner.compare_table("bvival_policy_summary.csv", a, b, self.rules)["passes_fixed_rules"])
        b = self.write("b.csv", [{**row, "mean_harmful_action_rate": "0.100002"}])
        self.assertFalse(runner.compare_table("bvival_policy_summary.csv", a, b, self.rules)["passes_fixed_rules"])

    def test_summary_auc_metric_cannot_change(self):
        row = {"policy": "toy", "error_budget_auc": "100", "auc_metric": "post_rmsle", "mean_harmful_action_rate": "0.1"}
        a, b = self.write("a.csv", [row]), self.write("b.csv", [row])
        self.assertFalse(runner.compare_table("bvival_policy_summary.csv", a, b, self.rules)["passes_fixed_rules"])

    def test_zero_action_summary_has_undefined_harmful_rate(self):
        row = {"policy": "score_no_acquisition", "error_budget_auc": "100", "auc_metric": "post_mae_price", "mean_harmful_action_rate": ""}
        a, b = self.write("a.csv", [row]), self.write("b.csv", [row])
        report = runner.compare_table("bvival_policy_summary.csv", a, b, self.rules)
        self.assertTrue(report["passes_fixed_rules"])
        self.assertEqual(report["matched_not_applicable_cells"], 1)
        row["policy"] = "score_mean_value"
        a, b = self.write("a.csv", [row]), self.write("b.csv", [row])
        with self.assertRaisesRegex(ValueError, "missing metric"):
            runner.compare_table("bvival_policy_summary.csv", a, b, self.rules)

    def stage_fixture(self, kind):
        prefix = "experiments/replays/mucars-reconstructed-toy-v1"
        script = "experiments/" + ("join_bvival_evaluation_outcomes.py" if kind == "join" else "evaluate_bvival_policies.py")
        plan = {"planned_output_root": prefix, "static_source_closure": {"local_source_sha256": {script: "a" * 64}}}
        stage = {"id": kind, "phase": "evaluation", "argv_template": [{"runtime_binding": "deferred"}, script]}
        if kind == "join":
            stage["argv_template"] += ["--sealed-labels", prefix + "/cohort/sealed_labels/test_labels.csv",
                                      "--expected-policy-scores-sha256", {"frozen_artifact_sha256": prefix + "/scores/frozen_policy_scores_absolute_price_error.csv"},
                                      "--output-dir", prefix + "/evaluation-inputs"]
        else:
            stage["argv_template"] += ["--evaluation-outcomes", prefix + "/evaluation-inputs/evaluation_outcomes.csv",
                                      "--output-dir", prefix + "/evaluation-mae"]
        return plan, stage

    def test_only_new_output_paths_remapped_and_new_hash_bound(self):
        plan, stage = self.stage_fixture("join")
        args = runner.outcome_command(self.root, plan, stage, Path("/toy/venv/bin/python"), self.root / "snapshot", self.root / "new", "b" * 64)
        self.assertIn("b" * 64, args)
        self.assertIn(str(self.root / "new/evaluation-inputs"), args)
        self.assertIn(str(self.root / plan["planned_output_root"] / "cohort/sealed_labels/test_labels.csv"), args)
        plan, stage = self.stage_fixture("evaluate")
        args = runner.outcome_command(self.root, plan, stage, Path("/toy/python"), self.root / "snapshot", self.root / "new", "b" * 64)
        self.assertIn(str(self.root / "new/evaluation-inputs/evaluation_outcomes.csv"), args)
        self.assertIn(str(self.root / "new/evaluation-mae"), args)

    def test_training_stage_wrong_source_and_deferred_binding_rejected(self):
        for change in ("stage", "source", "binding", "phase"):
            plan, stage = self.stage_fixture("join")
            if change == "stage":
                stage["id"] = "value"
            elif change == "source":
                stage["argv_template"][1] = "experiments/fit_bvival_value_policy.py"
            elif change == "binding":
                stage["argv_template"][5] = {"unbound": "wrong"}
            else:
                stage["phase"] = "scoring"
            with self.assertRaises(ValueError):
                runner.outcome_command(self.root, plan, stage, Path("/toy/python"), self.root, self.root / "new", "b" * 64)

    def test_bad_freeze_hash_stops_before_any_runtime_or_label_read(self):
        path = self.root / "toy/scoring-freeze.json"
        path.parent.mkdir()
        path.write_text("{}")
        with patch.object(runner, "static_import_closure") as closure:
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                runner.verify_scoring(self.root, "toy", "a" * 64)
            closure.assert_not_called()

    def test_numeric_false_boundary_flag_rejected(self):
        path = self.root / "toy/scoring-freeze.json"
        path.parent.mkdir()
        data = {"stage": "RECONSTRUCTED_MUCARS_SCORES_FROZEN_NO_OUTCOME_EVALUATION", "outcome_join_called": 0}
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "genuine false"):
            runner.verify_scoring(self.root, "toy", runner.sha256(path))


if __name__ == "__main__":
    unittest.main()
