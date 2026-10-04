"""Synthetic snapshots/aggregates only; no study label reads or model fits."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_jucars_outcome_replay as runner


class JUCarsOutcomeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="jucars-outcome-toy-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        quiet = patch.object(runner, "print", create=True)
        quiet.start()
        self.addCleanup(quiet.stop)
        self.prefix = runner.SCORING_ROOT
        self.scoring = self.root / self.prefix
        self.snapshot = self.scoring / "snapshot-workspace"
        self.output = "experiments/replays/jucars-reconstructed-primary-outcomes-toy-v1"
        self.rules = {"price_mae_absolute_tolerance_JOD": .05, "rmsle_absolute_tolerance": 1e-6,
                      "relative_gain_and_ci_endpoint_tolerance_percentage_points": .01,
                      "fraction_metrics_absolute_tolerance": 1e-6, "original_aggregate_reference": "toy-reference"}
        sources = {}
        for name in ("join_bvival_evaluation_outcomes.py", "evaluate_bvival_policies.py"):
            relative = "experiments/" + name
            self.write(self.prefix + "/snapshot-workspace/" + relative, "raise RuntimeError('Must not execute toy source')\n")
            self.write(relative, (self.snapshot / relative).read_text())
            sources[relative] = runner.sha256(self.snapshot / relative)
        self.plan = {"planned_output_root": self.prefix, "historical_tests_already_opened": True,
                     "static_study_source_closure": runner.static_import_closure(self.snapshot, list(sources)),
                     "stages": [], "engineering_concordance_rules": self.rules,
                     "expected_structural_contract": {"phase_counts": {"evaluation": {"listings": 2, "pairs": 4}}}}
        for name in ("join", "evaluate"):
            source = list(sources)[0 if name == "join" else 1]
            args = [dict(runner.RUNTIME), source]
            if name == "join":
                args += ["--sealed-labels", self.prefix + "/cohort/sealed_labels/test_labels.csv",
                         "--frozen-policy-scores", self.prefix + "/scores/frozen_policy_scores_absolute_price_error.csv",
                         "--expected-policy-scores-sha256", {"frozen_artifact_sha256": self.prefix + "/scores/frozen_policy_scores_absolute_price_error.csv"},
                         "--output-dir", self.prefix + "/evaluation-inputs"]
            else:
                args += ["--evaluation-outcomes", self.prefix + "/evaluation-inputs/evaluation_outcomes.csv",
                         "--output-dir", self.prefix + "/evaluation-mae"]
            self.plan["stages"].append({"id": name, "phase": "evaluation", "argv_template": args})
        for role, path in runner.AUDITS.items():
            self.write(path, "{}")
        recovery = {"audit_file_sha256": {p: runner.sha256(self.root / p) for p in runner.AUDITS.values()}}
        self.write(runner.RECEIPT, json.dumps(recovery))
        self.write(self.prefix + "/snapshot-workspace/" + runner.RECEIPT, json.dumps(recovery))
        self.write(runner.ADAPTER, "# synthetic adapter\n")
        self.write(self.prefix + "/snapshot-workspace/" + runner.ADAPTER, "# synthetic adapter\n")
        self.plan["recovery_receipt_sha256"] = runner.sha256(self.root / runner.RECEIPT)
        self.plan["schema_adapter_source"] = {"sha256": runner.sha256(self.root / runner.ADAPTER)}
        self.rules["reference_aggregate_hashes"] = {}
        for name in runner.TABLES:
            self.write("toy-reference/" + name, "toy-reference-not-parsed\n")
            self.rules["reference_aggregate_hashes"][name] = runner.sha256(self.root / "toy-reference" / name)
        self.write(self.prefix + "/bound-plan.json", json.dumps(self.plan))
        self.plan_hash = runner.sha256(self.scoring / "bound-plan.json")
        bound = {**sources, runner.ADAPTER: self.plan["schema_adapter_source"]["sha256"],
                 runner.RECEIPT: self.plan["recovery_receipt_sha256"]}
        self.write(self.prefix + "/scores/frozen_policy_scores_absolute_price_error.csv", "synthetic_scores\n")
        self.write(self.prefix + "/cohort/sealed_labels/test_labels.csv", "synthetic_labels_not_parsed\n")
        self.write(self.prefix + "/pairs/evaluation_prediction_pairs.csv", "synthetic_pairs_not_parsed\n")
        self.freeze = {"stage": "RECONSTRUCTED_JUCARS_SCORES_FROZEN_NO_OUTCOME_EVALUATION",
                       "historical_test_labels_already_opened": True, "original_source_and_dictionary_hashes_checked": True,
                       "completed_scoring_stages": [{"stage": s} for s in runner.SCORING_IDS],
                       "plan_sha256": self.plan_hash, "study_and_recovery_source_sha256": bound, "runner_and_helper_sha256": {},
                       "new_score_sha256": runner.sha256(self.scoring / "scores/frozen_policy_scores_absolute_price_error.csv")}
        for name in ("outcome_join_called", "test_performance_computed", "independent_replication_claimed", "submission_ready", "historical_source_identity_claimed", "secondary_score_guard_replayed"):
            self.freeze[name] = False
        self.freeze["artifact_sha256"] = {str(p.relative_to(self.scoring)): runner.sha256(p) for p in self.scoring.rglob("*") if p.is_file()}
        self.save_freeze()
        self.python = self.root / "toy-python"
        self.python.write_text("No executable; subprocess must be mocked")
        self.freeze["runtime_validation"] = {"python_executable_sha256": runner.sha256(self.python)}
        self.save_freeze()
        pin = patch.object(runner, "EXPECTED_PLAN", self.plan_hash)
        pin.start()
        self.addCleanup(pin.stop)

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def save_freeze(self):
        p = self.write(self.prefix + "/scoring-freeze.json", json.dumps(self.freeze))
        self.freeze_hash = runner.sha256(p)

    def verify(self):
        return runner.verify_scoring(self.root, self.prefix, self.freeze_hash)

    def command(self, stage):
        return runner.outcome_command(self.root, self.plan, stage, self.python, self.snapshot, self.root / self.output, "a" * 64)

    def table(self, name, row):
        path = self.root / name
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return path

    def test_valid_freeze_checks_metadata_and_opaque_bytes_only(self):
        scoring, f, p = self.verify()
        self.assertEqual(scoring, self.scoring)
        self.assertEqual(p, self.plan)
        self.assertFalse(f["test_performance_computed"])
        self.assertFalse((self.root / self.output).exists())

    def test_wrong_hash_stops_before_reading_plan(self):
        with self.assertRaisesRegex(ValueError, "freeze hash"):
            runner.verify_scoring(self.root, self.prefix, "a" * 64)

    def test_other_market_or_root_rejected(self):
        with self.assertRaisesRegex(ValueError, "bound JUCars"):
            runner.verify_scoring(self.root, "experiments/replays/mucars-reconstructed-primary-v1", self.freeze_hash)

    def test_numeric_false_or_unopened_status_rejected(self):
        for name in ("outcome_join_called", "test_performance_computed", "secondary_score_guard_replayed"):
            original = self.freeze[name]
            self.freeze[name] = 0
            self.save_freeze()
            with self.assertRaisesRegex(ValueError, "genuine false"):
                self.verify()
            self.freeze[name] = original
        self.freeze["historical_test_labels_already_opened"] = False
        self.save_freeze()
        with self.assertRaises(ValueError):
            self.verify()

    def test_incomplete_stages_and_artifact_drift_rejected(self):
        original = self.freeze["completed_scoring_stages"]
        self.freeze["completed_scoring_stages"] = original[:-1]
        self.save_freeze()
        with self.assertRaisesRegex(ValueError, "nine"):
            self.verify()
        self.freeze["completed_scoring_stages"] = original
        self.save_freeze()
        (self.scoring / "pairs/evaluation_prediction_pairs.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "artifact drift"):
            self.verify()

    def test_original_audit_and_reference_drift_rejected(self):
        path = self.root / next(iter(runner.AUDITS.values()))
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "audit drift"):
            self.verify()
        path.write_text("{}")
        (self.root / "toy-reference/bvival_action_mix.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "reference drift"):
            self.verify()

    def test_live_source_drift_rejected_even_if_snapshot_unchanged(self):
        path = self.root / next(iter(self.plan["static_study_source_closure"]["local_source_sha256"]))
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Source/helper"):
            self.verify()

    def test_command_uses_new_score_hash_and_remaps_only_outcome_outputs(self):
        args = self.command(self.plan["stages"][0])
        self.assertIn("a" * 64, args)
        self.assertIn(str(self.scoring / "cohort/sealed_labels/test_labels.csv"), args)
        self.assertIn(str(self.root / self.output / "evaluation-inputs"), args)
        args = self.command(self.plan["stages"][1])
        self.assertIn(str(self.root / self.output / "evaluation-inputs/evaluation_outcomes.csv"), args)

    def test_training_source_runtime_and_deferred_bindings_rejected(self):
        for change in ("stage", "source", "runtime", "deferred"):
            s = copy.deepcopy(self.plan["stages"][0])
            if change == "stage":
                s["id"] = "value"
            elif change == "source":
                s["argv_template"][1] = "experiments/fit_bvival_value_policy.py"
            elif change == "runtime":
                s["argv_template"][0] = {}
            else:
                s["argv_template"][s["argv_template"].index("--expected-policy-scores-sha256") + 1] = {"old_hash": "x"}
            with self.assertRaises(ValueError):
                self.command(s)

    def test_cached_secondary_score_guard_forbidden(self):
        s = copy.deepcopy(self.plan["stages"][0])
        s["argv_template"].extend(["--additional-frozen-policy-scores", "old.csv"])
        with self.assertRaisesRegex(ValueError, "secondary"):
            self.command(s)

    def test_symlink_outcome_component_rejected(self):
        (self.root / self.output).symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.command(self.plan["stages"][1])

    def test_comparator_uses_frozen_jod_price_tolerance_without_rule_mutation(self):
        row = {"policy": "toy", "error_budget_auc": "100", "auc_metric": "post_mae_price", "mean_harmful_action_rate": ".1"}
        a = self.table("a.csv", row)
        original = copy.deepcopy(self.rules)
        b = self.table("b.csv", {**row, "error_budget_auc": "100.04"})
        r = runner.compare_table("bvival_policy_summary.csv", a, b, self.rules)
        self.assertTrue(r["passes_fixed_rules"])
        self.assertEqual(r["price_tolerance_unit"], "JOD")
        self.assertEqual(self.rules, original)
        b = self.table("b.csv", {**row, "error_budget_auc": "100.06"})
        self.assertFalse(runner.compare_table("bvival_policy_summary.csv", a, b, self.rules)["passes_fixed_rules"])

    def test_interval_boundary_change_is_retained_inside_numeric_tolerance(self):
        row = {"policy": "toy", "reference_policy": "risk", "budget": ".1", "n": "2", "relative_mae_gain_percent": "0.1", "ci_lower_percent": "0.001", "ci_upper_percent": "1"}
        a, b = self.table("a.csv", row), self.table("b.csv", {**row, "ci_lower_percent": "-0.001"})
        result = runner.compare_table("bvival_reference_comparisons.csv", a, b, self.rules)
        self.assertFalse(result["mismatches"])
        self.assertFalse(result["passes_fixed_rules"])
        self.assertEqual(len(result["changed_interval_boundaries"]), 1)

    def test_join_input_gate_rejects_cached_secondary_and_false_score_guard(self):
        output = self.root / self.output
        joined = self.write(self.output + "/evaluation-inputs/evaluation_outcomes.csv", "synthetic\n")
        audit = {"output_sha256": runner.sha256(joined), "listing_count": 2, "listing_action_count": 4,
                 "labels_opened": True, "policy_score_hash_verified_before_label_read": True,
                 "inputs": {"frozen_policy_scores_sha256": self.freeze["new_score_sha256"],
                            "additional_frozen_policy_scores_sha256": None,
                            "prediction_pairs_sha256": self.freeze["artifact_sha256"]["pairs/evaluation_prediction_pairs.csv"],
                            "sealed_labels_sha256": self.freeze["artifact_sha256"]["cohort/sealed_labels/test_labels.csv"]}}
        self.write(self.output + "/evaluation-inputs/evaluation_outcome_join_audit.json", json.dumps(audit))
        self.assertEqual(runner.joined_outputs(output, self.freeze, self.plan), audit)
        audit["inputs"]["additional_frozen_policy_scores_sha256"] = "a" * 64
        self.write(self.output + "/evaluation-inputs/evaluation_outcome_join_audit.json", json.dumps(audit))
        with self.assertRaises(ValueError):
            runner.joined_outputs(output, self.freeze, self.plan)

    def test_failed_child_retained_without_refit_retry_or_overwrite(self):
        for name in runner.HELPERS:
            path = self.root / "release-bvival" / name
            if not path.exists():
                self.write("release-bvival/" + name, "# toy helper\n")
        with patch.object(runner.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["toy"])) as process:
            with self.assertRaises(subprocess.CalledProcessError):
                runner.execute(self.root, self.prefix, self.freeze_hash, self.scoring, self.freeze, self.plan, self.python, {}, self.output)
        self.assertEqual(process.call_count, 1)
        failure = json.loads((self.root / self.output / "failed-outcome-replay.json").read_text())
        self.assertEqual(failure["phase"], "join")
        self.assertTrue(failure["no_automatic_retry"])
        self.assertFalse((self.root / self.output / "evaluation-mae").exists())
        with self.assertRaises(ValueError):
            runner.execute(self.root, self.prefix, self.freeze_hash, self.scoring, self.freeze, self.plan, self.python, {}, self.output)

    def helper_fixture(self):
        for name in runner.HELPERS:
            if not (self.root / "release-bvival" / name).exists():
                self.write("release-bvival/" + name, "# synthetic helper\n")

    def test_join_gate_failure_prevents_evaluation(self):
        self.helper_fixture()
        with patch.object(runner.subprocess, "run") as child, patch.object(runner, "joined_outputs", side_effect=ValueError("synthetic join mismatch")):
            with self.assertRaisesRegex(ValueError, "synthetic join mismatch"):
                runner.execute(self.root, self.prefix, self.freeze_hash, self.scoring, self.freeze, self.plan, self.python, {}, self.output)
        self.assertEqual(child.call_count, 1)
        self.assertFalse((self.root / self.output / "evaluation-mae").exists())

    def test_success_receipt_is_primary_only_without_training_or_new_confirmation(self):
        self.helper_fixture()
        def child(args, **kwargs):
            directory = Path(args[args.index("--output-dir") + 1])
            directory.mkdir()
            if directory.name == "evaluation-inputs":
                path = directory / "evaluation_outcomes.csv"
                path.write_text("synthetic outcome\n")
                audit = {"output_sha256": runner.sha256(path), "listing_count": 2, "listing_action_count": 4,
                         "labels_opened": True, "policy_score_hash_verified_before_label_read": True,
                         "inputs": {"frozen_policy_scores_sha256": self.freeze["new_score_sha256"],
                                    "additional_frozen_policy_scores_sha256": None,
                                    "prediction_pairs_sha256": self.freeze["artifact_sha256"]["pairs/evaluation_prediction_pairs.csv"],
                                    "sealed_labels_sha256": self.freeze["artifact_sha256"]["cohort/sealed_labels/test_labels.csv"]}}
                (directory / "evaluation_outcome_join_audit.json").write_text(json.dumps(audit))
            else:
                joined = self.root / self.output / "evaluation-inputs/evaluation_outcomes.csv"
                audit = {"analysis_status": "post_test_exploratory", "objective": "absolute_price_error",
                         "reference_policy": "score_uncertainty_only", "bootstrap_repetitions": 10000,
                         "bootstrap_seed": 2026, "budgets": [.01, .05, .1, .2, .3],
                         "evaluation_outcomes_sha256": runner.sha256(joined), "frozen_policy_scores_sha256": self.freeze["new_score_sha256"]}
                (directory / "bvival_evaluation_audit.json").write_text(json.dumps(audit))
                for name in runner.TABLES:
                    (directory / name).write_text("toy table not parsed by mocked comparator\n")
            return subprocess.CompletedProcess(args, 0)
        with patch.object(runner.subprocess, "run", side_effect=child) as process, patch.object(runner, "compare_table", return_value={"passes_fixed_rules": True, "byte_hash_matches": True}):
            r = runner.execute(self.root, self.prefix, self.freeze_hash, self.scoring, self.freeze, self.plan, self.python, {}, self.output)
        self.assertEqual(process.call_count, 2)
        self.assertFalse(r["new_training_or_tuning_called"])
        self.assertFalse(r["independent_replication_claimed"])
        self.assertFalse(r["exact_full_historical_join_protocol_claimed"])
        self.assertTrue(r["test_performance_computed"])
        for relative, digest in r["artifact_sha256"].items():
            self.assertEqual(runner.sha256(self.root / self.output / relative), digest)


if __name__ == "__main__":
    unittest.main()
