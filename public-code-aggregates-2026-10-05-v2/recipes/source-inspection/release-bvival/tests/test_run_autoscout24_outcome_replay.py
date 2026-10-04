"""Synthetic metadata/aggregate tests; native study processes are never run."""
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
import run_autoscout24_outcome_replay as r
from plan_autoscout24_replay import RECEIPT
from check_autoscout24_reconstruction import CONFIG
from test_plan_autoscout24_replay import templates


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="autoscout-outcome-toy-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.scoring = self.root / r.SCORING_ROOT
        self.snapshot = self.scoring / "snapshot-workspace"
        self.output_relative = "experiments/replays/autoscout24-reconstructed-primary-outcomes-toy"
        self.output = self.root / self.output_relative
        self.python = self.write("toy-python", "not executable; mock processes")
        self.plan = {"planned_output_root": r.SCORING_ROOT, "historical_tests_already_opened": True,
                     "stages": templates(), "source_binding": {"sha256": "a" * 64},
                     "structural_contract": {"test_listing_count": 2, "test_action_pair_count": 4},
                     "outcome_concordance": {"reference_directory": "toy-reference", "reference_aggregate_hashes": {}}}
        self.plan["stages"] = copy.deepcopy(self.plan["stages"])
        for s in self.plan["stages"]:
            s["argv_template"] = [v.replace("experiments/replays/autoscout24-reconstructed-toy", r.SCORING_ROOT) if isinstance(v, str)
                                  else {k: v2.replace("experiments/replays/autoscout24-reconstructed-toy", r.SCORING_ROOT) for k, v2 in v.items()}
                                  for v in s["argv_template"]]
        sources = {}
        for stage in self.plan["stages"]:
            relative = stage["argv_template"][1]
            p = self.write(relative, "raise RuntimeError('Toy sources must not execute')\n")
            sources[relative] = r.sha256(p)
        self.plan["static_study_source_closure"] = {"local_source_sha256": sources}
        for relative in r.AUDITS.values():
            self.write(relative, "{}")
        config = self.write(CONFIG, "{}")
        note = self.write("notes/toy-rights.md", "not a permission")
        receipt = {"audit_file_sha256": {p: r.sha256(self.root / p) for p in r.AUDITS.values()},
                   "config_sha256": r.sha256(config), "rights_correction_note": "notes/toy-rights.md",
                   "rights_correction_note_sha256": r.sha256(note)}
        self.write(RECEIPT, json.dumps(receipt))
        self.plan["recovery_receipt_sha256"] = r.sha256(self.root / RECEIPT)
        bindings = r.source_checks(self.root, self.plan)
        for relative in bindings:
            self.write(r.SCORING_ROOT + "/snapshot-workspace/" + relative, (self.root / relative).read_text())
        for name in r.TABLES:
            p = self.write("toy-reference/" + name, "opaque synthetic bytes; not parsed\n")
            self.plan["outcome_concordance"]["reference_aggregate_hashes"][name] = r.sha256(p)
        scores = self.write(r.SCORING_ROOT + "/scores/frozen_policy_scores_absolute_price_error.csv", "synthetic score bytes\n")
        self.write(r.SCORING_ROOT + "/cohort/sealed_labels/test_labels.csv", "synthetic labels; do not parse\n")
        self.write(r.SCORING_ROOT + "/final-pairs/evaluation_prediction_pairs.csv", "synthetic pairs\n")
        self.plan["structural_contract"]["historical_primary_score_sha256"] = r.sha256(scores)
        plan_path = self.write(r.SCORING_ROOT + "/bound-plan.json", json.dumps(self.plan))
        self.freeze = {"stage": "RECONSTRUCTED_AUTOSCOUT24_SCORES_FROZEN_NO_OUTCOME_EVALUATION",
                       "completed_scoring_stages": [{"stage": s} for s in r.SCORING_IDS],
                       "plan_sha256": r.sha256(plan_path), "study_and_recovery_source_sha256": bindings,
                       "runner_and_helper_sha256": {}, "original_source_sha256_checked": "a" * 64,
                       "new_score_sha256": r.sha256(scores), "historical_tests_already_opened": True,
                       "score_bytes_match_recorded_historical_hash": True,
                       "runtime_validation": {"python_executable_sha256": r.sha256(self.python)}}
        for key in ("outcome_join_called", "test_performance_computed", "full_primary_table_concordance_verified",
                    "independent_replication_claimed", "exact_historical_source_identity_claimed",
                    "rights_adjudicated", "public_release_created", "submission_ready"):
            self.freeze[key] = False
        self.freeze["artifact_sha256"] = {str(p.relative_to(self.scoring)): r.sha256(p) for p in self.scoring.rglob("*") if p.is_file()}
        self.save_freeze()
        for target, replacement in (("EXPECTED_PLAN", r.sha256(plan_path)), ("validate_plan", None), ("print", None)):
            p = patch.object(r, target, replacement) if replacement is not None else patch.object(r, target, create=True)
            p.start(); self.addCleanup(p.stop)

    def write(self, relative, text):
        p = self.root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def save_freeze(self):
        p = self.write(r.SCORING_ROOT + "/scoring-freeze.json", json.dumps(self.freeze))
        self.freeze_hash = r.sha256(p)

    def verify(self):
        with patch.object(r, "EXPECTED_FREEZE", self.freeze_hash):
            return r.verify_scoring(self.root, r.SCORING_ROOT, self.freeze_hash)

    def command(self, stage):
        return r.outcome_command(self.root, self.plan, stage, self.python, self.snapshot, self.output, "b" * 64)

    def table(self, name, rows):
        p = self.root / name
        with p.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
        return p

    def helper_fixture(self):
        for name in r.HELPERS:
            if not (self.root / "release-bvival" / name).exists():
                self.write("release-bvival/" + name, "# Synthetic only\n")

    def execute(self):
        with patch.object(r, "EXPECTED_FREEZE", self.freeze_hash):
            return r.execute(self.root, r.SCORING_ROOT, self.freeze_hash, self.scoring, self.freeze,
                             self.plan, self.python, {}, self.output_relative)

    def test_valid_scoring_gate_reads_opaque_bytes_without_output_creation(self):
        scoring, f, p = self.verify()
        self.assertEqual(scoring, self.scoring)
        self.assertEqual(p, self.plan)
        self.assertFalse(f["test_performance_computed"])
        self.assertFalse(self.output.exists())

    def test_wrong_freeze_hash_or_market_rejected(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            r.verify_scoring(self.root, r.SCORING_ROOT, "0" * 64)
        with self.assertRaisesRegex(ValueError, "bound AutoScout24"):
            r.verify_scoring(self.root, "experiments/replays/jucars-reconstructed-primary-v1", self.freeze_hash)

    def test_numeric_false_not_boolean_and_unopened_status_rejected(self):
        for k in ("outcome_join_called", "rights_adjudicated", "full_primary_table_concordance_verified"):
            self.freeze[k] = 0; self.save_freeze()
            with self.assertRaisesRegex(ValueError, "genuine false"):
                self.verify()
            self.freeze[k] = False
        self.freeze["historical_tests_already_opened"] = False; self.save_freeze()
        with self.assertRaisesRegex(ValueError, "Opened-test"):
            self.verify()

    def test_all_twelve_stages_required(self):
        self.freeze["completed_scoring_stages"].pop(); self.save_freeze()
        with self.assertRaisesRegex(ValueError, "twelve"):
            self.verify()

    def test_generated_score_artifact_drift_stops_gate(self):
        (self.scoring / "scores/frozen_policy_scores_absolute_price_error.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "artifact drift"):
            self.verify()

    def test_historical_reference_drift_rejected(self):
        (self.root / "toy-reference/bvival_action_mix.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "reference drift"):
            self.verify()

    def test_live_source_drift_rejected(self):
        (self.root / next(iter(self.plan["static_study_source_closure"]["local_source_sha256"]))).write_text("changed")
        with self.assertRaises(ValueError):
            self.verify()

    def test_new_hash_used_and_only_outcome_outputs_remapped(self):
        args = self.command(self.plan["stages"][-2])
        self.assertIn("b" * 64, args)
        self.assertIn(str(self.scoring / "cohort/sealed_labels/test_labels.csv"), args)
        self.assertIn(str(self.output / "evaluation-inputs"), args)
        args = self.command(self.plan["stages"][-1])
        self.assertIn(str(self.output / "evaluation-inputs/evaluation_outcomes.csv"), args)
        self.assertEqual(args[args.index("--budgets") + 1], "0.1")

    def test_training_runtime_source_and_deferred_hash_rejected(self):
        for which in ("stage", "runtime", "source", "hash"):
            s = copy.deepcopy(self.plan["stages"][-2])
            if which == "stage": s["id"] = "value"
            elif which == "runtime": s["argv_template"][0] = {}
            elif which == "source": s["argv_template"][1] = "experiments/fit_bvival_value_policy.py"
            else: s["argv_template"][s["argv_template"].index("--expected-policy-scores-sha256") + 1] = {"old": "hash"}
            with self.assertRaises(ValueError): self.command(s)

    def test_cached_historical_inputs_secondary_scores_and_oracle_rejected(self):
        for flag in ("--additional-frozen-policy-scores", "--additional-policy-scores", "--include-oracle-diagnostic"):
            s = copy.deepcopy(self.plan["stages"][-2]); s["argv_template"].append(flag)
            with self.assertRaisesRegex(ValueError, "excluded"): self.command(s)
        s = copy.deepcopy(self.plan["stages"][-2]); s["argv_template"].extend(["--source", "experiments/outputs/old.csv"])
        with self.assertRaisesRegex(ValueError, "substitution"): self.command(s)

    def test_symlink_output_rejected(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.command(self.plan["stages"][-1])

    def summary(self, auc="", harmful=".1", policy="toy"):
        return {"policy": policy, "error_budget_auc": auc, "auc_metric": "post_mae_price", "mean_harmful_action_rate": harmful}

    def test_single_budget_auc_and_no_acquisition_na_preserved(self):
        rows = [self.summary(), self.summary(harmful="", policy="score_no_acquisition")]
        a, b = self.table("a.csv", rows), self.table("b.csv", rows)
        c = r.compare_table("bvival_policy_summary.csv", a, b)
        self.assertTrue(c["passes_exact_rules"])
        self.assertEqual(c["matched_not_applicable_cells"], 3)

    def test_auc_zero_imputation_rejected(self):
        a, b = self.table("a.csv", [self.summary()]), self.table("b.csv", [self.summary(auc="0")])
        with self.assertRaisesRegex(ValueError, "never zero-imputed"):
            r.compare_table("bvival_policy_summary.csv", a, b)

    def test_summary_numeric_changes_fail_with_zero_tolerance(self):
        a = self.table("a.csv", [self.summary()]); b = self.table("b.csv", [self.summary(harmful=".1000000001")])
        c = r.compare_table("bvival_policy_summary.csv", a, b)
        self.assertFalse(c["passes_exact_rules"])
        self.assertEqual(c["mismatches"][0]["tolerance"], 0)

    def test_numeric_equal_but_bytes_different_fails(self):
        a = self.table("a.csv", [self.summary(harmful="0.1")]); b = self.table("b.csv", [self.summary(harmful=".1")])
        c = r.compare_table("bvival_policy_summary.csv", a, b)
        self.assertTrue(c["passes_fixed_rules"])
        self.assertFalse(c["passes_exact_rules"])

    def test_missing_summary_rows_are_not_silently_omitted(self):
        a = self.table("a.csv", [self.summary(), self.summary(policy="other")]); b = self.table("b.csv", [self.summary()])
        self.assertFalse(r.compare_table("bvival_policy_summary.csv", a, b)["passes_exact_rules"])

    def test_nonfinite_or_unexpected_missing_summary_rejected(self):
        for value in ("", "nan", "inf"):
            a, b = self.table("a.csv", [self.summary(harmful=value)]), self.table("b.csv", [self.summary(harmful=value)])
            with self.assertRaises(ValueError): r.compare_table("bvival_policy_summary.csv", a, b)

    def test_reference_interval_and_n_match_exactly(self):
        row = {"policy": "toy", "reference_policy": "risk", "budget": ".1", "n": "2", "relative_mae_gain_percent": "2.3", "ci_lower_percent": "1.7", "ci_upper_percent": "2.9"}
        a, b = self.table("a.csv", [row]), self.table("b.csv", [row])
        self.assertTrue(r.compare_table("bvival_reference_comparisons.csv", a, b)["passes_exact_rules"])
        self.table(b.name, [{**row, "ci_lower_percent": "1.70000001"}])
        self.assertFalse(r.compare_table("bvival_reference_comparisons.csv", a, b)["passes_exact_rules"])

    def test_failed_process_preserved_and_repeating_output_refused(self):
        self.helper_fixture()
        with patch.object(r.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["toy"])) as child:
            with self.assertRaises(subprocess.CalledProcessError): self.execute()
            self.assertEqual(child.call_count, 1)
        failed = json.loads((self.output / "failed-outcome-replay.json").read_text())
        self.assertEqual(failed["phase"], "join")
        self.assertTrue(failed["partial_directory_retained"])
        with self.assertRaisesRegex(ValueError, "nonexistent"): self.execute()

    def test_join_gate_failure_prevents_evaluation(self):
        self.helper_fixture()
        with patch.object(r.subprocess, "run") as child, patch.object(r, "joined_outputs", side_effect=ValueError("toy join gate")):
            with self.assertRaisesRegex(ValueError, "toy join gate"): self.execute()
            self.assertEqual(child.call_count, 1)
        self.assertFalse((self.output / "evaluation-mae").exists())


if __name__ == "__main__":
    unittest.main()
