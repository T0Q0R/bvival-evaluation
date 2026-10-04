"""Synthetic checks only; no study data or native study training."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_autoscout24_scoring_replay as r
from test_plan_autoscout24_replay import templates


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.output = self.root / "experiments/replays/autoscout24-reconstructed-toy"
        self.stages = templates()
        self.plan = {"stages": self.stages, "planned_output_root": str(self.output.relative_to(self.root)),
                     "source_binding": {"path": "experiments/data/external/source.csv", "sha256": "a" * 64},
                     "static_study_source_closure": {"local_source_sha256": {s["argv_template"][1]: "a" * 64 for s in self.stages}},
                     "recovery_receipt_sha256": "a" * 64,
                     "structural_contract": {"allowed_cohort_metadata_differences": ["source_rights", "source_path"]}}
        self.python = self.root / "toy-python"
        self.python.write_text("not executable")
        self.snapshot = self.root / "snapshot"
        for path in self.plan["static_study_source_closure"]["local_source_sha256"]:
            p = self.snapshot / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# Never executed\n")

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def table(self, path, header, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def command(self, i):
        return r.stage_command(self.root, self.plan, self.stages[i], self.python, self.snapshot)

    def test_exact_twelve_scoring_stages_and_no_outcome_execution(self):
        self.assertEqual(len(r.SCORING_IDS), 12)
        for i in (12, 13):
            with self.assertRaisesRegex(ValueError, "forbidden"):
                self.command(i)

    def test_bound_source_snapshot_and_absolute_generated_paths(self):
        cmd = self.command(0)
        self.assertEqual(cmd[:3], [str(self.python), "-I", "-B"])
        self.assertIn(str(self.root / self.plan["source_binding"]["path"]), cmd)
        self.assertIn(str(self.output / "cohort"), cmd)
        self.assertIn(str(self.snapshot / self.stages[0]["argv_template"][1]), cmd)

    def test_all_scoring_templates_resolve_without_deferred_arguments(self):
        for i in range(12):
            self.assertTrue(all(isinstance(x, str) for x in self.command(i)))

    def test_old_target_substitution_rejected(self):
        self.stages[4]["argv_template"].extend(["--other", "experiments/outputs/old.csv"])
        with self.assertRaisesRegex(ValueError, "substitution"):
            self.command(4)

    def test_deferred_outcome_hash_rejected(self):
        self.stages[11]["argv_template"].append({"hash": "old"})
        with self.assertRaisesRegex(ValueError, "Deferred"):
            self.command(11)

    def test_unknown_source_runtime_or_phase_rejected(self):
        for which in (0, 1, "phase"):
            stage = copy.deepcopy(self.stages[0])
            if which == "phase":
                stage["phase"] = "evaluation"
            else:
                stage["argv_template"][which] = "unbound"
            with self.assertRaises(ValueError):
                r.stage_command(self.root, self.plan, stage, self.python, self.snapshot)

    def test_opaque_source_hash_checks_no_output(self):
        p = self.root / self.plan["source_binding"]["path"]
        p.parent.mkdir(parents=True)
        p.write_bytes(b"synthetic source only")
        self.plan["source_binding"]["sha256"] = r.sha256(p)
        r.original_input_checks(self.root, self.plan)
        p.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            r.original_input_checks(self.root, self.plan)
        self.assertFalse(self.output.exists())

    def test_audit_paths_and_model_serialization_only_are_ignored(self):
        old = {"inputs": {"source": {"path": "old", "sha256": "a"}},
               "models": [{"seed": 13, "model_path": "old", "sha256": "b"}], "count": 1}
        new = copy.deepcopy(old)
        new["inputs"]["source"]["path"] = "new"
        new["models"][0].update(model_path="new", sha256="c")
        r.compare_audit(new, old)
        new["inputs"]["source"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "contract differs"):
            r.compare_audit(new, old)

    def test_count_or_calibrator_change_is_not_ignored(self):
        for new in ({"count": 2, "calibrator": {"alpha": .1}}, {"count": 1, "calibrator": {"alpha": .2}}):
            with self.assertRaises(ValueError):
                r.compare_audit(new, {"count": 1, "calibrator": {"alpha": .1}})

    def cohort(self):
        p = self.table(self.output / "cohort/toy.csv", ["listing_id"], [["x"]])
        hashes = {"toy.csv": r.sha256(p)}
        old = {"output_hashes": hashes, "split_counts": {"train": 1}, "source_path": "old", "source_rights": "old", "raw_rows": 1}
        new = dict(old, source_path="new", source_rights="corrected")
        self.write(self.root / r.AUDITS["cohort"], old)
        self.write(self.output / "cohort/autoscout24_bvival_audit.json", new)
        self.plan["structural_contract"].update(six_cohort_data_hashes=hashes, split_counts={"train": 1})
        return p, new

    def test_cohort_rights_correction_allowed_but_not_data_drift(self):
        p, _ = self.cohort()
        result = r.cohort_checks(self.root, self.output, self.plan)
        self.assertEqual(result["metadata_differences"], ["source_path", "source_rights"])
        p.write_text("changed")
        with self.assertRaisesRegex(ValueError, "generated bytes"):
            r.cohort_checks(self.root, self.output, self.plan)

    def test_cohort_unapproved_metadata_drift_rejected(self):
        _, new = self.cohort()
        new["raw_rows"] = 2
        self.write(self.output / "cohort/autoscout24_bvival_audit.json", new)
        with self.assertRaisesRegex(ValueError, "beyond approved"):
            r.cohort_checks(self.root, self.output, self.plan)

    def test_metadata_exception_cannot_expand(self):
        self.cohort()
        self.plan["structural_contract"]["allowed_cohort_metadata_differences"].append("raw_rows")
        with self.assertRaisesRegex(ValueError, "not be widened"):
            r.cohort_checks(self.root, self.output, self.plan)

    def decision(self):
        a = {"selected_action": "mileage_km", "listing_count": 14160, "capacity": 1416, "evaluation_labels_read": False,
             "input_sha256": {}}
        for role, path in (("development_actions", "selection-validation-targets/bvival_action_dataset.csv"),
                           ("development_risk_scores", "selection-risk/evaluation_risk_scores.csv")):
            p = self.table(self.output / path, ["listing_id", "action_id"], [["x", "a"]])
            a["input_sha256"][role] = r.sha256(p)
        p = self.output / "fixed-field/selection.json"
        self.write(p, a)
        self.plan["structural_contract"].update(fixed_field_decision_sha256=r.sha256(p), validation_fixed_field="mileage_km",
                                              validation_listing_count=14160, validation_capacity=1416)
        return a

    def test_fixed_field_uses_new_validation_inputs(self):
        self.decision()
        r.fixed_field_checks(self.root, self.output, self.plan)
        (self.output / "selection-risk/evaluation_risk_scores.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "newly generated"):
            r.fixed_field_checks(self.root, self.output, self.plan)

    def test_test_capacity_cannot_replace_validation_capacity(self):
        a = self.decision()
        a.update(listing_count=14005, capacity=1401)
        self.write(self.output / "fixed-field/selection.json", a)
        self.plan["structural_contract"]["fixed_field_decision_sha256"] = r.sha256(self.output / "fixed-field/selection.json")
        with self.assertRaisesRegex(ValueError, "validation capacity"):
            r.fixed_field_checks(self.root, self.output, self.plan)

    def test_score_has_eight_not_seven_columns(self):
        p = self.table(self.root / "scores.csv", ["listing_id", "action_id", *r.SCORES], [["x", "a", *([0] * 8)]])
        self.assertTrue(r.score_checks(p, 1, 1)["eight_score_columns_validated"])
        self.table(p, ["listing_id", "action_id", *r.SCORES[:-1]], [["x", "a", *([0] * 7)]])
        with self.assertRaises(ValueError):
            r.score_checks(p, 1, 1)

    def test_score_nonfinite_rules(self):
        for value, column, allowed in (("-inf", "score_fixed_field_risk", True), ("nan", "score_mean_value", False),
                                       ("inf", "score_uncertainty_only", False), ("-inf", "score_mean_value", False)):
            row = ["x", "a", *([0] * 8)]
            row[2 + r.SCORES.index(column)] = value
            p = self.table(self.root / "scores.csv", ["listing_id", "action_id", *r.SCORES], [row])
            if allowed:
                r.score_checks(p, 1, 1)
            else:
                with self.assertRaises(ValueError):
                    r.score_checks(p, 1, 1)

    def test_wrong_plan_stops_before_execution_and_directory_creation(self):
        with patch.object(r.subprocess, "run", side_effect=AssertionError("No execution")):
            with self.assertRaisesRegex(ValueError, "plan identity"):
                r.execute_scoring(self.root, self.plan, b"{}", self.python, None, None, {})
        self.assertFalse(self.output.exists())

    def test_new_model_audit_hash_checked_not_just_file_count(self):
        p = self.output / "selection-pairs/train_only_base_seed_13.cbm"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"synthetic not a real model")
        records = [{"seed": 13, "sha256": r.sha256(p)}]
        r.verify_model_records(p.parent, records, train_only=True)
        p.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "model bytes"):
            r.verify_model_records(p.parent, records, train_only=True)

    def test_model_outside_new_stage_rejected(self):
        records = [{"seed": 13, "path": str(self.root / "old.cbm"), "sha256": "a" * 64}]
        with self.assertRaisesRegex(ValueError, "outside new stage"):
            r.verify_model_records(self.output / "final-pairs", records)

    def test_model_relative_to_new_stage_and_hash_are_required(self):
        p = self.output / "value/models/seed_13.cbm"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"toy")
        r.verify_model_records(self.output / "value", [{"seed": 13, "model_path": str(p), "sha256": r.sha256(p)}])

    def test_score_count_or_duplicate_rejected(self):
        p = self.table(self.root / "scores.csv", ["listing_id", "action_id", *r.SCORES], [["x", "a", *([0] * 8)]])
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            r.score_checks(p, 2, 1)
        self.table(p, ["listing_id", "action_id", *r.SCORES], [["x", "a", *([0] * 8)]] * 2)
        with self.assertRaises(ValueError):
            r.score_checks(p, 1, 2)

    def test_target_byte_mismatch_cannot_pass_on_matching_audit(self):
        stage = "selection-train-targets"
        p = self.table(self.output / stage / "bvival_action_dataset.csv", ["listing_id", "action_id"], [["x", "a"]])
        a = {"output_sha256": r.sha256(p)}
        self.write(self.output / stage / "bvival_action_dataset_audit.json", a)
        self.write(self.root / r.AUDITS["selection_train"], a)
        self.plan["structural_contract"]["four_generated_target_hashes"] = {"selection_train": "0" * 64}
        with patch.object(r, "pair_checks"):
            with self.assertRaisesRegex(ValueError, "target bytes differ"):
                r.target_checks(self.root, self.output, self.plan, stage)

    def test_cohort_failure_prevents_any_actual_study_fit(self):
        data = json.dumps(self.plan).encode()
        for n in r.HELPERS:
            self.write(self.root / "release-bvival" / n, {"synthetic": True})
        with patch.object(r, "EXPECTED_PLAN", hashlib.sha256(data).hexdigest()), patch.object(r, "validate_plan"), \
                patch.object(r, "source_checks", return_value={}), patch.object(r, "original_input_checks"), \
                patch.object(r, "stage_command", return_value=["toy"]), patch.object(r, "runtime_hash_checks"), \
                patch.object(r.subprocess, "run") as child, patch.object(r, "print", create=True), \
                patch.object(r, "cohort_checks", side_effect=ValueError("cohort mismatch")):
            with self.assertRaisesRegex(ValueError, "cohort mismatch"):
                r.execute_scoring(self.root, self.plan, data, self.python, self.python, self.python, {})
            self.assertEqual(child.call_count, 1)
            failed = json.loads((self.output / "failed-scoring-replay.json").read_text())
            self.assertEqual(failed["phase"], "cohort")
            self.assertEqual(failed["completed_scoring_stages"], [])

    def test_preflight_hash_failure_starts_no_process(self):
        p = self.root / "plan.json"
        p.write_text("{}")
        with patch.object(r.subprocess, "run", side_effect=AssertionError("No process")):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                r.preflight(self.root, p, self.python, p, p, "a" * 64, "a" * 64)

    def test_failed_process_preserved_and_retry_refused(self):
        data = json.dumps(self.plan).encode()
        for n in r.HELPERS:
            self.write(self.root / "release-bvival" / n, {"synthetic": True})
        with patch.object(r, "EXPECTED_PLAN", hashlib.sha256(data).hexdigest()), patch.object(r, "validate_plan"), \
                patch.object(r, "source_checks", return_value={}), patch.object(r, "original_input_checks"), \
                patch.object(r, "stage_command", return_value=["toy"]), \
                patch.object(r, "runtime_hash_checks"), patch.object(r.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["toy"])) as child, \
                patch.object(r, "print", create=True):
            with self.assertRaises(subprocess.CalledProcessError):
                r.execute_scoring(self.root, self.plan, data, self.python, self.python, self.python, {})
            failed = json.loads((self.output / "failed-scoring-replay.json").read_text())
            self.assertEqual(failed["phase"], "cohort")
            self.assertTrue(failed["partial_directory_retained_no_overwrite_or_automatic_retry"])
            self.assertFalse(failed["test_performance_computed"])
            self.assertEqual(child.call_count, 1)
            with self.assertRaises(FileExistsError):
                r.execute_scoring(self.root, self.plan, data, self.python, self.python, self.python, {})
            self.assertFalse((self.output / "evaluation-inputs").exists())


if __name__ == "__main__":
    unittest.main()
