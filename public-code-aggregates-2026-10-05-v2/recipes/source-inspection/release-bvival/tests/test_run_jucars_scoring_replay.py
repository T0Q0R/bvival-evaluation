"""Toy metadata/CSV tests; all process execution is mocked, no study training."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_jucars_scoring_replay as runner
import test_plan_jucars_replay as fixture_module


class JUCarsScoringRunnerTests(unittest.TestCase):
    def setUp(self):
        quiet = patch.object(runner, "print", create=True)
        quiet.start()
        self.addCleanup(quiet.stop)
        fixture = fixture_module.JUCarsPlanTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        fixture.receipt["recorded_primary_score_sha256"] = "1" * 64
        fixture.save()
        self.plan = fixture.plan()
        self.data = json.dumps(self.plan).encode()
        self.stages = {s["id"]: s for s in self.plan["stages"]}
        self.snapshot = self.root / "snapshot"
        for relative in {**self.plan["static_study_source_closure"]["local_source_sha256"], runner.ADAPTER: "", runner.RECEIPT: ""}:
            p = self.snapshot / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((self.root / relative).read_bytes())
        self.python = self.root / "toy-python"
        self.python.write_text("Not executable; processes must be mocked")
        self.lock = self.root / "toy.lock"
        self.lock.write_text("toy==1 --hash=sha256:" + "a" * 64 + "\n")
        self.install = self.root / "install.json"
        self.install.write_text("{}")
        self.runtime = {"requirements_lock_sha256": runner.sha256(self.lock),
                        "install_report_sha256": runner.sha256(self.install),
                        "python_executable_sha256": runner.sha256(self.python)}

    def table(self, relative, header, rows):
        p = self.root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        return p

    def command(self, name):
        return runner.stage_command(self.root, self.plan, self.stages[name], self.python, self.snapshot)

    def execute(self):
        return runner.execute_scoring(self.root, self.plan, self.data, self.python, self.lock, self.install, self.runtime)

    def allow_toy_execution(self):
        patches = [patch.object(runner, "EXPECTED_PLAN", hashlib.sha256(self.data).hexdigest()),
                   patch.object(runner, "validate_plan"), patch.object(runner, "original_input_checks")]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        for name in runner.HELPERS:
            path = self.root / "release-bvival" / name
            if not path.exists():
                path.write_text("# synthetic helper, never run\n")

    def test_nine_stage_allowlist_excludes_outcome_templates(self):
        self.assertEqual(len(runner.SCORING_IDS), 9)
        for name in ("join", "evaluate"):
            with self.assertRaisesRegex(ValueError, "forbidden"):
                self.command(name)

    def test_snapshot_absolute_source_dictionary_and_replay_paths(self):
        args = self.command("cohort")
        self.assertEqual(args[:3], [str(self.python), "-I", "-B"])
        self.assertIn(str(self.snapshot / self.stages["cohort"]["argv_template"][1]), args)
        for binding in self.plan["source_bindings_to_verify_before_execution"].values():
            self.assertIn(str(self.root / binding["path"]), args)
        self.assertIn(str(self.root / self.plan["planned_output_root"] / "cohort"), args)

    def test_projection_hash_resolved_only_from_own_new_target_audit(self):
        directory = self.plan["planned_output_root"] + "/development-current"
        path = self.table(directory + "/bvival_action_dataset.csv", ["listing_id", "action_id"], [["toy", "a"]])
        (path.parent / "bvival_action_dataset_audit.json").write_text(json.dumps({"output_sha256": runner.sha256(path)}))
        args = self.command("development-legacy")
        self.assertEqual(args[args.index("--expected-input-sha256") + 1], runner.sha256(path))
        self.assertIn(str(self.snapshot / runner.RECEIPT), args)
        path.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.command("development-legacy")

    def test_wrong_deferred_hash_source_and_outcome_hash_forbidden(self):
        stage = self.stages["development-legacy"]
        index = stage["argv_template"].index("--expected-input-sha256") + 1
        stage["argv_template"][index] = {"generated_target_hash_from_audit": "outside.json"}
        with self.assertRaises(ValueError):
            self.command("development-legacy")
        self.stages["scores"]["argv_template"].append({"frozen_artifact_sha256": "old.csv"})
        with self.assertRaises(ValueError):
            self.command("scores")

    def test_unknown_runtime_source_or_evaluation_phase_rejected(self):
        for which in ("runtime", "source", "phase"):
            stage = copy.deepcopy(self.stages["cohort"])
            if which == "runtime":
                stage["argv_template"][0] = {"runtime_binding": "other"}
            elif which == "source":
                stage["argv_template"][1] = "experiments/unbound.py"
            else:
                stage["phase"] = "evaluation"
            with self.assertRaises(ValueError):
                runner.stage_command(self.root, self.plan, stage, self.python, self.snapshot)

    def test_confined_paths_reject_traversal_and_symlink_components(self):
        for relative in ("../other", "/absolute", "a//b", "a/./b", "a\\b"):
            with self.assertRaises(ValueError):
                runner.confined(self.root, relative)
        (self.root / "linked").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            runner.confined(self.root, "linked/new-output")

    def test_raw_source_and_dictionary_both_required_and_checked(self):
        for binding in self.plan["source_bindings_to_verify_before_execution"].values():
            path = self.root / binding["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic-only\n")
            binding["sha256"] = runner.sha256(path)
        runner.original_input_checks(self.root, self.plan)
        binding = self.plan["source_bindings_to_verify_before_execution"]["dictionary"]
        (self.root / binding["path"]).write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "dictionary hash mismatch"):
            runner.original_input_checks(self.root, self.plan)
        self.assertFalse((self.root / self.plan["planned_output_root"]).exists())

    def test_plan_identity_mismatch_stops_before_output_creation(self):
        with self.assertRaisesRegex(ValueError, "plan identity"):
            self.execute()
        self.assertFalse((self.root / self.plan["planned_output_root"]).exists())

    def test_preflight_wrong_hash_starts_no_native_process(self):
        plan_file = self.root / "plan.json"
        plan_file.write_bytes(self.data)
        with patch.object(runner.subprocess, "run", side_effect=AssertionError("Must not run")):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                runner.preflight(self.root, plan_file, self.python, self.lock, self.install, "0" * 64,
                                 runner.sha256(self.lock), runner.sha256(self.install))

    def test_runtime_file_drift_rejected(self):
        runner.runtime_hash_checks(self.python, self.lock, self.install, self.runtime)
        self.lock.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Runtime changed"):
            runner.runtime_hash_checks(self.python, self.lock, self.install, self.runtime)

    def test_live_or_snapshot_helper_drift_rejected(self):
        bindings = runner.source_checks(self.root, self.plan)
        runner.verify_bindings(self.root, self.snapshot, bindings)
        (self.snapshot / runner.ADAPTER).write_text("changed")
        with self.assertRaisesRegex(ValueError, "drift"):
            runner.verify_bindings(self.root, self.snapshot, bindings)

    def test_generated_key_schema_and_rows_validation(self):
        path = self.table("toy.csv", ["listing_id", "action_id", "text"], [["x", "a", "1.2300"], ["x", "b", ""]])
        keys, listings = runner.table_keys(path, ["listing_id", "action_id", "text"])
        self.assertEqual(keys, {("x", "a"), ("x", "b")})
        self.assertEqual(listings, {"x"})
        for rows in ([["x", "a", 1], ["x", "a", 2]], [["", "a", 1]], [["x", "a"]], [["x", "a", 1, 2]]):
            path = self.table("toy.csv", ["listing_id", "action_id", "text"], rows)
            with self.assertRaises(ValueError):
                runner.table_keys(path)

    def test_target_checkpoint_requires_recorded_current_bytes_and_oof_flag(self):
        output = self.root / self.plan["planned_output_root"]
        output.mkdir(parents=True)
        directory = output / "development-current"
        directory.mkdir()
        path = directory / "bvival_action_dataset.csv"
        path.write_text("listing_id,action_id\nx,a\n")
        audit = {"output_sha256": runner.sha256(path)}
        (directory / "bvival_action_dataset_audit.json").write_text(json.dumps(audit))
        with patch.object(runner, "pair_checks", return_value={}):
            with self.assertRaisesRegex(ValueError, "Current target bytes differ"):
                runner.target_checkpoint(output, self.plan)

    def test_projection_checkpoint_rejects_numeric_recomputation(self):
        output = self.root / self.plan["planned_output_root"]
        c = self.plan["expected_structural_contract"]
        header = c["legacy_target_header"]
        row = ["x", "a", *(["0"] * (len(header) - 2))]
        for phase, role in (("development", "development_value"), ("calibration", "calibration")):
            path = self.table(self.plan["planned_output_root"] + "/" + phase + "-legacy/bvival_action_dataset.csv", header, [row])
            current = self.table(self.plan["planned_output_root"] + "/" + phase + "-current/bvival_action_dataset.csv", header, [row])
            c[phase + "_legacy_sha256"] = runner.sha256(path)
            c["phase_counts"][phase] = {"listings": 1, "pairs": 1}
            (path.parent / "legacy_projection_audit.json").write_text(json.dumps({
                "stage": "LEGACY_SCHEMA_PROJECTED_WITH_RECORDED_BYTE_HASH_MATCH", "role": role,
                "output_sha256": runner.sha256(path), "numeric_values_recomputed": phase == "calibration",
                "historical_target_rows_read": False, "fitting_called": False,
                "contract_sha256": self.plan["recovery_receipt_sha256"], "input_sha256": runner.sha256(current),
                "listing_count": 1, "action_pair_count": 1,
                "retained_cell_text_and_row_order_preserved": True}))
        with self.assertRaisesRegex(ValueError, "Legacy projection checkpoint"):
            runner.projection_checkpoint(output, self.plan)

    def test_cohort_hash_checks_real_generated_bytes_not_just_audit(self):
        output = self.root / "output"
        directory = output / "cohort"
        directory.mkdir(parents=True)
        path = directory / "toy.csv"
        path.write_text("old\n")
        c = self.plan["expected_structural_contract"]
        c["cohort_output_sha256"] = {"toy.csv": runner.sha256(path)}
        (directory / "jucars_bvival_audit.json").write_text(json.dumps({"split_counts": c["split_counts"], "outputs": c["cohort_output_sha256"]}))
        runner.cohort_checks(output, self.plan)
        path.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "Cohort bytes differ"):
            runner.cohort_checks(output, self.plan)

    def test_failed_process_retained_without_join_or_automatic_retry(self):
        self.allow_toy_execution()
        with patch.object(runner.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["toy"])) as child:
            with self.assertRaises(subprocess.CalledProcessError):
                self.execute()
        output = self.root / self.plan["planned_output_root"]
        failed = json.loads((output / "failed-scoring-replay.json").read_text())
        self.assertEqual(child.call_count, 1)
        self.assertEqual(failed["phase"], "cohort")
        self.assertFalse(failed["scores_frozen"])
        self.assertFalse(failed["test_performance_computed"])
        self.assertFalse((output / "evaluation-inputs").exists())
        with self.assertRaises(FileExistsError):
            self.execute()

    def test_target_gate_failure_prevents_projection_and_heads(self):
        self.allow_toy_execution()
        with patch.object(runner.subprocess, "run") as child, patch.object(runner, "cohort_checks"), patch.object(runner, "pair_checks"), patch.object(runner, "target_checkpoint", side_effect=ValueError("synthetic target mismatch")):
            with self.assertRaisesRegex(ValueError, "synthetic target mismatch"):
                self.execute()
        self.assertEqual(child.call_count, 4)
        output = self.root / self.plan["planned_output_root"]
        failed = json.loads((output / "failed-scoring-replay.json").read_text())
        self.assertEqual(failed["phase"], "development-legacy")
        self.assertFalse((output / "value").exists())

    def test_success_freezes_only_nine_stages_and_new_score_hash(self):
        self.allow_toy_execution()
        def fake_child(args, **kwargs):
            directory = Path(args[args.index("--output-dir") + 1])
            directory.mkdir()
            name = directory.name
            if name in {"development-current", "calibration-current"}:
                p = directory / "bvival_action_dataset.csv"
                p.write_text("listing_id,action_id\ntoy,a\n")
                (directory / "bvival_action_dataset_audit.json").write_text(json.dumps({"output_sha256": runner.sha256(p)}))
            if name in {"pairs", "scores"}:
                header = ["listing_id", "action_id"] + (fixture_module.planner.SCORES if name == "scores" else [])
                rows = [[f"toy-{i}", a, *([0] * 7 if name == "scores" else [])] for i in range(4) for a in ("a", "b", "c")]
                filename = "frozen_policy_scores_absolute_price_error.csv" if name == "scores" else "evaluation_prediction_pairs.csv"
                self.table(str((directory / filename).relative_to(self.root)), header, rows)
            return subprocess.CompletedProcess(args, 0)
        with patch.object(runner.subprocess, "run", side_effect=fake_child) as child, patch.object(runner, "cohort_checks"), patch.object(runner, "pair_checks"), patch.object(runner, "target_checkpoint", return_value={"toy": True}), patch.object(runner, "projection_checkpoint", return_value={"toy": True}), patch.object(runner, "head_checks"):
            result = self.execute()
        self.assertEqual(child.call_count, 9)
        self.assertEqual([s["stage"] for s in result["completed_scoring_stages"]], runner.SCORING_IDS)
        self.assertFalse(result["outcome_join_called"])
        self.assertFalse(result["test_performance_computed"])
        self.assertFalse(result["independent_replication_claimed"])
        output = self.root / self.plan["planned_output_root"]
        self.assertEqual(result["new_score_sha256"], runner.sha256(output / "scores/frozen_policy_scores_absolute_price_error.csv"))
        self.assertTrue((output / "scoring-freeze.json").is_file())
        for name, digest in result["artifact_sha256"].items():
            self.assertEqual(runner.sha256(output / name), digest)


if __name__ == "__main__":
    unittest.main()
