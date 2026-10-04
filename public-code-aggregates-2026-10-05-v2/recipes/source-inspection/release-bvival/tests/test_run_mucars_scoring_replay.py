"""Synthetic fixtures only; no source data, model fitting or outcome evaluation."""
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
import run_mucars_scoring_replay as runner


class ScoringRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="scoring-runner-toy-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.lock = self.root / "toy.lock"
        self.lock.write_text("# toy\n--only-binary=:all:\ntoy_pkg==1.0 --hash=sha256:" + "a" * 64 + "\n")
        self.pins = runner.lock_pins(self.lock.read_text())
        self.probe = {name: True for name in ("is_virtualenv", "all_distribution_metadata_inside_venv",
                                            "all_four_imports_inside_venv", "synthetic_native_fit_passed")}
        self.probe["installed_versions"] = {"toy-pkg": "1.0", "pip": "25.0"}
        self.install = {"install": [{"metadata": {"name": "toy_pkg", "version": "1.0"},
                      "download_info": {"archive_info": {"hashes": {"sha256": "a" * 64}}}}]}
        self.script = self.root / "experiments/toy.py"
        self.script.parent.mkdir()
        self.script.write_text("raise RuntimeError('Toy source must not execute')\n")
        self.source = self.root / "toy.csv"
        self.source.write_text("synthetic_only\n")
        self.plan = {"planned_output_root": "experiments/replays/mucars-reconstructed-toy-v1",
                     "static_source_closure": {"local_source_sha256": {"experiments/toy.py": runner.sha256(self.script)}},
                     "source_sha256_to_verify_before_execution": runner.sha256(self.source),
                     "stages": [{"id": name, "phase": "reconstruction" if name == "cohort" else "scoring",
                                 "argv_template": [{"runtime": "deferred"}, "experiments/toy.py", "--source", "toy.csv",
                                                   "--output-dir", "experiments/replays/mucars-reconstructed-toy-v1/" + name]}
                                for name in runner.SCORING_IDS]}

    def check_runtime(self):
        return runner.validate_runtime_records(self.pins, self.install, self.probe)

    def write_scores(self, values=None, columns=None, keys=None):
        path = self.root / "scores.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns or ["listing_id", "action_id", *runner.SCORE_COLUMNS])
            for key in keys or [("toy-listing", "toy-action")]:
                writer.writerow([*key, *(values or [0] * 7)])
        return path

    def test_exact_install_and_separate_bootstrap_pip_pass(self):
        self.assertEqual(self.check_runtime()["wheel_count"], 1)

    def test_floating_duplicate_or_empty_lock_rejected(self):
        for text in ("", "toy>=1.0", self.lock.read_text() + self.lock.read_text()):
            with self.assertRaises(ValueError):
                runner.lock_pins(text)

    def test_runtime_requires_real_booleans(self):
        for name in ("is_virtualenv", "all_distribution_metadata_inside_venv", "all_four_imports_inside_venv", "synthetic_native_fit_passed"):
            for value in (False, 1, "true"):
                probe = copy.deepcopy(self.probe)
                probe[name] = value
                with self.assertRaises(ValueError):
                    runner.validate_runtime_records(self.pins, self.install, probe)

    def test_missing_or_wrong_installed_version_rejected(self):
        for versions in ({"pip": "25.0"}, {"pip": "25.0", "toy_pkg": "2.0"}):
            self.probe["installed_versions"] = versions
            with self.assertRaises(ValueError):
                self.check_runtime()

    def test_unlocked_extra_distribution_rejected(self):
        self.probe["installed_versions"]["unexpected"] = "1.0"
        with self.assertRaises(ValueError):
            self.check_runtime()

    def test_archive_hash_and_version_mismatch_rejected(self):
        for change in ("hash", "version"):
            report = copy.deepcopy(self.install)
            if change == "hash":
                report["install"][0]["download_info"]["archive_info"]["hashes"]["sha256"] = "b" * 64
            else:
                report["install"][0]["metadata"]["version"] = "2.0"
            with self.assertRaises(ValueError):
                runner.validate_runtime_records(self.pins, report, self.probe)

    def test_missing_duplicate_install_record_rejected(self):
        for records in ([], self.install["install"] * 2):
            with self.assertRaises(ValueError):
                runner.validate_runtime_records(self.pins, {"install": records}, self.probe)

    def test_stage_launches_snapshot_and_absolute_new_paths(self):
        snapshot = self.root / "snapshot"
        args = runner.stage_command(self.root, self.plan, self.plan["stages"][0], Path("/toy/venv/bin/python"), snapshot)
        self.assertEqual(args[:3], ["/toy/venv/bin/python", "-I", "-B"])
        self.assertIn(str(snapshot / "experiments/toy.py"), args)
        self.assertIn(str(self.source), args)
        self.assertIn(str(self.root / self.plan["planned_output_root"] / "cohort"), args)

    def test_outcome_stages_forbidden(self):
        for name in ("join", "evaluation", "anything"):
            stage = copy.deepcopy(self.plan["stages"][0])
            stage["id"] = name
            with self.assertRaises(ValueError):
                runner.stage_command(self.root, self.plan, stage, Path("/toy/python"), self.root)

    def test_evaluation_phase_or_deferred_arg_forbidden(self):
        for change in ("phase", "argument", "source"):
            stage = copy.deepcopy(self.plan["stages"][0])
            if change == "phase":
                stage["phase"] = "evaluation"
            elif change == "argument":
                stage["argv_template"].append({"deferred": "hash"})
            else:
                stage["argv_template"][1] = "experiments/not-bound.py"
            with self.assertRaises(ValueError):
                runner.stage_command(self.root, self.plan, stage, Path("/toy/python"), self.root)

    def test_source_drift_stops_runner(self):
        self.script.write_text("changed\n")
        with self.assertRaises(ValueError):
            runner.source_checks(self.root, self.plan)

    def test_score_keys_and_original_policy_schema_pass(self):
        self.assertEqual(runner.check_score_file(self.write_scores(), 1, 1)["action_pair_count"], 1)

    def test_target_or_extra_header_rejected(self):
        columns = ["listing_id", "action_id", *runner.SCORE_COLUMNS, "target_log"]
        with self.assertRaises(ValueError):
            runner.check_score_file(self.write_scores(columns=columns), 1, 1)

    def test_duplicate_empty_or_wrong_count_rejected(self):
        for keys in ([("toy", "a"), ("toy", "a")], [("", "a")], [("toy", "a"), ("other", "b")]):
            with self.assertRaises(ValueError):
                runner.check_score_file(self.write_scores(keys=keys), 1, 1)

    def test_nan_positive_inf_and_unapproved_negative_inf_rejected(self):
        for value in ("nan", "inf", "-inf"):
            with self.assertRaises(ValueError):
                runner.check_score_file(self.write_scores(values=[value, *([0] * 6)]), 1, 1)

    def test_negative_inf_routing_sentinels_allowed(self):
        runner.check_score_file(self.write_scores(values=[0, 0, 0, "-inf", "-inf", 0, 0]), 1, 1)

    def test_short_or_long_row_rejected(self):
        for values in ([0] * 6, [0] * 8):
            with self.assertRaises(ValueError):
                runner.check_score_file(self.write_scores(values=values), 1, 1)

    def test_environment_drops_inherited_python_paths(self):
        with patch.dict("os.environ", {"PYTHONPATH": "/outside", "PYTHONHOME": "/outside", "PYTHONSTARTUP": "/outside"}):
            env = runner.child_environment(Path("/toy/run/venv/bin/python"))
        self.assertFalse({"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}.intersection(env))
        self.assertEqual(env["MPLCONFIGDIR"], "/toy/run/matplotlib-cache")

    def test_receipt_never_overwrites(self):
        path = self.root / "receipt.json"
        runner.write_new_json(path, {"one": 1})
        with self.assertRaises(FileExistsError):
            runner.write_new_json(path, {"two": 2})
        self.assertEqual(json.loads(path.read_text()), {"one": 1})

    def test_raw_source_mismatch_creates_no_output(self):
        self.source.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            runner.execute_scoring(self.root, self.plan, b"{}", Path("/toy/python"), self.lock, self.lock, {})
        self.assertFalse((self.root / self.plan["planned_output_root"]).exists())

    def test_failed_child_is_retained_without_evaluation_or_retry(self):
        with patch.object(runner.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["toy"])) as process:
            with self.assertRaises(subprocess.CalledProcessError):
                runner.execute_scoring(self.root, self.plan, b"{}", Path("/toy/run/venv/bin/python"), self.lock, self.lock, {})
        output = self.root / self.plan["planned_output_root"]
        failure = json.loads((output / "failed-scoring-replay.json").read_text())
        self.assertEqual(process.call_count, 1)
        self.assertEqual(failure["phase"], "cohort")
        self.assertFalse(failure["test_performance_computed"])
        self.assertFalse((output / "evaluation-inputs").exists())
        with self.assertRaises(FileExistsError):
            runner.execute_scoring(self.root, self.plan, b"{}", Path("/toy/run/venv/bin/python"), self.lock, self.lock, {})


if __name__ == "__main__":
    unittest.main()
