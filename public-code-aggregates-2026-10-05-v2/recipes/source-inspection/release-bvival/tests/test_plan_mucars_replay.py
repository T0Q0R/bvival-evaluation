"""Toy code/metadata fixtures only; no original data, models or experiments."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plan_mucars_replay as planner


FLAGS = (
    "source", "output-dir", "train", "validation", "calibration", "evaluation-features",
    "action-fields", "seeds", "folds", "iterations", "depth", "learning-rate", "l2-leaf-reg",
    "prediction-pairs", "labels", "pre-action-features", "feature-columns", "evidence-status",
    "train-actions", "calibration-actions", "categorical-columns", "alpha", "minimum-action-group-size",
    "objective", "development-actions", "value-scores", "risk-scores", "random-seed",
    "evaluation-prediction-pairs", "sealed-labels", "frozen-policy-scores", "expected-policy-scores-sha256",
    "freeze-id", "evaluation-outcomes", "score-columns", "reference-policy", "budgets",
    "bootstrap-repetitions", "bootstrap-seed", "analysis-status", "include-oracle-diagnostic",
)
FILES = ("build_mucars_bvival_manifest.py", "generate_bvival_prediction_pairs.py",
         "build_bvival_action_dataset.py", "fit_bvival_value_policy.py", "fit_bvival_risk_baseline.py",
         "assemble_bvival_policy_scores.py", "join_bvival_evaluation_outcomes.py", "evaluate_bvival_policies.py")


class ReplayPlanningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mucars-replay-plan-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        hashes = {}
        for relative in planner.AUDITS.values():
            data = json.dumps({"outputs": {"synthetic.csv": "1" * 64},
                               "split_counts": {"train": 10, "validation": 5, "calibration": 3, "test": 4},
                               "output_hashes": {"splits/train.csv": "1" * 64},
                               **{phase + "_listing_count": count for phase, count in (("development", 15), ("calibration", 3), ("evaluation", 4))},
                               **{phase + "_action_pair_count": count for phase, count in (("development", 45), ("calibration", 9), ("evaluation", 12))}}).encode()
            self.write(relative, data)
            hashes[relative] = hashlib.sha256(data).hexdigest()
        models = [{"role": role, "seed": seed,
                   "parameters": {"iterations": 200, "depth": 6, "l2_leaf_reg": 10,
                                  "learning_rate": 0.08 if role == "base" else 0.05,
                                  "random_seed": seed, "loss_function": "RMSE"}}
                  for role in ("base", "absolute_price_error", "squared_log_error", "risk_mae")
                  for seed in (13, 42)]
        self.receipt = {
            "stage": "HISTORICAL_MUCARS_METADATA_RECONSTRUCTION_NOT_EMPIRICAL_REPLAY",
            "full_empirical_retraining_verified": False, "historical_test_labels_already_opened": True,
            "audit_file_sha256": hashes, "model_parameter_inspection": {"records": models},
            "policy_feature_columns": ["action_id", "before_prediction_log", "make"],
            "policy_categorical_columns": ["action_id", "make"],
            "action_fields": ["field_a"], "source_doi": "synthetic-only", "source_sha256_recorded_not_row_checked": "1" * 64,
            "base_feature_columns": ["make"], "base_categorical_columns": ["make"],
        }
        self.save_receipt()
        toy = ("raise RuntimeError('This toy study source must never be executed')\n"
               "import numpy, pandas, catboost, scipy\nfrom toy_helper import fixture_only\n"
               "def main():\n    parser = argparse.ArgumentParser()\n"
               f"    for flag in {FLAGS!r}:\n        parser.add_argument('--' + flag)\n")
        for name in FILES:
            self.write("experiments/" + name, toy.encode())
        self.write("experiments/toy_helper.py", b"raise RuntimeError('No study imports')\ndef fixture_only(): pass\n")

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def save_receipt(self):
        self.write(planner.RECEIPT, json.dumps(self.receipt).encode())

    def plan(self):
        return planner.build_plan(self.root)

    def test_plan_has_nine_templates_without_executing_toy_code(self):
        plan = self.plan()
        self.assertEqual(len(plan["stages"]), 9)
        self.assertEqual(len(plan["static_source_closure"]["local_source_sha256"]), 9)
        self.assertFalse((self.root / plan["planned_output_root"]).exists())
        self.assertFalse(plan["executor_implemented"])
        self.assertEqual(planner.validate_plan(self.root, plan)["command_templates"], 9)

    def test_heads_explicitly_use_200_iterations_and_primary_objective(self):
        stages = {stage["id"]: stage for stage in self.plan()["stages"]}
        for name in ("value", "risk-mae"):
            args = stages[name]["argv_template"]
            self.assertEqual(args[args.index("--iterations") + 1], "200")
        args = stages["risk-mae"]["argv_template"]
        self.assertEqual(args[args.index("--objective") + 1], "absolute_price_error")

    def test_opened_test_not_called_new_confirmation(self):
        plan = self.plan()
        self.assertTrue(plan["historical_tests_previously_opened"])
        self.assertFalse(plan["new_preregistration"])
        args = plan["stages"][-1]["argv_template"]
        self.assertEqual(args[args.index("--analysis-status") + 1], "post_test_exploratory")

    def test_join_depends_on_new_scoring_freeze_with_deferred_hash(self):
        stage = next(stage for stage in self.plan()["stages"] if stage["id"] == "join")
        self.assertIn("new_scoring_freeze_checkpoint", stage["depends_on"])
        args = stage["argv_template"]
        self.assertIn("frozen_artifact_sha256", args[args.index("--expected-policy-scores-sha256") + 1])

    def test_all_output_dirs_are_inside_new_root(self):
        plan = self.plan()
        for stage in plan["stages"]:
            args = stage["argv_template"]
            self.assertTrue(args[args.index("--output-dir") + 1].startswith(plan["planned_output_root"] + "/"))

    def test_changed_tolerance_rejected(self):
        plan = self.plan()
        plan["engineering_concordance_rules"]["price_mae_absolute_tolerance_MAD"] = 50
        with self.assertRaisesRegex(ValueError, "Plan differs"):
            planner.validate_plan(self.root, plan)

    def test_changed_command_rejected(self):
        plan = self.plan()
        plan["stages"][1]["argv_template"].extend(["--use-test-labels", "yes"])
        with self.assertRaisesRegex(ValueError, "Plan differs"):
            planner.validate_plan(self.root, plan)

    def test_numeric_boolean_stand_ins_rejected(self):
        for key in ("execution_started", "executor_implemented", "submission_ready", "new_preregistration"):
            plan = self.plan()
            plan[key] = 0
            with self.assertRaisesRegex(ValueError, "Boolean plan"):
                planner.validate_plan(self.root, plan)
        plan = self.plan()
        plan["data_access_this_planning_command"]["row_files_read"] = 0
        with self.assertRaisesRegex(ValueError, "Boolean plan"):
            planner.validate_plan(self.root, plan)

    def test_source_drift_invalidates_saved_plan(self):
        plan = self.plan()
        self.write("experiments/toy_helper.py", b"def fixture_only(): return 2\n")
        with self.assertRaisesRegex(ValueError, "Plan differs"):
            planner.validate_plan(self.root, plan)

    def test_audit_drift_prevents_planning(self):
        relative = next(iter(planner.AUDITS.values()))
        self.write(relative, b'{"changed":true}')
        with self.assertRaisesRegex(ValueError, "Historical audit changed"):
            self.plan()

    def test_missing_audit_binding_rejected(self):
        self.receipt["audit_file_sha256"].pop(next(iter(planner.AUDITS.values())))
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "All nine"):
            self.plan()

    def test_wrong_recovered_head_parameters_rejected(self):
        self.receipt["model_parameter_inspection"]["records"][-1]["parameters"]["iterations"] = 400
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "parameters do not support"):
            self.plan()

    def test_existing_output_root_rejected(self):
        plan = self.plan()
        (self.root / plan["planned_output_root"]).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "no overwrite"):
            self.plan()

    def test_unsafe_names_and_symlink_parent_rejected(self):
        for name in ("../outside", "mucars-reconstructed-../outside", "/tmp/foo", "original"):
            with self.assertRaises(ValueError):
                planner.build_plan(self.root, name)
        (self.root / "experiments/replays").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink replay"):
            self.plan()

    def test_dynamic_import_requires_manual_review(self):
        self.write("experiments/toy_helper.py", b"value = __import__('unknown_runtime_module')\n")
        with self.assertRaisesRegex(ValueError, "Dynamic imports"):
            self.plan()

    def test_local_relative_import_requires_manual_review(self):
        self.write("experiments/toy_helper.py", b"from .other import value\n")
        with self.assertRaisesRegex(ValueError, "Relative import"):
            self.plan()

    def test_missing_required_cli_argument_rejected(self):
        self.write("experiments/required.py", b"parser.add_argument('--mandatory', required=True)\n")
        stage = {"argv_template": [dict(planner.RUNTIME_BINDING), "experiments/required.py"]}
        with self.assertRaisesRegex(ValueError, "required CLI flag"):
            planner.validate_declared_flags(self.root, stage)

    def test_unrecognised_planned_cli_argument_rejected(self):
        self.write("experiments/only.py", b"parser.add_argument('--only')\n")
        stage = {"argv_template": [dict(planner.RUNTIME_BINDING), "experiments/only.py", "--invented", "value"]}
        with self.assertRaisesRegex(ValueError, "undeclared"):
            planner.validate_declared_flags(self.root, stage)

    def test_dependency_packages_are_names_not_imported(self):
        imports = self.plan()["static_source_closure"]["external_import_roots"]
        self.assertEqual(imports, ["catboost", "numpy", "pandas", "scipy"])
        self.assertNotIn("toy_helper", sys.modules)


if __name__ == "__main__":
    unittest.main()
