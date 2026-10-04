"""Synthetic metadata and never-executed source fixtures; no study rows/models."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plan_jucars_replay as planner
from test_check_jucars_reconstruction import toy_audits
from test_plan_mucars_replay import FLAGS, FILES


class JUCarsPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="jucars-plan-toy-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.a = toy_audits()
        self.a["pairs"]["folds"] = 5
        self.a["pairs"]["model_parameters"].update(iterations=400, depth=8)
        self.a["evaluation"].update(budgets=[.01, .05, .10, .20, .30], bootstrap_repetitions=10000)
        hashes = {}
        for role, relative in planner.AUDITS.items():
            self.write(relative, json.dumps(self.a[role]))
            hashes[relative] = hashlib.sha256((self.root / relative).read_bytes()).hexdigest()
        flags = FLAGS + ("data-dictionary", "source-record-sha256", "dictionary-record-sha256")
        toy = ("raise RuntimeError('Never execute synthetic study modules')\n"
               "import numpy, pandas, catboost, scipy\nfrom toy_helper import marker\n"
               f"for flag in {flags!r}:\n    parser.add_argument('--' + flag)\n")
        files = ["build_jucars_bvival_manifest.py" if f == "build_mucars_bvival_manifest.py" else f for f in FILES]
        for name in files:
            self.write("experiments/" + name, toy)
        self.write("experiments/toy_helper.py", "raise RuntimeError('No import allowed')\nmarker = None\n")
        self.write(planner.ADAPTER, (Path(planner.__file__).parent / Path(planner.ADAPTER).name).read_text())
        models = []
        for role in ("base", "absolute_price_error", "squared_log_error", "risk_mae"):
            features = self.a["pairs" if role == "base" else "value"]["feature_columns"] + (["action_mask"] if role == "base" else [])
            cats = self.a["pairs" if role == "base" else "value"]["categorical_columns"] + (["action_mask"] if role == "base" else [])
            for seed in planner.SEEDS:
                models.append({"role": role, "seed": seed, "tree_count": 400,
                               "feature_names": features, "categorical_feature_indices": [features.index(c) for c in cats],
                               "parameters": {"iterations": 400, "depth": 8 if role == "base" else 6,
                                   "learning_rate": .08 if role == "base" else .05, "l2_leaf_reg": 10,
                                   "random_seed": seed, "loss_function": "RMSE", "task_type": "CPU"}})
        header = ["listing_id", "action_id", "before_prediction_log", "make", "year", "value"]
        self.receipt = {
            "stage": "HISTORICAL_JUCARS_METADATA_RECOVERY_NOT_EMPIRICAL_REPLAY",
            "empirical_replay_completed": False, "historical_test_labels_already_opened": True,
            "audit_file_sha256": hashes, "model_parameter_inspection": {"records": models},
            "base_parameters_from_run_audit": {"seeds": planner.SEEDS, "folds": 5, **self.a["pairs"]["model_parameters"]},
            "base_feature_columns": self.a["pairs"]["feature_columns"], "base_categorical_columns": self.a["pairs"]["categorical_columns"],
            "policy_feature_columns": self.a["value"]["feature_columns"], "policy_categorical_columns": self.a["value"]["categorical_columns"],
            "action_fields": self.a["pairs"]["action_fields"],
            "phase_counts": {p: {"listings": self.a["pairs"][p + "_listing_count"], "pairs": self.a["pairs"][p + "_action_pair_count"]} for p in ("development", "calibration", "evaluation")},
            "target_schema_inspection": {"records": [{"role": role, "header": header + (["risk_target_absolute_price_error_before"] if role == "development_risk" else []),
                                                       "sha256": self.a[role]["output_sha256"]} for role in ("development_value", "development_risk", "calibration")]},
            "source_path": "experiments/data/external/jucars_2024_v2/cars_jordan.csv",
            "dictionary_path": "experiments/data/external/jucars_2024_v2/data_dictionary.csv",
            "source_sha256_recorded_not_raw_bytes_checked": "1" * 64,
            "dictionary_sha256_recorded_not_bytes_checked": "2" * 64,
            "source_doi": "synthetic-only", "historical_additional_secondary_score_sha256": "3" * 64,
            "current_static_import_closure": planner.static_import_closure(self.root, ["experiments/" + f for f in files])}
        self.save()

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)

    def save(self):
        self.write(planner.RECEIPT, json.dumps(self.receipt))

    def plan(self):
        return planner.build_plan(self.root)

    def stages(self):
        return {s["id"]: s for s in self.plan()["stages"]}

    def test_eleven_templates_without_study_import_or_output_creation(self):
        plan = self.plan()
        self.assertEqual(len(plan["stages"]), 11)
        self.assertEqual(planner.validate_plan(self.root, plan)["command_templates"], 11)
        self.assertFalse((self.root / plan["planned_output_root"]).exists())
        self.assertNotIn("toy_helper", sys.modules)
        self.assertFalse(plan["orchestrator_implemented"])
        self.assertTrue(all(v is False for v in plan["data_access_this_planning_step"].values()))

    def test_correct_dual_target_branches_and_three_seeds(self):
        stages = self.stages()
        for role, branch in (("value", "development-legacy"), ("risk-mae", "development-current")):
            args = stages[role]["argv_template"]
            self.assertIn("/" + branch + "/", args[args.index("--train-actions") + 1])
            self.assertEqual(args[args.index("--seeds") + 1], "13,42,2026")
            self.assertEqual(args[args.index("--iterations") + 1], "400")
        self.assertEqual(stages["pairs"]["argv_template"][stages["pairs"]["argv_template"].index("--folds") + 1], "5")

    def test_source_and_dictionary_hashes_both_bound(self):
        args = self.stages()["cohort"]["argv_template"]
        self.assertEqual(args[args.index("--source-record-sha256") + 1], "1" * 64)
        self.assertEqual(args[args.index("--dictionary-record-sha256") + 1], "2" * 64)

    def test_projection_and_freeze_require_future_bindings(self):
        stages = self.stages()
        args = stages["calibration-legacy"]["argv_template"]
        self.assertIsInstance(args[args.index("--expected-input-sha256") + 1], dict)
        self.assertIn("target_schema_checkpoint", stages["calibration-legacy"]["depends_on"])
        self.assertIn("new_scoring_freeze_checkpoint", stages["join"]["depends_on"])
        self.assertNotIn("both projections", " ".join(self.plan()["target_schema_checkpoint"]["conditions"]))

    def test_narrow_join_and_engineering_limits_not_historical_identity(self):
        plan = self.plan()
        self.assertFalse(plan["primary_only_join_contract"]["secondary_score_verification_replayed"])
        self.assertNotIn("--additional-frozen-policy-scores", self.stages()["join"]["argv_template"])
        self.assertEqual(plan["engineering_concordance_rules"]["price_mae_absolute_tolerance_JOD"], .05)
        self.assertFalse(plan["historical_full_source_identity_claimed"])
        self.assertEqual(self.stages()["evaluate"]["argv_template"][-2:], ["post_test_exploratory", "--include-oracle-diagnostic"])

    def test_output_paths_scoped_and_existing_root_rejected(self):
        plan = self.plan()
        for s in plan["stages"]:
            args = s["argv_template"]
            self.assertTrue(args[args.index("--output-dir") + 1].startswith(plan["planned_output_root"] + "/"))
        (self.root / plan["planned_output_root"]).mkdir(parents=True)
        with self.assertRaises(ValueError):
            self.plan()

    def test_saved_plan_mutation_and_numeric_booleans_rejected(self):
        original = self.plan()
        for key, value in (("execution_started", 0), ("new_preregistration", True), ("submission_ready", True)):
            plan = copy.deepcopy(original)
            plan[key] = value
            with self.assertRaises(ValueError):
                planner.validate_plan(self.root, plan)
        plan = copy.deepcopy(original)
        plan["engineering_concordance_rules"]["price_mae_absolute_tolerance_JOD"] = 50
        with self.assertRaises(ValueError):
            planner.validate_plan(self.root, plan)

    def test_receipt_schema_phase_and_parameter_drift_rejected(self):
        original = copy.deepcopy(self.receipt)
        for change in (lambda r: r["policy_feature_columns"].reverse(),
                       lambda r: r["phase_counts"]["development"].update(listings=1),
                       lambda r: r["base_parameters_from_run_audit"].update(folds=3),
                       lambda r: r["model_parameter_inspection"]["records"][0].update(tree_count=399)):
            self.receipt = copy.deepcopy(original)
            change(self.receipt)
            self.save()
            with self.assertRaises(ValueError):
                self.plan()

    def test_audit_source_adapter_drift_and_missing_bindings_rejected(self):
        original = self.plan()
        self.write("experiments/toy_helper.py", "marker = 1\n")
        with self.assertRaisesRegex(ValueError, "Study code changed"):
            planner.validate_plan(self.root, original)
        self.write("experiments/toy_helper.py", "raise RuntimeError('No import allowed')\nmarker = None\n")
        self.write(planner.ADAPTER, (self.root / planner.ADAPTER).read_text() + "\n# changed\n")
        with self.assertRaises(ValueError):
            planner.validate_plan(self.root, original)
        self.write(next(iter(planner.AUDITS.values())), "{}")
        with self.assertRaisesRegex(ValueError, "Historical audit drift"):
            self.plan()

    def test_unsafe_name_symlink_and_nonstdlib_adapter_rejected(self):
        for name in ("../escape", "original", "jucars-reconstructed-../escape"):
            with self.assertRaises(ValueError):
                planner.build_plan(self.root, name)
        self.write(planner.ADAPTER, (self.root / planner.ADAPTER).read_text() + "\nimport catboost\n")
        with self.assertRaisesRegex(ValueError, "standard-library"):
            self.plan()
        (self.root / "experiments/replays").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.plan()

    def test_bad_legacy_schema_and_numeric_historical_status_rejected(self):
        self.receipt["target_schema_inspection"]["records"][0]["header"].reverse()
        self.save()
        with self.assertRaises(ValueError):
            self.plan()
        self.receipt["empirical_replay_completed"] = 0
        self.save()
        with self.assertRaises(ValueError):
            self.plan()


if __name__ == "__main__":
    unittest.main()
