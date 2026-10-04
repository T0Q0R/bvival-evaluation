"""Synthetic-only AutoScout two-branch template/graph tests."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import plan_autoscout24_replay as p


def templates():
    r = {"source_path": "experiments/data/external/source.csv", "action_fields": ["mileage_km", "transmission", "fuel_type", "service_history"]}
    a = {"value": {"feature_columns": ["action_id", "make", "before_prediction_log"], "categorical_columns": ["action_id", "make"]},
         "evaluation": {"score_columns": ["score_mean_value", "score_uncertainty_only", "score_fixed_field_risk"]}}
    with patch.object(p, "validate_declared_flags") as flags:
        stages = p.make_stages(Path("/synthetic"), "experiments/replays/autoscout24-reconstructed-toy", r, a)
        assert flags.call_count == 14
    return stages


class PlanTests(unittest.TestCase):
    def test_exact_order_and_canonical_graph(self):
        s = templates()
        p.validate_structure(s)
        self.assertEqual([x["id"] for x in s[:12]], p.SCORING_IDS)
        self.assertEqual([x["phase"] for x in s[-2:]], ["evaluation", "evaluation"])

    def test_selection_cli_receives_train_validation_only(self):
        args = templates()[1]["argv_template"]
        self.assertIn("--train", args)
        self.assertIn("--validation", args)
        self.assertNotIn("--calibration", args)
        self.assertNotIn("--evaluation-features", args)

    def test_final_fit_waits_for_separate_fixed_choice(self):
        self.assertEqual(templates()[6]["depends_on"], ["cohort", "fixed-field"])

    def test_fixed_selection_uses_validation_targets_not_final_dev(self):
        args = templates()[5]["argv_template"]
        self.assertTrue(args[args.index("--development-actions") + 1].endswith('/selection-validation-targets/bvival_action_dataset.csv'))
        self.assertEqual(args[args.index("--budget-fraction") + 1], "0.1")

    def test_risk_only_train_targets(self):
        s = templates()
        s[4]["depends_on"] = ["development-targets"]
        with self.assertRaises(ValueError):
            p.validate_structure(s)

    def test_selection_cannot_use_test_inputs(self):
        s = templates()
        s[1]["argv_template"] += ["--evaluation-features", "sealed-test.csv"]
        with self.assertRaisesRegex(ValueError, "Selection branch"):
            p.validate_structure(s)

    def test_no_shortcut_hardcoded_mileage_in_assembly(self):
        s = templates()
        args = s[11]["argv_template"]
        args[args.index("--fixed-action-file")] = "--fixed-action"
        with self.assertRaisesRegex(ValueError, "generated fixed-field"):
            p.validate_structure(s)

    def test_no_oracle(self):
        s = templates()
        s[-1]["argv_template"].append("--include-oracle-diagnostic")
        with self.assertRaisesRegex(ValueError, "single-budget"):
            p.validate_structure(s)

    def test_no_borrowed_multibudget_settings(self):
        s = templates()
        args = s[-1]["argv_template"]
        args[args.index("--budgets") + 1] = "0.01,0.05,0.1,0.2,0.3"
        with self.assertRaisesRegex(ValueError, "single-budget"):
            p.validate_structure(s)

    def test_join_hash_not_filled_with_old_score(self):
        s = templates()
        args = s[-2]["argv_template"]
        args[args.index("--expected-policy-scores-sha256") + 1] = "a" * 64
        with self.assertRaisesRegex(ValueError, "new generated score freeze"):
            p.validate_structure(s)

    def test_missing_selection_stage_rejected(self):
        s = templates()
        s.pop(5)
        with self.assertRaisesRegex(ValueError, "fourteen-stage"):
            p.validate_structure(s)

    def test_forward_dependency_rejected(self):
        s = templates()
        s[1]["depends_on"] = ["value"]
        with self.assertRaisesRegex(ValueError, "forward"):
            p.validate_structure(s)

    def test_new_prefix_no_directory_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            path = p.new_prefix(root, "autoscout24-reconstructed-toy")
            self.assertEqual(path, "experiments/replays/autoscout24-reconstructed-toy")
            self.assertFalse((root / "experiments").exists())

    def test_existing_prefix_refused_unless_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            name = "autoscout24-reconstructed-toy"
            (root / "experiments/replays" / name).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "nonexistent"):
                p.new_prefix(root, name)
            self.assertEqual(p.new_prefix(root, name, allow_existing=True), "experiments/replays/" + name)

    def test_unsafe_names_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("../anything", "mucars-reconstructed-v1", "autoscout24-reconstructed-", "/tmp/out"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    p.new_prefix(Path(d).resolve(), name)

    def test_symlink_prefix_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "experiments").mkdir()
            (root / "other").mkdir()
            (root / "experiments/replays").symlink_to(root / "other")
            with self.assertRaisesRegex(ValueError, "Symlink"):
                p.new_prefix(root, "autoscout24-reconstructed-toy")

    def test_plan_mutation_rejected(self):
        original = {"replay_name": "autoscout24-reconstructed-toy", "stages": templates()}
        changed = copy.deepcopy(original)
        changed["stages"][-1]["argv_template"].append("changed")
        with patch.object(p, "build_plan", return_value=original):
            with self.assertRaisesRegex(ValueError, "canonical bound"):
                p.validate_plan(Path("/synthetic"), changed)


if __name__ == "__main__":
    unittest.main()
