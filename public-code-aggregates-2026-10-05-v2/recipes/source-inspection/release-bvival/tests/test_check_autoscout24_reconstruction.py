"""Synthetic-only final versus strict-selection reconstruction guards."""
import copy
from pathlib import Path
import tempfile
import unittest

import check_autoscout24_reconstruction as c


def selection_fixture():
    h = "a" * 64
    p = {"inputs": {"train_sha256": h, "validation_sha256": h},
         "train_listing_count": 51045, "validation_listing_count": 14160,
         "train_pair_count": 203731, "validation_pair_count": 56522,
         "outputs": {phase: {"features": h, "labels": h, "pairs": h} for phase in ("train_oof", "validation_heldout")},
         "train_validation_id_overlap": 0, "validation_labels_used_in_base_fit": False,
         "sealed_test_input_read": False, "feature_columns": ["make"],
         "model_parameters": {"iterations": 400, "depth": 8}, "seeds": c.SEEDS,
         "action_fields": ["mileage_km", "transmission", "fuel_type", "service_history"]}
    a = {"selection_pairs": p, "cohort": {"output_hashes": {"splits/train.csv": h, "splits/validation.csv": h},
         "split_counts": {"train": 51045, "validation": 14160}}}
    for phase, role, key in (("train_oof", "selection_train", "train"), ("validation_heldout", "selection_validation", "validation")):
        a[role] = {"inputs": {name: {"sha256": h} for name in ("prediction_pairs", "pre_action_features", "labels")},
                   "all_predictions_oof": True, "listing_count": p[key + "_listing_count"],
                   "listing_action_count": p[key + "_pair_count"], "output_sha256": h}
    a["selection_risk"] = {"inputs": {"train_actions_sha256": h, "evaluation_features_sha256": h},
                            "evaluation_risk_scores_sha256": h, "objective": "absolute_price_error",
                            "feature_columns": ["action_id", "make"], "seeds": c.SEEDS}
    a["decision"] = {"input_sha256": {"development_actions": h, "development_risk_scores": h},
                     "selected_action": "mileage_km", "evaluation_labels_read": False,
                     "evidence_status": "development_selection_only", "listing_count": 14160,
                     "budget_fraction": .1, "capacity": 1416,
                     "candidate_fields": [{"action_id": f, "development_post_mae": i + 1.} for i, f in enumerate(p["action_fields"])]}
    a["assembly"] = {"fixed_action_file": {"sha256": h}, "resolved_fixed_action": "mileage_km"}
    a["pairs"] = {"feature_columns": p["feature_columns"], "model_parameters": p["model_parameters"], "seeds": c.SEEDS}
    a["risk"] = {"feature_columns": a["selection_risk"]["feature_columns"], "seeds": c.SEEDS}
    a["value"] = {"seeds": c.SEEDS}
    a["evaluation"] = {"budgets": [.1]}
    return a, {"action_fields": p["action_fields"], "primary_budget_fraction": .1}, {c.AUDITS["decision"]: h}


class RecoveryTests(unittest.TestCase):
    def test_strict_selection_graph_passes_without_mutation(self):
        a, cfg, h = selection_fixture()
        before = copy.deepcopy(a)
        checks = c.selection_checks(a, cfg, h)
        self.assertIn("validation_labels_not_base_fit", checks)
        self.assertIn("minimum_validation_post_mae_with_lexical_tie", checks)
        self.assertEqual(a, before)

    def test_reused_full_development_targets_rejected(self):
        a, cfg, h = selection_fixture()
        a["selection_risk"]["inputs"]["train_actions_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "train_targets_to_selection_risk"):
            c.selection_checks(a, cfg, h)

    def test_validation_labels_used_to_fit_selection_base_rejected(self):
        a, cfg, h = selection_fixture()
        a["selection_pairs"]["validation_labels_used_in_base_fit"] = True
        with self.assertRaisesRegex(ValueError, "validation_labels_not_base_fit"):
            c.selection_checks(a, cfg, h)

    def test_test_access_in_selection_rejected(self):
        a, cfg, h = selection_fixture()
        a["selection_pairs"]["sealed_test_input_read"] = True
        with self.assertRaisesRegex(ValueError, "selection_test_not_read"):
            c.selection_checks(a, cfg, h)

    def test_selected_field_cannot_disagree_with_assembly(self):
        a, cfg, h = selection_fixture()
        a["assembly"]["resolved_fixed_action"] = "fuel_type"
        with self.assertRaisesRegex(ValueError, "fixed_field_matches_decision"):
            c.selection_checks(a, cfg, h)

    def test_capacity_denominator_not_final_test_size(self):
        a, cfg, h = selection_fixture()
        a["decision"]["capacity"] = 1401
        with self.assertRaisesRegex(ValueError, "fixed_capacity"):
            c.selection_checks(a, cfg, h)

    def test_all_field_candidates_required(self):
        a, cfg, h = selection_fixture()
        a["decision"]["candidate_fields"].pop()
        with self.assertRaisesRegex(ValueError, "four candidate"):
            c.selection_checks(a, cfg, h)

    def test_recorded_selection_must_minimize_full_candidate_list(self):
        a, cfg, h = selection_fixture()
        a["decision"]["candidate_fields"][1]["development_post_mae"] = .5
        with self.assertRaisesRegex(ValueError, "minimum_validation"):
            c.selection_checks(a, cfg, h)

    def test_duplicate_field_candidate_rejected(self):
        a, cfg, h = selection_fixture()
        a["decision"]["candidate_fields"][1]["action_id"] = "mileage_km"
        with self.assertRaisesRegex(ValueError, "four candidate"):
            c.selection_checks(a, cfg, h)

    def test_bool_integer_overlap_flag_rejected(self):
        a, cfg, h = selection_fixture()
        a["selection_pairs"]["train_validation_id_overlap"] = False
        with self.assertRaisesRegex(ValueError, "overlap"):
            c.selection_checks(a, cfg, h)

    def test_malformed_same_hashes_still_rejected(self):
        a, cfg, h = selection_fixture()
        a["cohort"]["output_hashes"]["splits/train.csv"] = a["selection_pairs"]["inputs"]["train_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "Malformed"):
            c.selection_checks(a, cfg, h)

    def test_four_target_headers_without_parsing_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            a = {}
            for role in ("development", "calibration", "selection_train", "selection_validation"):
                p = root / Path(c.AUDITS[role]).parent / "bvival_action_dataset.csv"
                p.parent.mkdir(parents=True)
                p.write_text('listing_id,action_id,make\nnot,a,valid,data,row\n')
                a[role] = {"output_sha256": c.digest(p), "feature_allowlist": ["make"]}
            report = c.inspect_headers(root, a)
            self.assertTrue(report["all_four_headers_equal"])
            self.assertFalse(report["data_rows_parsed"])
            self.assertEqual(len(report["records"]), 4)

    def test_header_hash_guard_before_header_access(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            p = root / Path(c.AUDITS["development"]).parent / "bvival_action_dataset.csv"
            p.parent.mkdir(parents=True)
            p.write_text('listing_id,action_id,make\n')
            with self.assertRaisesRegex(ValueError, "Target-table bytes"):
                c.inspect_headers(root, {"development": {"output_sha256": "f" * 64}})


if __name__ == "__main__":
    unittest.main()
