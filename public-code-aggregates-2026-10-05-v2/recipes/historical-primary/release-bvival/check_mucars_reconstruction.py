"""Read-only MUCars historical metadata/model-parameter reconstruction.

Default: read nine fixed audit JSONs, a freeze list and code/document bytes.
Optional --inspect-model-parameters: additionally hash/load eight bound local
CatBoost models and expose selected parameters, never trees or predictions.
No row-level table is opened and no training or inference is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


AUDITS = {
    "cohort": "experiments/data/processed/mucars_bvival_v1/mucars_bvival_audit.json",
    "pairs": "experiments/outputs/mucars_bvival_v5_pairs/prediction_pair_generation_audit.json",
    "development": "experiments/outputs/mucars_bvival_v5_actions_development/bvival_action_dataset_audit.json",
    "calibration": "experiments/outputs/mucars_bvival_v5_actions_calibration/bvival_action_dataset_audit.json",
    "value": "experiments/outputs/mucars_bvival_v5_value/value_policy_fit_audit.json",
    "risk": "experiments/outputs/mucars_bvival_v5_risk_mae/risk_baseline_fit_audit.json",
    "assembly": "experiments/outputs/mucars_bvival_v5_frozen_scores/policy_score_assembly_audit_absolute_price_error.json",
    "join": "experiments/outputs/mucars_bvival_v5_evaluation_inputs/evaluation_outcome_join_audit.json",
    "evaluation": "experiments/outputs/mucars_bvival_v5_confirmatory_mae/bvival_evaluation_audit.json",
}
FREEZE = "experiments/mucars_bvival_v5_pretest_freeze.sha256"
FROZEN_CODE_AND_METADATA = {
    "experiments/build_mucars_bvival_manifest.py",
    "notes/analysis/bvival-v5-development-decision-2026-09-25.md",
    "experiments/generate_bvival_prediction_pairs.py",
    "experiments/build_bvival_action_dataset.py",
    "experiments/fit_bvival_value_policy.py",
    "experiments/fit_bvival_risk_baseline.py",
    "experiments/assemble_bvival_policy_scores.py",
    "experiments/bvi_val.py", "experiments/bvi_val_baselines.py",
    "experiments/join_bvival_evaluation_outcomes.py",
    "experiments/evaluate_bvival_policies.py", "experiments/evaluate_bvival_subgroups.py",
    "experiments/configs/mucars_bvival_v5_subgroups.json",
}
FROZEN_ROW_PATHS = {
    "experiments/data/processed/mucars_bvival_v1/splits/test_features.csv",
    "experiments/data/processed/mucars_bvival_v1/sealed_labels/test_labels.csv",
    "experiments/outputs/mucars_bvival_v5_frozen_scores/frozen_policy_scores_absolute_price_error.csv",
    "experiments/outputs/mucars_bvival_v5_frozen_scores/frozen_policy_scores_squared_log_error.csv",
}
PARAMETER_KEYS = (
    "iterations", "depth", "learning_rate", "l2_leaf_reg", "loss_function",
    "random_seed", "task_type", "bootstrap_type", "boosting_type",
)


def safe_path(root, relative):
    parts = PurePosixPath(relative).parts
    if (not parts or ".." in parts or "\\" in relative
            or PurePosixPath(relative).is_absolute() or relative != "/".join(parts)):
        raise ValueError("Unsafe relative path")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink input/component rejected")
    if not path.is_file():
        raise ValueError(f"Missing reconstruction input: {relative}")
    return path


def recorded_graph_checks(audits):
    """Check recorded producer/consumer metadata; do not hash row artifacts."""
    c, p, d, k, v, r, s, j, e = (audits[n] for n in AUDITS)
    passed = []

    def equal(name, left, right):
        if left is None or right is None or left != right:
            raise ValueError(f"Recorded dependency mismatch: {name}")
        passed.append(name)

    def same_hash(name, left, right):
        for value in (left, right):
            if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"Invalid recorded SHA256: {name}")
        equal(name, left, right)

    for split, key in (("train", "train"), ("validation", "validation"),
                       ("calibration", "calibration"), ("test_features", "evaluation_features")):
        same_hash(f"cohort_{split}_to_pairs", c["output_hashes"]["splits/" + split + ".csv"], p["inputs"][key + "_sha256"])
    for phase, action in (("development", d), ("calibration", k)):
        for producer, consumer in (("pairs", "prediction_pairs"), ("features", "pre_action_features"), ("labels", "labels")):
            same_hash(f"pairs_{phase}_{producer}_to_action", p["outputs"][phase][producer], action["inputs"][consumer]["sha256"])
        equal(f"{phase}_pair_count", p[phase + "_action_pair_count"], action["listing_action_count"])
        equal(f"{phase}_listing_count", p[phase + "_listing_count"], action["listing_count"])
    same_hash("development_targets_to_value", d["output_sha256"], v["inputs"]["train_actions_sha256"])
    same_hash("calibration_targets_to_value", k["output_sha256"], v["inputs"]["calibration_actions_sha256"])
    same_hash("development_targets_to_risk", d["output_sha256"], r["inputs"]["train_actions_sha256"])
    for name, head in (("value", v), ("risk", r)):
        same_hash(f"evaluation_features_to_{name}", p["outputs"]["evaluation"]["features"], head["inputs"]["evaluation_features_sha256"])
        equal(f"{name}_feature_allowlist", ["action_id"] + d["feature_allowlist"], head["feature_columns"])
        equal(f"{name}_seeds", p["seeds"], head["seeds"])
    equal("action_allowlists_match", d["feature_allowlist"], k["feature_allowlist"])
    same_hash("value_scores_to_assembly", v["evaluation_action_scores_sha256"], s["inputs"]["value_scores"]["sha256"])
    same_hash("risk_scores_to_assembly", r["evaluation_risk_scores_sha256"], s["inputs"]["risk_scores"]["sha256"])
    same_hash("development_targets_to_assembly", d["output_sha256"], s["inputs"]["development_actions"]["sha256"])
    same_hash("evaluation_features_to_assembly", p["outputs"]["evaluation"]["features"], s["inputs"]["evaluation_features"]["sha256"])
    same_hash("assembly_scores_to_join", s["output"]["sha256"], j["inputs"]["frozen_policy_scores_sha256"])
    same_hash("evaluation_pairs_to_join", p["outputs"]["evaluation"]["pairs"], j["inputs"]["prediction_pairs_sha256"])
    same_hash("cohort_test_labels_to_join", c["output_hashes"]["sealed_labels/test_labels.csv"], j["inputs"]["sealed_labels_sha256"])
    same_hash("joined_outcomes_to_evaluation", j["output_sha256"], e["evaluation_outcomes_sha256"])
    same_hash("assembly_scores_to_evaluation", s["output"]["sha256"], e["frozen_policy_scores_sha256"])
    for name, stage in (("risk", r), ("assembly", s), ("evaluation", e)):
        equal(f"{name}_primary_objective", stage["objective"], "absolute_price_error")
    equal("evaluation_reference_policy", e["reference_policy"], "score_uncertainty_only")
    equal("join_evaluation_freeze_id", j["freeze_id"], e["freeze_id"])
    equal("cohort_development_count", c["split_counts"]["train"] + c["split_counts"]["validation"], p["development_listing_count"])
    equal("cohort_calibration_count", c["split_counts"]["calibration"], p["calibration_listing_count"])
    equal("cohort_evaluation_count", c["split_counts"]["test"], p["evaluation_listing_count"])
    equal("evaluation_join_listing_count", p["evaluation_listing_count"], j["listing_count"])
    equal("evaluation_join_pair_count", p["evaluation_action_pair_count"], j["listing_action_count"])
    if j["labels_opened"] is not True:
        raise ValueError("Historical test status must not be described as unopened")
    passed.append("historical_test_opening_retained")
    return passed


def inspect_models(root, audits):
    # Deliberate opt-in: not imported by the default metadata-only check.
    from catboost import CatBoostRegressor
    from importlib.metadata import version

    groups = [("base", audits["pairs"]["models"])]
    groups += [(name, value["models"]) for name, value in audits["value"]["objectives"].items()]
    groups.append(("risk_mae", audits["risk"]["models"]))
    result = []
    for role, models in groups:
        if role not in {"base", "absolute_price_error", "squared_log_error", "risk_mae"}:
            raise ValueError("Unexpected model role")
        if sorted(m["seed"] for m in models) != [13, 42]:
            raise ValueError("Expected two historical seeds for each model role")
        for saved in models:
            seed = saved["seed"]
            relative = saved.get("model_path", saved.get("path"))
            expected = (f"experiments/outputs/mucars_bvival_v5_pairs/base_models/mask_aware_catboost_seed_{seed}.cbm"
                        if role == "base" else
                        f"experiments/outputs/mucars_bvival_v5_risk_mae/models/seed_{seed}.cbm"
                        if role == "risk_mae" else
                        f"experiments/outputs/mucars_bvival_v5_value/models/{role}/seed_{seed}.cbm")
            if relative != expected:
                raise ValueError("Model path differs from the fixed role/seed binding")
            path = safe_path(root, relative)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != saved["sha256"]:
                raise ValueError("Saved model hash differs from its historical audit")
            model = CatBoostRegressor()
            model.load_model(str(path))
            parameters = model.get_all_params()
            raw_parameters = json.loads(model.get_metadata()["params"])
            expected_features = (audits["pairs"]["feature_columns"] + ["action_mask"]
                                 if role == "base" else audits["risk" if role == "risk_mae" else "value"]["feature_columns"])
            if model.feature_names_ != expected_features:
                raise ValueError("Model feature names differ from the audited order")
            # Compare serialized float32 rate with the reported nominal rate.
            expected_rate = audits["pairs"]["model_parameters"]["learning_rate"] if role == "base" else 0.05
            if (parameters["random_seed"] != seed or parameters["iterations"] != 200
                    or model.tree_count_ != 200 or parameters["depth"] != 6
                    or parameters["l2_leaf_reg"] != 10 or parameters["loss_function"] != "RMSE"
                    or not math.isclose(parameters["learning_rate"], expected_rate, rel_tol=1e-6)):
                raise ValueError("Saved model parameters differ from the reconstruction")
            result.append({"role": role, "seed": seed, "path": relative, "sha256": digest,
                           "tree_count": model.tree_count_,
                           "parameters": {name: parameters[name] for name in PARAMETER_KEYS},
                           "serialized_thread_count": raw_parameters.get("system_options", {}).get("thread_count"),
                           "model_feature_names": model.feature_names_})
    if len(result) != 8:
        raise ValueError("Expected eight historical model parameter records")
    return {"records": result, "reader_catboost_version": version("catboost"),
            "reader_version_is_not_training_environment_certification": True,
            "training_called": False, "prediction_called": False}


def reconstruct(root, include_models=False):
    root = Path(root).resolve(strict=True)
    audits, audit_hashes = {}, {}
    for name, relative in AUDITS.items():
        data = safe_path(root, relative).read_bytes()
        audits[name] = json.loads(data)
        audit_hashes[relative] = hashlib.sha256(data).hexdigest()
    checks = recorded_graph_checks(audits)
    freeze_bytes = safe_path(root, FREEZE).read_bytes()
    comparison, skipped, seen = [], [], set()
    for line in freeze_bytes.decode().splitlines():
        digest, relative = line.split(None, 1)
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("Invalid SHA256 in the pretest freeze")
        if relative in seen:
            raise ValueError("Duplicate freeze entry")
        seen.add(relative)
        if relative in FROZEN_CODE_AND_METADATA:
            current = hashlib.sha256(safe_path(root, relative).read_bytes()).hexdigest()
            comparison.append({"path": relative, "frozen_sha256": digest,
                               "current_sha256": current, "matches": digest == current})
        elif relative in FROZEN_ROW_PATHS:
            skipped.append({"path": relative, "recorded_sha256": digest, "row_bytes_checked": False})
        else:
            raise ValueError("Unexpected freeze path; do not open unclassified inputs")
    if {item["path"] for item in comparison} != FROZEN_CODE_AND_METADATA:
        raise ValueError("Incomplete code/metadata freeze inventory")
    if {item["path"] for item in skipped} != FROZEN_ROW_PATHS:
        raise ValueError("Incomplete recorded row-path inventory")
    p, v, r, e = (audits[name] for name in ("pairs", "value", "risk", "evaluation"))
    report = {
        "stage": "HISTORICAL_MUCARS_METADATA_RECONSTRUCTION_NOT_EMPIRICAL_REPLAY",
        "source_doi": audits["cohort"]["doi"], "source_sha256_recorded_not_row_checked": audits["cohort"]["source_sha256"],
        "audit_file_sha256": audit_hashes, "pretest_freeze_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
        "recorded_dependency_checks": checks,
        "recorded_dependencies_match": True, "row_artifact_bytes_verified": False,
        "base_parameters_from_run_audit": {"seeds": p["seeds"], "folds": p["folds"], **p["model_parameters"]},
        "base_feature_columns": p["feature_columns"], "base_categorical_columns": p["categorical_columns"],
        "action_fields": p["action_fields"], "policy_feature_columns": v["feature_columns"],
        "policy_categorical_columns": v["categorical_columns"],
        "risk_objective": r["objective"], "calibration_parameters": v["objectives"]["absolute_price_error"]["calibrator"],
        "evaluation_parameters": {name: e[name] for name in ("objective", "reference_policy", "budgets", "bootstrap_repetitions", "bootstrap_seed", "oracle_diagnostic_included")},
        "live_code_vs_pretest_freeze": comparison, "row_entries_deliberately_not_opened": skipped,
        "historical_test_labels_already_opened": True,
        "calibration_fold_semantics": "fold -1 denotes held-out calibration scored by final development ensemble; historical prediction_is_oof flag is broader than internal cross-fitting",
        "saved_model_parameters_inspected": include_models,
        "listing_rows_read": False, "fitting_called": False,
        "inference_called": False, "study_modules_imported": False,
        "full_empirical_retraining_verified": False, "submission_ready": False,
    }
    if include_models:
        report["model_parameter_inspection"] = inspect_models(root, audits)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--inspect-model-parameters", action="store_true")
    args = parser.parse_args()
    try:
        report = reconstruct(args.project_root, args.inspect_model_parameters)
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(1, f"MUCars reconstruction failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
