"""Recover JUCars historical primary MAE metadata without empirical replay.

Default reads ten audit JSONs, source/document bytes and a historical freeze.
Opt-ins inspect 12 bound final models and/or three target-table headers/hashes.
No study data rows are parsed, no fitting, prediction or outcome evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mucars_reconstruction import safe_path, PARAMETER_KEYS
from plan_mucars_replay import static_import_closure

AUDITS = {
    "cohort": "experiments/data/processed/jucars_bvival_v1/jucars_bvival_audit.json",
    "pairs": "experiments/outputs/jucars_bvival_prediction_pairs_v1/prediction_pair_generation_audit.json",
    "development_value": "experiments/outputs/jucars_bvival_actions_development_v1/bvival_action_dataset_audit.json",
    "development_risk": "experiments/outputs/jucars_bvival_actions_development_v2/bvival_action_dataset_audit.json",
    "calibration": "experiments/outputs/jucars_bvival_actions_calibration_v1/bvival_action_dataset_audit.json",
    "value": "experiments/outputs/jucars_bvival_value_policy_v1/value_policy_fit_audit.json",
    "risk": "experiments/outputs/jucars_bvival_risk_mae_matched_v1/risk_baseline_fit_audit.json",
    "assembly": "experiments/outputs/jucars_bvival_v5_frozen_mae_scores/policy_score_assembly_audit_absolute_price_error.json",
    "join": "experiments/outputs/jucars_bvival_v5_evaluation_inputs/evaluation_outcome_join_audit.json",
    "evaluation": "experiments/outputs/jucars_bvival_v5_stress_mae/bvival_evaluation_audit.json",
}
FREEZE = "experiments/jucars_bvival_v5_pretest_freeze.sha256"
FROZEN_CODE_AND_METADATA = {
    "notes/analysis/jucars-bvival-v5-external-stress-decision-2026-09-25.md",
    "experiments/join_bvival_evaluation_outcomes.py", "experiments/evaluate_bvival_policies.py",
}
FROZEN_ROWS = {
    "experiments/outputs/jucars_bvival_v5_frozen_mae_scores/frozen_policy_scores_absolute_price_error.csv",
    "experiments/outputs/jucars_bvival_frozen_scores_v1/frozen_policy_scores_squared_log_error.csv",
    "experiments/data/processed/jucars_bvival_v1/sealed_labels/test_labels.csv",
    "experiments/data/processed/jucars_bvival_v1/splits/test_features.csv",
    "experiments/outputs/jucars_bvival_prediction_pairs_v1/evaluation_prediction_pairs.csv",
}
SEEDS = [13, 42, 2026]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recorded_graph_checks(a):
    c, p, dv, dr, k, v, r, s, j, e = (a[n] for n in AUDITS)
    passed = []

    def equal(name, left, right):
        if left is None or right is None or left != right:
            raise ValueError("Recorded dependency mismatch: " + name)
        passed.append(name)

    def same_hash(name, left, right):
        if any(not isinstance(x, str) or len(x) != 64 or any(ch not in "0123456789abcdef" for ch in x) for x in (left, right)):
            raise ValueError("Malformed dependency hash: " + name)
        equal(name, left, right)

    for split, key in (("train", "train"), ("validation", "validation"), ("calibration", "calibration"), ("test_features", "evaluation_features")):
        same_hash("cohort_" + split + "_to_pairs", c["outputs"]["splits/" + split + ".csv"], p["inputs"][key + "_sha256"])
    for phase, name, target in (("development", "value", dv), ("development", "risk", dr), ("calibration", "calibration", k)):
        for producer, consumer in (("pairs", "prediction_pairs"), ("features", "pre_action_features"), ("labels", "labels")):
            same_hash("pairs_" + phase + "_" + producer + "_to_" + name,
                      p["outputs"][phase][producer], target["inputs"][consumer]["sha256"])
        equal(name + "_pair_count", p[phase + "_action_pair_count"], target["listing_action_count"])
        equal(name + "_listing_count", p[phase + "_listing_count"], target["listing_count"])
        if target["all_predictions_oof"] is not True:
            raise ValueError("Expected historical out-of-sample target flag")
        passed.append(name + "_out_of_sample_flag")
    equal("value_risk_pre_action_allowlist", dv["feature_allowlist"], dr["feature_allowlist"])
    equal("calibration_pre_action_allowlist", dv["feature_allowlist"], k["feature_allowlist"])
    same_hash("development_v1_to_value", dv["output_sha256"], v["inputs"]["train_actions_sha256"])
    same_hash("development_v2_to_risk", dr["output_sha256"], r["inputs"]["train_actions_sha256"])
    same_hash("development_v2_to_assembly", dr["output_sha256"], s["inputs"]["development_actions"]["sha256"])
    same_hash("calibration_v1_to_value", k["output_sha256"], v["inputs"]["calibration_actions_sha256"])
    for name, head in (("value", v), ("risk", r)):
        same_hash("evaluation_features_to_" + name, p["outputs"]["evaluation"]["features"], head["inputs"]["evaluation_features_sha256"])
        equal(name + "_feature_order", ["action_id"] + dv["feature_allowlist"], head["feature_columns"])
        equal(name + "_seeds", head["seeds"], SEEDS)
    equal("base_seeds", p["seeds"], SEEDS)
    equal("head_categorical_order", v["categorical_columns"], r["categorical_columns"])
    same_hash("value_scores_to_assembly", v["evaluation_action_scores_sha256"], s["inputs"]["value_scores"]["sha256"])
    same_hash("risk_scores_to_assembly", r["evaluation_risk_scores_sha256"], s["inputs"]["risk_scores"]["sha256"])
    same_hash("evaluation_features_to_assembly", p["outputs"]["evaluation"]["features"], s["inputs"]["evaluation_features"]["sha256"])
    same_hash("assembly_to_join", s["output"]["sha256"], j["inputs"]["frozen_policy_scores_sha256"])
    same_hash("evaluation_pairs_to_join", p["outputs"]["evaluation"]["pairs"], j["inputs"]["prediction_pairs_sha256"])
    same_hash("test_labels_to_join", c["outputs"]["sealed_labels/test_labels.csv"], j["inputs"]["sealed_labels_sha256"])
    same_hash("join_to_evaluation", j["output_sha256"], e["evaluation_outcomes_sha256"])
    same_hash("scores_to_evaluation", s["output"]["sha256"], e["frozen_policy_scores_sha256"])
    for name, stage in (("risk", r), ("assembly", s), ("evaluation", e)):
        equal(name + "_objective", stage["objective"], "absolute_price_error")
    equal("reference_policy", e["reference_policy"], "score_uncertainty_only")
    equal("freeze_id", j["freeze_id"], e["freeze_id"])
    for phase, count in (("development", c["split_counts"]["train"] + c["split_counts"]["validation"]),
                         ("calibration", c["split_counts"]["calibration"]), ("evaluation", c["split_counts"]["test"])):
        equal("cohort_" + phase + "_count", p[phase + "_listing_count"], count)
    equal("join_listing_count", j["listing_count"], p["evaluation_listing_count"])
    equal("join_pair_count", j["listing_action_count"], p["evaluation_action_pair_count"])
    if j["labels_opened"] is not True:
        raise ValueError("Historical test cannot be relabelled unopened")
    passed.append("historical_test_opening_retained")
    return passed


def freeze_check(root, a):
    data = safe_path(root, FREEZE).read_bytes()
    rows, code, seen = [], [], set()
    for line in data.decode().splitlines():
        digest, relative = line.split(None, 1)
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest) or relative in seen:
            raise ValueError("Malformed or duplicate freeze entry")
        seen.add(relative)
        if relative in FROZEN_CODE_AND_METADATA:
            actual = sha256(safe_path(root, relative))
            code.append({"path": relative, "frozen_sha256": digest, "current_sha256": actual, "matches": actual == digest})
        elif relative in FROZEN_ROWS:
            rows.append({"path": relative, "recorded_sha256": digest, "row_bytes_checked": False})
        else:
            raise ValueError("Unclassified freeze entry")
    if seen != FROZEN_CODE_AND_METADATA | FROZEN_ROWS:
        raise ValueError("Incomplete JUCars freeze inventory")
    row_hashes = {r["path"]: r["recorded_sha256"] for r in rows}
    expected = {
        "experiments/outputs/jucars_bvival_v5_frozen_mae_scores/frozen_policy_scores_absolute_price_error.csv": a["assembly"]["output"]["sha256"],
        "experiments/outputs/jucars_bvival_frozen_scores_v1/frozen_policy_scores_squared_log_error.csv": a["join"]["inputs"]["additional_frozen_policy_scores_sha256"],
        "experiments/data/processed/jucars_bvival_v1/sealed_labels/test_labels.csv": a["cohort"]["outputs"]["sealed_labels/test_labels.csv"],
        "experiments/data/processed/jucars_bvival_v1/splits/test_features.csv": a["cohort"]["outputs"]["splits/test_features.csv"],
        "experiments/outputs/jucars_bvival_prediction_pairs_v1/evaluation_prediction_pairs.csv": a["pairs"]["outputs"]["evaluation"]["pairs"],
    }
    if row_hashes != expected:
        raise ValueError("Freeze rows do not bind the recorded graph")
    return {"sha256": hashlib.sha256(data).hexdigest(), "code_and_metadata_comparison": code,
            "row_entries_not_opened": rows, "recorded_row_bindings_match_audits": True,
            "upstream_base_value_source_byte_versions_bound_by_this_freeze": False}


def inspect_models(root, a):
    from catboost import CatBoostRegressor
    from importlib.metadata import version
    result = []
    groups = [("base", a["pairs"]["models"])]
    groups += [(role, obj["models"]) for role, obj in a["value"]["objectives"].items()]
    groups += [("risk_mae", a["risk"]["models"])]
    if {role for role, _ in groups} != {"base", "absolute_price_error", "squared_log_error", "risk_mae"} or len(groups) != 4:
        raise ValueError("All four model roles required")
    for role, models in groups:
        if sorted(m["seed"] for m in models) != SEEDS:
            raise ValueError("Three original seeds required for each role")
        for saved in models:
            seed = saved["seed"]
            expected = (f"experiments/outputs/jucars_bvival_prediction_pairs_v1/base_models/mask_aware_catboost_seed_{seed}.cbm"
                        if role == "base" else f"experiments/outputs/jucars_bvival_risk_mae_matched_v1/models/seed_{seed}.cbm"
                        if role == "risk_mae" else f"experiments/outputs/jucars_bvival_value_policy_v1/models/{role}/seed_{seed}.cbm")
            if saved.get("model_path", saved.get("path")) != expected:
                raise ValueError("Model path does not match fixed role/seed")
            path = safe_path(root, expected)
            digest = sha256(path)
            if digest != saved["sha256"]:
                raise ValueError("Saved model hash mismatch")
            model = CatBoostRegressor()
            model.load_model(str(path))
            audit = a["pairs" if role == "base" else "risk" if role == "risk_mae" else "value"]
            features = audit["feature_columns"] + (["action_mask"] if role == "base" else [])
            categorical = audit["categorical_columns"] + (["action_mask"] if role == "base" else [])
            if model.feature_names_ != features or model.get_cat_feature_indices() != [features.index(c) for c in categorical]:
                raise ValueError("Saved feature/categorical order mismatch")
            params = model.get_all_params()
            if params["random_seed"] != seed or model.tree_count_ != params["iterations"] or params["loss_function"] != "RMSE":
                raise ValueError("Model seed/tree/loss mismatch")
            if role == "base":
                for name, value in a["pairs"]["model_parameters"].items():
                    if not math.isclose(params[name], value, rel_tol=1e-6):
                        raise ValueError("Base model parameters differ from run audit")
            metadata = json.loads(model.get_metadata()["params"])
            result.append({"role": role, "seed": seed, "path": expected, "sha256": digest,
                           "tree_count": model.tree_count_, "parameters": {k: params[k] for k in PARAMETER_KEYS},
                           "serialized_thread_count": metadata.get("system_options", {}).get("thread_count"),
                           "feature_names": features, "categorical_feature_indices": model.get_cat_feature_indices()})
    return {"records": result, "reader_catboost_version": version("catboost"),
            "historical_training_environment_certified": False, "fitting_called": False, "prediction_called": False}


def inspect_target_schemas(root, a):
    headers, records = {}, []
    for role in ("development_value", "development_risk", "calibration"):
        relative = AUDITS[role].rsplit("/", 1)[0] + "/bvival_action_dataset.csv"
        path = safe_path(root, relative)
        if sha256(path) != a[role]["output_sha256"]:
            raise ValueError("Target table bytes differ from recorded hash")
        # Parse the header only; hashing above necessarily reads all opaque bytes.
        with path.open(encoding="utf-8", newline="") as stream:
            header = next(csv.reader([stream.readline()]))
        if len(header) != len(set(header)) or not set(a[role]["feature_allowlist"]).issubset(header):
            raise ValueError("Invalid target-table header")
        headers[role] = header
        records.append({"role": role, "path": relative, "sha256": a[role]["output_sha256"], "header": header})
    left, right = headers["development_value"], headers["development_risk"]
    return {"records": records, "added_in_risk_v2": [c for c in right if c not in left],
            "removed_in_risk_v2": [c for c in left if c not in right],
            "common_column_order_preserved": [c for c in left if c in right] == [c for c in right if c in left],
            "row_artifact_opaque_bytes_hashed": True, "data_rows_parsed": False,
            "common_column_row_values_compared": False}


def reconstruct(root, include_models=False, include_schemas=False):
    root = Path(root).resolve(strict=True)
    a, hashes = {}, {}
    for name, relative in AUDITS.items():
        path = safe_path(root, relative)
        a[name], hashes[relative] = json.loads(path.read_text()), sha256(path)
    checks = recorded_graph_checks(a)
    entries = ["experiments/" + name for name in (
        "build_jucars_bvival_manifest.py", "generate_bvival_prediction_pairs.py", "build_bvival_action_dataset.py",
        "fit_bvival_value_policy.py", "fit_bvival_risk_baseline.py", "assemble_bvival_policy_scores.py",
        "join_bvival_evaluation_outcomes.py", "evaluate_bvival_policies.py")]
    report = {
        "date": "2026-10-01", "stage": "HISTORICAL_JUCARS_METADATA_RECOVERY_NOT_EMPIRICAL_REPLAY",
        "audit_file_sha256": hashes, "recorded_dependency_checks": checks,
        "recorded_dependencies_match": True,
        "source_doi": a["cohort"]["dataset_doi"], "source_path": a["cohort"]["source_file"],
        "source_sha256_recorded_not_raw_bytes_checked": a["cohort"]["source_file_sha256"],
        "dictionary_path": a["cohort"]["data_dictionary_file"],
        "dictionary_sha256_recorded_not_bytes_checked": a["cohort"]["data_dictionary_sha256"],
        "cohort_split_counts": a["cohort"]["split_counts"], "recorded_cohort_output_hashes": a["cohort"]["outputs"],
        "phase_counts": {phase: {"listings": a["pairs"][phase + "_listing_count"], "pairs": a["pairs"][phase + "_action_pair_count"]}
                         for phase in ("development", "calibration", "evaluation")},
        "base_parameters_from_run_audit": {"seeds": a["pairs"]["seeds"], "folds": a["pairs"]["folds"], **a["pairs"]["model_parameters"]},
        "base_feature_columns": a["pairs"]["feature_columns"], "base_categorical_columns": a["pairs"]["categorical_columns"],
        "policy_feature_columns": a["value"]["feature_columns"], "policy_categorical_columns": a["value"]["categorical_columns"],
        "action_fields": a["pairs"]["action_fields"], "target_branches": {
            "value_training": AUDITS["development_value"], "risk_training_and_assembly": AUDITS["development_risk"],
            "value_calibration": AUDITS["calibration"],
            "upstream_inputs_equal_but_target_table_bytes_different": a["development_value"]["inputs"] == a["development_risk"]["inputs"]
                and a["development_value"]["output_sha256"] != a["development_risk"]["output_sha256"]},
        "calibrator_parameters": a["value"]["objectives"]["absolute_price_error"]["calibrator"],
        "evaluation_parameters": {k: a["evaluation"][k] for k in ("objective", "reference_policy", "budgets", "bootstrap_repetitions", "bootstrap_seed", "oracle_diagnostic_included")},
        "recorded_primary_score_sha256": a["assembly"]["output"]["sha256"],
        "historical_additional_secondary_score_sha256": a["join"]["inputs"]["additional_frozen_policy_scores_sha256"],
        "primary_aggregate_reference_hashes": a["evaluation"]["outputs"],
        "pretest_freeze": freeze_check(root, a), "current_static_import_closure": static_import_closure(root, entries),
        "historical_test_labels_already_opened": True,
        "historical_v5_reused_earlier_pairs_and_value_models": True, "uniform_historical_v5_full_refit_claimed": False,
        "calibration_flag_semantics": "Fold -1 is held-out calibration from final development models, not internal OOF",
        "saved_final_models_inspected": include_models, "target_headers_inspected": include_schemas,
        "raw_source_or_dictionary_bytes_read": False, "data_rows_parsed": False,
        "fitting_or_prediction_called": False, "new_outcome_metrics_computed": False,
        "empirical_replay_completed": False, "independent_replication_claimed": False,
        "manuscript_or_original_outputs_changed": False, "public_release_created": False, "submission_ready": False,
    }
    if include_models:
        report["model_parameter_inspection"] = inspect_models(root, a)
    if include_schemas:
        report["target_schema_inspection"] = inspect_target_schemas(root, a)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--inspect-model-parameters", action="store_true")
    parser.add_argument("--inspect-target-schemas", action="store_true")
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    try:
        report = reconstruct(args.project_root, args.inspect_model_parameters, args.inspect_target_schemas)
        if args.output_report:
            with args.output_report.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
            print(json.dumps({"stage": report["stage"], "report_path": str(args.output_report),
                              "report_sha256": sha256(args.output_report), "dependency_checks": len(report["recorded_dependency_checks"]),
                              "model_records": len(report.get("model_parameter_inspection", {}).get("records", [])),
                              "data_rows_parsed": False, "empirical_replay_completed": False}))
        else:
            print(json.dumps(report, indent=2))
    except (ValueError, OSError, KeyError, ImportError, SyntaxError) as error:
        parser.exit(1, f"JUCars reconstruction failed: {error}\n")


if __name__ == "__main__":
    main()
