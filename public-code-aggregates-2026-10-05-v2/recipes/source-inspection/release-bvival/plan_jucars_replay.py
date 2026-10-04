"""Freeze/check JUCars primary reconstruction templates, without execution.

Only code and audit/parameter/schema JSONs are read. No source records, models,
old targets, scores or test labels are opened. No replay directory is created.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_jucars_reconstruction import AUDITS, recorded_graph_checks, SEEDS
from check_mucars_reconstruction import safe_path
from plan_mucars_replay import static_import_closure, validate_declared_flags
from check_rerun_entrypoints import inspect_interface

RECEIPT = "release-bvival/jucars-reconstruction-2026-10-01.json"
ADAPTER = "release-bvival/project_jucars_legacy_targets.py"
DEFAULT_NAME = "jucars-reconstructed-primary-v1"
RUNTIME = {"runtime_binding": "validated_jucars_python"}
SCORES = ["score_no_acquisition", "score_random", "score_global_field_prior", "score_uncertainty_only",
          "score_disagreement_only", "score_mean_value", "score_conservative_lower_value"]


def new_prefix(root, name):
    if not isinstance(name, str) or not re.fullmatch(r"jucars-reconstructed-[a-z0-9][a-z0-9-]{0,60}", name):
        raise ValueError("Scoped jucars-reconstructed-* name required")
    relative = "experiments/replays/" + name
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink replay component rejected")
    if path.exists():
        raise ValueError("New nonexistent replay root required; no overwrite")
    return relative


def adapter_interface(root, stage):
    data = safe_path(root, ADAPTER).read_text()
    for node in ast.walk(ast.parse(data)):
        names = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
        if isinstance(node, ast.ImportFrom) and node.level or any(n.split(".")[0] not in sys.stdlib_module_names for n in names):
            raise ValueError("Adapter must remain standard-library-only")
        if isinstance(node, ast.Call) and (isinstance(node.func, ast.Name) and node.func.id == "__import__" or isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"):
            raise ValueError("Adapter dynamic imports require manual review")
    flags = {x["flag"]: x["required"] for x in inspect_interface(data)["flags"]}
    present = [x for x in stage["argv_template"][2:] if isinstance(x, str) and x.startswith("--")]
    if len(present) != len(set(present)) or any(f not in flags for f in present) or any(f not in present for f, required in flags.items() if required):
        raise ValueError("Adapter CLI template differs from declarations")


def build_plan(root, name=DEFAULT_NAME):
    root = Path(root).resolve(strict=True)
    prefix = new_prefix(root, name)
    data = safe_path(root, RECEIPT).read_bytes()
    receipt = json.loads(data)
    if receipt.get("stage") != "HISTORICAL_JUCARS_METADATA_RECOVERY_NOT_EMPIRICAL_REPLAY" or receipt.get("empirical_replay_completed") is not False or receipt.get("historical_test_labels_already_opened") is not True:
        raise ValueError("Bound non-replayed historical recovery receipt required")
    if set(receipt["audit_file_sha256"]) != set(AUDITS.values()):
        raise ValueError("All ten historical audit bindings required")
    audits = {}
    for role, relative in AUDITS.items():
        path = safe_path(root, relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["audit_file_sha256"][relative]:
            raise ValueError("Historical audit drift")
        audits[role] = json.loads(path.read_text())
    checks = recorded_graph_checks(audits)
    for field, audit_role, audit_field in (
            ("base_feature_columns", "pairs", "feature_columns"),
            ("base_categorical_columns", "pairs", "categorical_columns"),
            ("policy_feature_columns", "value", "feature_columns"),
            ("policy_categorical_columns", "value", "categorical_columns"),
            ("action_fields", "pairs", "action_fields")):
        if receipt[field] != audits[audit_role][audit_field]:
            raise ValueError("Recovered schema differs from audit: " + field)
    phases = {phase: {"listings": audits["pairs"][phase + "_listing_count"],
                      "pairs": audits["pairs"][phase + "_action_pair_count"]}
              for phase in ("development", "calibration", "evaluation")}
    if receipt["phase_counts"] != phases:
        raise ValueError("Recovered phase counts differ from audits")
    base_contract = {"seeds": SEEDS, "folds": 5, "iterations": 400,
                     "depth": 8, "learning_rate": .08, "l2_leaf_reg": 10}
    if receipt["base_parameters_from_run_audit"] != base_contract or audits["pairs"]["model_parameters"] != {k: v for k, v in base_contract.items() if k not in {"seeds", "folds"}} or audits["pairs"]["folds"] != 5:
        raise ValueError("Five-fold 400-tree base audit contract required")
    models = receipt["model_parameter_inspection"]["records"]
    roles = {"base", "absolute_price_error", "squared_log_error", "risk_mae"}
    if len(models) != 12 or {(m["role"], m["seed"]) for m in models} != {(role, seed) for role in roles for seed in SEEDS}:
        raise ValueError("Twelve model/seed parameter records required")
    for m in models:
        p = m["parameters"]
        features = receipt["base_feature_columns"] + ["action_mask"] if m["role"] == "base" else receipt["policy_feature_columns"]
        cats = receipt["base_categorical_columns"] + ["action_mask"] if m["role"] == "base" else receipt["policy_categorical_columns"]
        if m["feature_names"] != features or m["categorical_feature_indices"] != [features.index(c) for c in cats] or m["tree_count"] != 400:
            raise ValueError("Recovered model feature/category/tree contract differs")
        if (p["iterations"] != 400 or p["depth"] != (8 if m["role"] == "base" else 6)
            or p["random_seed"] != m["seed"] or p["l2_leaf_reg"] != 10 or p["loss_function"] != "RMSE"
            or p["task_type"] != "CPU" or not math.isclose(p["learning_rate"], .08 if m["role"] == "base" else .05, rel_tol=1e-6)):
            raise ValueError("Recovered models do not support fixed templates")
    if receipt["base_parameters_from_run_audit"]["folds"] != 5 or receipt["base_parameters_from_run_audit"]["seeds"] != SEEDS:
        raise ValueError("Five-fold three-seed base contract required")
    schema = receipt["target_schema_inspection"]
    headers = {r["role"]: r for r in schema["records"]}
    if len(schema["records"]) != 3 or set(headers) != {"development_value", "development_risk", "calibration"}:
        raise ValueError("Three recorded target headers required")
    current = headers["development_risk"]["header"]
    expected_legacy = [c for c in current if c != "risk_target_absolute_price_error_before"]
    if len(current) != len(set(current)) or "risk_target_absolute_price_error_before" not in current:
        raise ValueError("Current target header invalid")
    for role in ("development_value", "calibration"):
        if headers[role]["header"] != expected_legacy or headers[role]["sha256"] != audits[role]["output_sha256"]:
            raise ValueError("Legacy header/hash differs from role contract")
    if headers["development_risk"]["sha256"] != audits["development_risk"]["output_sha256"]:
        raise ValueError("Current development target hash differs")
    calibrator = audits["value"]["objectives"]["absolute_price_error"]["calibrator"]
    if calibrator["alpha"] != .1 or calibrator["minimum_group_size"] != 100:
        raise ValueError("Calibrator differs from fixed contract")
    e = audits["evaluation"]
    if (e["budgets"] != [.01, .05, .10, .20, .30] or e["bootstrap_repetitions"] != 10000
            or e["bootstrap_seed"] != 2026 or e["oracle_diagnostic_included"] is not True):
        raise ValueError("Historical evaluation settings differ from templates")
    if receipt["historical_additional_secondary_score_sha256"] != audits["join"]["inputs"]["additional_frozen_policy_scores_sha256"]:
        raise ValueError("Historical secondary-score record differs")
    dirs = {k: prefix + "/" + k for k in ("cohort", "pairs", "development-current", "calibration-current",
            "development-legacy", "calibration-legacy", "value", "risk-mae", "scores", "evaluation-inputs", "evaluation-mae")}
    stages = []
    def stage(identifier, file, args, dependencies, phase):
        s = {"id": identifier, "argv_template": [dict(RUNTIME), file, *args], "depends_on": dependencies, "phase": phase}
        (adapter_interface if file == ADAPTER else validate_declared_flags)(root, s)
        stages.append(s)
    c = audits["cohort"]
    source, dictionary = receipt["source_path"], receipt["dictionary_path"]
    if source != "experiments/data/external/jucars_2024_v2/cars_jordan.csv" or dictionary != "experiments/data/external/jucars_2024_v2/data_dictionary.csv":
        raise ValueError("Expected original v2 source/dictionary paths")
    if receipt["source_sha256_recorded_not_raw_bytes_checked"] != c["source_file_sha256"] or receipt["dictionary_sha256_recorded_not_bytes_checked"] != c["data_dictionary_sha256"]:
        raise ValueError("Recorded raw source/dictionary hash mismatch")
    stage("cohort", "experiments/build_jucars_bvival_manifest.py", ["--source", source, "--data-dictionary", dictionary,
          "--source-record-sha256", c["source_file_sha256"], "--dictionary-record-sha256", c["data_dictionary_sha256"],
          "--output-dir", dirs["cohort"]], [], "reconstruction")
    stage("pairs", "experiments/generate_bvival_prediction_pairs.py", ["--train", dirs["cohort"] + "/splits/train.csv",
          "--validation", dirs["cohort"] + "/splits/validation.csv", "--calibration", dirs["cohort"] + "/splits/calibration.csv",
          "--evaluation-features", dirs["cohort"] + "/splits/test_features.csv", "--output-dir", dirs["pairs"],
          "--action-fields", ",".join(receipt["action_fields"]), "--seeds", "13,42,2026", "--folds", "5", "--iterations", "400",
          "--depth", "8", "--learning-rate", "0.08", "--l2-leaf-reg", "10"], ["cohort"], "scoring")
    features, cats = ",".join(receipt["policy_feature_columns"]), ",".join(receipt["policy_categorical_columns"])
    for phase in ("development", "calibration"):
        stage(phase + "-current", "experiments/build_bvival_action_dataset.py", ["--prediction-pairs", dirs["pairs"] + f"/{phase}_prediction_pairs.csv",
              "--labels", dirs["pairs"] + f"/{phase}_labels.csv", "--pre-action-features", dirs["pairs"] + f"/{phase}_pre_action_features.csv",
              "--feature-columns", ",".join(receipt["policy_feature_columns"][1:]), "--output-dir", dirs[phase + "-current"],
              "--evidence-status", "development_only"], ["pairs"], "scoring")
    contract_hash = hashlib.sha256(data).hexdigest()
    for phase, role in (("development", "development_value"), ("calibration", "calibration")):
        stage(phase + "-legacy", ADAPTER, ["--input", dirs[phase + "-current"] + "/bvival_action_dataset.csv",
              "--contract", RECEIPT, "--expected-contract-sha256", contract_hash, "--role", role,
              "--expected-input-sha256", {"generated_target_hash_from_audit": dirs[phase + "-current"] + "/bvival_action_dataset_audit.json"},
              "--expected-output-sha256", headers[role]["sha256"],
              "--expected-listings", str(receipt["phase_counts"][phase]["listings"]),
              "--expected-pairs", str(receipt["phase_counts"][phase]["pairs"]), "--output-dir", dirs[phase + "-legacy"]],
              [phase + "-current", "target_schema_checkpoint"], "scoring")
    eval_features = dirs["pairs"] + "/evaluation_pre_action_features.csv"
    fits = ["--seeds", "13,42,2026", "--iterations", "400", "--depth", "6", "--learning-rate", "0.05", "--l2-leaf-reg", "10"]
    stage("value", "experiments/fit_bvival_value_policy.py", ["--train-actions", dirs["development-legacy"] + "/bvival_action_dataset.csv",
          "--calibration-actions", dirs["calibration-legacy"] + "/bvival_action_dataset.csv", "--evaluation-features", eval_features,
          "--feature-columns", features, "--categorical-columns", cats, "--output-dir", dirs["value"], *fits,
          "--alpha", "0.1", "--minimum-action-group-size", "100"], ["development-legacy", "calibration-legacy"], "scoring")
    stage("risk-mae", "experiments/fit_bvival_risk_baseline.py", ["--train-actions", dirs["development-current"] + "/bvival_action_dataset.csv",
          "--evaluation-features", eval_features, "--feature-columns", features, "--categorical-columns", cats,
          "--objective", "absolute_price_error", "--output-dir", dirs["risk-mae"], *fits], ["development-current", "target_schema_checkpoint"], "scoring")
    scores = dirs["scores"] + "/frozen_policy_scores_absolute_price_error.csv"
    stage("scores", "experiments/assemble_bvival_policy_scores.py", ["--development-actions", dirs["development-current"] + "/bvival_action_dataset.csv",
          "--value-scores", dirs["value"] + "/evaluation_action_scores.csv", "--risk-scores", dirs["risk-mae"] + "/evaluation_risk_scores.csv",
          "--evaluation-features", eval_features, "--objective", "absolute_price_error", "--random-seed", "2026", "--output-dir", dirs["scores"]],
          ["value", "risk-mae"], "scoring")
    stage("join", "experiments/join_bvival_evaluation_outcomes.py", ["--evaluation-prediction-pairs", dirs["pairs"] + "/evaluation_prediction_pairs.csv",
          "--sealed-labels", dirs["cohort"] + "/sealed_labels/test_labels.csv", "--frozen-policy-scores", scores,
          "--expected-policy-scores-sha256", {"frozen_artifact_sha256": scores}, "--output-dir", dirs["evaluation-inputs"], "--freeze-id", name],
          ["scores", "new_scoring_freeze_checkpoint"], "evaluation")
    stage("evaluate", "experiments/evaluate_bvival_policies.py", ["--evaluation-outcomes", dirs["evaluation-inputs"] + "/evaluation_outcomes.csv",
          "--frozen-policy-scores", scores, "--score-columns", ",".join(SCORES), "--reference-policy", "score_uncertainty_only",
          "--objective", "absolute_price_error", "--budgets", "0.01,0.05,0.10,0.20,0.30", "--bootstrap-repetitions", "10000",
          "--bootstrap-seed", "2026", "--output-dir", dirs["evaluation-mae"], "--freeze-id", name,
          "--analysis-status", "post_test_exploratory", "--include-oracle-diagnostic"], ["join"], "evaluation")
    closure = static_import_closure(root, [s["argv_template"][1] for s in stages if s["argv_template"][1] != ADAPTER])
    if closure != receipt["current_static_import_closure"]:
        raise ValueError("Study code changed after source-specific recovery")
    return {
        "date": "2026-10-01", "schema_version": 1, "stage": "DRAFT_JUCARS_PRIMARY_RECONSTRUCTION_PLAN_NOT_EXECUTED",
        "replay_name": name, "planned_output_root": prefix, "execution_started": False, "orchestrator_implemented": False,
        "historical_tests_already_opened": True, "new_preregistration": False, "historical_full_source_identity_claimed": False,
        "uniform_historical_v5_full_refit_claimed": False, "public_release_created": False, "submission_ready": False,
        "source_doi": receipt["source_doi"], "recovery_receipt_sha256": contract_hash,
        "source_bindings_to_verify_before_execution": {"source": {"path": source, "sha256": c["source_file_sha256"]},
                                                       "dictionary": {"path": dictionary, "sha256": c["data_dictionary_sha256"]}},
        "static_study_source_closure": closure,
        "schema_adapter_source": {"path": ADAPTER, "sha256": hashlib.sha256(safe_path(root, ADAPTER).read_bytes()).hexdigest(), "standard_library_only": True},
        "expected_structural_contract": {"split_counts": c["split_counts"], "cohort_output_sha256": c["outputs"],
            "phase_counts": receipt["phase_counts"], "base_feature_columns": receipt["base_feature_columns"],
            "base_categorical_columns": receipt["base_categorical_columns"], "appended_base_feature_and_category": "action_mask",
            "policy_feature_columns": receipt["policy_feature_columns"], "policy_categorical_columns": receipt["policy_categorical_columns"],
            "new_target_header": current, "legacy_target_header": expected_legacy,
            "development_current_sha256": headers["development_risk"]["sha256"],
            "development_legacy_sha256": headers["development_value"]["sha256"], "calibration_legacy_sha256": headers["calibration"]["sha256"]},
        "stages": stages,
        "target_schema_checkpoint": {"before": ["development-legacy", "calibration-legacy", "risk-mae"], "conditions": [
            "current development bytes match recorded v2 hash and all phase listing/action counts match",
            "both current target headers match recorded v2 schema and listing/action keys are valid",
            "cohort/pair feature and categorical schemas match before fitting heads"],
            "on_failure": "retain discrepancy and stop; no value adjustments, alternative projection or silent template change"},
        "legacy_projection_checkpoints": {"before": "value", "conditions": [
            "both projection audits pass recorded legacy byte hashes, counts, headers and unique keys",
            "numeric cell text was not recomputed and no historical target rows were used"]},
        "new_scoring_freeze_checkpoint": {"required_before": "join", "bind": ["plan_sha256", "study_source_hashes", "adapter_source_hash",
            "runtime_lock_hash", "generated_inputs_and_projection_hashes", "new_score_sha256"], "implemented_by_planner": False},
        "primary_only_join_contract": {"historical_join_also_checked_secondary_score": True,
            "historical_secondary_score_sha256": receipt["historical_additional_secondary_score_sha256"],
            "secondary_score_verification_replayed": False, "exact_full_historical_join_protocol_claimed": False,
            "reason": "Primary-MAE numerical reconstruction only; no cached historical secondary-policy artifact dependency or secondary routing refit"},
        "engineering_concordance_rules": {"integer_counts_and_aggregate_key_sets": "exact", "price_mae_absolute_tolerance_JOD": .05,
            "rmsle_absolute_tolerance": 1e-6, "relative_gain_and_ci_endpoint_tolerance_percentage_points": .01,
            "fraction_metrics_absolute_tolerance": 1e-6, "original_aggregate_reference": "experiments/outputs/jucars_bvival_v5_stress_mae",
            "reference_aggregate_hashes": audits["evaluation"]["outputs"], "interval_boundary_changes_at_zero_or_two_percent_must_be_disclosed": True,
            "author_chosen_engineering_tolerances_after_historical_results_known": True, "not_journal_or_statistical_equivalence_rules": True,
            "on_failure": "retain and disclose; no retuning, tolerance widening, seed selection or original result replacement"},
        "unresolved_bindings": ["validated_jucars_python", "generated_target_hash_from_audit", "target_schema_checkpoint",
                                "new_scoring_freeze_checkpoint", "frozen_artifact_sha256"],
        "data_access_this_planning_step": {"original_data_rows_read": False, "models_loaded": False, "study_modules_imported": False,
            "training_or_inference_called": False, "replay_output_directory_created": False},
        "scope_exclusions": ["secondary routing evaluation", "subgroup bootstrap", "new comparators", "unopened independent source",
            "full historical source-byte recovery", "public deposit and licence", "scientific novelty claim"],
        "recorded_dependency_checks_revalidated": len(checks),
    }


def validate_plan(root, plan):
    expected = build_plan(root, plan.get("replay_name"))
    def check_booleans(actual, reference):
        if type(reference) is bool and type(actual) is not bool:
            raise ValueError("Boolean boundaries cannot be numeric/string stand-ins")
        if isinstance(reference, dict) and isinstance(actual, dict):
            for k, v in reference.items():
                check_booleans(actual.get(k), v)
        if isinstance(reference, list) and isinstance(actual, list):
            for x, v in zip(actual, reference):
                check_booleans(x, v)
    check_booleans(plan, expected)
    if plan != expected:
        raise ValueError("Plan differs from fixed code, schemas, parameters or engineering rules")
    return {"stage": "JUCARS_PLAN_STATIC_CHECK_PASSED_NO_EXECUTION", "command_templates": len(plan["stages"]),
            "study_source_files": len(plan["static_study_source_closure"]["local_source_sha256"]),
            "schema_adapter_source_bound": True, "execution_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--replay-name", default=DEFAULT_NAME)
    parser.add_argument("--check-plan", type=Path)
    parser.add_argument("--output-plan", type=Path)
    args = parser.parse_args()
    try:
        if args.check_plan and args.output_plan:
            raise ValueError("Check and create are separate operations")
        result = validate_plan(args.project_root, json.loads(args.check_plan.read_text())) if args.check_plan else build_plan(args.project_root, args.replay_name)
        if args.output_plan:
            with args.output_plan.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
                stream.write("\n")
            print(json.dumps({"stage": result["stage"], "plan_path": str(args.output_plan),
                              "plan_sha256": hashlib.sha256(args.output_plan.read_bytes()).hexdigest(), "execution_started": False}))
        else:
            print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, SyntaxError) as error:
        parser.exit(1, f"JUCars planning failed: {error}\n")


if __name__ == "__main__":
    main()
