"""Build/validate a non-executing reconstructed MUCars primary replay plan.

Reads code and existing audit/parameter JSONs only. No study imports, subprocess
execution, output directory creation, row-file reads, training or inference.
Commands are argv templates with unresolved runtime/score-hash bindings.
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

# -I omits the script directory. Allow only this reviewed helper directory,
# not the experiments directory or the process working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mucars_reconstruction import AUDITS, safe_path
from check_rerun_entrypoints import inspect_interface, safe_source_path


RECEIPT = "release-bvival/mucars-reconstruction-2026-10-01.json"
DEFAULT_NAME = "mucars-reconstructed-primary-v1"
RUNTIME_BINDING = {"runtime_binding": "validated_main_python"}


def output_prefix(root, name):
    if not isinstance(name, str) or not re.fullmatch(r"mucars-reconstructed-[a-z0-9][a-z0-9-]{0,60}", name):
        raise ValueError("Use a scoped mucars-reconstructed-* replay name")
    relative = "experiments/replays/" + name
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink replay target/component rejected")
    if path.exists():
        raise ValueError("Replay output must be new and nonexistent; no overwrite")
    return relative


def static_import_closure(root, entrypoints):
    pending = list(entrypoints)
    hashes, external, dynamic = {}, set(), []
    while pending:
        relative = pending.pop()
        if relative in hashes:
            continue
        path = safe_source_path(root, relative)
        data = path.read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        tree = ast.parse(data)
        for node in ast.walk(tree):
            imports = []
            if isinstance(node, ast.Import):
                imports = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError("Relative import requires manual closure review")
                imports = [node.module] if node.module else []
            elif isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Name) and node.func.id == "__import__"
                        or isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"):
                    dynamic.append({"source": relative, "line": node.lineno})
            for imported in imports:
                module = imported.split(".")[0]
                if module in sys.stdlib_module_names:
                    continue
                candidate = "experiments/" + module + ".py"
                candidate_path = root / candidate
                if candidate_path.exists() or candidate_path.is_symlink():
                    pending.append(candidate)
                elif (root / "experiments" / module).exists():
                    raise ValueError("Local package import requires manual closure review")
                else:
                    external.add(module)
    if dynamic:
        raise ValueError("Dynamic imports require manual closure review")
    return {"local_source_sha256": dict(sorted(hashes.items())),
            "external_import_roots": sorted(external),
            "static_only_not_runtime_or_installation_validation": True}


def validate_declared_flags(root, stage):
    args = stage["argv_template"]
    interface = inspect_interface(safe_source_path(root, args[1]).read_text(encoding="utf-8"))
    declared = {item["flag"]: item["required"] for item in interface["flags"]}
    present = [arg for arg in args[2:] if isinstance(arg, str) and arg.startswith("--")]
    if len(present) != len(set(present)) or any(flag not in declared for flag in present):
        raise ValueError("Repeated or undeclared planned CLI flag")
    if any(flag not in present for flag, required in declared.items() if required):
        raise ValueError("A declared required CLI flag is missing")
    for group in interface["mutually_exclusive_groups"]:
        count = sum(flag in present for flag in group["flags"])
        if count > 1 or group["required"] and count != 1:
            raise ValueError("Planned mutually exclusive arguments conflict")


def build_plan(project_root, name=DEFAULT_NAME):
    root = Path(project_root).resolve(strict=True)
    prefix = output_prefix(root, name)
    receipt_bytes = safe_path(root, RECEIPT).read_bytes()
    receipt = json.loads(receipt_bytes)
    if (receipt.get("stage") != "HISTORICAL_MUCARS_METADATA_RECONSTRUCTION_NOT_EMPIRICAL_REPLAY"
            or receipt.get("full_empirical_retraining_verified") is not False
            or receipt.get("historical_test_labels_already_opened") is not True):
        raise ValueError("Expected bounded historical reconstruction receipt")
    if set(receipt["audit_file_sha256"]) != set(AUDITS.values()):
        raise ValueError("All nine historical audit bindings are required")
    for relative, expected in receipt["audit_file_sha256"].items():
        # Paths are not blindly followed from the receipt.
        if relative not in AUDITS.values():
            raise ValueError("Unclassified audit path in reconstruction receipt")
        if hashlib.sha256(safe_path(root, relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Historical audit changed since parameter recovery")
    models = receipt["model_parameter_inspection"]["records"]
    bindings = {(item["role"], item["seed"]) for item in models}
    expected_bindings = {(role, seed) for role in ("base", "absolute_price_error", "squared_log_error", "risk_mae") for seed in (13, 42)}
    if len(models) != 8 or bindings != expected_bindings:
        raise ValueError("Recovered model parameters incomplete")
    for item in models:
        params = item["parameters"]
        rate = 0.08 if item["role"] == "base" else 0.05
        if (params["iterations"] != 200 or params["depth"] != 6 or params["l2_leaf_reg"] != 10
                or params["random_seed"] != item["seed"] or params["loss_function"] != "RMSE"
                or not math.isclose(params["learning_rate"], rate, rel_tol=1e-6)):
            raise ValueError("Recovered model parameters do not support these templates")
    features = ",".join(receipt["policy_feature_columns"])
    action_features = ",".join(receipt["policy_feature_columns"][1:])
    categorical = ",".join(receipt["policy_categorical_columns"])
    actions = ",".join(receipt["action_fields"])
    dirs = {key: prefix + "/" + key for key in ("cohort", "pairs", "development-actions", "calibration-actions", "value", "risk-mae", "scores", "evaluation-inputs", "evaluation-mae")}
    source = "experiments/data/external/mucars_2024_v2/cars_dataframe.csv"
    eval_features = dirs["pairs"] + "/evaluation_pre_action_features.csv"
    dev_actions = dirs["development-actions"] + "/bvival_action_dataset.csv"
    cal_actions = dirs["calibration-actions"] + "/bvival_action_dataset.csv"
    scores = dirs["scores"] + "/frozen_policy_scores_absolute_price_error.csv"
    fit_flags = ["--seeds", "13,42", "--iterations", "200", "--depth", "6", "--learning-rate", "0.05", "--l2-leaf-reg", "10"]
    stages = []

    def stage(identifier, file, args, depends, phase, access):
        item = {"id": identifier, "argv_template": [dict(RUNTIME_BINDING), "experiments/" + file, *args],
                "depends_on": depends, "phase": phase, "declared_data_access": access}
        validate_declared_flags(root, item)
        stages.append(item)

    stage("cohort", "build_mucars_bvival_manifest.py",
          ["--source", source, "--output-dir", dirs["cohort"]], [], "reconstruction",
          "Historical full-source prices are read for quality filtering/split construction; not an unseen-label claim")
    stage("pairs", "generate_bvival_prediction_pairs.py",
          ["--train", dirs["cohort"] + "/splits/train.csv", "--validation", dirs["cohort"] + "/splits/validation.csv",
           "--calibration", dirs["cohort"] + "/splits/calibration.csv", "--evaluation-features", dirs["cohort"] + "/splits/test_features.csv",
           "--output-dir", dirs["pairs"], "--action-fields", actions, "--seeds", "13,42", "--folds", "3",
           "--iterations", "200", "--depth", "6", "--learning-rate", "0.08", "--l2-leaf-reg", "10"],
          ["cohort"], "scoring", "Development/calibration data and label-free evaluation features; no evaluation loss")
    for phase, identifier in (("development", "development-actions"), ("calibration", "calibration-actions")):
        stage(identifier, "build_bvival_action_dataset.py",
              ["--prediction-pairs", dirs["pairs"] + f"/{phase}_prediction_pairs.csv", "--labels", dirs["pairs"] + f"/{phase}_labels.csv",
               "--pre-action-features", dirs["pairs"] + f"/{phase}_pre_action_features.csv", "--feature-columns", action_features,
               "--output-dir", dirs[identifier], "--evidence-status", "development_only"],
              ["pairs"], "scoring", "Permitted development or held-out calibration targets, not evaluation outcomes")
    stage("value", "fit_bvival_value_policy.py",
          ["--train-actions", dev_actions, "--calibration-actions", cal_actions, "--evaluation-features", eval_features,
           "--feature-columns", features, "--categorical-columns", categorical, "--output-dir", dirs["value"],
           *fit_flags, "--alpha", "0.1", "--minimum-action-group-size", "100"],
          ["development-actions", "calibration-actions"], "scoring", "Fits both original value objectives; no evaluation targets")
    stage("risk-mae", "fit_bvival_risk_baseline.py",
          ["--train-actions", dev_actions, "--evaluation-features", eval_features,
           "--feature-columns", features, "--categorical-columns", categorical,
           "--output-dir", dirs["risk-mae"], "--objective", "absolute_price_error", *fit_flags],
          ["development-actions"], "scoring", "Original MAE risk objective; no new development-risk OOF selection")
    stage("scores", "assemble_bvival_policy_scores.py",
          ["--development-actions", dev_actions, "--value-scores", dirs["value"] + "/evaluation_action_scores.csv",
           "--risk-scores", dirs["risk-mae"] + "/evaluation_risk_scores.csv", "--evaluation-features", eval_features,
           "--objective", "absolute_price_error", "--random-seed", "2026", "--output-dir", dirs["scores"]],
          ["value", "risk-mae"], "scoring", "Original seven policy columns; no new fixed-field comparator or outcome-based selection")
    stage("join", "join_bvival_evaluation_outcomes.py",
          ["--evaluation-prediction-pairs", dirs["pairs"] + "/evaluation_prediction_pairs.csv",
           "--sealed-labels", dirs["cohort"] + "/sealed_labels/test_labels.csv", "--frozen-policy-scores", scores,
           "--expected-policy-scores-sha256", {"frozen_artifact_sha256": scores},
           "--output-dir", dirs["evaluation-inputs"], "--freeze-id", name],
          ["scores", "new_scoring_freeze_checkpoint"], "evaluation", "Already-opened historical labels; new score hash must be frozen before this outcome join")
    score_columns = ["score_no_acquisition", "score_random", "score_global_field_prior", "score_uncertainty_only",
                     "score_disagreement_only", "score_mean_value", "score_conservative_lower_value"]
    stage("evaluate", "evaluate_bvival_policies.py",
          ["--evaluation-outcomes", dirs["evaluation-inputs"] + "/evaluation_outcomes.csv", "--frozen-policy-scores", scores,
           "--score-columns", ",".join(score_columns), "--reference-policy", "score_uncertainty_only",
           "--objective", "absolute_price_error", "--budgets", "0.01,0.05,0.10,0.20,0.30",
           "--bootstrap-repetitions", "10000", "--bootstrap-seed", "2026", "--output-dir", dirs["evaluation-mae"],
           "--freeze-id", name, "--analysis-status", "post_test_exploratory", "--include-oracle-diagnostic"],
          ["join"], "evaluation", "Historical primary-MAE reconstruction, not a fresh confirmatory experiment")
    closure = static_import_closure(root, [item["argv_template"][1] for item in stages])
    cohort_audit = json.loads(safe_path(root, AUDITS["cohort"]).read_text())
    pair_audit = json.loads(safe_path(root, AUDITS["pairs"]).read_text())
    return {
        "schema_version": 1, "stage": "DRAFT_RECONSTRUCTED_PRIMARY_REPLAY_PLAN_NOT_EXECUTED", "date": "2026-10-01",
        "replay_name": name, "planned_output_root": prefix,
        "execution_started": False, "executor_implemented": False, "submission_ready": False,
        "historical_tests_previously_opened": True, "new_preregistration": False,
        "historical_source_byte_identity_claimed": False, "source_doi": receipt["source_doi"],
        "source_sha256_to_verify_before_execution": receipt["source_sha256_recorded_not_row_checked"],
        "parameter_recovery_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "expected_structural_contract": {
            "split_listing_counts": cohort_audit["split_counts"],
            "recorded_cohort_output_sha256_to_check_after_rebuilding": cohort_audit["output_hashes"],
            "phase_counts": {phase: {"listings": pair_audit[phase + "_listing_count"],
                                     "action_pairs": pair_audit[phase + "_action_pair_count"]}
                             for phase in ("development", "calibration", "evaluation")},
            "base_feature_columns": receipt["base_feature_columns"],
            "base_categorical_columns": receipt["base_categorical_columns"],
            "final_valuation_model_appended_feature": "action_mask",
            "policy_feature_columns": receipt["policy_feature_columns"],
            "policy_categorical_columns": receipt["policy_categorical_columns"]},
        "static_source_closure": closure, "stages": stages,
        "unresolved_runtime_bindings": ["validated_main_python", "new_scoring_freeze_checkpoint", "frozen_artifact_sha256"],
        "runtime_role": {"task": "CPU", "original_serialized_thread_count": 10,
                         "thread_count_cli_option_available": False,
                         "record_actual_parallelism_and_implicit_catboost_defaults": True,
                         "complete_historical_training_environment_certified": False},
        "pre_execution_gates": ["validate a dedicated environment against this static closure and archive its actual lock",
                                "verify original source bytes against the recorded SHA256 and confirm legitimate access",
                                "snapshot current bound code and plan before running; do not claim historical source identity",
                                "use a new output root; original outputs/freezes/manuscript remain read-only"],
        "new_scoring_freeze_checkpoint": {
            "required_before": "join", "bind": ["plan_sha256", "code_closure_sha256", "runtime_lock_sha256", "new_score_sha256", "generated_input_hashes"],
            "feature_schema_and_cohort_counts_must_match_recovered_metadata": True,
            "not_an_external_preregistration_or_unseen_label_security_guarantee": True},
        "engineering_concordance_rules": {
            "author_chosen_not_journal_or_statistical_significance_thresholds": True,
            "historical_results_already_known_when_rules_chosen": True,
            "integer_counts_and_aggregate_key_sets": "exact",
            "price_mae_absolute_tolerance_MAD": 0.05,
            "rmsle_absolute_tolerance": 1e-6,
            "relative_gain_and_ci_endpoint_tolerance_percentage_points": 0.01,
            "fraction_metrics_absolute_tolerance": 1e-6,
            "original_aggregate_reference": "experiments/outputs/mucars_bvival_v5_confirmatory_mae",
            "reference_aggregate_hashes": json.loads(safe_path(root, "experiments/outputs/mucars_bvival_v5_confirmatory_mae/bvival_evaluation_audit.json").read_text())["outputs"],
            "a_changed_zero_or_two_percent_interval_boundary_must_be_disclosed": True,
            "on_failure": "Retain and explain discrepancies; no threshold relaxation, seed selection, tuning or replacement of original results",
            "passing_does_not_establish_independent_scientific_replication": True},
        "scope_exclusions": ["secondary RMSLE routing refit", "subgroup bootstrap", "post-test new comparators", "cross-source reruns",
                             "real workflow/cost outcomes", "public deposit or software licensing"],
        "data_access_this_planning_command": {"row_files_read": False, "models_loaded": False, "study_modules_imported": False,
                                             "training_called": False, "inference_called": False, "directories_created": False},
    }


def validate_plan(project_root, plan):
    expected = build_plan(project_root, plan.get("replay_name"))
    def strict_boolean_metadata(actual, reference):
        if type(reference) is bool:
            if type(actual) is not bool:
                raise ValueError("Boolean plan boundaries cannot be numeric/string stand-ins")
        elif isinstance(reference, dict) and isinstance(actual, dict):
            for key, value in reference.items():
                strict_boolean_metadata(actual.get(key), value)
        elif isinstance(reference, list) and isinstance(actual, list):
            for item, value in zip(actual, reference):
                strict_boolean_metadata(item, value)
    strict_boolean_metadata(plan, expected)
    if plan != expected:
        raise ValueError("Plan differs from bound code/metadata/templates/concordance rules")
    return {"plan_matches_current_bound_sources_and_metadata": True,
            "command_templates": len(plan["stages"]),
            "local_source_files": len(plan["static_source_closure"]["local_source_sha256"]),
            "execution_started": False, "empirical_replay_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--replay-name", default=DEFAULT_NAME)
    parser.add_argument("--check-plan", type=Path)
    args = parser.parse_args()
    try:
        result = (validate_plan(args.project_root, json.loads(args.check_plan.read_text(encoding="utf-8")))
                  if args.check_plan else build_plan(args.project_root, args.replay_name))
    except (ValueError, OSError, KeyError, SyntaxError) as error:
        parser.exit(1, f"Replay planning failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
