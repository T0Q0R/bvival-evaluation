"""Bind AutoScout24 primary reconstruction CLI templates, without execution.

Only recovered metadata and source code are read. No raw rows, model loading,
training, prediction, label joining, bootstrap or replay output creation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_autoscout24_reconstruction import AUDITS, CONFIG, selection_checks, SEEDS
from check_mucars_reconstruction import safe_path, recorded_graph_checks
from check_remaining_source_metadata import AUTO
from plan_mucars_replay import static_import_closure, validate_declared_flags

RECEIPT = "release-bvival/autoscout24-reconstruction-2026-10-01.json"
RECEIPT_HASH = "3956332b97d5be334e1f97f3e83e3f5b0f9b1bf7f8df106374e421557b7404fd"
DEFAULT_NAME = "autoscout24-reconstructed-primary-v1"
RUNTIME = {"runtime_binding": "validated_autoscout24_main_python"}
SCORING_IDS = ["cohort", "selection-pairs", "selection-train-targets", "selection-validation-targets",
               "selection-risk", "fixed-field", "final-pairs", "development-targets", "calibration-targets",
               "value", "risk-mae", "scores"]


def new_prefix(root, name, allow_existing=False):
    if not isinstance(name, str) or not re.fullmatch(r"autoscout24-reconstructed-[a-z0-9][a-z0-9-]{0,60}", name):
        raise ValueError("Scoped autoscout24-reconstructed-* name required")
    relative = "experiments/replays/" + name
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink replay component rejected")
    if path.exists() and not allow_existing:
        raise ValueError("New nonexistent replay root required")
    return relative


def make_stages(root, prefix, receipt, a):
    dirs = {name: prefix + "/" + name for name in SCORING_IDS}
    stages = []
    def add(name, module, args, deps, phase="scoring"):
        stage = {"id": name, "argv_template": [dict(RUNTIME), "experiments/" + module + ".py", *args],
                 "depends_on": deps, "phase": phase}
        validate_declared_flags(root, stage)
        stages.append(stage)
    source = receipt["source_path"]
    add("cohort", "build_autoscout24_bvival_manifest", ["--source", source, "--output-dir", dirs["cohort"]], [], "reconstruction")
    split = dirs["cohort"] + "/splits/"
    base = ["--action-fields", ",".join(receipt["action_fields"]), "--seeds", "13,42,2026", "--folds", "5",
            "--iterations", "400", "--depth", "8", "--learning-rate", "0.08", "--l2-leaf-reg", "10"]
    head = ["--seeds", "13,42,2026", "--iterations", "400", "--depth", "6", "--learning-rate", "0.05", "--l2-leaf-reg", "10"]
    features = a["value"]["feature_columns"]
    cats = a["value"]["categorical_columns"]
    schema = ["--feature-columns", ",".join(features), "--categorical-columns", ",".join(cats)]
    add("selection-pairs", "generate_bvival_field_selection_pairs", ["--train", split + "train.csv",
        "--validation", split + "validation.csv", "--output-dir", dirs["selection-pairs"], *base], ["cohort"])
    def targets(name, pairdir, phase, deps):
        add(name, "build_bvival_action_dataset", ["--prediction-pairs", pairdir + f"/{phase}_prediction_pairs.csv",
            "--labels", pairdir + f"/{phase}_labels.csv", "--pre-action-features", pairdir + f"/{phase}_pre_action_features.csv",
            "--feature-columns", ",".join(features[1:]), "--evidence-status", "development_only", "--output-dir", dirs[name]], deps)
    targets("selection-train-targets", dirs["selection-pairs"], "train_oof", ["selection-pairs"])
    targets("selection-validation-targets", dirs["selection-pairs"], "validation_heldout", ["selection-pairs"])
    add("selection-risk", "fit_bvival_risk_baseline", ["--train-actions", dirs["selection-train-targets"] + "/bvival_action_dataset.csv",
        "--evaluation-features", dirs["selection-pairs"] + "/validation_heldout_pre_action_features.csv",
        *schema, "--objective", "absolute_price_error", "--output-dir", dirs["selection-risk"], *head], ["selection-train-targets"])
    decision = dirs["fixed-field"] + "/selection.json"
    add("fixed-field", "select_bvival_fixed_field_development", ["--development-actions", dirs["selection-validation-targets"] + "/bvival_action_dataset.csv",
        "--development-risk-scores", dirs["selection-risk"] + "/evaluation_risk_scores.csv", "--budget-fraction", "0.1", "--output", decision],
        ["selection-validation-targets", "selection-risk"])
    add("final-pairs", "generate_bvival_prediction_pairs", ["--train", split + "train.csv", "--validation", split + "validation.csv",
        "--calibration", split + "calibration.csv", "--evaluation-features", split + "test_features.csv", "--output-dir", dirs["final-pairs"], *base],
        ["cohort", "fixed-field"])
    targets("development-targets", dirs["final-pairs"], "development", ["final-pairs"])
    targets("calibration-targets", dirs["final-pairs"], "calibration", ["final-pairs"])
    evaluation_features = dirs["final-pairs"] + "/evaluation_pre_action_features.csv"
    add("value", "fit_bvival_value_policy", ["--train-actions", dirs["development-targets"] + "/bvival_action_dataset.csv",
        "--calibration-actions", dirs["calibration-targets"] + "/bvival_action_dataset.csv", "--evaluation-features", evaluation_features,
        *schema, "--output-dir", dirs["value"], *head, "--alpha", "0.1", "--minimum-action-group-size", "100"],
        ["development-targets", "calibration-targets"])
    add("risk-mae", "fit_bvival_risk_baseline", ["--train-actions", dirs["development-targets"] + "/bvival_action_dataset.csv",
        "--evaluation-features", evaluation_features, *schema, "--objective", "absolute_price_error", "--output-dir", dirs["risk-mae"], *head], ["development-targets"])
    scores = dirs["scores"] + "/frozen_policy_scores_absolute_price_error.csv"
    add("scores", "assemble_bvival_policy_scores", ["--development-actions", dirs["development-targets"] + "/bvival_action_dataset.csv",
        "--value-scores", dirs["value"] + "/evaluation_action_scores.csv", "--risk-scores", dirs["risk-mae"] + "/evaluation_risk_scores.csv",
        "--evaluation-features", evaluation_features, "--fixed-action-file", decision, "--objective", "absolute_price_error",
        "--random-seed", "2026", "--output-dir", dirs["scores"]], ["fixed-field", "value", "risk-mae"])
    join = prefix + "/evaluation-inputs"
    add("join", "join_bvival_evaluation_outcomes", ["--evaluation-prediction-pairs", dirs["final-pairs"] + "/evaluation_prediction_pairs.csv",
        "--sealed-labels", dirs["cohort"] + "/sealed_labels/test_labels.csv", "--frozen-policy-scores", scores,
        "--expected-policy-scores-sha256", {"new_scoring_freeze_sha256_for": scores}, "--output-dir", join, "--freeze-id", Path(prefix).name],
        ["scores", "new_score_freeze_checkpoint"], "evaluation")
    add("evaluate", "evaluate_bvival_policies", ["--evaluation-outcomes", join + "/evaluation_outcomes.csv", "--frozen-policy-scores", scores,
        "--objective", "absolute_price_error", "--score-columns", ",".join(a["evaluation"]["score_columns"]),
        "--reference-policy", "score_uncertainty_only", "--budgets", "0.1", "--bootstrap-repetitions", "10000", "--bootstrap-seed", "2026",
        "--output-dir", prefix + "/evaluation-mae", "--freeze-id", Path(prefix).name, "--analysis-status", "post_test_exploratory"], ["join"], "evaluation")
    return stages


def validate_structure(stages):
    if [s["id"] for s in stages] != SCORING_IDS + ["join", "evaluate"]:
        raise ValueError("Exact two-branch fourteen-stage graph required")
    by_id = {s["id"]: s for s in stages}
    seen = set()
    for s in stages:
        if not set(s["depends_on"]).issubset(seen | {"new_score_freeze_checkpoint"}):
            raise ValueError("Stage has a forward or missing dependency")
        seen.add(s["id"])
    if by_id["selection-risk"]["depends_on"] != ["selection-train-targets"]:
        raise ValueError("Selection risk must be fitted on train-only targets")
    args = by_id["selection-pairs"]["argv_template"]
    if "--calibration" in args or "--evaluation-features" in args:
        raise ValueError("Selection branch cannot receive held-out evaluation inputs")
    args = by_id["scores"]["argv_template"]
    if "--fixed-action-file" not in args or "--fixed-action" in args:
        raise ValueError("Final assembly must consume generated fixed-field decision")
    args = by_id["evaluate"]["argv_template"]
    if args[args.index("--budgets") + 1] != "0.1" or "--include-oracle-diagnostic" in args:
        raise ValueError("Original single-budget/no-oracle contract required")
    if not isinstance(by_id["join"]["argv_template"][by_id["join"]["argv_template"].index("--expected-policy-scores-sha256") + 1], dict):
        raise ValueError("Join must wait for a new generated score freeze")


def build_plan(root, name=DEFAULT_NAME, allow_existing=False):
    root = Path(root).resolve(strict=True)
    prefix = new_prefix(root, name, allow_existing)
    raw = safe_path(root, RECEIPT).read_bytes()
    if hashlib.sha256(raw).hexdigest() != RECEIPT_HASH:
        raise ValueError("Bound source-specific recovery receipt changed")
    r = json.loads(raw)
    a = {}
    for role, relative in AUDITS.items():
        data = safe_path(root, relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != r["audit_file_sha256"][relative]:
            raise ValueError("Recovered audit drift")
        a[role] = json.loads(data)
    recorded_graph_checks({k: a[k] for k in AUTO})
    config_bytes = safe_path(root, CONFIG).read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != r["config_sha256"]:
        raise ValueError("Recovered configuration drift")
    selection_checks(a, json.loads(config_bytes), r["audit_file_sha256"])
    models = r["saved_model_parameter_inspection"]["records"]
    if len(models) != 18 or any(m["tree_count"] != 400 or m["serialized_thread_count"] != 10 for m in models):
        raise ValueError("All eighteen fixed-model parameter records required")
    headers = r["target_header_inspection"]
    if not headers["all_four_headers_equal"] or len(headers["records"]) != 4 or len(headers["records"][0]["header"]) != 23:
        raise ValueError("Four equal recovered target schemas required")
    stages = make_stages(root, prefix, r, a)
    validate_structure(stages)
    closure = static_import_closure(root, [s["argv_template"][1] for s in stages])
    if closure != r["current_static_import_closure"]:
        raise ValueError("Current study code changed after recovery")
    if a["evaluation"]["budgets"] != [.1] or a["evaluation"]["bootstrap_repetitions"] != 10000 or a["evaluation"]["bootstrap_seed"] != 2026 or a["evaluation"]["oracle_diagnostic_included"] is not False:
        raise ValueError("Historical evaluation differs from proposed primary-only scope")
    return {"date": "2026-10-01", "stage": "DRAFT_AUTOSCOUT24_TWO_BRANCH_PRIMARY_PLAN_NOT_EXECUTED",
        "replay_name": name, "planned_output_root": prefix, "recovery_receipt_sha256": RECEIPT_HASH,
        "static_study_source_closure": closure, "stages": stages,
        "source_binding": {"path": r["source_path"], "sha256": r["recorded_source_sha256_not_raw_bytes_checked"]},
        "structural_contract": {"split_counts": a["cohort"]["split_counts"], "six_cohort_data_hashes": a["cohort"]["output_hashes"],
            "cohort_audit_metadata_byte_identity_required": False, "allowed_cohort_metadata_differences": ["source_rights", "source_path"],
            "target_header": headers["records"][0]["header"],
            "four_generated_target_hashes": {x["role"]: x["sha256"] for x in headers["records"]},
            "validation_fixed_field": r["validation_fixed_field"], "validation_listing_count": 14160, "validation_capacity": 1416,
            "fixed_field_decision_sha256": r["audit_file_sha256"][AUDITS["decision"]],
            "test_listing_count": 14005, "test_action_pair_count": 55892,
            "historical_primary_score_sha256": a["assembly"]["output"]["sha256"]},
        "required_checkpoints": {"before_any_fit": ["raw opaque source SHA256", "separate validated main runtime", "source closure and plan bindings", "rebuilt six cohort hashes/counts"],
            "before_selection_risk": ["new train-only target header/hash/counts"],
            "before_fixed_field": ["new held-out validation targets/hash/counts", "new train-fitted risk scores"],
            "before_final_pairs": ["new fixed decision matches historical field/capacity/bytes"],
            "before_final_heads": ["new development/calibration target headers/hashes/counts"],
            "before_join": ["new immutable generated-score/input/model/code/runtime hash manifest"],
            "on_failure": "retain and stop; no cached historical target substitution, retuning or tolerance widening"},
        "outcome_concordance": {"reference_directory": "experiments/outputs/autoscout24_bvival_v2_full_mae_test",
            "reference_aggregate_hashes": a["evaluation"]["outputs"], "rule": "exact aggregate keys and bytes; disclose any mismatch without replacing history"},
        "historical_evaluator_source_differs": True, "exact_historical_source_identity_claimed": False,
        "historical_tests_already_opened": True, "new_confirmatory_evidence": False,
        "new_output_created": False, "execution_started": False, "runtime_bound": False,
        "orchestrator_implemented": False, "rows_or_models_read_by_planner": False,
        "rights_adjudicated": False, "public_release_created": False, "submission_ready": False}


def validate_plan(root, plan):
    expected = build_plan(root, plan["replay_name"], allow_existing=True)
    if plan != expected:
        raise ValueError("Plan differs from canonical bound reconstruction contract")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    plan = build_plan(args.root, args.name)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"stage": plan["stage"], "stages": len(plan["stages"]), "source_files": len(plan["static_study_source_closure"]["local_source_sha256"]),
                          "plan_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
