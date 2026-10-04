"""Replay only AutoScout24's frozen primary outcome contract; never refit.

Explicit execution opt-in, exclusive output tree, zero-tolerance table and
byte comparisons. All historical tests were already opened. No rights grant.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_autoscout24_reconstruction import AUDITS
from check_mucars_reconstruction import safe_path
from plan_autoscout24_replay import new_prefix, validate_plan, RUNTIME
from run_autoscout24_scoring_replay import (
    EXPECTED_PLAN, SCORING_IDS, HELPERS as SCORING_HELPERS,
    confined, source_checks, verify_bindings,
)
from run_mucars_scoring_replay import sha256, child_environment, write_new_json, LAUNCHER
from run_mucars_outcome_replay import TABLES, check_runtime, read_table, compare_table as shared_compare

SCORING_ROOT = "experiments/replays/autoscout24-reconstructed-primary-v1"
EXPECTED_FREEZE = "abd7ce52944f4855e70da690aec455aab072d3e39ea166ff2ae44bb11408da30"
HELPERS = ["run_autoscout24_outcome_replay.py", "run_mucars_outcome_replay.py", *SCORING_HELPERS]
ZERO_RULES = {"price_mae_absolute_tolerance_MAD": 0,
              "rmsle_absolute_tolerance": 0,
              "relative_gain_and_ci_endpoint_tolerance_percentage_points": 0,
              "fraction_metrics_absolute_tolerance": 0}


def verify_scoring(root, scoring_relative, expected_freeze):
    if scoring_relative != SCORING_ROOT:
        raise ValueError("Only the bound AutoScout24 scoring root is permitted")
    path = safe_path(root, scoring_relative + "/scoring-freeze.json")
    if expected_freeze != EXPECTED_FREEZE or sha256(path) != EXPECTED_FREEZE:
        raise ValueError("Pinned scoring freeze hash mismatch")
    f = json.loads(path.read_text())
    if f["stage"] != "RECONSTRUCTED_AUTOSCOUT24_SCORES_FROZEN_NO_OUTCOME_EVALUATION":
        raise ValueError("Completed AutoScout24 scoring checkpoint required")
    for name in ("outcome_join_called", "test_performance_computed", "full_primary_table_concordance_verified",
                 "independent_replication_claimed", "exact_historical_source_identity_claimed",
                 "rights_adjudicated", "public_release_created", "submission_ready"):
        if f.get(name) is not False:
            raise ValueError("Boundary flags must be genuine false booleans")
    if f.get("historical_tests_already_opened") is not True or f.get("score_bytes_match_recorded_historical_hash") is not True:
        raise ValueError("Opened-test and scoring-concordance gates required")
    if [s["stage"] for s in f["completed_scoring_stages"]] != SCORING_IDS:
        raise ValueError("All twelve scoring stages required")
    scoring = path.parent
    for relative, expected in f["artifact_sha256"].items():
        if sha256(safe_path(scoring, relative)) != expected:
            raise ValueError("Scoring artifact drift: " + relative)
    plan_path = safe_path(scoring, "bound-plan.json")
    if sha256(plan_path) != EXPECTED_PLAN or f["plan_sha256"] != EXPECTED_PLAN:
        raise ValueError("Pinned AutoScout24 plan hash required")
    plan = json.loads(plan_path.read_text())
    validate_plan(root, plan)
    if plan["planned_output_root"] != scoring_relative or plan["historical_tests_already_opened"] is not True:
        raise ValueError("Scoring root/opened-test contract differs")
    sources = source_checks(root, plan)
    if sources != f["study_and_recovery_source_sha256"]:
        raise ValueError("Recovered source/reference bindings differ")
    verify_bindings(root, scoring / "snapshot-workspace", {**sources, **f["runner_and_helper_sha256"]})
    if f["original_source_sha256_checked"] != plan["source_binding"]["sha256"]:
        raise ValueError("Original source verification record differs")
    rules = plan["outcome_concordance"]
    if set(rules["reference_aggregate_hashes"]) != set(TABLES):
        raise ValueError("Exactly four primary aggregate references required")
    for name, expected in rules["reference_aggregate_hashes"].items():
        if sha256(safe_path(root, rules["reference_directory"] + "/" + name)) != expected:
            raise ValueError("Historical reference drift")
    if (sha256(safe_path(scoring, "scores/frozen_policy_scores_absolute_price_error.csv")) != f["new_score_sha256"]
            or f["new_score_sha256"] != plan["structural_contract"]["historical_primary_score_sha256"]):
        raise ValueError("New/historical score hash differs")
    return scoring, f, plan


def outcome_command(root, plan, stage, python, snapshot, output, score_hash):
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Symlink outcome component rejected")
    if stage["id"] not in {"join", "evaluate"} or stage["phase"] != "evaluation":
        raise ValueError("Only frozen outcome stages permitted")
    args = stage["argv_template"]
    script = "experiments/" + ("join_bvival_evaluation_outcomes.py" if stage["id"] == "join" else "evaluate_bvival_policies.py")
    if args[0] != RUNTIME or args[1] != script or script not in plan["static_study_source_closure"]["local_source_sha256"]:
        raise ValueError("Unbound outcome source/runtime")
    if any(v in args for v in ("--additional-frozen-policy-scores", "--additional-policy-scores", "--include-oracle-diagnostic")):
        raise ValueError("Secondary scores/oracle excluded by original contract")
    prefix = plan["planned_output_root"]
    bound = []
    for value in args[2:]:
        if isinstance(value, dict):
            if stage["id"] != "join" or value != {"new_scoring_freeze_sha256_for": prefix + "/scores/frozen_policy_scores_absolute_price_error.csv"}:
                raise ValueError("Unknown deferred outcome hash")
            value = score_hash
        elif not isinstance(value, str):
            raise ValueError("Invalid argv value")
        elif value == prefix + "/evaluation-inputs" or value.startswith(prefix + "/evaluation-inputs/"):
            value = str(confined(output, value[len(prefix) + 1:]))
        elif value == prefix + "/evaluation-mae":
            value = str(confined(output, "evaluation-mae"))
        elif value.startswith(prefix + "/"):
            value = str(safe_path(root, value))
        elif value.startswith("experiments/"):
            raise ValueError("Historical data substitution forbidden")
        bound.append(value)
    return [str(python), "-I", "-B", "-c", LAUNCHER, str(safe_path(snapshot, script)), *bound]


def compare_table(name, reference, replay):
    # The shared comparator's legacy MAD key is an internal spelling only:
    # tolerance is exactly zero, with no conversion of the EUR values.
    if name != "bvival_policy_summary.csv":
        result = shared_compare(name, reference, replay, dict(ZERO_RULES))
    else:
        # A curve AUC is undefined for the original ONE budget. Preserve the
        # seven genuine blank AUCs, not zero-impute them or borrow other budgets.
        old, new = read_table(reference, TABLES[name]), read_table(replay, TABLES[name])
        missing, extra = set(old) - set(new), set(new) - set(old)
        mismatches = [{"kind": "missing_key", "key": list(k)} for k in sorted(missing)]
        mismatches += [{"kind": "extra_key", "key": list(k)} for k in sorted(extra)]
        cells, blanks, maximum = 0, 0, 0
        for key in sorted(set(old) & set(new)):
            a, b = old[key], new[key]
            if a["error_budget_auc"] != "" or b["error_budget_auc"] != "":
                raise ValueError("Single-budget AUC must remain not applicable, never zero-imputed")
            cells += 2
            blanks += 1
            if a["auc_metric"] != b["auc_metric"] or a["auc_metric"] != "post_mae_price":
                mismatches.append({"kind": "text", "key": list(key), "column": "auc_metric"})
            x, y = a["mean_harmful_action_rate"], b["mean_harmful_action_rate"]
            if x == y == "" and key == ("score_no_acquisition",):
                blanks += 1
            elif "" in (x, y):
                raise ValueError("Unexpected missing summary metric")
            else:
                x, y = float(x), float(y)
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("Nonfinite summary metric")
                maximum = max(maximum, abs(x - y))
                if x != y:
                    mismatches.append({"kind": "numeric", "key": list(key), "column": "mean_harmful_action_rate",
                                       "reference": x, "replay": y, "tolerance": 0})
        result = {"file": name, "reference_row_count": len(old), "replay_row_count": len(new),
                  "key_sets_match": not missing and not extra, "numeric_cells_checked": cells,
                  "matched_not_applicable_cells": blanks,
                  "max_absolute_delta_by_column": {"mean_harmful_action_rate": maximum},
                  "mismatches": mismatches, "changed_interval_boundaries": [],
                  "passes_fixed_rules": not mismatches, "byte_hash_matches": sha256(reference) == sha256(replay)}
    result["price_unit"] = "EUR"
    result["passes_exact_rules"] = result["passes_fixed_rules"] and result["byte_hash_matches"]
    return result


def joined_outputs(root, output, freeze, plan):
    join = json.loads(safe_path(output, "evaluation-inputs/evaluation_outcome_join_audit.json").read_text())
    old = json.loads(safe_path(root, AUDITS["join"]).read_text())
    if {k: v for k, v in join.items() if k != "freeze_id"} != {k: v for k, v in old.items() if k != "freeze_id"}:
        raise ValueError("Joined outcome audit differs beyond new freeze identifier")
    c, inputs = plan["structural_contract"], join["inputs"]
    if (sha256(safe_path(output, "evaluation-inputs/evaluation_outcomes.csv")) != join["output_sha256"]
            or join["listing_count"] != c["test_listing_count"] or join["listing_action_count"] != c["test_action_pair_count"]
            or join["labels_opened"] is not True or join["policy_score_hash_verified_before_label_read"] is not True
            or inputs["frozen_policy_scores_sha256"] != freeze["new_score_sha256"]
            or inputs["additional_frozen_policy_scores_sha256"] is not None
            or inputs["prediction_pairs_sha256"] != freeze["artifact_sha256"]["final-pairs/evaluation_prediction_pairs.csv"]
            or inputs["sealed_labels_sha256"] != freeze["artifact_sha256"]["cohort/sealed_labels/test_labels.csv"]):
        raise ValueError("Joined outcome hash/universe/input gate differs")
    return join


def evaluation_checks(root, output, freeze, join):
    audit = json.loads(safe_path(output, "evaluation-mae/bvival_evaluation_audit.json").read_text())
    old = json.loads(safe_path(root, AUDITS["evaluation"]).read_text())
    for key in ("objective", "auc_metric", "reference_policy", "score_columns", "oracle_diagnostic_included",
                "oracle_diagnostic_name", "budgets", "bootstrap_repetitions", "bootstrap_seed", "nonclaim"):
        if audit[key] != old[key]:
            raise ValueError("Evaluation setting changed: " + key)
    if (audit["analysis_status"] != "post_test_exploratory" or audit["budgets"] != [.1]
            or audit["bootstrap_repetitions"] != 10000 or audit["bootstrap_seed"] != 2026
            or audit["additional_policy_scores_sha256"] is not None
            or audit["oracle_diagnostic_included"] is not False
            or audit["evaluation_outcomes_sha256"] != join["output_sha256"]
            or audit["frozen_policy_scores_sha256"] != freeze["new_score_sha256"]
            or set(audit["outputs"]) != set(TABLES)):
        raise ValueError("Original single-budget outcome contract required")
    for name, expected in audit["outputs"].items():
        if sha256(safe_path(output, "evaluation-mae/" + name)) != expected:
            raise ValueError("Generated aggregate audit/hash differs")
    return audit


def execute(root, scoring_relative, expected_freeze, scoring, freeze, plan, python, runtime, output_relative):
    if (output_relative != new_prefix(root, Path(output_relative).name)
            or not Path(output_relative).name.startswith("autoscout24-reconstructed-primary-outcomes-")):
        raise ValueError("New scoped outcome-only root required")
    verify_scoring(root, scoring_relative, expected_freeze)
    helpers = {"release-bvival/" + n: sha256(safe_path(root, "release-bvival/" + n)) for n in HELPERS}
    output = confined(root, output_relative)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    snapshot, logs = output / "snapshot-workspace", output / "logs"
    logs.mkdir()
    phase, completed = "snapshot", []
    try:
        sources = plan["static_study_source_closure"]["local_source_sha256"]
        for relative, expected in {**sources, **helpers}.items():
            origin = safe_path(scoring / "snapshot-workspace", relative) if relative in sources else safe_path(root, relative)
            path = confined(snapshot, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, path)
            if sha256(path) != expected:
                raise ValueError("Outcome snapshot drift")
        write_new_json(output / "runtime-validation.json", runtime)
        write_new_json(output / "execution-binding.json", {
            "scoring_freeze_sha256": expected_freeze, "plan_sha256": EXPECTED_PLAN,
            "new_score_sha256": freeze["new_score_sha256"], "runner_and_helper_sha256": helpers,
            "outcome_concordance": plan["outcome_concordance"], "numeric_tolerances": ZERO_RULES,
            "price_unit": "EUR", "byte_identity_required": True,
            "historical_tests_already_opened": True, "new_preregistration": False})
        for stage in plan["stages"][-2:]:
            phase = stage["id"]
            verify_scoring(root, scoring_relative, expected_freeze)
            verify_bindings(root, snapshot, {**sources, **helpers})
            if sha256(python) != freeze["runtime_validation"]["python_executable_sha256"]:
                raise ValueError("Runtime launcher changed")
            print(json.dumps({"stage": phase, "status": "started"}), flush=True)
            started = time.monotonic()
            with (logs / (phase + ".txt")).open("x") as log:
                subprocess.run(outcome_command(root, plan, stage, python, snapshot, output, freeze["new_score_sha256"]),
                               cwd=snapshot, env=child_environment(python), stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=1800)
            if phase == "join":
                joined_outputs(root, output, freeze, plan)
            completed.append({"stage": phase, "elapsed_seconds": round(time.monotonic() - started, 3)})
            print(json.dumps({**completed[-1], "status": "completed"}), flush=True)
        phase = "concordance"
        verify_scoring(root, scoring_relative, expected_freeze)
        verify_bindings(root, snapshot, {**sources, **helpers})
        check_runtime(scoring, freeze, python)  # Recheck; this phase never fits.
        join = joined_outputs(root, output, freeze, plan)
        evaluation_checks(root, output, freeze, join)
        rules = plan["outcome_concordance"]
        comparisons = [compare_table(name, safe_path(root, rules["reference_directory"] + "/" + name),
                                     safe_path(output, "evaluation-mae/" + name)) for name in TABLES]
        passed = all(c["passes_exact_rules"] for c in comparisons)
        write_new_json(output / "aggregate-concordance.json", {
            "tables": comparisons, "all_tables_pass_exact_rules": passed, "zero_numeric_tolerances": True,
            "byte_identity_required": True, "price_unit": "EUR", "independent_scientific_replication": False})
        if not passed:
            raise ValueError("Exact primary aggregate concordance failed; retained without tolerance widening")
        artifacts = {}
        for p in sorted(output.rglob("*")):
            if p.is_symlink():
                raise ValueError("Generated symlink rejected")
            if p.is_file():
                artifacts[str(p.relative_to(output))] = sha256(p)
        receipt = {"date": "2026-10-01", "stage": "RECONSTRUCTED_AUTOSCOUT24_PRIMARY_OUTCOMES_EXACT_MATCH",
                   "scoring_freeze_sha256": expected_freeze, "plan_sha256": EXPECTED_PLAN,
                   "completed_stages": completed, "artifact_sha256": artifacts,
                   "joined_outcomes_sha256": join["output_sha256"], "listing_count": join["listing_count"],
                   "listing_action_count": join["listing_action_count"], "all_primary_tables_pass_exact_rules": True,
                   "all_primary_table_byte_hashes_match": True,
                   "finite_numeric_cells_matched": sum(c["numeric_cells_checked"] - c["matched_not_applicable_cells"] for c in comparisons),
                   "not_applicable_cells_matched": sum(c["matched_not_applicable_cells"] for c in comparisons),
                   "outcome_join_called": True, "test_performance_computed": True,
                   "historical_tests_already_opened": True, "analysis_status": "post_test_exploratory",
                   "bootstrap_unit": "listing", "bootstrap_repetitions": 10000, "bootstrap_seed": 2026,
                   "budgets": [.1], "new_training_or_tuning_called": False,
                   "exact_historical_source_identity_claimed": False, "independent_replication_claimed": False,
                   "original_results_replaced": False, "rights_adjudicated": False,
                   "public_release_created": False, "submission_ready": False}
        write_new_json(output / "outcome-replay-receipt.json", receipt)
        return receipt
    except Exception as error:
        write_new_json(output / "failed-outcome-replay.json", {
            "phase": phase, "exception_class": type(error).__name__, "reason": str(error),
            "completed_stages": completed, "partial_directory_retained": True,
            "outcome_label_read_may_have_occurred": phase != "snapshot", "no_automatic_retry": True})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--expected-scoring-freeze-sha256", required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--output-name", default="autoscout24-reconstructed-primary-outcomes-v1")
    p.add_argument("--execute-outcomes", action="store_true")
    args = p.parse_args()
    try:
        root, python = args.project_root.resolve(strict=True), args.python.absolute()
        output = new_prefix(root, args.output_name)
        scoring, freeze, plan = verify_scoring(root, SCORING_ROOT, args.expected_scoring_freeze_sha256)
        runtime = check_runtime(scoring, freeze, python)
        if args.execute_outcomes:
            receipt = execute(root, SCORING_ROOT, args.expected_scoring_freeze_sha256, scoring, freeze, plan, python, runtime, output)
            print(json.dumps({"stage": receipt["stage"], "output_root": output,
                              "finite_numeric_cells_matched": receipt["finite_numeric_cells_matched"]}), flush=True)
        else:
            print(json.dumps({"stage": "AUTOSCOUT24_OUTCOME_PREFLIGHT_ONLY",
                              "scoring_artifact_hashes_verified": len(freeze["artifact_sha256"]),
                              "runtime_verified": True, "outcome_join_called": False,
                              "test_performance_computed": False, "output_created": False}))
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        p.exit(1, f"AutoScout24 outcome replay failed: {error}\n")


if __name__ == "__main__":
    main()
