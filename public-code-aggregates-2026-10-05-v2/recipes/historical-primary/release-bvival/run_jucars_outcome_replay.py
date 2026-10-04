"""Reconstruct only the bound JUCars primary outcome tables; never refit.

Defaults verify scoring/runtime without joining labels or evaluating. Historical
tests are already opened. The primary-only join omits the old secondary guard.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mucars_reconstruction import safe_path
from check_jucars_reconstruction import AUDITS
from plan_jucars_replay import new_prefix, static_import_closure, ADAPTER, RECEIPT, RUNTIME
from run_jucars_scoring_replay import EXPECTED_PLAN, SCORING_IDS, confined
from run_mucars_scoring_replay import sha256, child_environment, write_new_json, LAUNCHER
from run_mucars_outcome_replay import TABLES, check_runtime, compare_table as shared_compare

SCORING_ROOT = "experiments/replays/jucars-reconstructed-primary-v1"
HELPERS = ["run_jucars_outcome_replay.py", "run_mucars_outcome_replay.py", "run_mucars_scoring_replay.py",
           "run_jucars_scoring_replay.py", "check_jucars_reconstruction.py", "check_mucars_reconstruction.py",
           "plan_jucars_replay.py", "plan_mucars_replay.py", "check_rerun_entrypoints.py"]


def verify_scoring(root, scoring_relative, expected_freeze):
    if scoring_relative != SCORING_ROOT:
        raise ValueError("Only the bound JUCars scoring root is permitted")
    path = safe_path(root, scoring_relative + "/scoring-freeze.json")
    if sha256(path) != expected_freeze:
        raise ValueError("Scoring freeze hash mismatch")
    f = json.loads(path.read_text())
    if f["stage"] != "RECONSTRUCTED_JUCARS_SCORES_FROZEN_NO_OUTCOME_EVALUATION":
        raise ValueError("Completed JUCars scoring-only checkpoint required")
    for name in ("outcome_join_called", "test_performance_computed", "independent_replication_claimed",
                 "submission_ready", "historical_source_identity_claimed", "secondary_score_guard_replayed"):
        if f.get(name) is not False:
            raise ValueError("Boundary flags must be genuine false booleans")
    if f.get("historical_test_labels_already_opened") is not True or f.get("original_source_and_dictionary_hashes_checked") is not True:
        raise ValueError("Opened-test/source gate status required")
    if [s["stage"] for s in f["completed_scoring_stages"]] != SCORING_IDS:
        raise ValueError("All nine scoring stages required")
    scoring = path.parent
    for relative, expected in f["artifact_sha256"].items():
        if sha256(safe_path(scoring, relative)) != expected:
            raise ValueError("Scoring artifact drift: " + relative)
    plan_path = safe_path(scoring, "bound-plan.json")
    if sha256(plan_path) != EXPECTED_PLAN or f["plan_sha256"] != EXPECTED_PLAN:
        raise ValueError("Fixed JUCars plan hash required")
    p = json.loads(plan_path.read_text())
    if p["planned_output_root"] != scoring_relative or p["historical_tests_already_opened"] is not True:
        raise ValueError("Root/opened-test contract differs")
    if [s["id"] for s in p["stages"][-2:]] != ["join", "evaluate"]:
        raise ValueError("Bound join/evaluate templates required")
    snapshot = scoring / "snapshot-workspace"
    entries = [s["argv_template"][1] for s in p["stages"] if s["argv_template"][1] != ADAPTER]
    if static_import_closure(snapshot, entries) != p["static_study_source_closure"]:
        raise ValueError("Snapshot study closure differs")
    bindings = dict(p["static_study_source_closure"]["local_source_sha256"])
    bindings[ADAPTER] = p["schema_adapter_source"]["sha256"]
    bindings[RECEIPT] = p["recovery_receipt_sha256"]
    if bindings != f["study_and_recovery_source_sha256"]:
        raise ValueError("Frozen source/adapter/recovery bindings differ")
    for relative, expected in {**bindings, **f["runner_and_helper_sha256"]}.items():
        if sha256(safe_path(snapshot, relative)) != expected or sha256(safe_path(root, relative)) != expected:
            raise ValueError("Source/helper/recovery drift")
    recovery = json.loads(safe_path(snapshot, RECEIPT).read_text())
    if set(recovery["audit_file_sha256"]) != set(AUDITS.values()):
        raise ValueError("Ten historical audit bindings required")
    for relative, expected in recovery["audit_file_sha256"].items():
        if sha256(safe_path(root, relative)) != expected:
            raise ValueError("Historical audit drift")
    rules = p["engineering_concordance_rules"]
    if set(rules["reference_aggregate_hashes"]) != set(TABLES):
        raise ValueError("Four primary aggregate references required")
    for name, expected in rules["reference_aggregate_hashes"].items():
        if sha256(safe_path(root, rules["original_aggregate_reference"] + "/" + name)) != expected:
            raise ValueError("Historical reference drift")
    if sha256(safe_path(scoring, "scores/frozen_policy_scores_absolute_price_error.csv")) != f["new_score_sha256"]:
        raise ValueError("New score hash differs")
    return scoring, f, p


def outcome_command(root, plan, stage, python, snapshot, output, score_hash):
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Symlink outcome root/component rejected")
    if stage["id"] not in {"join", "evaluate"} or stage["phase"] != "evaluation":
        raise ValueError("Only frozen outcome stages permitted")
    args = stage["argv_template"]
    script = "experiments/" + ("join_bvival_evaluation_outcomes.py" if stage["id"] == "join" else "evaluate_bvival_policies.py")
    if args[0] != RUNTIME or args[1] != script or script not in plan["static_study_source_closure"]["local_source_sha256"]:
        raise ValueError("Unbound outcome source/runtime")
    if "--additional-frozen-policy-scores" in args:
        raise ValueError("Historical secondary-score guard is excluded from this primary-only join")
    prefix = plan["planned_output_root"]
    bound = []
    for value in args[2:]:
        if isinstance(value, dict):
            if stage["id"] != "join" or value != {"frozen_artifact_sha256": prefix + "/scores/frozen_policy_scores_absolute_price_error.csv"}:
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
        bound.append(value)
    return [str(python), "-I", "-B", "-c", LAUNCHER, str(safe_path(snapshot, script)), *bound]


def compare_table(name, reference, replay, rules):
    # Reuse the tested numeric comparator without altering its frozen MUCars
    # source. Its internal price-key spelling is legacy; values stay in JOD.
    internal = dict(rules)
    internal["price_mae_absolute_tolerance_MAD"] = rules["price_mae_absolute_tolerance_JOD"]
    result = shared_compare(name, reference, replay, internal)
    result["price_tolerance_unit"] = "JOD"
    return result


def joined_outputs(output, freeze, plan):
    join = json.loads(safe_path(output, "evaluation-inputs/evaluation_outcome_join_audit.json").read_text())
    counts = plan["expected_structural_contract"]["phase_counts"]["evaluation"]
    inputs = join["inputs"]
    if (sha256(safe_path(output, "evaluation-inputs/evaluation_outcomes.csv")) != join["output_sha256"]
            or join["listing_count"] != counts["listings"] or join["listing_action_count"] != counts["pairs"]
            or join["labels_opened"] is not True or join["policy_score_hash_verified_before_label_read"] is not True
            or inputs["frozen_policy_scores_sha256"] != freeze["new_score_sha256"]
            or inputs["additional_frozen_policy_scores_sha256"] is not None
            or inputs["prediction_pairs_sha256"] != freeze["artifact_sha256"]["pairs/evaluation_prediction_pairs.csv"]
            or inputs["sealed_labels_sha256"] != freeze["artifact_sha256"]["cohort/sealed_labels/test_labels.csv"]):
        raise ValueError("Joined outcome hash/universe/input gate differs")
    return join


def snapshot_checks(root, snapshot, sources, helpers):
    for relative, expected in {**sources, **helpers}.items():
        if sha256(safe_path(snapshot, relative)) != expected:
            raise ValueError("Outcome snapshot drift")
        if relative in helpers and sha256(safe_path(root, relative)) != expected:
            raise ValueError("Outcome helper drift")


def execute(root, scoring_relative, expected_freeze, scoring, freeze, plan, python, runtime, output_relative):
    output = confined(root, output_relative)
    if output_relative != new_prefix(root, Path(output_relative).name) or not Path(output_relative).name.startswith("jucars-reconstructed-primary-outcomes-"):
        raise ValueError("New scoped outcome-only root required")
    verify_scoring(root, scoring_relative, expected_freeze)
    helper_hashes = {"release-bvival/" + n: sha256(safe_path(root, "release-bvival/" + n)) for n in HELPERS}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    snapshot, logs = output / "snapshot-workspace", output / "logs"
    logs.mkdir()
    phase, completed = "snapshot", []
    try:
        sources = plan["static_study_source_closure"]["local_source_sha256"]
        for relative, expected in {**sources, **helper_hashes}.items():
            origin = safe_path(scoring / "snapshot-workspace", relative) if relative in sources else safe_path(root, relative)
            target = confined(snapshot, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
            if sha256(target) != expected:
                raise ValueError("Outcome snapshot drift")
        write_new_json(output / "runtime-validation.json", runtime)
        write_new_json(output / "execution-binding.json", {"scoring_freeze_sha256": expected_freeze,
                       "plan_sha256": freeze["plan_sha256"], "score_sha256": freeze["new_score_sha256"],
                       "runner_and_helper_sha256": helper_hashes, "engineering_rules": plan["engineering_concordance_rules"],
                       "historical_tests_already_opened": True, "secondary_guard_replayed": False, "new_preregistration": False})
        for stage in plan["stages"][-2:]:
            phase = stage["id"]
            verify_scoring(root, scoring_relative, expected_freeze)
            snapshot_checks(root, snapshot, sources, helper_hashes)
            command = outcome_command(root, plan, stage, python, snapshot, output, freeze["new_score_sha256"])
            print(json.dumps({"stage": phase, "status": "started"}), flush=True)
            started = time.monotonic()
            with (logs / (phase + ".txt")).open("x", encoding="utf-8") as log:
                subprocess.run(command, cwd=snapshot, env=child_environment(python), stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=1800)
            completed.append({"stage": phase, "elapsed_seconds": round(time.monotonic() - started, 3)})
            if phase == "join":
                joined_outputs(output, freeze, plan)
            print(json.dumps({"stage": phase, "status": "completed"}), flush=True)
        phase = "concordance"
        verify_scoring(root, scoring_relative, expected_freeze)
        snapshot_checks(root, snapshot, sources, helper_hashes)
        if sha256(python) != freeze["runtime_validation"]["python_executable_sha256"]:
            raise ValueError("Runtime launcher changed during evaluation")
        join = joined_outputs(output, freeze, plan)
        audit = json.loads(safe_path(output, "evaluation-mae/bvival_evaluation_audit.json").read_text())
        if (audit["analysis_status"] != "post_test_exploratory" or audit["objective"] != "absolute_price_error"
                or audit["reference_policy"] != "score_uncertainty_only" or audit["bootstrap_repetitions"] != 10000
                or audit["bootstrap_seed"] != 2026 or audit["budgets"] != [.01, .05, .1, .2, .3]
                or audit["evaluation_outcomes_sha256"] != join["output_sha256"]
                or audit["frozen_policy_scores_sha256"] != freeze["new_score_sha256"]):
            raise ValueError("Evaluation settings/input contract differs")
        rules = plan["engineering_concordance_rules"]
        comparisons = [compare_table(name, safe_path(root, rules["original_aggregate_reference"] + "/" + name),
                                     safe_path(output, "evaluation-mae/" + name), rules) for name in TABLES]
        write_new_json(output / "aggregate-concordance.json", {"tables": comparisons,
                       "all_tables_pass_fixed_rules": all(r["passes_fixed_rules"] for r in comparisons),
                       "price_tolerance_unit": "JOD", "rules_unchanged": True, "independent_scientific_replication": False})
        artifacts = {}
        for p in sorted(output.rglob("*")):
            if p.is_symlink():
                raise ValueError("Generated symlink rejected")
            if p.is_file():
                artifacts[str(p.relative_to(output))] = sha256(p)
        receipt = {"date": "2026-10-01", "stage": "RECONSTRUCTED_JUCARS_PRIMARY_OUTCOME_COMPARISON_COMPLETED",
                   "scoring_freeze_sha256": expected_freeze, "plan_sha256": freeze["plan_sha256"],
                   "completed_stages": completed, "artifact_sha256": artifacts,
                   "joined_outcomes_sha256": join["output_sha256"], "listing_count": join["listing_count"],
                   "listing_action_count": join["listing_action_count"],
                   "all_primary_tables_pass_fixed_rules": all(r["passes_fixed_rules"] for r in comparisons),
                   "all_primary_table_byte_hashes_match": all(r["byte_hash_matches"] for r in comparisons),
                   "outcome_join_called": True, "test_performance_computed": True,
                   "historical_tests_already_opened": True, "analysis_status": "post_test_exploratory",
                   "secondary_score_guard_replayed": False, "exact_full_historical_join_protocol_claimed": False,
                   "new_training_or_tuning_called": False, "historical_source_identity_claimed": False,
                   "independent_replication_claimed": False, "original_results_replaced": False,
                   "public_release_created": False, "submission_ready": False}
        write_new_json(output / "outcome-replay-receipt.json", receipt)
        return receipt
    except Exception as error:
        write_new_json(output / "failed-outcome-replay.json", {"phase": phase, "exception_class": type(error).__name__,
                       "completed_stages": completed, "partial_directory_retained": True,
                       "outcome_label_read_may_have_occurred": phase != "snapshot", "no_automatic_retry": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--expected-scoring-freeze-sha256", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-name", default="jucars-reconstructed-primary-outcomes-v1")
    parser.add_argument("--execute-outcomes", action="store_true")
    args = parser.parse_args()
    try:
        root, python = args.project_root.resolve(strict=True), args.python.absolute()
        output = new_prefix(root, args.output_name)
        scoring, freeze, plan = verify_scoring(root, SCORING_ROOT, args.expected_scoring_freeze_sha256)
        runtime = check_runtime(scoring, freeze, python)
        if args.execute_outcomes:
            receipt = execute(root, SCORING_ROOT, args.expected_scoring_freeze_sha256, scoring, freeze, plan, python, runtime, output)
            print(json.dumps({"stage": receipt["stage"], "output_root": output,
                              "all_primary_tables_pass_fixed_rules": receipt["all_primary_tables_pass_fixed_rules"],
                              "all_primary_table_byte_hashes_match": receipt["all_primary_table_byte_hashes_match"]}), flush=True)
        else:
            print(json.dumps({"stage": "JUCARS_OUTCOME_PREFLIGHT_ONLY", "artifacts_verified": len(freeze["artifact_sha256"]),
                              "runtime_validated": True, "outcome_join_called": False, "test_performance_computed": False,
                              "output_created": False}))
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"JUCars outcome replay failed: {error}\n")


if __name__ == "__main__":
    main()
