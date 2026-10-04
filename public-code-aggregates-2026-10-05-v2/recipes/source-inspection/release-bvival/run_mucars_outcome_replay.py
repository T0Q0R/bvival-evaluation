"""Gated reconstructed MUCars outcome replay and fixed-tolerance comparison.

No training, tuning, new policies, or replacement of historical results.
Historical tests are already open. Defaults check inputs without evaluating.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mucars_reconstruction import safe_path, AUDITS
from plan_mucars_replay import output_prefix, static_import_closure
from run_mucars_scoring_replay import sha256, canonical, lock_pins, validate_runtime_records, child_environment, write_new_json, LAUNCHER

TABLES = {
    "bvival_error_budget_curve.csv": {
        "keys": ["policy", "budget"], "integer": ["listing_count", "action_count"],
        "price": ["before_mae_price", "post_mae_price"], "rmsle": ["before_rmsle", "post_rmsle"],
        "percent": ["relative_rmsle_improvement_percent"],
        "fraction": ["action_rate", "harmful_action_rate", "positive_gain_capture"], "text": []},
    "bvival_action_mix.csv": {
        "keys": ["policy", "budget", "action_id"], "integer": ["count"], "price": [], "rmsle": [],
        "percent": [], "fraction": ["share_of_actions"], "text": []},
    "bvival_policy_summary.csv": {
        "keys": ["policy"], "integer": [], "price": ["error_budget_auc"], "rmsle": [],
        "percent": [], "fraction": ["mean_harmful_action_rate"], "text": ["auc_metric"]},
    "bvival_reference_comparisons.csv": {
        "keys": ["policy", "reference_policy", "budget"], "integer": ["n"], "price": [], "rmsle": [],
        "percent": ["relative_mae_gain_percent", "ci_lower_percent", "ci_upper_percent"], "fraction": [], "text": []},
}
PROBE = '''
import json,sys,platform
from pathlib import Path
from importlib import metadata
import numpy,pandas,catboost,scipy
prefix=Path(sys.prefix).resolve()
inside=lambda p: Path(p).resolve().is_relative_to(prefix)
ds=list(metadata.distributions())
print(json.dumps({"python":platform.python_version(),"platform":platform.platform(),"machine":platform.machine(),
 "is_virtualenv":sys.prefix!=sys.base_prefix,"all_distribution_metadata_inside_venv":all(inside(d._path) for d in ds),
 "all_four_imports_inside_venv":all(inside(m.__file__) for m in [numpy,pandas,catboost,scipy]),
 "installed_versions":{d.metadata["Name"]:d.version for d in ds},
 "module_versions":{n:m.__version__ for n,m in [("numpy",numpy),("pandas",pandas),("catboost",catboost),("scipy",scipy)]}}))
'''


def verify_scoring(root, scoring_relative, expected_freeze):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_freeze):
        raise ValueError("Explicit scoring-freeze SHA256 required")
    path = safe_path(root, scoring_relative + "/scoring-freeze.json")
    if sha256(path) != expected_freeze:
        raise ValueError("Scoring freeze receipt hash mismatch")
    freeze = json.loads(path.read_text())
    if freeze["stage"] != "RECONSTRUCTED_MUCARS_SCORES_FROZEN_NO_OUTCOME_EVALUATION":
        raise ValueError("Completed scoring-only checkpoint required")
    for name in ("outcome_join_called", "test_performance_computed", "independent_replication_claimed", "submission_ready"):
        if freeze.get(name) is not False:
            raise ValueError("Scoring checkpoint boundary flags must be genuine false booleans")
    if freeze.get("historical_test_labels_already_opened") is not True:
        raise ValueError("Previously opened historical-test status required")
    scoring = path.parent
    for relative, expected in freeze["artifact_sha256"].items():
        if sha256(safe_path(scoring, relative)) != expected:
            raise ValueError("Frozen scoring artifact changed: " + relative)
    if sha256(safe_path(scoring, "bound-plan.json")) != freeze["plan_sha256"]:
        raise ValueError("Bound plan hash mismatch")
    plan = json.loads((scoring / "bound-plan.json").read_text())
    if plan["planned_output_root"] != scoring_relative or plan["historical_tests_previously_opened"] is not True:
        raise ValueError("Scoring root/opened-test status mismatch")
    if [s["id"] for s in plan["stages"][-2:]] != ["join", "evaluate"]:
        raise ValueError("Bound join/evaluate templates required")
    closure = plan["static_source_closure"]["local_source_sha256"]
    if closure != freeze["bound_local_source_sha256"]:
        raise ValueError("Frozen source closure mismatch")
    snapshot = scoring / "snapshot-workspace"
    actual = static_import_closure(snapshot, [s["argv_template"][1] for s in plan["stages"]])
    if actual != plan["static_source_closure"]:
        raise ValueError("Snapshot source closure changed")
    for name, expected in freeze["runner_and_helper_sha256"].items():
        if sha256(safe_path(snapshot, "release-bvival/" + name)) != expected:
            raise ValueError("Scoring runner snapshot mismatch")
    rules = plan["engineering_concordance_rules"]
    if set(rules["reference_aggregate_hashes"]) != set(TABLES):
        raise ValueError("All four fixed historical aggregate references required")
    recovery = safe_path(root, "release-bvival/mucars-reconstruction-2026-10-01.json")
    if sha256(recovery) != plan["parameter_recovery_receipt_sha256"]:
        raise ValueError("Parameter recovery receipt changed")
    for relative, expected in json.loads(recovery.read_text())["audit_file_sha256"].items():
        if relative not in AUDITS.values() or sha256(safe_path(root, relative)) != expected:
            raise ValueError("Historical audit drift")
    for name, expected in rules["reference_aggregate_hashes"].items():
        if sha256(safe_path(root, rules["original_aggregate_reference"] + "/" + name)) != expected:
            raise ValueError("Historical reference aggregate drift")
    scores = safe_path(scoring, "scores/frozen_policy_scores_absolute_price_error.csv")
    if sha256(scores) != freeze["new_score_sha256"]:
        raise ValueError("Frozen score hash mismatch")
    return scoring, freeze, plan


def check_runtime(scoring, freeze, python):
    prior = freeze["runtime_validation"]
    if sha256(python) != prior["python_executable_sha256"]:
        raise ValueError("Interpreter differs from scoring runtime")
    lock = safe_path(scoring, "runtime.lock")
    report = safe_path(scoring, "install-report.json")
    if sha256(lock) != prior["requirements_lock_sha256"] or sha256(report) != prior["install_report_sha256"]:
        raise ValueError("Runtime lock/install report mismatch")
    result = subprocess.run([str(python), "-I", "-B", "-c", PROBE], check=True, capture_output=True,
                            text=True, timeout=120, env=child_environment(python))
    fresh = json.loads(result.stdout)
    previous = prior["probe"]
    for key in ("python", "platform", "machine", "module_versions"):
        if fresh[key] != previous[key]:
            raise ValueError("Runtime differs since scoring: " + key)
    if {canonical(k): v for k, v in fresh["installed_versions"].items()} != {canonical(k): v for k, v in previous["installed_versions"].items()}:
        raise ValueError("Installed distribution set/versions changed")
    # Reuse the already recorded toy-fit result; this outcome phase fits nothing.
    validate_runtime_records(lock_pins(lock.read_text()), json.loads(report.read_text()),
                             {**fresh, "synthetic_native_fit_passed": previous["synthetic_native_fit_passed"]})
    subprocess.run([str(python), "-I", "-B", "-m", "pip", "--isolated", "check"], check=True,
                   capture_output=True, text=True, timeout=60, env=child_environment(python))
    return {"fresh_import_probe": fresh, "wheel_lock_and_archives_rechecked": True, "pip_check_passed": True,
            "native_toy_fit_result_reused_from_scoring_not_repeated": True, "study_fitting_called": False}


def outcome_command(root, plan, stage, python, snapshot, output, score_hash):
    if stage["id"] not in {"join", "evaluate"} or stage["phase"] != "evaluation":
        raise ValueError("Only frozen join/evaluate stages permitted")
    template = stage["argv_template"]
    script = template[1]
    required = "experiments/" + ("join_bvival_evaluation_outcomes.py" if stage["id"] == "join" else "evaluate_bvival_policies.py")
    if script != required or script not in plan["static_source_closure"]["local_source_sha256"]:
        raise ValueError("Unexpected outcome source")
    prefix = plan["planned_output_root"]
    args = []
    for value in template[2:]:
        if isinstance(value, dict):
            if stage["id"] != "join" or value != {"frozen_artifact_sha256": prefix + "/scores/frozen_policy_scores_absolute_price_error.csv"}:
                raise ValueError("Unrecognized deferred binding")
            value = score_hash
        elif not isinstance(value, str):
            raise ValueError("Invalid bound argv value")
        elif value == prefix + "/evaluation-inputs" or value.startswith(prefix + "/evaluation-inputs/"):
            value = str(output / value[len(prefix) + 1:])
        elif value == prefix + "/evaluation-mae":
            value = str(output / "evaluation-mae")
        elif value.startswith(prefix + "/"):
            value = str(root / value)
        args.append(value)
    return [str(python), "-I", "-B", "-c", LAUNCHER, str(snapshot / script), *args]


def read_table(path, schema):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = set().union(*(set(v) for v in schema.values()))
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or set(reader.fieldnames) != expected:
            raise ValueError("Unrecognized aggregate schema: " + path.name)
        rows = {}
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed aggregate row")
            key = tuple(float(row[k]) if k == "budget" else row[k] for k in schema["keys"])
            if key in rows or any(k == "" or isinstance(k, float) and not math.isfinite(k) for k in key):
                raise ValueError("Invalid/duplicate aggregate key")
            rows[key] = row
    return rows


def interval_side(lower, upper, threshold):
    if lower > upper or not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("Invalid bootstrap interval")
    return "above" if lower > threshold else "below" if upper < threshold else "includes_boundary"


def compare_table(name, reference_path, replay_path, rules):
    schema = TABLES[name]
    reference, replay = read_table(reference_path, schema), read_table(replay_path, schema)
    missing, extra = set(reference) - set(replay), set(replay) - set(reference)
    mismatches = [{"kind": "missing_key", "key": list(k)} for k in sorted(missing)]
    mismatches += [{"kind": "extra_key", "key": list(k)} for k in sorted(extra)]
    tolerances = {"integer": 0, "price": rules["price_mae_absolute_tolerance_MAD"],
                  "rmsle": rules["rmsle_absolute_tolerance"],
                  "percent": rules["relative_gain_and_ci_endpoint_tolerance_percentage_points"],
                  "fraction": rules["fraction_metrics_absolute_tolerance"]}
    cells, equal_blanks, maximum = 0, 0, {}
    boundaries = []
    for key in sorted(set(reference) & set(replay)):
        a, b = reference[key], replay[key]
        for column in schema["text"]:
            if a[column] != b[column] or column == "auc_metric" and a[column] != "post_mae_price":
                mismatches.append({"kind": "text", "key": list(key), "column": column, "reference": a[column], "replay": b[column]})
        for group, tolerance in tolerances.items():
            for column in schema[group]:
                cells += 1
                if a[column] == b[column] == "":
                    zero_action_rate = (column == "harmful_action_rate" and a.get("action_count") == b.get("action_count") == "0")
                    zero_action_summary = (column == "mean_harmful_action_rate"
                                           and a.get("policy") == b.get("policy") == "score_no_acquisition")
                    if not (zero_action_rate or zero_action_summary):
                        raise ValueError("Unexpected missing metric")
                    equal_blanks += 1
                    continue
                if "" in (a[column], b[column]):
                    mismatches.append({"kind": "missing_value", "key": list(key), "column": column})
                    continue
                if group == "integer" and not all(re.fullmatch(r"\d+", v) for v in (a[column], b[column])):
                    raise ValueError("Noninteger count")
                x, y = (int(a[column]), int(b[column])) if group == "integer" else (float(a[column]), float(b[column]))
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("Nonfinite aggregate metric")
                delta = abs(x - y)
                maximum[column] = max(maximum.get(column, 0), delta)
                if delta > tolerance:
                    mismatches.append({"kind": "numeric", "key": list(key), "column": column,
                                       "reference": x, "replay": y, "absolute_delta": delta, "tolerance": tolerance})
        if name == "bvival_reference_comparisons.csv":
            for threshold in (0, 2):
                old = interval_side(float(a["ci_lower_percent"]), float(a["ci_upper_percent"]), threshold)
                new = interval_side(float(b["ci_lower_percent"]), float(b["ci_upper_percent"]), threshold)
                if old != new:
                    boundaries.append({"key": list(key), "threshold_percent": threshold, "reference": old, "replay": new})
    return {"file": name, "reference_row_count": len(reference), "replay_row_count": len(replay),
            "key_sets_match": not missing and not extra, "numeric_cells_checked": cells,
            "matched_not_applicable_cells": equal_blanks, "max_absolute_delta_by_column": maximum,
            "mismatches": mismatches, "changed_interval_boundaries": boundaries,
            "passes_fixed_rules": not mismatches and not boundaries,
            "byte_hash_matches": sha256(reference_path) == sha256(replay_path)}


def execute(root, scoring_relative, expected_freeze, scoring, freeze, plan, python, runtime, output_relative):
    output = root / output_relative
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    snapshot = output / "snapshot-workspace"
    logs = output / "logs"
    logs.mkdir()
    phase, completed = "snapshot", []
    try:
        for relative in plan["static_source_closure"]["local_source_sha256"]:
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(scoring / "snapshot-workspace", relative), target)
        target = snapshot / "release-bvival/run_mucars_outcome_replay.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).resolve(), target)
        write_new_json(output / "runtime-validation.json", runtime)
        write_new_json(output / "execution-binding.json", {"scoring_freeze_sha256": expected_freeze,
                       "bound_plan_sha256": freeze["plan_sha256"], "new_score_sha256": freeze["new_score_sha256"],
                       "output_root": output_relative, "engineering_rules": plan["engineering_concordance_rules"],
                       "historical_tests_already_opened": True, "new_preregistration": False})
        for stage in plan["stages"][-2:]:
            phase = stage["id"]
            verify_scoring(root, scoring_relative, expected_freeze)
            for relative, expected in freeze["bound_local_source_sha256"].items():
                if sha256(safe_path(snapshot, relative)) != expected:
                    raise ValueError("Outcome snapshot drift")
            command = outcome_command(root, plan, stage, python, snapshot, output, freeze["new_score_sha256"])
            print(json.dumps({"stage": phase, "status": "started"}), flush=True)
            started = time.monotonic()
            with (logs / (phase + ".txt")).open("x", encoding="utf-8") as log:
                subprocess.run(command, cwd=snapshot, env=child_environment(python), stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=1800)
            completed.append({"stage": phase, "elapsed_seconds": round(time.monotonic() - started, 3)})
            print(json.dumps({"stage": phase, "status": "completed"}), flush=True)
        phase = "concordance"
        verify_scoring(root, scoring_relative, expected_freeze)
        rules = plan["engineering_concordance_rules"]
        comparisons = [compare_table(name, safe_path(root, rules["original_aggregate_reference"] + "/" + name),
                                     output / "evaluation-mae" / name, rules) for name in TABLES]
        write_new_json(output / "aggregate-concordance.json", {"tables": comparisons,
                       "all_tables_pass_fixed_rules": all(r["passes_fixed_rules"] for r in comparisons),
                       "rules_unchanged": True, "independent_scientific_replication": False})
        artifacts = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
        join = json.loads((output / "evaluation-inputs/evaluation_outcome_join_audit.json").read_text())
        receipt = {"date": "2026-10-01", "stage": "RECONSTRUCTED_MUCARS_PRIMARY_OUTCOME_COMPARISON_COMPLETED",
                   "scoring_freeze_sha256": expected_freeze, "plan_sha256": freeze["plan_sha256"],
                   "completed_stages": completed, "artifact_sha256": artifacts,
                   "joined_outcomes_sha256": join["output_sha256"], "listing_count": join["listing_count"],
                   "listing_action_count": join["listing_action_count"],
                   "all_primary_tables_pass_fixed_rules": all(r["passes_fixed_rules"] for r in comparisons),
                   "all_primary_table_byte_hashes_match": all(r["byte_hash_matches"] for r in comparisons),
                   "outcome_join_called": True, "test_performance_computed": True,
                   "historical_tests_already_opened": True, "analysis_status": "post_test_exploratory",
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
    parser.add_argument("--scoring-root", default="experiments/replays/mucars-reconstructed-primary-v1")
    parser.add_argument("--expected-scoring-freeze-sha256", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-name", default="mucars-reconstructed-primary-outcomes-v1")
    parser.add_argument("--execute-outcomes", action="store_true")
    args = parser.parse_args()
    try:
        root, python = args.project_root.resolve(strict=True), args.python.absolute()
        output = output_prefix(root, args.output_name)
        scoring, freeze, plan = verify_scoring(root, args.scoring_root, args.expected_scoring_freeze_sha256)
        runtime = check_runtime(scoring, freeze, python)
        if args.execute_outcomes:
            receipt = execute(root, args.scoring_root, args.expected_scoring_freeze_sha256,
                              scoring, freeze, plan, python, runtime, output)
            print(json.dumps({"stage": receipt["stage"], "output_root": output,
                              "all_primary_tables_pass_fixed_rules": receipt["all_primary_tables_pass_fixed_rules"],
                              "all_primary_table_byte_hashes_match": receipt["all_primary_table_byte_hashes_match"]}), flush=True)
        else:
            print(json.dumps({"stage": "OUTCOME_REPLAY_PREFLIGHT_ONLY", "scoring_artifacts_verified": len(freeze["artifact_sha256"]),
                              "runtime_validated": True, "outcome_join_called": False,
                              "new_outcome_performance_computed": False, "output_created": False}))
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Outcome replay failed: {error}\n")


if __name__ == "__main__":
    main()
