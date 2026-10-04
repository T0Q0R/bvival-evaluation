"""Run nine bound JUCars scoring stages, never an outcome join or evaluator.

Explicit execution opt-in; new exclusive root; retain any failure. Procedural
isolation of trusted code, not an OS sandbox or a new unopened-label study.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mucars_reconstruction import safe_path
from plan_jucars_replay import validate_plan, RECEIPT, ADAPTER, RUNTIME
from run_mucars_scoring_replay import (
    sha256, lock_pins, validate_runtime_records, child_environment, PROBE, LAUNCHER,
    check_score_file, write_new_json,
)

SCORING_IDS = ["cohort", "pairs", "development-current", "calibration-current",
               "development-legacy", "calibration-legacy", "value", "risk-mae", "scores"]
HELPERS = ["run_jucars_scoring_replay.py", "plan_jucars_replay.py", "project_jucars_legacy_targets.py",
           "check_jucars_reconstruction.py", "check_mucars_reconstruction.py", "plan_mucars_replay.py",
           "check_rerun_entrypoints.py", "run_mucars_scoring_replay.py"]
EXPECTED_PLAN = "201c8fbd418034aa545c0ee5aa70dbd1f476cfd359dbf4c2c9db36eb8eaa81b3"


def confined(root, relative):
    """Reject malformed relative paths and all symlink components, even outputs."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or "\\" in relative:
        raise ValueError("Unsafe relative replay path")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Unsafe relative replay component")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink replay component rejected")
    return path


def source_checks(root, plan):
    bindings = dict(plan["static_study_source_closure"]["local_source_sha256"])
    bindings[ADAPTER] = plan["schema_adapter_source"]["sha256"]
    bindings[RECEIPT] = plan["recovery_receipt_sha256"]
    for relative, expected in bindings.items():
        if sha256(safe_path(root, relative)) != expected:
            raise ValueError("Bound source/adapter/recovery receipt drift")
    return bindings


def original_input_checks(root, plan):
    bindings = plan["source_bindings_to_verify_before_execution"]
    if set(bindings) != {"source", "dictionary"}:
        raise ValueError("Both original source and dictionary must be bound")
    for role, item in bindings.items():
        if sha256(safe_path(root, item["path"])) != item["sha256"]:
            raise ValueError("Original " + role + " hash mismatch; no fitting permitted")


def verify_bindings(root, snapshot, bindings):
    for relative, expected in bindings.items():
        if sha256(safe_path(snapshot, relative)) != expected or sha256(safe_path(root, relative)) != expected:
            raise ValueError("Source/helper drift during execution")


def runtime_hash_checks(python, lock, install, runtime):
    for path, key in ((lock, "requirements_lock_sha256"), (install, "install_report_sha256"), (python, "python_executable_sha256")):
        if sha256(path) != runtime[key]:
            raise ValueError("Runtime changed after preflight")


def stage_command(root, plan, stage, python, snapshot):
    if stage["id"] not in SCORING_IDS or stage["phase"] not in {"reconstruction", "scoring"}:
        raise ValueError("Outcome/evaluation stage forbidden")
    template = stage["argv_template"]
    if template[0] != RUNTIME:
        raise ValueError("Unknown runtime binding")
    relative = template[1]
    if relative not in plan["static_study_source_closure"]["local_source_sha256"] and relative != ADAPTER:
        raise ValueError("Unbound stage source")
    raw_paths = {i["path"] for i in plan["source_bindings_to_verify_before_execution"].values()}
    args = []
    for value in template[2:]:
        if isinstance(value, dict):
            if stage["id"] not in {"development-legacy", "calibration-legacy"} or set(value) != {"generated_target_hash_from_audit"}:
                raise ValueError("Only projection input hashes may be resolved in scoring")
            phase = stage["id"].split("-")[0]
            expected = plan["planned_output_root"] + "/" + phase + "-current/bvival_action_dataset_audit.json"
            if value["generated_target_hash_from_audit"] != expected:
                raise ValueError("Projection hash binding is not its own current target")
            audit = json.loads(safe_path(root, expected).read_text())
            target = safe_path(root, expected.rsplit("/", 1)[0] + "/bvival_action_dataset.csv")
            if sha256(target) != audit["output_sha256"]:
                raise ValueError("Generated target hash differs from its audit")
            value = audit["output_sha256"]
        if not isinstance(value, str):
            raise ValueError("Unsupported deferred scoring argument")
        if value == RECEIPT:
            value = str(safe_path(snapshot, RECEIPT))
        elif value in raw_paths or value.startswith(plan["planned_output_root"] + "/"):
            value = str(confined(root, value))
        args.append(value)
    return [str(python), "-I", "-B", "-c", LAUNCHER, str(safe_path(snapshot, relative)), *args]


def table_keys(path, header=None):
    keys, listings = set(), set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if (reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames))
                or not {"listing_id", "action_id"}.issubset(reader.fieldnames)
                or header is not None and reader.fieldnames != header):
            raise ValueError("Generated table schema differs")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed generated table row")
            key = row["listing_id"], row["action_id"]
            if not all(key) or key in keys:
                raise ValueError("Generated keys must be nonempty and unique")
            keys.add(key)
            listings.add(key[0])
    return keys, listings


def cohort_checks(output, plan):
    contract = plan["expected_structural_contract"]
    audit = json.loads(safe_path(output, "cohort/jucars_bvival_audit.json").read_text())
    if audit["split_counts"] != contract["split_counts"]:
        raise ValueError("Cohort counts differ")
    if audit["outputs"] != contract["cohort_output_sha256"]:
        raise ValueError("Cohort reported hashes differ")
    for relative, expected in contract["cohort_output_sha256"].items():
        if sha256(safe_path(output / "cohort", relative)) != expected:
            raise ValueError("Cohort bytes differ from historical contract")
    return {"split_counts_and_all_recorded_output_hashes_match": True,
            "historical_source_prices_read_by_builder": True}


def pair_checks(output, plan):
    contract = plan["expected_structural_contract"]
    p = json.loads(safe_path(output, "pairs/prediction_pair_generation_audit.json").read_text())
    for phase, count in contract["phase_counts"].items():
        if p[phase + "_listing_count"] != count["listings"] or p[phase + "_action_pair_count"] != count["pairs"]:
            raise ValueError("Rebuilt pair counts differ")
        for name, suffix in (("pairs", "prediction_pairs"), ("features", "pre_action_features"), ("labels", "labels")):
            if name == "labels" and phase == "evaluation":
                continue  # No evaluation labels are read by the pair generator.
            if sha256(safe_path(output, f"pairs/{phase}_{suffix}.csv")) != p["outputs"][phase][name]:
                raise ValueError("Pair output bytes differ from generated audit")
    for k, expected in (("feature_columns", contract["base_feature_columns"]),
                        ("categorical_columns", contract["base_categorical_columns"]),
                        ("seeds", [13, 42, 2026]), ("folds", 5)):
        if p[k] != expected:
            raise ValueError("Base feature/category/seed/fold contract differs")
    if p["evaluation_labels_read"] is not False or p["performance_metrics_computed"] is not False:
        raise ValueError("Unexpected pair-generator outcome access")
    return p


def target_checkpoint(output, plan):
    contract = plan["expected_structural_contract"]
    p = pair_checks(output, plan)
    results = {}
    for phase in ("development", "calibration"):
        directory = output / (phase + "-current")
        a = json.loads(safe_path(directory, "bvival_action_dataset_audit.json").read_text())
        path = safe_path(directory, "bvival_action_dataset.csv")
        digest = sha256(path)
        if digest != a["output_sha256"] or phase == "development" and digest != contract["development_current_sha256"]:
            raise ValueError("Current target bytes differ")
        if a["feature_allowlist"] != contract["policy_feature_columns"][1:] or a["all_predictions_oof"] is not True:
            raise ValueError("Current target feature/OOF contract differs")
        for producer, consumer in (("pairs", "prediction_pairs"), ("features", "pre_action_features"), ("labels", "labels")):
            if a["inputs"][consumer]["sha256"] != p["outputs"][phase][producer]:
                raise ValueError("Target input hashes differ from generated pairs")
        keys, listings = table_keys(path, contract["new_target_header"])
        pair_keys, _ = table_keys(safe_path(output, f"pairs/{phase}_prediction_pairs.csv"))
        count = contract["phase_counts"][phase]
        if keys != pair_keys or len(keys) != count["pairs"] or len(listings) != count["listings"] or a["listing_count"] != len(listings) or a["listing_action_count"] != len(keys):
            raise ValueError("Current target universe/counts differ")
        results[phase] = {"sha256": digest, "listing_count": len(listings), "pair_count": len(keys)}
    return {"stage": "CURRENT_TARGET_SCHEMA_CHECKPOINT_PASSED", "targets": results,
            "historical_target_rows_read": False, "generated_development_and_calibration_rows_parsed": True}


def projection_checkpoint(output, plan):
    contract = plan["expected_structural_contract"]
    records = {}
    for phase, role in (("development", "development_value"), ("calibration", "calibration")):
        path = safe_path(output, phase + "-legacy/bvival_action_dataset.csv")
        a = json.loads(safe_path(output, phase + "-legacy/legacy_projection_audit.json").read_text())
        keys, listings = table_keys(path, contract["legacy_target_header"])
        current_keys, _ = table_keys(safe_path(output, phase + "-current/bvival_action_dataset.csv"))
        digest = sha256(path)
        count = contract["phase_counts"][phase]
        if (digest != contract[phase + "_legacy_sha256"] or digest != a["output_sha256"]
                or a["stage"] != "LEGACY_SCHEMA_PROJECTED_WITH_RECORDED_BYTE_HASH_MATCH" or a["role"] != role
                or a["numeric_values_recomputed"] is not False or a["historical_target_rows_read"] is not False
                or a["fitting_called"] is not False or a["retained_cell_text_and_row_order_preserved"] is not True
                or a["contract_sha256"] != plan["recovery_receipt_sha256"]
                or a["input_sha256"] != sha256(safe_path(output, phase + "-current/bvival_action_dataset.csv"))
                or len(keys) != count["pairs"] or len(listings) != count["listings"]
                or a["listing_count"] != len(listings) or a["action_pair_count"] != len(keys)
                or keys != current_keys):
            raise ValueError("Legacy projection checkpoint failed")
        records[phase] = {"sha256": digest, "listing_count": len(listings), "pair_count": len(keys)}
    return {"stage": "LEGACY_PROJECTION_CHECKPOINT_PASSED", "projections": records}


def head_checks(output, plan):
    c = plan["expected_structural_contract"]
    for directory, audit_file, branch in (("value", "value_policy_fit_audit.json", "development-legacy"),
                                          ("risk-mae", "risk_baseline_fit_audit.json", "development-current")):
        a = json.loads(safe_path(output, directory + "/" + audit_file).read_text())
        if a["feature_columns"] != c["policy_feature_columns"] or a["categorical_columns"] != c["policy_categorical_columns"] or a["seeds"] != [13, 42, 2026]:
            raise ValueError("Head feature/category/seed contract differs")
        if a["inputs"]["train_actions_sha256"] != sha256(safe_path(output, branch + "/bvival_action_dataset.csv")):
            raise ValueError("Head trained on wrong target branch")
        if directory == "value" and a["inputs"]["calibration_actions_sha256"] != c["calibration_legacy_sha256"]:
            raise ValueError("Value calibration branch differs")
        if directory == "risk-mae" and (a["objective"] != "absolute_price_error" or a["development_oof"] is not None):
            raise ValueError("Unexpected risk objective/development selection")


def preflight(root, plan_file, python, lock, install, expected_plan, expected_lock, expected_install):
    for path, expected in ((plan_file, expected_plan), (lock, expected_lock), (install, expected_install)):
        if sha256(path) != expected:
            raise ValueError("Plan/lock/install receipt hash mismatch")
    data = plan_file.read_bytes()
    plan = json.loads(data)
    validate_plan(root, plan)
    if [s["id"] for s in plan["stages"][:9]] != SCORING_IDS or len(plan["stages"]) != 11:
        raise ValueError("Exactly nine scoring and two forbidden outcome templates required")
    probe = subprocess.run([str(python), "-I", "-B", "-c", PROBE], check=True, capture_output=True,
                           text=True, timeout=60, env=child_environment(python))
    runtime = validate_runtime_records(lock_pins(lock.read_text()), json.loads(install.read_text()), json.loads(probe.stdout))
    pip = subprocess.run([str(python), "-I", "-B", "-m", "pip", "--isolated", "check"], check=True,
                         capture_output=True, text=True, timeout=60, env=child_environment(python))
    runtime.update({"pip_check_passed": pip.returncode == 0, "requirements_lock_sha256": expected_lock,
                    "install_report_sha256": expected_install, "python_executable_sha256": sha256(python)})
    return plan, data, runtime


def execute_scoring(root, plan, data, python, lock, install, runtime):
    if hashlib.sha256(data).hexdigest() != EXPECTED_PLAN or json.loads(data) != plan:
        raise ValueError("Frozen JUCars v1 plan identity required")
    validate_plan(root, plan)
    bindings = source_checks(root, plan)
    original_input_checks(root, plan)
    runtime_hash_checks(python, lock, install, runtime)
    helper_hashes = {"release-bvival/" + n: sha256(safe_path(root, "release-bvival/" + n)) for n in HELPERS}
    output = confined(root, plan["planned_output_root"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    snapshot = output / "snapshot-workspace"
    logs = output / "logs"
    logs.mkdir()
    phase, completed = "snapshot", []
    try:
        for relative, expected in {**bindings, **helper_hashes}.items():
            target = confined(snapshot, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(root, relative), target)
            if sha256(target) != expected:
                raise ValueError("Source snapshot differs")
        with (output / "bound-plan.json").open("xb") as stream:
            stream.write(data)
        shutil.copyfile(lock, output / "runtime.lock")
        shutil.copyfile(install, output / "install-report.json")
        write_new_json(output / "runtime-validation.json", runtime)
        for stage in plan["stages"][:9]:
            phase = stage["id"]
            source_checks(root, plan)
            verify_bindings(root, snapshot, {**bindings, **helper_hashes})
            if phase == "development-legacy":
                write_new_json(output / "target-schema-checkpoint.json", target_checkpoint(output, plan))
            if phase in {"value", "risk-mae"}:
                target_checkpoint(output, plan)
                projection_checkpoint(output, plan)
                if phase == "value":
                    write_new_json(output / "legacy-projection-checkpoint.json", projection_checkpoint(output, plan))
            command = stage_command(root, plan, stage, python, snapshot)
            print(json.dumps({"stage": phase, "status": "started"}), flush=True)
            started = time.monotonic()
            with (logs / (phase + ".txt")).open("x", encoding="utf-8") as log:
                subprocess.run(command, cwd=snapshot, env=child_environment(python), stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=1800)
            completed.append({"stage": phase, "elapsed_seconds": round(time.monotonic() - started, 3)})
            if phase == "cohort":
                cohort_checks(output, plan)
            if phase == "pairs":
                pair_checks(output, plan)
            print(json.dumps({"stage": phase, "status": "completed", "elapsed_seconds": completed[-1]["elapsed_seconds"]}), flush=True)
        phase = "freeze_checks"
        source_checks(root, plan)
        verify_bindings(root, snapshot, {**bindings, **helper_hashes})
        runtime_hash_checks(python, lock, install, runtime)
        cohort_checks(output, plan)
        target_checkpoint(output, plan)
        projection_checkpoint(output, plan)
        head_checks(output, plan)
        scores = safe_path(output, "scores/frozen_policy_scores_absolute_price_error.csv")
        count = plan["expected_structural_contract"]["phase_counts"]["evaluation"]
        score_check = check_score_file(scores, count["listings"], count["pairs"])
        score_keys, _ = table_keys(scores)
        pair_keys, _ = table_keys(safe_path(output, "pairs/evaluation_prediction_pairs.csv"))
        if score_keys != pair_keys:
            raise ValueError("Score/pair action universe differs")
        if (output / "evaluation-inputs").exists() or (output / "evaluation-mae").exists():
            raise ValueError("Forbidden outcome output created")
        artifacts = {}
        for path in sorted(output.rglob("*")):
            if path.is_symlink():
                raise ValueError("Generated symlink rejected")
            if path.is_file():
                artifacts[str(path.relative_to(output))] = sha256(path)
        receipt = json.loads(safe_path(root, RECEIPT).read_text())
        freeze = {"stage": "RECONSTRUCTED_JUCARS_SCORES_FROZEN_NO_OUTCOME_EVALUATION", "date": "2026-10-01",
                  "replay_name": plan["replay_name"], "plan_sha256": hashlib.sha256(data).hexdigest(),
                  "study_and_recovery_source_sha256": bindings, "runner_and_helper_sha256": helper_hashes,
                  "runtime_validation": runtime, "runtime_lock_sha256": sha256(lock),
                  "original_source_and_dictionary_hashes_checked": True, "completed_scoring_stages": completed,
                  "new_score_sha256": sha256(scores), "recorded_historical_score_sha256": receipt["recorded_primary_score_sha256"],
                  "score_bytes_match_recorded_historical_hash": sha256(scores) == receipt["recorded_primary_score_sha256"],
                  "score_schema_check": score_check, "artifact_sha256": artifacts,
                  "source_prices_processed_for_cohort_and_training": True, "new_target_rows_parsed": True,
                  "historical_test_labels_already_opened": True, "historical_target_rows_parsed": False,
                  "inherited_historical_protocol_tags_are_not_new_confirmation": True,
                  "outcome_join_called": False, "test_performance_computed": False,
                  "full_primary_table_concordance_verified": False, "secondary_score_guard_replayed": False,
                  "historical_source_identity_claimed": False, "independent_replication_claimed": False,
                  "original_outputs_or_manuscript_changed": False, "public_release_created": False, "submission_ready": False}
        write_new_json(output / "scoring-freeze.json", freeze)
        return freeze
    except Exception as error:
        write_new_json(output / "failed-scoring-replay.json", {"phase": phase, "exception_class": type(error).__name__,
                       "completed_scoring_stages": completed, "scores_frozen": False,
                       "outcome_join_called": False, "test_performance_computed": False,
                       "partial_directory_retained_no_overwrite_or_automatic_retry": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--requirements-lock", type=Path, required=True)
    parser.add_argument("--install-report", type=Path, required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--expected-install-report-sha256", required=True)
    parser.add_argument("--execute-scoring", action="store_true")
    args = parser.parse_args()
    try:
        root = args.project_root.resolve(strict=True)
        python = args.python.absolute()  # Keep the venv launcher, not its symlink target.
        plan, data, runtime = preflight(root, args.plan, python, args.requirements_lock, args.install_report,
                                       EXPECTED_PLAN, args.expected_lock_sha256, args.expected_install_report_sha256)
        if args.execute_scoring:
            result = execute_scoring(root, plan, data, python, args.requirements_lock, args.install_report, runtime)
            print(json.dumps({"stage": result["stage"], "score_sha256": result["new_score_sha256"],
                              "output_root": plan["planned_output_root"], "test_performance_computed": False}), flush=True)
        else:
            print(json.dumps({"stage": "JUCARS_SCORING_RUNTIME_PREFLIGHT_ONLY", "runtime": runtime,
                              "original_data_rows_read": False, "study_training_started": False,
                              "replay_output_created": False}, indent=2))
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"JUCars scoring replay failed: {error}\n")


if __name__ == "__main__":
    main()
