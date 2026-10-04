"""Execute twelve frozen AutoScout24 scoring stages; never join/evaluate.

Trusted-code procedural isolation, not an OS sandbox. Every mismatch stops
with the exclusive attempt retained. Historical tests remain already opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_autoscout24_reconstruction import AUDITS, CONFIG
from check_mucars_reconstruction import safe_path
from plan_autoscout24_replay import validate_plan, RECEIPT, RUNTIME, SCORING_IDS
from run_jucars_scoring_replay import confined, verify_bindings, runtime_hash_checks, table_keys
from run_mucars_scoring_replay import (
    sha256, lock_pins, validate_runtime_records, child_environment, PROBE,
    LAUNCHER, write_new_json,
)

EXPECTED_PLAN = "aaaa6753a119c1e081a70b71d7eee04b5ce17a0f60d9ea1a2907e635262dd683"
HELPERS = ["run_autoscout24_scoring_replay.py", "plan_autoscout24_replay.py",
           "check_autoscout24_reconstruction.py", "check_remaining_source_metadata.py",
           "check_mucars_reconstruction.py", "plan_mucars_replay.py", "check_rerun_entrypoints.py",
           "run_mucars_scoring_replay.py", "run_jucars_scoring_replay.py",
           "plan_jucars_replay.py", "check_jucars_reconstruction.py", "project_jucars_legacy_targets.py"]
PAIR_AUDITS = {"selection-pairs": ("selection_pairs", "field_selection_pair_audit.json"),
               "final-pairs": ("pairs", "prediction_pair_generation_audit.json")}
TARGETS = {"selection-train-targets": ("selection_train", "selection-pairs", "train_oof"),
           "selection-validation-targets": ("selection_validation", "selection-pairs", "validation_heldout"),
           "development-targets": ("development", "final-pairs", "development"),
           "calibration-targets": ("calibration", "final-pairs", "calibration")}
HEADS = {"selection-risk": ("selection_risk", "risk_baseline_fit_audit.json", "selection-train-targets"),
         "value": ("value", "value_policy_fit_audit.json", "development-targets"),
         "risk-mae": ("risk", "risk_baseline_fit_audit.json", "development-targets")}
SCORES = ["score_no_acquisition", "score_random", "score_global_field_prior",
          "score_uncertainty_only", "score_disagreement_only", "score_fixed_field_risk",
          "score_mean_value", "score_conservative_lower_value"]


def source_checks(root, plan):
    bindings = dict(plan["static_study_source_closure"]["local_source_sha256"])
    bindings[RECEIPT] = plan["recovery_receipt_sha256"]
    receipt = json.loads(safe_path(root, RECEIPT).read_text())
    bindings.update(receipt["audit_file_sha256"])
    bindings[CONFIG] = receipt["config_sha256"]
    bindings[receipt["rights_correction_note"]] = receipt["rights_correction_note_sha256"]
    for relative, expected in bindings.items():
        if sha256(safe_path(root, relative)) != expected:
            raise ValueError("Bound source/recovery drift")
    return bindings


def original_input_checks(root, plan):
    item = plan["source_binding"]
    if sha256(safe_path(root, item["path"])) != item["sha256"]:
        raise ValueError("Original opaque source hash mismatch; no study fitting permitted")


def stage_command(root, plan, stage, python, snapshot):
    if stage["id"] not in SCORING_IDS or stage["phase"] not in {"scoring", "reconstruction"}:
        raise ValueError("Outcome/evaluation stage forbidden")
    template = stage["argv_template"]
    if template[0] != RUNTIME or template[1] not in plan["static_study_source_closure"]["local_source_sha256"]:
        raise ValueError("Unbound runtime or study source")
    args = []
    for value in template[2:]:
        if not isinstance(value, str):
            raise ValueError("Deferred outcome arguments forbidden in scoring")
        if value.startswith("experiments/"):
            if value != plan["source_binding"]["path"] and not value.startswith(plan["planned_output_root"] + "/"):
                raise ValueError("Historical data substitution forbidden")
            value = str(confined(root, value))
        args.append(value)
    return [str(python), "-I", "-B", "-c", LAUNCHER,
            str(safe_path(snapshot, template[1])), *args]


def read_audit(root, relative):
    return json.loads(safe_path(root, relative).read_text())


def canonical_audit(value, model=False):
    """Only relocate documented file paths and allow new CBM serialization hashes.

    Training data, predictions, calibration and score hashes stay exact. CBM
    metadata can contain fresh IDs/timestamps; their new hashes are bound below.
    """
    if isinstance(value, list):
        return [canonical_audit(v, model) for v in value]
    if isinstance(value, dict):
        return {k: canonical_audit(v, k in {"models", "train_only_base_models"})
                for k, v in value.items()
                if k not in {"path", "model_path"} and not (model and k == "sha256")}
    return value


def compare_audit(new, reference):
    if canonical_audit(new) != canonical_audit(reference):
        raise ValueError("Rebuilt audit contract differs (paths/CBM serialization excluded only)")


def verify_model_records(directory, records, train_only=False):
    for record in records:
        if train_only:
            path = safe_path(directory, f"train_only_base_seed_{record['seed']}.cbm")
        else:
            raw = record.get("model_path", record.get("path"))
            path = Path(raw)
            try:
                relative = str(path.relative_to(directory))
            except ValueError as error:
                raise ValueError("Saved model path outside new stage") from error
            path = safe_path(directory, relative)
        if sha256(path) != record["sha256"]:
            raise ValueError("New model bytes differ from generated model audit")


def cohort_checks(root, output, plan):
    c = plan["structural_contract"]
    new = read_audit(output, "cohort/autoscout24_bvival_audit.json")
    old = read_audit(root, AUDITS["cohort"])
    allowed = {"source_rights", "source_path"}
    if set(c["allowed_cohort_metadata_differences"]) != allowed:
        raise ValueError("Cohort metadata exception must not be widened")
    if {k: v for k, v in new.items() if k not in allowed} != {k: v for k, v in old.items() if k not in allowed}:
        raise ValueError("Cohort metadata/counts differ beyond approved rights/path correction")
    if new["split_counts"] != c["split_counts"] or new["output_hashes"] != c["six_cohort_data_hashes"]:
        raise ValueError("Cohort structural contract differs")
    for relative, expected in c["six_cohort_data_hashes"].items():
        if sha256(safe_path(output / "cohort", relative)) != expected:
            raise ValueError("Cohort generated bytes differ")
    return {"six_data_hashes_and_counts_match": True,
            "metadata_differences": sorted(k for k in allowed if new.get(k) != old.get(k))}


def pair_checks(root, output, stage):
    role, name = PAIR_AUDITS[stage]
    a = read_audit(output, stage + "/" + name)
    compare_audit(a, read_audit(root, AUDITS[role]))
    for phase, outputs in a["outputs"].items():
        for key, suffix in (("pairs", "prediction_pairs"), ("features", "pre_action_features"), ("labels", "labels")):
            if key in outputs and sha256(safe_path(output, f"{stage}/{phase}_{suffix}.csv")) != outputs[key]:
                raise ValueError("Generated prediction bundle bytes differ")
    if stage == "selection-pairs":
        verify_model_records(output / stage, a["train_only_base_models"], train_only=True)
        if a["sealed_test_input_read"] is not False or a["validation_labels_used_in_base_fit"] is not False:
            raise ValueError("Selection branch leakage")
    elif a["evaluation_labels_read"] is not False:
        raise ValueError("Unexpected evaluation labels")
    if stage == "final-pairs":
        verify_model_records(output / stage, a["models"])
    if a["performance_metrics_computed"] is not False:
        raise ValueError("Unexpected test performance")
    return a


def target_checks(root, output, plan, stage):
    role, pairdir, phase = TARGETS[stage]
    pair_checks(root, output, pairdir)
    a = read_audit(output, stage + "/bvival_action_dataset_audit.json")
    compare_audit(a, read_audit(root, AUDITS[role]))
    path = safe_path(output, stage + "/bvival_action_dataset.csv")
    digest = sha256(path)
    if digest != a["output_sha256"] or digest != plan["structural_contract"]["four_generated_target_hashes"][role]:
        raise ValueError("New target bytes differ; historical substitution forbidden")
    keys, listings = table_keys(path, plan["structural_contract"]["target_header"])
    pairkeys, _ = table_keys(safe_path(output, f"{pairdir}/{phase}_prediction_pairs.csv"))
    if keys != pairkeys or len(keys) != a["listing_action_count"] or len(listings) != a["listing_count"] or a["all_predictions_oof"] is not True:
        raise ValueError("Generated target universe/OOF contract differs")
    return {"sha256": digest, "listings": len(listings), "pairs": len(keys), "historical_target_rows_parsed": False}


def head_checks(root, output, stage):
    role, name, train = HEADS[stage]
    a = read_audit(output, stage + "/" + name)
    compare_audit(a, read_audit(root, AUDITS[role]))
    if a["inputs"]["train_actions_sha256"] != sha256(safe_path(output, train + "/bvival_action_dataset.csv")) or a["evaluation_outcomes_read"] is not False:
        raise ValueError("Wrong head training branch or unexpected outcomes")
    groups = a["objectives"].values() if stage == "value" else [a]
    for group in groups:
        verify_model_records(output / stage, group["models"])
    return a


def fixed_field_checks(root, output, plan):
    a = read_audit(output, "fixed-field/selection.json")
    c = plan["structural_contract"]
    if (sha256(safe_path(output, "fixed-field/selection.json")) != c["fixed_field_decision_sha256"]
            or a["selected_action"] != c["validation_fixed_field"] or a["listing_count"] != c["validation_listing_count"]
            or a["capacity"] != c["validation_capacity"] or a["evaluation_labels_read"] is not False):
        raise ValueError("Fixed choice/validation capacity/bytes differ")
    for role, path in (("development_actions", "selection-validation-targets/bvival_action_dataset.csv"),
                       ("development_risk_scores", "selection-risk/evaluation_risk_scores.csv")):
        if a["input_sha256"][role] != sha256(safe_path(output, path)):
            raise ValueError("Fixed decision not bound to newly generated validation inputs")
    return a


def score_checks(path, listing_count, pair_count):
    keys, listings = table_keys(path, ["listing_id", "action_id", *SCORES])
    if len(keys) != pair_count or len(listings) != listing_count:
        raise ValueError("Score universe/count mismatch")
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            for column in SCORES:
                value = float(row[column])
                if not math.isfinite(value) and not (value == -math.inf and column in {
                        "score_uncertainty_only", "score_disagreement_only", "score_fixed_field_risk"}):
                    raise ValueError("Invalid generated policy score")
    return {"listings": len(listings), "pairs": len(keys), "eight_score_columns_validated": True}


def preflight(root, plan_file, python, lock, install, expected_lock, expected_install):
    for path, expected in ((plan_file, EXPECTED_PLAN), (lock, expected_lock), (install, expected_install)):
        if sha256(path) != expected:
            raise ValueError("Plan/lock/install hash mismatch")
    data = plan_file.read_bytes()
    plan = json.loads(data)
    validate_plan(root, plan)
    if [s["id"] for s in plan["stages"]] != SCORING_IDS + ["join", "evaluate"]:
        raise ValueError("Twelve scoring plus two forbidden outcome stages required")
    probe = subprocess.run([str(python), "-I", "-B", "-c", PROBE], check=True,
                           capture_output=True, text=True, timeout=60, env=child_environment(python))
    runtime = validate_runtime_records(lock_pins(lock.read_text()), json.loads(install.read_text()), json.loads(probe.stdout))
    subprocess.run([str(python), "-I", "-B", "-m", "pip", "--isolated", "check"], check=True,
                   capture_output=True, text=True, timeout=60, env=child_environment(python))
    runtime.update({"pip_check_passed": True, "requirements_lock_sha256": expected_lock,
                    "install_report_sha256": expected_install, "python_executable_sha256": sha256(python)})
    return plan, data, runtime


def execute_scoring(root, plan, data, python, lock, install, runtime):
    if hashlib.sha256(data).hexdigest() != EXPECTED_PLAN or json.loads(data) != plan:
        raise ValueError("Frozen AutoScout24 plan identity required")
    validate_plan(root, plan)
    bindings = source_checks(root, plan)
    original_input_checks(root, plan)
    runtime_hash_checks(python, lock, install, runtime)
    helpers = {"release-bvival/" + n: sha256(safe_path(root, "release-bvival/" + n)) for n in HELPERS}
    output = confined(root, plan["planned_output_root"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)  # Existing attempts are never resumed or overwritten.
    snapshot = output / "snapshot-workspace"
    (output / "logs").mkdir()
    phase, completed, checkpoints = "snapshot", [], {}
    try:
        for relative, digest in {**bindings, **helpers}.items():
            path = confined(snapshot, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(root, relative), path)
            if sha256(path) != digest:
                raise ValueError("Snapshot mismatch")
        with (output / "bound-plan.json").open("xb") as stream:
            stream.write(data)
        shutil.copyfile(lock, output / "runtime.lock")
        shutil.copyfile(install, output / "install-report.json")
        write_new_json(output / "runtime-validation.json", runtime)
        for stage in plan["stages"][:12]:
            phase = stage["id"]
            if not set(stage["depends_on"]).issubset({s["stage"] for s in completed}):
                raise ValueError("Incomplete stage dependency")
            verify_bindings(root, snapshot, {**bindings, **helpers})
            runtime_hash_checks(python, lock, install, runtime)
            if phase == "selection-pairs":
                cohort_checks(root, output, plan)
            if phase == "selection-risk":
                target_checks(root, output, plan, "selection-train-targets")
            if phase == "fixed-field":
                target_checks(root, output, plan, "selection-validation-targets")
                head_checks(root, output, "selection-risk")
            if phase == "final-pairs":
                fixed_field_checks(root, output, plan)
            if phase in {"value", "risk-mae"}:
                for target in ("development-targets", "calibration-targets"):
                    target_checks(root, output, plan, target)
            print(json.dumps({"stage": phase, "status": "started"}), flush=True)
            started = time.monotonic()
            with (output / "logs" / (phase + ".txt")).open("x") as log:
                subprocess.run(stage_command(root, plan, stage, python, snapshot), cwd=snapshot,
                               env=child_environment(python), stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=3600)
            if phase == "cohort":
                checkpoints[phase] = cohort_checks(root, output, plan)
            elif phase in PAIR_AUDITS:
                pair_checks(root, output, phase)
            elif phase in TARGETS:
                checkpoints[phase] = target_checks(root, output, plan, phase)
            elif phase in HEADS:
                a = head_checks(root, output, phase)
                name = "evaluation_action_scores.csv" if phase == "value" else "evaluation_risk_scores.csv"
                key = "evaluation_action_scores_sha256" if phase == "value" else "evaluation_risk_scores_sha256"
                if sha256(safe_path(output, phase + "/" + name)) != a[key]:
                    raise ValueError("Head score output hash differs")
            elif phase == "fixed-field":
                fixed_field_checks(root, output, plan)
            completed.append({"stage": phase, "elapsed_seconds": round(time.monotonic() - started, 3)})
            print(json.dumps({**completed[-1], "status": "completed"}), flush=True)
        phase = "freeze_checks"
        verify_bindings(root, snapshot, {**bindings, **helpers})
        original_input_checks(root, plan)
        runtime_hash_checks(python, lock, install, runtime)
        assembly = read_audit(output, "scores/policy_score_assembly_audit_absolute_price_error.json")
        compare_audit(assembly, read_audit(root, AUDITS["assembly"]))
        scores = safe_path(output, "scores/frozen_policy_scores_absolute_price_error.csv")
        c = plan["structural_contract"]
        checked = score_checks(scores, c["test_listing_count"], c["test_action_pair_count"])
        if sha256(scores) != c["historical_primary_score_sha256"] or sha256(scores) != assembly["output"]["sha256"]:
            raise ValueError("Reconstructed score bytes differ from historical hash")
        if table_keys(scores)[0] != table_keys(safe_path(output, "final-pairs/evaluation_prediction_pairs.csv"))[0]:
            raise ValueError("Score action universe differs from generated pairs")
        if (output / "evaluation-inputs").exists() or (output / "evaluation-mae").exists():
            raise ValueError("Forbidden outcome output created")
        artifacts = {}
        for path in sorted(output.rglob("*")):
            if path.is_symlink():
                raise ValueError("Generated symlink rejected")
            if path.is_file():
                artifacts[str(path.relative_to(output))] = sha256(path)
        models = {k: v for k, v in artifacts.items() if k.endswith(".cbm")}
        if len(models) != 18:
            raise ValueError("Exactly eighteen new saved models required")
        freeze = {"stage": "RECONSTRUCTED_AUTOSCOUT24_SCORES_FROZEN_NO_OUTCOME_EVALUATION",
                  "date": "2026-10-01", "replay_name": plan["replay_name"], "plan_sha256": EXPECTED_PLAN,
                  "study_and_recovery_source_sha256": bindings, "runner_and_helper_sha256": helpers,
                  "runtime_validation": runtime, "original_source_sha256_checked": plan["source_binding"]["sha256"],
                  "completed_scoring_stages": completed, "checkpoints": checkpoints,
                  "new_score_sha256": sha256(scores), "score_bytes_match_recorded_historical_hash": True,
                  "score_schema_check": checked, "new_saved_model_sha256": models, "artifact_sha256": artifacts,
                  "source_prices_processed_for_cohort_and_training": True, "new_target_rows_parsed": True,
                  "historical_target_rows_parsed": False, "historical_tests_already_opened": True,
                  "inherited_protocol_tags_are_not_new_confirmation": True, "outcome_join_called": False,
                  "test_performance_computed": False, "full_primary_table_concordance_verified": False,
                  "exact_historical_source_identity_claimed": False, "independent_replication_claimed": False,
                  "original_outputs_or_manuscript_changed": False, "rights_adjudicated": False,
                  "public_release_created": False, "submission_ready": False}
        write_new_json(output / "scoring-freeze.json", freeze)
        return freeze
    except Exception as error:
        write_new_json(output / "failed-scoring-replay.json", {
            "phase": phase, "exception_class": type(error).__name__, "reason": str(error),
            "completed_scoring_stages": completed, "scores_frozen": False,
            "outcome_join_called": False, "test_performance_computed": False,
            "partial_directory_retained_no_overwrite_or_automatic_retry": True})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "plan", "python", "requirements-lock", "install-report"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-lock-sha256", required=True)
    p.add_argument("--expected-install-report-sha256", required=True)
    p.add_argument("--execute-scoring", action="store_true")
    args = p.parse_args()
    try:
        root, python = args.project_root.resolve(strict=True), args.python.absolute()
        plan, data, runtime = preflight(root, args.plan, python, args.requirements_lock, args.install_report,
                                       args.expected_lock_sha256, args.expected_install_report_sha256)
        if args.execute_scoring:
            result = execute_scoring(root, plan, data, python, args.requirements_lock, args.install_report, runtime)
            print(json.dumps({"stage": result["stage"], "score_sha256": result["new_score_sha256"]}), flush=True)
        else:
            print(json.dumps({"stage": "AUTOSCOUT24_SCORING_RUNTIME_PREFLIGHT_ONLY", "runtime": runtime,
                              "study_training_started": False, "replay_output_created": False}, indent=2))
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        p.exit(1, f"AutoScout24 scoring replay failed: {error}\n")


if __name__ == "__main__":
    main()
