"""Saved-model pre-action scoring only, not a held-out release/freeze receipt.

Inputs contain record keys and the exact initial-field allowlist, no prices,
validity, hidden acquisition values or after predictions. No refits/reselection.
Only locally produced completed packages may be deserialized after hash checks.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import numpy as np

from run_turkey_development import (RUNTIME_SOURCES, allocations_from_scores, jsonable,
                                   probe_neural_environment, save_json, verify_files)
from turkey_development_io import sha256
from turkey_execution_contract import ACTIONS, validate_plan
from turkey_information_boundary import initial_context, policy_context, valuation_state
from turkey_policy_components import score_head
from turkey_valuation_components import ensemble_prediction


def initial_records(records, config):
    """Validate all input roles before calling any model; canonical key order."""
    if not isinstance(records, list) or not records:
        raise ValueError("Nonempty list of initial-only records required")
    seen, values = set(), {}
    for row in records:
        if not isinstance(row, dict) or set(row) != {"record_key", "initial"}:
            raise ValueError("Exact record_key/initial input roles required")
        key = row["record_key"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Unique nonempty string record keys required")
        initial = initial_context(row["initial"], config)
        if any(not isinstance(value, str) for value in initial.values()):
            raise ValueError("Normalized string initial fields required")
        seen.add(key)
        values[key] = initial
    keys = sorted(values)
    return keys, [values[key] for key in keys]


def before_contexts(records, models, plan, config):
    validate_plan(plan, config)
    keys, initial = initial_records(records, config)
    if set(models) != {str(seed) for seed in plan["seeds"]}:
        raise ValueError("Exactly three final saved valuation models required")
    states = [valuation_state(row, config) for row in initial]
    logs = np.asarray([models[str(seed)].predict(states) for seed in plan["seeds"]], dtype=float)
    if logs.shape != (3, len(keys)):
        raise ValueError("Aligned three-seed all-hidden prediction vectors required")
    summary = ensemble_prediction(logs)
    contexts = [policy_context(row, config, before_prediction_log=float(summary["prediction_log"][i]),
                               before_disagreement_log=float(summary["disagreement_log"][i]))
                for i, row in enumerate(initial)]
    return {"keys": keys, "contexts": contexts,
            "before_prediction_log": summary["prediction_log"],
            "before_disagreement_log": summary["disagreement_log"],
            "per_seed_before_raw_log": {str(seed): logs[i] for i, seed in enumerate(plan["seeds"])}}


def score_saved_heads(contexts, models, selection, plan, config):
    expected = {(kind, str(seed)) for kind in ("risk", "value", "posterror") for seed in plan["seeds"]}
    if set(models) != expected or set(selection) != {"common_training_scale", "fixed_field", "global_field"}:
        raise ValueError("Complete saved heads and unchanged selection metadata required")
    if selection["fixed_field"] not in ACTIONS or selection["global_field"] not in ACTIONS:
        raise ValueError("Declared saved fields required; do not reselect on scoring cohort")
    per_seed = {kind: {str(seed): score_head(models[(kind, str(seed))], contexts, plan, config,
                                           kind=kind, scale=selection["common_training_scale"])
                        for seed in plan["seeds"]} for kind in ("risk", "value", "posterror")}
    return {**selection, "per_seed_scores": per_seed,
            "scores": {kind: np.mean(list(predictions.values()), axis=0) for kind, predictions in per_seed.items()}}


def verify_completed_package(directory, *, project_root):
    audit_path = directory / "development_execution_audit.json"
    if not audit_path.is_file():
        raise ValueError("Completed development package required; never load partial models")
    audit = json.loads(audit_path.read_text())
    stages = {"SYNTHETIC_FULL_DEVELOPMENT_CONTROLLER_PASSED": "synthetic_only",
              "SOURCE_DEVELOPMENT_COMPLETED_NOT_TEST_FROZEN": "authorized_development_only"}
    if (audit.get("stage") not in stages or audit.get("evidence_status") != stages[audit["stage"]]
            or audit.get("calibration_or_test_labels_read") is not False
            or audit.get("validation_is_selection_not_replication") is not True
            or [audit.get(k) for k in ("valuation_fits", "head_fits", "neural_fits")] != [258, 21, 6]
            or audit.get("reload", {}).get("all_saved_predictions_and_allocations_match") is not True
            or audit.get("reload", {}).get("refits") != 0):
        raise ValueError("Complete fitting/reload and development-only provenance required")
    verify_files(directory, audit["artifact_sha256"], ignored=[audit_path.name])
    for name in RUNTIME_SOURCES:
        if (directory / (name + ".snapshot")).read_bytes() != (project_root / "experiments" / name).read_bytes():
            raise ValueError("Saved-model dependency differs from current runtime")
    plan = json.loads((directory / "execution_config.json").read_text())
    config = json.loads((directory / "cohort_config.json").read_text())
    validate_plan(plan, config)
    return audit, plan, config


def score_completed_models(directory, records, output, neural_python, *, project_root):
    """New private inference package. Caller must separately bind split universe.

    This function never reads a workbook/price CSV, selects fields or evaluates
    outcomes. Its output alone does NOT authorize cal/test price access.
    """
    if output.exists():
        raise ValueError("New nonexistent scoring directory required")
    audit, plan, config = verify_completed_package(directory, project_root=project_root)
    initial_records(records, config)
    # Version checks precede deserialization in either environment.
    import importlib.metadata
    import platform
    observed = {"python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in audit["runtime_environment"]["packages"]}}
    expected = audit["runtime_environment"]
    if "platform" in expected:
        observed["platform"] = platform.platform()
    if observed != expected:
        raise ValueError("Saved-model main runtime changed")
    probe_neural_environment(neural_python, audit["neural_runtime_environment"])
    output.mkdir(parents=True, mode=0o700)
    phase = "model_load_started"
    try:
        for name in ("turkey_price_free_scoring.py", "turkey_price_free_neural_scoring.py"):
            with (output / (name + ".snapshot")).open("xb") as stream:
                stream.write((project_root / "experiments" / name).read_bytes())
        base = {str(seed): joblib.load(directory / "valuation_models" / f"final_seed_{seed}.joblib") for seed in plan["seeds"]}
        before = before_contexts(records, base, plan, config)
        saved = json.loads((directory / "head_results.json").read_text())
        selection = {name: saved[name] for name in ("common_training_scale", "fixed_field", "global_field")}
        models = {(kind, str(seed)): joblib.load(directory / "head_models" / f"{kind}_seed_{seed}.joblib")
                  for kind in ("risk", "value", "posterror") for seed in plan["seeds"]}
        heads = score_saved_heads(before["contexts"], models, selection, plan, config)
        phase = "neural_inference_started"
        task = {"stage": "INITIAL_ONLY_INFERENCE_NO_LABEL_RELEASE", "contexts": before["contexts"],
                "plan": plan, "config": config, "development_audit_sha256": sha256(directory / "development_execution_audit.json")}
        save_json(output / "initial_only_neural_task.json", task)
        child = subprocess.run([str(neural_python.absolute()), str(project_root / "experiments/turkey_price_free_neural_scoring.py"),
                                "--task", str((output / "initial_only_neural_task.json").absolute()),
                                "--development-run", str(directory.absolute())],
                               check=True, capture_output=True, text=True, timeout=120)
        neural = json.loads(child.stdout)
        allocations = allocations_from_scores(before["keys"], heads, neural, plan)
        result = jsonable({"before": before, "heads": heads, "neural": neural, "allocations": allocations,
                           "input_evidence_status": audit["evidence_status"], "refits": 0, "reselection": False,
                           "calibration_or_test_labels_read": False, "heldout_label_release_allowed": False,
                           "split_universe_not_bound_by_this_component": True})
        save_json(output / "initial_only_scores.json", result)
        save_json(output / "initial_only_scoring_audit.json", {
            "stage": "INITIAL_ONLY_SAVED_MODEL_INFERENCE_NOT_HELDOUT_FREEZE_RECEIPT",
            "input_evidence_status": audit["evidence_status"], "development_audit_sha256": task["development_audit_sha256"],
            "record_count": len(before["keys"]), "refits": 0, "reselection": False,
            "calibration_or_test_labels_read": False, "heldout_label_release_allowed": False,
            "artifact_sha256": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}})
        return result
    except Exception as error:
        save_json(output / "failed_initial_only_scoring.json", {"phase": phase, "exception_class": type(error).__name__,
                                                                "heldout_label_release_allowed": False,
                                                                "calibration_or_test_labels_read": False})
        raise
    finally:
        for path in output.rglob("*"):
            if path.is_file():
                path.chmod(0o400)
