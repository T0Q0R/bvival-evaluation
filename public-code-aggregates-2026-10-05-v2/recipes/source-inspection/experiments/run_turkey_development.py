"""Guarded full development controller. No calibration/test label access.

Readiness and both environments are checked before target access. Failures are
preserved, not resumed silently. Reload checks never refit or reselect models.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np

from train_turkey_nested_valuation import _prediction_records, run_nested_valuation, validate_dataset
from turkey_development_io import bound_plan, export_development_prices, load_development_dataset, require_development_receipt, sha256
from turkey_execution_contract import ACTIONS, select_actions
from turkey_policy_components import allocate_policies, fit_policy_heads, post_mae, prepare_policy_material, score_head


RUNTIME_SOURCES = [
    "archive_turkey_candidate.py", "audit_turkey_feature_only.py", "build_turkey_price_free_manifest.py",
    "verify_turkey_price_free_bundle.py", "turkey_information_boundary.py", "turkey_execution_contract.py",
    "turkey_valuation_components.py", "turkey_price_cells.py", "turkey_development_io.py",
    "train_turkey_nested_valuation.py", "turkey_policy_components.py", "turkey_neural_policy.py",
    "turkey_neural_replay.py", "run_turkey_development.py", "freeze_turkey_development_readiness.py",
]


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, value):
    with path.open("x") as stream:
        json.dump(jsonable(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def file_hashes(directory):
    return {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def verify_files(directory, hashes, *, ignored=()):
    if not hashes:
        raise ValueError("Nonempty artifact bindings required")
    inventory = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if inventory != set(hashes) | set(ignored):
        raise ValueError("Saved artifact inventory changed")
    for name, expected in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or (directory / relative).is_symlink():
            raise ValueError("Unsafe artifact path")
        if sha256(directory / relative) != expected:
            raise ValueError("Saved artifact hash changed")


def probe_neural_environment(neural_python, expected):
    packages = list(expected["packages"])
    script = "import json,platform,importlib.metadata; observed={'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in " + repr(packages) + "}}; "
    if "platform" in expected:
        script += "observed['platform']=platform.platform(); "
    script += "print(json.dumps(observed))"
    result = subprocess.run([str(neural_python.absolute()), "-c", script], check=True, capture_output=True, text=True, timeout=30)
    if json.loads(result.stdout) != expected:
        raise ValueError("Required execution environment changed; stop before prices")


def require_controller_receipt(receipt, binding, *, project_root):
    require_development_receipt(receipt, binding, project_root=project_root)
    if (set(receipt["source_sha256"]) != set(RUNTIME_SOURCES)
            or receipt.get("full_controller_and_reload_integration_passed") is not True
            or receipt.get("calibration_or_test_release_allowed") is not False
            or receipt.get("runtime_environment") != binding["runtime_environment"]
            or receipt.get("neural_runtime_environment") != binding["neural_runtime_environment"]):
        raise ValueError("Complete controller/runtime/reload readiness required")


def verify_readiness_package(readiness, binding, *, project_root):
    receipt = json.loads(readiness.read_text())
    require_controller_receipt(receipt, binding, project_root=project_root)
    audit_path = readiness.parent / "readiness_package_audit.json"
    audit = json.loads(audit_path.read_text())
    if (audit.get("stage") != "LOCAL_DEVELOPMENT_READINESS_PACKAGE_COMPLETE"
            or audit.get("readiness_receipt_sha256") != sha256(readiness)
            or audit.get("source_price_values_parsed") != 0
            or audit.get("plan_sha256") != binding["plan_sha256"]):
        raise ValueError("Matching complete local readiness package required")
    verify_files(readiness.parent, audit["artifact_sha256"], ignored=[audit_path.name])
    return receipt


def allocations_from_scores(keys, heads, neural, plan):
    ensemble = allocate_policies(keys, heads["scores"]["risk"], heads["scores"]["value"], heads["scores"]["posterror"],
                                 neural["probability"], plan, fixed_field=heads["fixed_field"], global_field=heads["global_field"])
    per_seed = {str(seed): allocate_policies(keys, heads["per_seed_scores"]["risk"][str(seed)],
                                            heads["per_seed_scores"]["value"][str(seed)], heads["per_seed_scores"]["posterror"][str(seed)],
                                            neural["per_seed_probability"][str(seed)], plan,
                                            fixed_field=heads["fixed_field"], global_field=heads["global_field"], seed=seed)
                for seed in plan["seeds"]}
    for allocations in [ensemble, *per_seed.values()]:
        for policies in allocations.values():
            common = {k for k, _ in policies["risk_fixed_validation_best"]}
            if any({k for k, _ in policies[p]} != common for p in
                   ("risk_global_train_field", "risk_benefit_field", "risk_neural_field")):
                raise ValueError("Common risk listing cohort changed")
    return {"ensemble": ensemble, "per_seed": per_seed}


def prediction_records_match(actual, expected):
    # Parallel tree reduction may differ by a few floating-point ulps on replay.
    # Only log-prediction roundoff is tolerated; allocation equality stays exact.
    if set(actual) != set(expected):
        return False
    fields = {"before_prediction_log", "before_disagreement_log", "after_prediction_log", "per_seed_state_logs"}
    for key in actual:
        if set(actual[key]) != fields or set(expected[key]) != fields:
            return False
        for field in fields:
            first, second = np.asarray(actual[key][field], dtype=float), np.asarray(expected[key][field], dtype=float)
            if (first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all()
                    or not np.allclose(first, second, rtol=1e-12, atol=1e-12)):
                return False
    return True


def reload_check(output, data, plan, config, neural_python, *, project_root, hashes=None):
    """Verify bytes first; then replay all saved predictions and allocations."""
    if hashes is None:
        audit = json.loads((output / "development_execution_audit.json").read_text())
        if audit["stage"] not in {"SYNTHETIC_FULL_DEVELOPMENT_CONTROLLER_PASSED", "SOURCE_DEVELOPMENT_COMPLETED_NOT_TEST_FROZEN"}:
            raise ValueError("Only completed development packages can be replayed")
        hashes = audit["artifact_sha256"]
        verify_files(output, hashes, ignored=["development_execution_audit.json"])
    else:
        verify_files(output, hashes)
    validate_dataset(data, plan, config)
    saved = json.loads((output / "valuation_results.json").read_text())
    contexts = json.loads((output / "neural_task.json").read_text())["validation"]["contexts"]
    keys = sorted(r["record_key"] for r in data["rows"] if r["split"] == "validation")
    if json.loads((output / "neural_task.json").read_text())["validation"]["keys"] != keys:
        raise ValueError("Saved validation role mismatch")
    # All fifteen OOF models and three final models are replayed, not refitted.
    for outer in range(5):
        group_keys = sorted(r["record_key"] for r in data["rows"] if r["split"] == "train" and int(r["oof_fold"]) == outer)
        models = [joblib.load(output / "valuation_models" / f"outer_{outer}_seed_{s}.joblib") for s in plan["seeds"]]
        actual = _prediction_records(models, data, group_keys, config)
        if not prediction_records_match(actual, {key: saved["oof"][key] for key in group_keys}):
            raise ValueError("Reloaded OOF prediction mismatch")
    models = [joblib.load(output / "valuation_models" / f"final_seed_{s}.joblib") for s in plan["seeds"]]
    if not prediction_records_match(_prediction_records(models, data, keys, config), saved["validation"]):
        raise ValueError("Reloaded final valuation prediction mismatch")
    heads = json.loads((output / "head_results.json").read_text())
    for kind in ("risk", "value", "posterror"):
        predictions = []
        for seed in plan["seeds"]:
            model = joblib.load(output / "head_models" / f"{kind}_seed_{seed}.joblib")
            actual = score_head(model, contexts, plan, config, kind=kind, scale=heads["common_training_scale"])
            if not np.array_equal(actual, np.asarray(heads["per_seed_scores"][kind][str(seed)])):
                raise ValueError("Reloaded head prediction mismatch")
            predictions.append(actual)
        if not np.array_equal(np.mean(predictions, axis=0), np.asarray(heads["scores"][kind])):
            raise ValueError("Reloaded ensemble head mismatch")
    result = subprocess.run([str(neural_python.absolute()), str(project_root / "experiments/turkey_neural_replay.py"),
                             "--task", str((output / "neural_task.json").absolute()), "--model-dir", str((output / "neural").absolute())],
                            check=True, capture_output=True, text=True, timeout=120)
    neural = json.loads(result.stdout)
    actual_allocations = jsonable(allocations_from_scores(keys, heads, neural, plan))
    if actual_allocations != json.loads((output / "validation_allocations.json").read_text()):
        raise ValueError("Reloaded allocation mismatch")
    return {"valuation_models_reloaded": 18, "head_models_reloaded": 9, "neural_models_reloaded": 3,
            "all_saved_predictions_and_allocations_match": True, "valuation_log_rtol": 1e-12, "valuation_log_atol": 1e-12,
            "head_and_neural_predictions_exact": True, "allocation_equality_exact": True, "refits": 0, "reselection": False}


def execute_pipeline(data, plan, config, binding, output, neural_python, *, project_root, evidence_status, provenance):
    if evidence_status not in {"synthetic_only", "authorized_development_only"}:
        raise ValueError("No calibration/test execution modes")
    if output.exists() and any(output.iterdir()):
        raise ValueError("New empty execution directory required; preserve previous attempts")
    validate_dataset(data, plan, config)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    started, phase = time.perf_counter(), "created"

    def checkpoint(name):
        nonlocal phase
        phase = name
        save_json(output / f"checkpoint_{name}.json", {"phase": name, "evidence_status": evidence_status,
                                                      "test_prices_read": False, "provenance": provenance})
        print("development phase: " + name, flush=True)

    try:
        for name in RUNTIME_SOURCES:
            with (output / (name + ".snapshot")).open("xb") as stream:
                stream.write((project_root / "experiments" / name).read_bytes())
        save_json(output / "execution_config.json", plan)
        save_json(output / "cohort_config.json", config)
        checkpoint("valuation_started")
        base_dir, head_dir = output / "valuation_models", output / "head_models"
        base_dir.mkdir(mode=0o700)
        head_dir.mkdir(mode=0o700)
        valuation = run_nested_valuation(data, plan, config, model_directory=base_dir)
        save_json(output / "valuation_results.json", valuation)
        checkpoint("heads_started")
        train, validation = prepare_policy_material(data, valuation, plan, config)
        heads = fit_policy_heads(train, validation, plan, config, model_directory=head_dir)
        save_json(output / "head_results.json", {k: v for k, v in heads.items() if k != "models"})
        common = select_actions(validation["keys"], ACTIONS, np.repeat(heads["scores"]["risk"][:, None], 3, axis=1),
                                plan["primary_budget"], fixed_action=heads["fixed_field"])
        task = {"stage": evidence_status, "scope": ["train", "validation"], "train": train, "validation": validation,
                "plan": plan, "config": config, "cohort": [k for k, _ in common],
                "expected_environment": binding["neural_runtime_environment"]}
        save_json(output / "neural_task.json", task)
        checkpoint("neural_started")
        subprocess.run([str(neural_python.absolute()), str(project_root / "experiments/turkey_neural_policy.py"),
                        "--task", str((output / "neural_task.json").absolute()), "--output-dir", str((output / "neural").absolute())],
                       check=True, capture_output=True, text=True, timeout=1200)
        neural = json.loads((output / "neural/neural_audit.json").read_text())
        if (neural["network_fit_count"] != 6 or neural["risk_cohort_keys"] != task["cohort"]
                or neural["runtime_environment"] != binding["neural_runtime_environment"]
                or any(a["validation_labels_used"] for a in neural["final_fit_audit"])):
            raise ValueError("Required neural comparator incomplete or wrong scope")
        if valuation["fit_count"] != 258 or len(heads["fit_audit"]) != 21:
            raise ValueError("Incomplete frozen fitting schedule")
        checkpoint("allocations_started")
        allocations = allocations_from_scores(validation["keys"], heads, neural, plan)
        save_json(output / "validation_allocations.json", allocations)
        save_json(output / "validation_selection_losses.json", {
            "selection_not_independent_replication": True,
            "post_mae": {b: {p: post_mae(validation, a) for p, a in policies.items()}
                         for b, policies in allocations["ensemble"].items()}})
        checkpoint("reload_started")
        replay = reload_check(output, data, plan, config, neural_python, project_root=project_root, hashes=file_hashes(output))
        checkpoint("completed")
        audit = {"stage": "SYNTHETIC_FULL_DEVELOPMENT_CONTROLLER_PASSED" if evidence_status == "synthetic_only"
                 else "SOURCE_DEVELOPMENT_COMPLETED_NOT_TEST_FROZEN", "evidence_status": evidence_status,
                 "provenance": provenance, "plan_sha256": binding["plan_sha256"], "cohort_audit_sha256": binding["cohort_audit_sha256"],
                 "runtime_environment": binding["runtime_environment"], "neural_runtime_environment": binding["neural_runtime_environment"],
                 "valuation_fits": 258, "head_fits": 21, "neural_fits": 6, "reload": replay,
                 "elapsed_seconds": time.perf_counter() - started, "source_prices_parsed_this_pipeline": 0,
                 "targets_loaded_from_authorized_release": evidence_status == "authorized_development_only",
                 "calibration_or_test_labels_read": False, "test_scores_or_actions_frozen": False,
                 "official_GDFS_reproduction": False, "validation_is_selection_not_replication": True,
                 "artifact_sha256": file_hashes(output)}
        save_json(output / "development_execution_audit.json", audit)
    except Exception as error:
        save_json(output / "failed_development_execution.json", {
            "stage": "FAILED_DEVELOPMENT_EXECUTION", "last_phase": phase, "exception_class": type(error).__name__,
            "evidence_status": evidence_status, "provenance": provenance, "calibration_or_test_labels_read": False,
            "attempt_artifact_sha256": file_hashes(output), "no_silent_resume_or_comparator_deletion": True})
        raise
    finally:
        for path in output.rglob("*"):
            if path.is_file():
                path.chmod(0o400)
    return audit


def run_authorized(source, labels, plan_freeze, readiness, output, neural_python, *, project_root):
    if (source is None) == (labels is None):
        raise ValueError("Exactly one raw-source export or existing development label release required")
    if output.exists():
        raise ValueError("New nonexistent development attempt directory required")
    plan, config, cohort, binding = bound_plan(plan_freeze, project_root=project_root)
    verify_readiness_package(readiness, binding, project_root=project_root)
    probe_neural_environment(neural_python, binding["neural_runtime_environment"])
    # All source/code/environment checks above precede target access.
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    phase = "before_label_access"
    try:
        with (output / "readiness_snapshot.json").open("xb") as stream:
            stream.write(readiness.read_bytes())
        if source is not None:
            phase, labels = "development_export_started", output / "labels"
            export_development_prices(source, plan_freeze, readiness, labels, project_root=project_root)
        phase = "development_label_load_started"
        label_audit = json.loads((labels / "development_label_release_audit.json").read_text())
        if (label_audit["plan_sha256"] != binding["plan_sha256"]
                or label_audit["readiness_receipt_sha256"] != sha256(readiness)):
            raise ValueError("Development label release bound to a different readiness/plan")
        data = load_development_dataset(cohort, labels, config)
        phase = "development_training_started"
        result = execute_pipeline(data, plan, config, binding, output / "training", neural_python,
                                  project_root=project_root, evidence_status="authorized_development_only",
                                  provenance={"development_label_audit_sha256": sha256(labels / "development_label_release_audit.json"),
                                              "readiness_sha256": sha256(readiness), "labels_already_released": True})
        save_json(output / "source_attempt_audit.json", {"stage": result["stage"], "calibration_or_test_labels_read": False,
                                                        "label_release_audit_sha256": sha256(labels / "development_label_release_audit.json"),
                                                        "artifact_sha256": file_hashes(output)})
    except Exception as error:
        # Price access may have begun: do not misleadingly claim zero source reads.
        save_json(output / "failed_source_attempt.json", {"stage": "FAILED_SOURCE_DEVELOPMENT_ATTEMPT", "last_phase": phase,
                                                          "exception_class": type(error).__name__, "price_access_may_have_started": phase != "before_label_access",
                                                          "calibration_or_test_labels_read": False, "attempt_artifact_sha256": file_hashes(output)})
        raise
    finally:
        for path in output.rglob("*"):
            if path.is_file():
                path.chmod(0o400)
    return result


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path)
    group.add_argument("--development-labels", type=Path)
    parser.add_argument("--plan-freeze", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neural-python", type=Path, required=True)
    args = parser.parse_args()
    result = run_authorized(args.source, args.development_labels, args.plan_freeze, args.readiness, args.output_dir,
                            args.neural_python, project_root=Path(__file__).resolve().parents[1])
    print(json.dumps({"stage": result["stage"], "calibration_or_test_labels_read": False}))


if __name__ == "__main__":
    main()
