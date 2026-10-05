"""Companion to completed portable development: scoring, then historical replay.

Never changes the development recipe, original ledger or old release authority.
Only the historical calibration/test partitions are eligible, after score and
complete before/after prediction replay. No new policy, budget or seed is added.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_source_bundle import safe_file
from check_turkey_portable_outcomes import verify
from run_turkey_portable_development import (
    LAUNCHER, bound_workspace, completed_phase, run_phase, validate_fit, validate_score,
    verify_hashes, write_new,
)
from turkey_portable_development_contract import COHORT, DEVELOPMENT, PLAN, SOURCE_SHA, sha
from turkey_portable_outcome_contract import (
    DEVELOPMENT_MANIFEST_SHA, DEVELOPMENT_PACKAGE, MISSING_SNAPSHOT, SCOPE, VERSION,
)


def new_output(path, package, workspace):
    path = Path(path).absolute()
    if (path.exists() or path.is_symlink() or ".." in path.parts
            or any(parent.is_symlink() for parent in path.parents)
            or path.is_relative_to(package) or path.is_relative_to(workspace)):
        raise ValueError("New separate non-symlink private output required")
    return path


def preflight(package, workspace, output):
    package = Path(package).resolve(strict=True)
    verify(package)
    workspace, prepared = bound_workspace(workspace)
    if sha(safe_file(workspace, "manifest.json")) != DEVELOPMENT_MANIFEST_SHA:
        raise ValueError("Exact v2 development package binding required")
    completed_phase(workspace, "readiness")
    fit = completed_phase(workspace, "fit")
    validate_fit(json.loads(safe_file(workspace, DEVELOPMENT + "/training/development_execution_audit.json").read_text()))
    for name in ("failed-portable-preparation.json", "failed-portable-readiness.json", "failed-portable-fit.json"):
        if (workspace / name).exists():
            raise ValueError("Failed original development attempt cannot become a successful input")
    output = new_output(output, package, workspace)
    return package, workspace, output, prepared, fit


def prepare(package, workspace, output, execute=False):
    package, workspace, output, prepared, fit = preflight(package, workspace, output)
    if not execute:
        return {"stage": "HISTORICAL_COMPANION_PREFLIGHT_ONLY", "output_created": False,
                "source_cells_parsed": False, "new_independent_confirmation": False}
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    try:
        hashes = {}
        # Projection of freshly source-rebuilt features/code only. No old study
        # rows, development target CSV or fitted models are copied into this tree.
        for name, expected in prepared["protected_sha256"].items():
            source = safe_file(workspace, name)
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            hashes[name] = expected
        companion_manifest = json.loads(safe_file(package, "manifest.json").read_text())
        for record in companion_manifest["files"]:
            name = record["file"]
            if name.startswith(DEVELOPMENT_PACKAGE + "/"):
                continue  # Original package retained inside the public companion.
            if name in {"README.md", "LICENSE", "LICENSING_SCOPE.md", "THIRD_PARTY_NOTICES.md"}:
                continue  # Do not replace the byte-preserved development docs.
            source = safe_file(package, name)
            destination = output / name
            if destination.exists():
                if sha(safe_file(output, name)) != record["sha256"]:
                    raise ValueError("Companion cannot replace different protected development code")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            hashes[name] = record["sha256"]
        verify_hashes(output, hashes)
        if MISSING_SNAPSHOT not in hashes:
            raise ValueError("Original path-read test snapshot must be explicit")
        receipt = {"version": VERSION, "scope": SCOPE, "stage": "HISTORICAL_COMPANION_PREPARED_NO_SCORES_OR_PRICES",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "output": str(output),
            "development_workspace": str(workspace), "companion_package": str(package),
            "companion_manifest_sha256": sha(package / "manifest.json"),
            "development_preparation_receipt_sha256": sha(workspace / "portable-preparation-receipt.json"),
            "development_fit_receipt_sha256": sha(workspace / "portable-fit-receipt.json"),
            "source_sha256": SOURCE_SHA, "protected_sha256": hashes,
            "whole_source_features_rebuilt_in_previous_bound_stage": True,
            "original_study_caches_or_models_required": False,
            "new_reconstruction_models_used_only_after_hash_checks": True,
            "original_v2_workspace_unmodified": True,
            "heldout_price_values_decoded": 0, "original_once_release_authority_reissued": False,
            "historical_tests_already_opened": True, "new_independent_confirmation": False}
        write_new(output / "historical-companion-preparation.json", receipt)
        return {k: v for k, v in receipt.items() if k != "protected_sha256"}
    except BaseException as error:
        write_new(output / "failed-historical-companion-preparation.json", {
            "exception_class": type(error).__name__, "reason": str(error),
            "failure_retained": True, "heldout_prices_decoded_here": False})
        raise


def bound_output(output):
    output = Path(output).absolute()
    if output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise ValueError("Non-symlink bound companion required")
    receipt = json.loads(safe_file(output, "historical-companion-preparation.json").read_text())
    if (receipt.get("version") != VERSION or receipt.get("scope") != SCOPE
            or receipt.get("output") != str(output) or receipt.get("source_sha256") != SOURCE_SHA
            or receipt.get("original_once_release_authority_reissued") is not False
            or receipt.get("historical_tests_already_opened") is not True
            or receipt.get("new_independent_confirmation") is not False):
        raise ValueError("Separate historical companion binding required")
    verify_hashes(output, receipt["protected_sha256"])
    workspace, prepared = bound_workspace(receipt["development_workspace"])
    if (receipt["development_preparation_receipt_sha256"] != sha(workspace / "portable-preparation-receipt.json")
            or receipt["development_fit_receipt_sha256"] != sha(workspace / "portable-fit-receipt.json")):
        raise ValueError("Original reconstructed development receipts changed")
    completed_phase(workspace, "readiness")
    completed_phase(workspace, "fit")
    validate_fit(json.loads(safe_file(workspace, DEVELOPMENT + "/training/development_execution_audit.json").read_text()))
    return output, receipt, workspace, prepared


def completed_score(output):
    value = json.loads(safe_file(output, "historical-score-receipt.json").read_text())
    if (value.get("version") != VERSION or value.get("phase") != "score"
            or value.get("stage") != "HISTORICAL_COMPANION_PHASE_COMPLETED"
            or value.get("heldout_price_values_decoded") != 0
            or value.get("new_independent_confirmation") is not False
            or value.get("preparation_sha256") != sha(output / "historical-companion-preparation.json")
            or not value.get("artifact_sha256")
            or (output / "failed-historical-score.json").exists()):
        raise ValueError("Completed price-free companion score required")
    verify_hashes(output, value["artifact_sha256"])
    validate_score(json.loads(safe_file(output, "score-freeze/heldout_score_freeze_audit.json").read_text()))
    return value


def compare_aggregates(reference, actual, path="root"):
    """Fixed structural comparison, original 1e-12 roundoff bound, no tuning."""
    if isinstance(reference, bool) or reference is None or isinstance(reference, str):
        if type(reference) is not type(actual) or reference != actual:
            raise ValueError("Historical aggregate differs: " + path)
        return 1
    if isinstance(reference, (int, float)):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isfinite(reference) or not math.isfinite(actual)
                or isinstance(reference, int) and (type(actual) is not int or reference != actual)
                or not math.isclose(reference, actual, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("Historical numerical aggregate differs: " + path)
        return 1
    if isinstance(reference, list):
        if not isinstance(actual, list) or len(reference) != len(actual):
            raise ValueError("Historical list scope differs: " + path)
        return sum(compare_aggregates(a, b, path + "/" + str(i))
                   for i, (a, b) in enumerate(zip(reference, actual)))
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or set(reference) != set(actual):
            raise ValueError("Historical aggregate key universe differs: " + path)
        return sum(compare_aggregates(reference[k], actual[k], path + "/" + str(k)) for k in reference)
    raise ValueError("Unsupported aggregate type: " + path)


def require_worker_entry(output, phase):
    started = json.loads(safe_file(output, "historical-" + phase + "-started.json").read_text())
    if (started.get("version") != VERSION or started.get("phase") != phase
            or started.get("scope") != SCOPE or started.get("new_independent_confirmation") is not False
            or started.get("preparation_sha256") != sha(output / "historical-companion-preparation.json")
            or any((output / name).exists() for name in (
                "historical-" + phase + "-receipt.json", "failed-historical-" + phase + ".json"))):
        raise ValueError("Bound new in-progress parent phase required, no worker restart")
    outputs = (["score-freeze", "score-worker-result.json"] if phase == "score" else
               ["historical-prediction-freeze.json", "calibration-complete-predictions.json",
                "test-complete-predictions.json", "outcomes-worker-result.json"])
    if any((output / name).exists() for name in outputs):
        raise ValueError("Partial or completed worker outputs cannot be resumed")


def score_worker(output):
    output, receipt, workspace, prepared = bound_output(output)
    require_worker_entry(output, "score")
    from freeze_turkey_heldout_scores import run_source_freeze
    result = run_source_freeze(workspace / DEVELOPMENT / "training", output / PLAN,
        output / "score-freeze", Path(prepared["neural_python"]), project_root=output)
    validate_score(result)
    write_new(output / "score-worker-result.json", {"whole_partitions": result["partition_counts"],
        "source_workbook_opened_here": False, "heldout_price_values_decoded": 0,
        "refits": 0, "reselection": False})


def outcomes_worker(output):
    output, receipt, workspace, prepared = bound_output(output)
    completed_score(output)
    require_worker_entry(output, "outcomes")
    import joblib
    import zipfile
    import audit_turkey_feature_only as reader
    from evaluate_turkey_released_phase import phase_candidate_values, verify_prediction_replay
    from run_turkey_development import jsonable
    from train_turkey_nested_valuation import _prediction_records
    from turkey_frozen_outcome_evaluation import evaluate_frozen_outcomes
    from turkey_price_access_accounting_v2 import read_selected_prices
    from turkey_source_access_binding import bind_original_source_coordinates
    from verify_turkey_heldout_score_freeze import verify_score_freeze

    development = workspace / DEVELOPMENT / "training"
    frozen = verify_score_freeze(output / "score-freeze", development,
        project_root=output, plan_directory=output / PLAN)
    coordinates = bind_original_source_coordinates(output / PLAN, project_root=output)
    all_keys = [r["record_key"] for r in coordinates["manifest"]]
    # Complete valuation predictions for BOTH phases must be frozen before any
    # selected price cell is parsed. These values never reach field selection.
    predictions, ordinals = {}, {}
    for phase in ("calibration", "test"):
        score, material = frozen["scores"][phase], frozen["partitions"][phase]
        keys = score["before"]["keys"]
        mapping = {r["source_data_ordinal"]: r["record_key"] for r in coordinates["manifest"] if r["split"] == phase}
        if sorted(mapping.values()) != keys or sorted(material["groups"]) != keys:
            raise ValueError("Complete source coordinate phase, not selected actions, required")
        initial = {r["record_key"]: r["initial"] for r in material["records"]}
        candidates = phase_candidate_values(output / COHORT, frozen["config"], all_keys, keys)
        replay = []
        for _ in range(2):
            models = [joblib.load(development / "valuation_models" / f"final_seed_{seed}.joblib")
                      for seed in frozen["plan"]["seeds"]]
            replay.append(_prediction_records(models, {"initial": initial, "actions": candidates}, keys, frozen["config"]))
        verify_prediction_replay(*replay, keys, score["before"], frozen["plan"]["seeds"])
        predictions[phase], ordinals[phase] = replay[0], mapping
        write_new(output / (phase + "-complete-predictions.json"), replay[0])
    prediction_hashes = {phase + "-complete-predictions.json": sha(output / (phase + "-complete-predictions.json"))
                         for phase in predictions}
    write_new(output / "historical-prediction-freeze.json", {
        "stage": "ALL_HISTORICAL_PHASE_PREDICTIONS_FROZEN_BEFORE_PRICE_RECONSTRUCTION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "artifact_sha256": prediction_hashes,
        "score_receipt_sha256": sha(output / "historical-score-receipt.json"),
        "both_entire_phase_predictions_reloaded": True, "refits": 0, "reselection": False,
        "source_cells_parsed_here": False, "new_independent_confirmation": False})
    bound_output(output)
    completed_score(output)
    verify_hashes(output, prediction_hashes)
    comparisons, access_receipts = {}, {}
    for phase in ("calibration", "test"):
        mapping = ordinals[phase]
        # This is a separately disclosed HISTORICAL reread. No canonical source
        # ledger/controller is reset, copied, called or claimed as new authority.
        with zipfile.ZipFile(prepared["source"]) as archive:
            header = reader.read_schema(archive, reader.sheet_path(archive))
            if set(header.values()) != set(frozen["config"]["expected_headers"]):
                raise ValueError("Original complete workbook schema required")
            selected, access = read_selected_prices(archive, header, set(mapping), expected_data_rows=52051)
        labels = {mapping[index]: value for index, value in selected.items()}
        keys = frozen["scores"][phase]["before"]["keys"]
        pure = {k: {f: predictions[phase][k][f] for f in ("before_prediction_log", "after_prediction_log")} for k in keys}
        outcome = jsonable(evaluate_frozen_outcomes(keys, frozen["partitions"][phase]["groups"],
            labels, pure, frozen["scores"][phase]["allocations"], frozen["plan"], frozen["config"],
            split=phase, evidence_status="authorized_heldout_only"))
        outcome = json.loads(json.dumps(outcome, allow_nan=False))  # Normalize integer JSON keys only.
        reference = json.loads(safe_file(output, "aggregates/" + phase + "-fixed-evaluation.json").read_text())
        write_new(output / (phase + "-reconstructed-fixed-evaluation.json"), outcome)
        comparisons[phase] = {"verified_scalar_leaves": compare_aggregates(reference, outcome),
            "all_original_policies_budgets_seeds_retained": True, "rtol": 1e-12, "atol": 1e-12}
        access_receipts[phase] = access
    bound_output(output)
    completed_score(output)
    verify_hashes(output, prediction_hashes)
    verify_score_freeze(output / "score-freeze", development, project_root=output, plan_directory=output / PLAN)
    write_new(output / "outcomes-worker-result.json", {
        "concordance": comparisons, "source_cell_access": access_receipts,
        "prices_reconstructed_from_pinned_source_not_old_cache": True,
        "both_predictions_frozen_before_any_price_parse": True, "original_once_release_controller_invoked": False,
        "original_ledger_read_or_written": False, "refits": 0, "reselection": False,
        "historical_tests_already_opened": True, "new_independent_confirmation": False})


def execute_phase(output, phase):
    if phase not in {"score", "outcomes"}:
        raise ValueError("Only companion score/historical outcomes phases exist")
    output, receipt, workspace, prepared = bound_output(output)
    if any((output / name).exists() for name in (
        "historical-" + phase + "-started.json", "historical-" + phase + "-receipt.json",
        "failed-historical-" + phase + ".json")):
        raise ValueError("Existing companion attempt cannot be resumed or overwritten")
    if phase == "outcomes":
        completed_score(output)
    started = {"version": VERSION, "phase": phase, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preparation_sha256": sha(output / "historical-companion-preparation.json"),
        "scope": SCOPE, "source_historical_prices_reread_in_this_phase": phase == "outcomes",
        "original_once_release_authority_reissued": False, "new_independent_confirmation": False}
    write_new(output / ("historical-" + phase + "-started.json"), started)
    try:
        command = [prepared["main_python"], "-I", "-B", "-c", LAUNCHER,
            str(output / "release-bvival/run_turkey_portable_outcomes.py"),
            "--output", str(output), "--worker", phase]
        execution = run_phase(command, output, output / (phase + "-worker-log.txt"))
        bound_output(output)
        result = json.loads(safe_file(output, phase + "-worker-result.json").read_text())
        if phase == "score":
            validate_score(json.loads(safe_file(output, "score-freeze/heldout_score_freeze_audit.json").read_text()))
        elif set(result.get("concordance", {})) != {"calibration", "test"}:
            raise ValueError("Both historical phase reconstructions required")
        generated = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*"))
                     if p.is_file() and p.relative_to(output).as_posix() not in receipt["protected_sha256"]
                     and p.name not in {"historical-companion-preparation.json"}}
        final = {**started, "stage": "HISTORICAL_COMPANION_PHASE_COMPLETED",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(), "execution": execution,
            "artifact_sha256": generated, "result": result,
            "heldout_price_values_decoded": 0 if phase == "score" else 15616,
            "original_v2_workspace_unmodified": True, "private_outputs_not_public_payload": True,
            "independent_machine_or_fresh_installation": False,
            "full_study_empirical_pipeline_verified_here": False}
        write_new(output / ("historical-" + phase + "-receipt.json"), final)
        return {k: v for k, v in final.items() if k != "artifact_sha256"}
    except BaseException as error:
        write_new(output / ("failed-historical-" + phase + ".json"), {
            "phase": phase, "exception_class": type(error).__name__, "reason": str(error),
            "attempt_retained": True, "automatic_retry_or_threshold_widening": False,
            "historical_prices_may_have_been_parsed": phase == "outcomes",
            "new_independent_confirmation": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--development-workspace", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--score", action="store_true")
    mode.add_argument("--outcomes", action="store_true")
    mode.add_argument("--worker", choices=["score", "outcomes"])
    args = parser.parse_args()
    if args.score or args.outcomes or args.worker:
        if args.package_root is not None or args.development_workspace is not None:
            parser.error("Bound phases cannot replace package/development/source/runtime inputs")
        if args.worker:
            # Exact script/hash binding is checked again by the bound output.
            (score_worker if args.worker == "score" else outcomes_worker)(args.output)
            result = {"stage": "HISTORICAL_COMPANION_WORKER_RETURNED"}
        else:
            result = execute_phase(args.output, "score" if args.score else "outcomes")
    else:
        if args.package_root is None or args.development_workspace is None:
            parser.error("Preflight/preparation requires package and completed development workspace")
        result = prepare(args.package_root, args.development_workspace, args.output, args.prepare)
    print(json.dumps(result, indent=2))
