"""Separate source-directed development recipe; held-out outcomes forbidden.

No original project root, cached real rows/models or once-release ledger is an
input. Preserve historical metadata separately from newly produced receipts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_source_bundle import safe_file
from check_turkey_portable_development import verify
from turkey_portable_development_contract import (
    COHORT, COHORT_REFERENCE, DEVELOPMENT, FEATURE_CONFIG, FEATURE_PROTOCOL,
    PLAN, READINESS, SCORES, SOURCE_SHA, VERSION, sha,
)

# Study subprocesses use only the copied tree's explicit local module paths.
# This is not an operating-system sandbox against malicious trusted code.
LAUNCHER = (
    "import pathlib,runpy,sys; p=pathlib.Path(sys.argv[1]); "
    "sys.path[:0]=[str(p.parent),str(p.parents[1]/'experiments'),"
    "str(p.parents[1]/'release-bvival')]; sys.argv=sys.argv[1:]; "
    "runpy.run_path(str(p),run_name='__main__')"
)


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def new_workspace(path, package):
    path = Path(path).absolute()
    if (path.exists() or path.is_symlink() or ".." in path.parts
            or any(parent.is_symlink() for parent in path.parents)
            or path.is_relative_to(package)):
        raise ValueError("New non-symlink workspace outside package required")
    return path


def source_file(path):
    path = Path(path).absolute()
    if (not path.is_file() or path.is_symlink()
            or any(parent.is_symlink() for parent in path.parents)
            or sha(path) != SOURCE_SHA):
        raise ValueError("Original non-symlink pinned source file required")
    return path


def runtime_probe(python, expected):
    python = Path(python).absolute()  # Keep venv identity; do not resolve symlink.
    if not python.is_file():
        raise ValueError("Existing role-specific interpreter required")
    code = ("import json,platform,importlib.metadata as m; expected=" + repr(expected)
            + "; actual={'python':platform.python_version(),'packages':"
            + "{p:m.version(p) for p in expected['packages']}}; "
            + "actual.update({'platform':platform.platform()} if 'platform' in expected else {}); "
            + "print(json.dumps(actual))")
    result = subprocess.run([str(python), "-I", "-B", "-c", code],
                            check=True, capture_output=True, text=True, timeout=30)
    actual = json.loads(result.stdout)
    if actual != expected:
        raise ValueError("Frozen role runtime versions differ")
    return {"environment": actual, "interpreter_sha256": sha(python)}


def run_phase(command, cwd, log):
    started = time.monotonic()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            result = process.wait(timeout=60)
        except BaseException:
            process.terminate()
            process.wait(timeout=60)
            raise
    if result:
        raise subprocess.CalledProcessError(result, command)
    return {"elapsed_seconds": round(time.monotonic() - started, 3), "log_sha256": sha(log)}


def command(python, workspace, relative, *arguments):
    safe_file(workspace, relative)
    return [str(python), "-I", "-B", "-c", LAUNCHER,
            str(workspace / relative), *map(str, arguments)]


def verify_hashes(root, hashes):
    for relative, expected in hashes.items():
        if sha(safe_file(root, relative)) != expected:
            raise ValueError("Bound copied/generated input changed: " + relative)


def prepare(package, workspace, source, main_python, neural_python, execute=False):
    package = Path(package).resolve(strict=True)
    package_check = verify(package)
    workspace = new_workspace(workspace, package)
    source = source_file(source)
    binding = json.loads(safe_file(package, PLAN + "/execution_plan_freeze_audit.json").read_text())
    main_python, neural_python = Path(main_python).absolute(), Path(neural_python).absolute()
    runtimes = {"main": runtime_probe(main_python, binding["runtime_environment"]),
                "neural": runtime_probe(neural_python, binding["neural_runtime_environment"])}
    if not execute:
        return {**package_check, "stage": "PORTABLE_DEVELOPMENT_PREFLIGHT_NO_EXECUTION",
                "source_bytes_verified": True, "source_cells_parsed": False,
                "runtime_versions_verified": True, "workspace_created": False}
    workspace.mkdir(parents=True, mode=0o700, exist_ok=False)
    stage = "copy-code-and-reference-metadata"
    try:
        manifest_path = safe_file(package, "manifest.json")
        manifest = json.loads(manifest_path.read_text())
        protected = {record["file"]: record["sha256"] for record in manifest["files"]}
        protected["manifest.json"] = sha(manifest_path)
        for relative in protected:
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_file(package, relative), target)
        verify_hashes(workspace, protected)
        logs = workspace / "portable-execution/logs"
        logs.mkdir(parents=True, mode=0o700)
        stage = "rebuild-all-nonprice-feature-files"
        execution = run_phase(command(main_python, workspace,
            "experiments/build_turkey_price_free_manifest.py", "--source", source,
            "--config", workspace / FEATURE_CONFIG, "--protocol", workspace / FEATURE_PROTOCOL,
            "--output-dir", workspace / COHORT), workspace, logs / "features.txt")
        actual_audit = safe_file(workspace, COHORT + "/cohort_freeze_audit.json")
        actual = json.loads(actual_audit.read_text())
        reference = json.loads(safe_file(workspace, COHORT_REFERENCE).read_text())
        if (actual["price_cell_values_decoded"] != 0 or actual["labels_exported"] is not False
                or actual["output_sha256"] != reference["output_sha256"]
                or actual["split_listing_counts"] != reference["split_listing_counts"]
                or len(actual["output_sha256"]) != 10):
            raise ValueError("Original ten feature/config/code hashes or no-price boundary differ")
        for name, expected in actual["output_sha256"].items():
            if Path(name).name != name or sha(safe_file(workspace, COHORT + "/" + name)) != expected:
                raise ValueError("Produced feature bytes differ")
        # Retain the actual new producer receipt outside the historically bound
        # cohort. Reuse reference metadata only after all ten bytes are verified.
        producer = workspace / "portable-execution/nonprice-producer-audit.json"
        actual_audit.rename(producer)
        with actual_audit.open("xb") as stream:
            stream.write(safe_file(workspace, COHORT_REFERENCE).read_bytes())
        protected.update({COHORT + "/" + name: value for name, value in actual["output_sha256"].items()})
        protected[COHORT + "/cohort_freeze_audit.json"] = sha(actual_audit)
        protected["portable-execution/nonprice-producer-audit.json"] = sha(producer)
        verify_hashes(workspace, protected)
        receipt = {"version": VERSION, "stage": "PORTABLE_SOURCE_FEATURES_REBUILT_NO_TARGET_ACCESS",
                   "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                   "workspace": str(workspace), "source": str(source), "source_sha256": SOURCE_SHA,
                   "main_python": str(main_python), "neural_python": str(neural_python),
                   "runtime_probes": runtimes, "protected_sha256": protected,
                   "feature_execution": execution,
                   "actual_producer_receipt_sha256": sha(producer),
                   "historical_cohort_audit_is_reference_only": True,
                   "feature_files_rebuilt_not_copied": True, "price_values_decoded": 0,
                   "original_study_artifacts_or_ledger_required": False,
                   "heldout_price_access_allowed": False, "historical_tests_already_opened": True,
                   "new_independent_confirmation": False, "full_study_empirical_pipeline_portable": False}
        write_new(workspace / "portable-preparation-receipt.json", receipt)
        return {key: value for key, value in receipt.items() if key != "protected_sha256"}
    except BaseException as error:
        write_new(workspace / "failed-portable-preparation.json", {
            "stage": stage, "exception_class": type(error).__name__, "reason": str(error),
            "failure_retained": True, "new_independent_confirmation": False,
            "heldout_price_access_allowed": False})
        raise


def bound_workspace(workspace):
    workspace = Path(workspace).absolute()
    if workspace.is_symlink() or any(parent.is_symlink() for parent in workspace.parents):
        raise ValueError("Non-symlink prepared workspace required")
    prepared_path = safe_file(workspace, "portable-preparation-receipt.json")
    prepared = json.loads(prepared_path.read_text())
    if (prepared.get("version") != VERSION or prepared.get("workspace") != str(workspace)
            or prepared.get("stage") != "PORTABLE_SOURCE_FEATURES_REBUILT_NO_TARGET_ACCESS"
            or prepared.get("heldout_price_access_allowed") is not False
            or prepared.get("new_independent_confirmation") is not False):
        raise ValueError("Matching separate development-only preparation required")
    verify_hashes(workspace, prepared["protected_sha256"])
    source_file(prepared["source"])
    binding = json.loads(safe_file(workspace, PLAN + "/execution_plan_freeze_audit.json").read_text())
    for role, expected in (("main", binding["runtime_environment"]),
                           ("neural", binding["neural_runtime_environment"])):
        if runtime_probe(prepared[role + "_python"], expected) != prepared["runtime_probes"][role]:
            raise ValueError("Bound runtime changed")
    return workspace, prepared


def completed_phase(workspace, name):
    receipt = json.loads(safe_file(workspace, "portable-" + name + "-receipt.json").read_text())
    if (receipt.get("version") != VERSION or receipt.get("phase") != name
            or receipt.get("stage") != "PORTABLE_BOUND_PHASE_COMPLETED"
            or receipt.get("heldout_price_access_allowed") is not False
            or receipt.get("new_independent_confirmation") is not False
            or not receipt.get("generated_artifact_sha256")
            or receipt.get("preparation_receipt_sha256") != sha(safe_file(workspace, "portable-preparation-receipt.json"))):
        raise ValueError("Completed bound phase required: " + name)
    verify_hashes(workspace, receipt["generated_artifact_sha256"])
    return receipt


def validate_fit(audit):
    if (audit.get("stage") != "SOURCE_DEVELOPMENT_COMPLETED_NOT_TEST_FROZEN"
            or audit.get("calibration_or_test_labels_read") is not False
            or (audit.get("valuation_fits"), audit.get("head_fits"), audit.get("neural_fits")) != (258, 21, 6)):
        raise ValueError("Complete unchanged 258/21/6 development-only schedule required")


def validate_score(audit):
    if (audit.get("stage") != "TURKEY_HELDOUT_SCORES_ACTIONS_FROZEN_NO_PRICE_RELEASE"
            or audit.get("evidence_status") != "authorized_development_only"
            or audit.get("calibration_or_test_labels_read") is not False
            or audit.get("heldout_label_release_allowed") is not False
            or audit.get("partition_counts") != {"calibration": 7814, "test": 7802}
            or audit.get("all_budget_policy_seed_allocations_replayed_exactly") is not True
            or audit.get("refits") != 0 or audit.get("reselection") is not False):
        raise ValueError("Complete price-free whole-partition score/replay required")


def phase_arguments(workspace, prepared, phase):
    if phase not in {"readiness", "fit", "score"}:
        raise ValueError("Only development/readiness/price-free scoring modes exist")
    output = {"readiness": READINESS, "fit": DEVELOPMENT, "score": SCORES}[phase]
    arguments = ["--plan-freeze", workspace / PLAN, "--output-dir", workspace / output,
                 "--neural-python", prepared["neural_python"]]
    if phase == "readiness":
        arguments += ["--main-python", prepared["main_python"]]
    elif phase == "fit":
        arguments += ["--source", prepared["source"], "--readiness", workspace / READINESS / "readiness_receipt.json"]
    else:
        arguments += ["--development-run", workspace / DEVELOPMENT / "training"]
    return arguments


def execute_bound_phase(workspace, phase):
    if phase not in {"readiness", "fit", "score"}:
        raise ValueError("Only development/readiness/price-free scoring modes exist")
    workspace, prepared = bound_workspace(workspace)
    if any((workspace / name).exists() for name in (
        "portable-" + phase + "-started.json", "portable-" + phase + "-receipt.json",
        "failed-portable-" + phase + ".json")):
        raise ValueError("Existing phase attempt cannot be restarted or overwritten")
    if phase != "readiness":
        completed_phase(workspace, "readiness")
    if phase == "score":
        completed_phase(workspace, "fit")
        validate_fit(json.loads(safe_file(workspace, DEVELOPMENT + "/training/development_execution_audit.json").read_text()))
    output_relative = {"readiness": READINESS, "fit": DEVELOPMENT, "score": SCORES}[phase]
    if (workspace / output_relative).exists():
        raise ValueError("New phase output required")
    preparation_sha = sha(safe_file(workspace, "portable-preparation-receipt.json"))
    write_new(workspace / ("portable-" + phase + "-started.json"), {
        "phase": phase, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "controller_pid": os.getpid(), "preparation_receipt_sha256": preparation_sha,
        "heldout_price_access_allowed": False, "new_independent_confirmation": False})
    scripts = {"readiness": "experiments/freeze_turkey_development_readiness.py",
               "fit": "experiments/run_turkey_development.py",
               "score": "experiments/freeze_turkey_heldout_scores.py"}
    arguments = phase_arguments(workspace, prepared, phase)
    try:
        execution = run_phase(command(prepared["main_python"], workspace, scripts[phase], *arguments),
                              workspace, workspace / "portable-execution/logs" / (phase + ".txt"))
        if phase == "fit":
            validate_fit(json.loads(safe_file(workspace, DEVELOPMENT + "/training/development_execution_audit.json").read_text()))
        elif phase == "score":
            validate_score(json.loads(safe_file(workspace, SCORES + "/heldout_score_freeze_audit.json").read_text()))
        bound_workspace(workspace)
        generated = {path.relative_to(workspace).as_posix(): sha(path)
                     for path in sorted((workspace / output_relative).rglob("*")) if path.is_file()}
        if not generated:
            raise ValueError("No completed phase artifacts")
        receipt = {"version": VERSION, "stage": "PORTABLE_BOUND_PHASE_COMPLETED", "phase": phase,
                   "finished_at_utc": datetime.now(timezone.utc).isoformat(), "execution": execution,
                   "preparation_receipt_sha256": preparation_sha, "generated_artifact_sha256": generated,
                   "historical_tests_already_opened": True, "heldout_price_access_allowed": False,
                   "new_independent_confirmation": False, "full_study_empirical_pipeline_portable": False,
                   "original_study_artifacts_or_ledger_required": False,
                   "private_output_not_public_payload": True}
        write_new(workspace / ("portable-" + phase + "-receipt.json"), receipt)
        return {key: value for key, value in receipt.items() if key != "generated_artifact_sha256"}
    except BaseException as error:
        write_new(workspace / ("failed-portable-" + phase + ".json"), {
            "phase": phase, "exception_class": type(error).__name__, "reason": str(error),
            "attempt_retained": True, "no_retry_or_automatic_resume": True,
            "heldout_price_access_allowed": False, "new_independent_confirmation": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--main-python", type=Path)
    parser.add_argument("--neural-python", type=Path)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--prepare", action="store_true")
    choices.add_argument("--readiness", action="store_true")
    choices.add_argument("--fit", action="store_true")
    choices.add_argument("--score", action="store_true")
    args = parser.parse_args()
    phase = next((name for name in ("readiness", "fit", "score") if getattr(args, name)), None)
    inputs = (args.package_root, args.source, args.main_python, args.neural_python)
    if phase:
        if any(inputs):
            parser.error("Bound phase cannot replace package/source/runtime inputs")
        result = execute_bound_phase(args.workspace, phase)
    else:
        if not all(inputs):
            parser.error("Preflight/preparation requires package, source and both runtime roles")
        result = prepare(args.package_root, args.workspace, args.source,
                         args.main_python, args.neural_python, args.prepare)
    print(json.dumps(result, indent=2))
