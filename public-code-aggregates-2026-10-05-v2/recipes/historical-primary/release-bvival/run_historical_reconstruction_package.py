"""Source-directed copied-tree reconstruction, explicit opt-in, no overwrite.

Packages contain only code, audit metadata and aggregate references. Source
rows are supplied separately to a new PRIVATE workspace and are never copied
back to the package. This is trusted-code isolation, not an OS sandbox.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_source_bundle import safe_file
from historical_reconstruction_contract import PLAN_SHAS, LOCK, LOCK_SHA, sha256
from check_historical_reconstruction_package import verify


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def input_bindings(source, raw_source, dictionary):
    if source not in PLAN_SHAS:
        raise ValueError("Unknown source")
    if (source == "jucars") != (dictionary is not None):
        raise ValueError("Dictionary is required ONLY for JUCars")
    result = {"source": Path(raw_source).absolute()}
    if dictionary is not None:
        result["dictionary"] = Path(dictionary).absolute()
    for path in result.values():
        if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():
            raise ValueError("Inputs must be existing non-symlink files")
    return result


def run_phase(command, cwd, log_path):
    started = time.monotonic()
    # No shell, inherited PYTHONPATH is ignored by the isolated interpreter.
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait(timeout=60)
        except BaseException:
            process.terminate()
            process.wait(timeout=60)
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return {"elapsed_seconds": round(time.monotonic() - started, 3), "log_sha256": sha256(log_path)}


def reconstruct(package, workspace, source, raw_source, dictionary, python, install_report, execute=False):
    package, workspace = Path(package).resolve(strict=True), Path(workspace).absolute()
    package_check = verify(package)
    manifest = json.loads(safe_file(package, "manifest.json").read_text())
    profile = manifest["sources"].get(source)
    if profile is None:
        raise ValueError("Source not in this package")
    inputs = input_bindings(source, raw_source, dictionary)
    if set(inputs) != set(profile["inputs"]):
        raise ValueError("Source input roles differ")
    for role, path in inputs.items():
        if sha256(path) != profile["inputs"][role]["sha256"]:
            raise ValueError("Original input SHA256 differs: " + role)
    if (workspace.exists() or workspace.is_symlink() or any(p.is_symlink() for p in workspace.parents)
            or workspace.is_relative_to(package)):
        raise ValueError("New workspace outside package required")
    python, install_report = Path(python).absolute(), Path(install_report).absolute()
    if not python.is_file() or not install_report.is_file():
        raise ValueError("Existing dedicated runtime and its actual install report required")
    if not execute:
        return {**package_check, "stage": "COPIED_TREE_PREFLIGHT_NO_EXECUTION",
                "original_input_byte_hashes_verified": True, "source_records_parsed": False,
                "workspace_created": False, "runtime_installation_verified_here": False}
    workspace.mkdir(parents=True, mode=0o700, exist_ok=False)
    phase, completed = "copy", []
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        for record in manifest["files"]:
            relative = record["file"]
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_file(package, relative), target)
            if sha256(target) != record["sha256"]:
                raise ValueError("Copied package drift")
        shutil.copyfile(safe_file(package, "manifest.json"), workspace / "manifest.json")
        for role, original in inputs.items():
            target = workspace / profile["inputs"][role]["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with original.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            if sha256(target) != profile["inputs"][role]["sha256"]:
                raise ValueError("Copied source drift")
        logs = workspace / "copied-tree-logs"
        logs.mkdir()
        install_hash = sha256(install_report)
        scoring = [str(python), "-I", "-B", str(workspace / f"release-bvival/run_{source}_scoring_replay.py"),
                   "--project-root", str(workspace), "--plan", str(workspace / profile["plan"]),
                   "--python", str(python), "--requirements-lock", str(workspace / LOCK),
                   "--install-report", str(install_report), "--execute-scoring"]
        if source != "mucars":
            scoring += ["--expected-lock-sha256", LOCK_SHA, "--expected-install-report-sha256", install_hash]
        phase = "scoring"
        completed.append({"phase": phase, **run_phase(scoring, workspace, logs / "scoring.txt")})
        freeze_path = safe_file(workspace, profile["scoring_root"] + "/scoring-freeze.json")
        freeze_sha = sha256(freeze_path)
        outcome_name = source + "-reconstructed-primary-outcomes-copied-tree-v1"
        outcome_relative = "experiments/replays/" + outcome_name
        outcome = [str(python), "-I", "-B", str(workspace / f"release-bvival/run_{source}_outcome_replay.py"),
                   "--project-root", str(workspace), "--expected-scoring-freeze-sha256", freeze_sha,
                   "--python", str(python), "--output-name", outcome_name, "--execute-outcomes"]
        if source == "mucars":
            outcome += ["--scoring-root", profile["scoring_root"]]
        elif source == "autoscout24":
            outcome += ["--reconstructed-scoring-freeze"]
        phase = "outcomes"
        completed.append({"phase": phase, **run_phase(outcome, workspace, logs / "outcomes.txt")})
        receipt_path = safe_file(workspace, outcome_relative + "/outcome-replay-receipt.json")
        result = json.loads(receipt_path.read_text())
        pass_key = "all_primary_tables_pass_exact_rules" if source == "autoscout24" else "all_primary_tables_pass_fixed_rules"
        if result.get(pass_key) is not True or result.get("all_primary_table_byte_hashes_match") is not True:
            raise ValueError("Original fixed rules or aggregate byte concordance failed")
        for relative, expected in result["artifact_sha256"].items():
            if sha256(safe_file(workspace, outcome_relative + "/" + relative)) != expected:
                raise ValueError("Generated outcome artifact drift")
        for record in manifest["files"]:
            if sha256(safe_file(workspace, record["file"])) != record["sha256"]:
                raise ValueError("Copied package changed during execution")
        tables = {}
        for filename, expected in profile["reference_aggregate_hashes"].items():
            path = safe_file(workspace, outcome_relative + "/evaluation-mae/" + filename)
            if sha256(path) != expected:
                raise ValueError("Primary aggregate bytes differ: " + filename)
            tables[filename] = expected
        receipt = {"stage": "COPIED_TREE_PRIMARY_SOURCE_TO_RESULTS_VERIFIED", "source": source,
                   "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                   "package_manifest_sha256": sha256(safe_file(package, "manifest.json")),
                   "plan_sha256": profile["plan_sha256"], "completed_phases": completed,
                   "scoring_freeze_sha256": freeze_sha, "outcome_receipt_sha256": sha256(receipt_path),
                   "primary_aggregate_sha256": tables, "all_four_primary_tables_byte_identical": True,
                   "source_input_hashes": {k: v["sha256"] for k, v in profile["inputs"].items()},
                   "price_unit": profile["price_unit"], "evaluation_listings": result["listing_count"],
                   "platform": platform.platform(), "machine": platform.machine(),
                   "interpreter_sha256": sha256(python), "install_report_sha256": install_hash,
                   "requirements_lock_sha256": LOCK_SHA,
                   "historical_receipt_dates_not_rewritten": True,
                   "same_host_execution_not_external_confirmation": True,
                   "fresh_install_performed_here": False, "new_independent_confirmation": False,
                   "publicly_archived": False, "rights_or_software_license_granted": False,
                   "full_study_including_supplement_reconstructed": False, "submission_ready": False}
        write_new(workspace / "copied-tree-reconstruction-receipt.json", receipt)
        return receipt
    except BaseException as error:
        write_new(workspace / "failed-copied-tree-reconstruction.json",
                  {"phase": phase, "exception_class": type(error).__name__, "reason": str(error),
                   "completed_phases": completed, "failure_directory_retained": True,
                   "no_automatic_retry_or_tolerance_widening": True, "source": source})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package-root", "workspace", "raw-source", "python", "install-report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source", choices=list(PLAN_SHAS), required=True)
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.package_root, args.workspace, args.source, args.raw_source,
                                 args.dictionary, args.python, args.install_report, args.execute), indent=2))
