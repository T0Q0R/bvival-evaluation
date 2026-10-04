"""Replay original Turkey fixed policies using saved models/released labels.

This is cached outcome reconstruction, NOT fresh development/head/neural fits.
Original provenance guards are retained. Additional worker guards reject XLSX
parsing and new source-price release. All historical inputs are hashed before
and after execution. Original directories/ledger are never modified.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mucars_reconstruction import safe_path
from check_remaining_source_metadata import TURKEY, digest
from plan_mucars_replay import static_import_closure
from run_mucars_scoring_replay import PROBE, child_environment, lock_pins, validate_runtime_records

REPORT = "release-bvival/remaining-source-metadata-2026-10-01.json"
REPORT_HASH = "481df04401d54070f094750658f6530e058e07d96b9d1141ba7cb2d9904e2a2b"
LOCK = "release-bvival/output/hash-lock-check-2026-10-01-X1Ohm9/resolved-main-macos-arm64.lock"
LOCK_HASH = "206739a71c7554641dabf2c41d1a8ec184af9a7cf0481b233f1cb343abfced0a"
INSTALL = "release-bvival/output/mucars-scoring-runtime-2026-10-01-AS1AKH/install-report.json"
INSTALL_HASH = "bc1853388e2e5035f1bfad3af220cc548642721f26021f254891fcf73e5f4b19"
DEFAULT_OUTPUT = "experiments/replays/turkey-cached-fixed-outcomes-v1"
DIRS = {k: str(Path(TURKEY[k]).parent) for k in ("freeze", "development", "plan", "release", "erratum")}
DIRS["readiness"] = "experiments/outputs/turkey_source_release_readiness_2026-09-30_v1"
WORKER = '''
import json,sys,zipfile
from pathlib import Path
root,output=map(Path,sys.argv[1:])
sys.path.insert(0,str(root/"experiments"))
import evaluate_turkey_released_phase as runner
import turkey_source_price_release as price
import turkey_price_cells as cells
def forbidden(*a,**k):
    raise RuntimeError("RAW_WORKBOOK_OR_NEW_PRICE_RELEASE_FORBIDDEN_IN_CACHED_REPLAY")
zipfile.ZipFile=forbidden
price.release_source_phase=forbidden
price._release_once_engine=forbidden
price.read_selected_prices=forbidden
price.verify_archive=forbidden
cells.read_selected_prices=forbidden
base=root/"experiments/outputs"
result=runner.evaluate_source_phase(
    base/"turkey_heldout_score_freeze_source_2026-09-30_v1",
    base/"turkey_source_development_2026-09-30_v1/training",
    base/"turkey_execution_plan_freeze_v1_2026-09-30",
    base/"turkey_test_release_2026-10-01_v1",
    base/"turkey_test_accounting_erratum_2026-10-01_v1",
    base/"turkey_source_release_readiness_2026-09-30_v1",
    output,phase="test",project_root=root)
print(json.dumps({"stage":result["stage"],"feature_records":result["feature_records"],
                  "primary_comparisons":result["primary_comparisons"],"new_price_release":False}))
'''


def save_new(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def output_path(root, relative):
    root = Path(root).resolve(strict=True)
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or str(p) != relative or not relative.startswith("experiments/replays/"):
        raise ValueError("New output must be a canonical experiments/replays child")
    result = root / p
    if result == root / "experiments/replays":
        raise ValueError("Replay collection root cannot be an output")
    for item in (result, *result.parents):
        if item.is_symlink():
            raise ValueError("Symlink output/ancestor rejected")
    if result.exists():
        raise ValueError("Existing replay output preserved; choose a new name")
    return result


def protected_files(root, report):
    """Hash opaque artifacts including caches/models; do not deserialize here."""
    expected = dict(report["audit_file_sha256"])
    expected[REPORT] = REPORT_HASH
    for relative in TURKEY.values():
        a = json.loads(safe_path(root, relative).read_text())
        if "artifact_sha256" in a or "frozen_snapshot_sha256" in a:
            bindings = a.get("artifact_sha256", a.get("frozen_snapshot_sha256"))
            for name, sha in bindings.items():
                path = str(Path(relative).parent / name)
                if path in expected and expected[path] != sha:
                    raise ValueError("Conflicting protected-artifact binding")
                expected[path] = sha
    ready = DIRS["readiness"] + "/source_release_readiness_audit.json"
    a = json.loads(safe_path(root, ready).read_text())
    expected[ready] = digest(safe_path(root, ready))
    expected.update({DIRS["readiness"] + "/" + k: v for k, v in a["artifact_sha256"].items()})
    for role in ("calibration_release", "release"):
        a = json.loads(safe_path(root, TURKEY[role]).read_text())
        claim = Path(a["ledger_claim_directory"])
        canonical = root / "experiments/access_ledgers/turkey_heldout_v1"
        if claim.parent != canonical or not claim.name.endswith("." + a["phase"]):
            raise ValueError("Original canonical ledger binding required")
        for name, field in (("reservation.json", "reservation_sha256"), ("terminal.json", "terminal_sha256")):
            relative = str((claim / name).relative_to(root))
            expected[relative] = a[field]
    config = json.loads(safe_path(root, TURKEY["config"]).read_text())
    cohort_audit = config["cohort_directory"] + "/cohort_freeze_audit.json"
    expected[cohort_audit] = config["cohort_audit_sha256"]
    a = json.loads(safe_path(root, cohort_audit).read_text())
    expected.update({config["cohort_directory"] + "/" + k: v for k, v in a["output_sha256"].items()})
    for relative in ("paper-bvival/main.tex", "paper-bvival/supplement.tex"):
        expected[relative] = digest(safe_path(root, relative))
    verify_hashes(root, expected)
    return expected


def verify_hashes(root, expected):
    if not expected:
        raise ValueError("Nonempty protected inventory required")
    for path, sha in expected.items():
        if digest(safe_path(root, path)) != sha:
            raise ValueError("Protected historical input changed: " + path)


def compare_aggregates(old, new):
    """Exact nested JSON comparison; no post-run numerical tolerances."""
    counts = {"finite_numeric_cells": 0, "other_leaf_cells": 0}

    def walk(a, b, path):
        if type(a) is not type(b):
            raise ValueError("Aggregate type mismatch: " + path)
        if isinstance(a, dict):
            if set(a) != set(b):
                raise ValueError("Aggregate key mismatch: " + path)
            for k in a:
                walk(a[k], b[k], path + "/" + k)
        elif isinstance(a, list):
            if len(a) != len(b):
                raise ValueError("Aggregate length mismatch: " + path)
            for i, (left, right) in enumerate(zip(a, b)):
                walk(left, right, path + "/" + str(i))
        else:
            if isinstance(a, (int, float)) and not isinstance(a, bool):
                if not math.isfinite(a) or not math.isfinite(b):
                    raise ValueError("Nonfinite aggregate cell")
                counts["finite_numeric_cells"] += 1
            else:
                counts["other_leaf_cells"] += 1
            if a != b:
                raise ValueError("Aggregate value mismatch: " + path)
    walk(old, new, "root")
    return {**counts, "exact_values_match": True, "numerical_tolerance": 0}


def execute(root, python, relative):
    root = Path(root).resolve(strict=True)
    output = output_path(root, relative)
    report_path = safe_path(root, REPORT)
    if digest(report_path) != REPORT_HASH:
        raise ValueError("Bound metadata gate changed")
    report = json.loads(report_path.read_text())
    protected = protected_files(root, report)
    closure = static_import_closure(root, ["experiments/evaluate_turkey_released_phase.py"])
    protected.update(closure["local_source_sha256"])
    if digest(safe_path(root, LOCK)) != LOCK_HASH or digest(safe_path(root, INSTALL)) != INSTALL_HASH:
        raise ValueError("Bound main runtime records changed")
    probe = subprocess.run([str(python), "-I", "-B", "-c", PROBE], check=True, capture_output=True,
                           text=True, timeout=60, env=child_environment(python))
    runtime = validate_runtime_records(lock_pins(safe_path(root, LOCK).read_text()),
                                      json.loads(safe_path(root, INSTALL).read_text()), json.loads(probe.stdout))
    runtime.update(lock_sha256=LOCK_HASH, install_report_sha256=INSTALL_HASH, python_executable_sha256=digest(python))
    verify_hashes(root, protected)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    phase = "bound_inputs"
    try:
        save_new(output / "bound-inputs.json", {"stage": "CACHED_TEST_REPLAY_NOT_FRESH_TRAINING",
                 "protected_input_sha256": protected, "static_import_closure": closure, "runtime": runtime,
                 "historical_tests_already_opened": True, "worker_source": WORKER,
                 "runner_sha256": digest(Path(__file__)), "requires_original_provenance_guards": True})
        phase = "original_fixed_evaluation"
        child = subprocess.run([str(python), "-I", "-B", "-c", WORKER, str(root), str(output / "fixed-evaluation")],
                               capture_output=True, text=True, env=child_environment(python))
        with (output / "worker.stdout.txt").open("x") as stream:
            stream.write(child.stdout)
        with (output / "worker.stderr.txt").open("x") as stream:
            stream.write(child.stderr)
        if child.returncode:
            raise RuntimeError("Original cached evaluator failed; inspect retained worker.stderr.txt")
        phase = "concordance"
        new = output / "fixed-evaluation"
        a = json.loads((new / "released_phase_evaluation_audit.json").read_text())
        for name, sha in a["artifact_sha256"].items():
            if digest(safe_path(new, name)) != sha:
                raise ValueError("Generated evaluation artifact mismatch")
        historical = root / Path(TURKEY["evaluation"]).parent
        old_audit = json.loads(safe_path(root, TURKEY["evaluation"]).read_text())
        exact = {name: a["artifact_sha256"].get(name) == sha for name, sha in old_audit["artifact_sha256"].items()}
        if set(a["artifact_sha256"]) != set(exact) or not all(exact.values()):
            raise ValueError("New output artifact hashes differ from history; no tolerance substitution")
        old = json.loads((historical / "phase_fixed_evaluation.json").read_text())
        result = json.loads((new / "phase_fixed_evaluation.json").read_text())
        comparison = compare_aggregates(old, result)
        verify_hashes(root, protected)
        receipt = {"stage": "TURKEY_CACHED_FIXED_OUTCOME_REPLAY_COMPLETED", "date": "2026-10-01",
                   "scope": "original_saved_valuation_and_frozen_actions_recomputed_not_refitted",
                   "historical_tests_already_opened": True, "feature_records": a["feature_records"],
                   "price_valid_records": a["price_valid_records"], "primary_comparisons": result["primary_comparisons"],
                   "aggregate_comparison": comparison, "all_ten_historical_artifact_hashes_match": exact,
                   "generated_evaluation_audit_sha256": digest(new / "released_phase_evaluation_audit.json"),
                   "protected_input_hashes_reverified": len(protected), "runtime": runtime,
                   "study_models_refitted": 0, "policies_reselected": False,
                   "raw_workbook_parsed_or_new_source_price_release": False, "ledger_modified": False,
                   "cached_test_labels_decoded": True, "unique_saved_final_valuation_models_loaded": 3,
                   "valuation_prediction_passes": 2,
                   "neural_runtime_or_policy_scoring_reconstructed": False,
                   "fresh_development_training_reproduced": False, "independent_replication": False,
                   "new_confirmatory_evidence": False, "public_release_created": False, "submission_ready": False}
        save_new(output / "cached-replay-receipt.json", receipt)
        return receipt
    except Exception as error:
        save_new(output / "failed-cached-replay.json", {"phase": phase, "exception_class": type(error).__name__,
                 "message": str(error), "original_results_not_replaced": True})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--execute-cached-outcomes", action="store_true", required=True)
    args = p.parse_args()
    result = execute(args.root, args.python, args.output)
    print(json.dumps({"stage": result["stage"], "protected_hashes": result["protected_input_hashes_reverified"],
                      "numeric_cells": result["aggregate_comparison"]["finite_numeric_cells"], "study_refits": 0}))


if __name__ == "__main__":
    main()
