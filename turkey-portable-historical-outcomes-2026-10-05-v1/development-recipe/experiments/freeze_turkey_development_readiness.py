"""Issue local train/validation readiness only after tests/full synthetic replay.

No source workbook access. This is not external preregistration, authorization
by a third party, or a calibration/test-opening receipt.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from run_turkey_development import RUNTIME_SOURCES, execute_pipeline, file_hashes, probe_neural_environment, reload_check, save_json, sha256, verify_readiness_package
from run_turkey_policy_synthetic_integration import smoke_plan
from run_turkey_synthetic_integration import synthetic_dataset, synthetic_price_check
from turkey_development_io import bound_plan


def run_tests(python, files, report, *, project_root):
    result = subprocess.run([str(python.absolute()), "-m", "pytest", *files, "-q"], cwd=project_root,
                            check=False, capture_output=True, text=True, timeout=300)
    with report.open("x") as stream:
        stream.write(result.stdout)
        stream.write(result.stderr)
    counts = re.findall(r"(\d+) passed", result.stdout)
    if result.returncode != 0 or not counts or int(counts[-1]) == 0:
        raise ValueError("Required regression tests failed; readiness not issued")
    return {"passed": int(counts[-1]), "report_sha256": sha256(report), "returncode": result.returncode}


def issue_readiness(plan_freeze, output, main_python, neural_python, *, project_root):
    if output.exists():
        raise ValueError("New nonexistent readiness directory required")
    plan, config, _, binding = bound_plan(plan_freeze, project_root=project_root)
    probe_neural_environment(main_python, binding["runtime_environment"])
    probe_neural_environment(neural_python, binding["neural_runtime_environment"])
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    phase = "sources"
    try:
        source_hashes = {name: sha256(project_root / "experiments" / name) for name in RUNTIME_SOURCES}
        snapshots = output / "source_and_test_snapshots"
        snapshots.mkdir(mode=0o700)
        files = [(project_root / "experiments" / name, name) for name in RUNTIME_SOURCES]
        files += [(p, "tests__" + p.name) for p in sorted((project_root / "experiments/tests").glob("test_*.py"))]
        files += [(project_root / "experiments" / name, name) for name in
                  ("run_turkey_synthetic_integration.py", "run_turkey_policy_synthetic_integration.py")]
        for path, name in files:
            with (snapshots / name).open("xb") as stream:
                stream.write(path.read_bytes())
        phase = "main_tests"
        print("readiness: full main-environment tests", flush=True)
        main_tests = run_tests(main_python, ["experiments/tests"], output / "main_tests.txt", project_root=project_root)
        phase = "neural_tests"
        print("readiness: neural and controller regression tests", flush=True)
        neural_tests = run_tests(neural_python, ["experiments/tests/test_turkey_policy_components.py",
                                                "experiments/tests/test_turkey_neural_policy.py",
                                                "experiments/tests/test_turkey_development_controller.py"],
                                 output / "neural_tests.txt", project_root=project_root)
        phase = "synthetic_controller"
        access = synthetic_price_check()
        data = synthetic_dataset(config)
        reduced = smoke_plan(plan)
        synthetic = execute_pipeline(data, reduced, config, binding, output / "synthetic_execution", neural_python,
                                     project_root=project_root, evidence_status="synthetic_only",
                                     provenance={"source_workbook_opened": False, "synthetic_smoke_hyperparameters_reduced": True})
        phase = "completed_package_reload"
        replay = reload_check(output / "synthetic_execution", data, reduced, config, neural_python, project_root=project_root)
        if (synthetic["stage"] != "SYNTHETIC_FULL_DEVELOPMENT_CONTROLLER_PASSED"
                or not replay["all_saved_predictions_and_allocations_match"] or replay["refits"] != 0
                or access["unselected_price_values_decoded"] != 0):
            raise ValueError("Full synthetic access/controller/replay evidence required")
        if source_hashes != {name: sha256(project_root / "experiments" / name) for name in RUNTIME_SOURCES}:
            raise ValueError("Source files changed while readiness checks ran")
        for path, name in files:
            if sha256(path) != sha256(snapshots / name):
                raise ValueError("Test or supporting source changed while checks ran")
        receipt = {"stage": "TURKEY_FULL_DEVELOPMENT_INTEGRATION_READY", "development_label_release_allowed": True,
                   "price_release_scope": ["train", "validation"], "calibration_or_test_release_allowed": False,
                   "policy_and_neural_integration_passed": True, "nested_valuation_integration_passed": True,
                   "full_controller_and_reload_integration_passed": True, "source_price_values_parsed": 0,
                   "source_sha256": source_hashes, "plan_sha256": binding["plan_sha256"],
                   "cohort_audit_sha256": binding["cohort_audit_sha256"], "runtime_environment": binding["runtime_environment"],
                   "neural_runtime_environment": binding["neural_runtime_environment"],
                   "main_tests": main_tests, "neural_tests": neural_tests, "synthetic_access_check": access,
                   "synthetic_execution_audit_sha256": sha256(output / "synthetic_execution/development_execution_audit.json"),
                   "completed_package_reload": replay, "external_registration": False,
                   "local_discipline_not_cryptographic_or_third_party_authorization": True,
                   "official_GDFS_reproduction_completed": False}
        save_json(output / "readiness_receipt.json", receipt)
        save_json(output / "readiness_package_audit.json", {
            "stage": "LOCAL_DEVELOPMENT_READINESS_PACKAGE_COMPLETE", "plan_sha256": binding["plan_sha256"],
            "source_price_values_parsed": 0, "readiness_receipt_sha256": sha256(output / "readiness_receipt.json"),
            "artifact_sha256": file_hashes(output)})
        verify_readiness_package(output / "readiness_receipt.json", binding, project_root=project_root)
    except Exception as error:
        save_json(output / "failed_readiness_attempt.json", {"stage": "FAILED_READINESS_ATTEMPT", "last_phase": phase,
                                                            "exception_class": type(error).__name__, "source_price_values_parsed": 0,
                                                            "calibration_or_test_release_allowed": False})
        raise
    finally:
        for path in output.rglob("*"):
            if path.is_file():
                path.chmod(0o400)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--main-python", type=Path, required=True)
    parser.add_argument("--neural-python", type=Path, required=True)
    args = parser.parse_args()
    receipt = issue_readiness(args.plan_freeze, args.output_dir, args.main_python, args.neural_python,
                              project_root=Path(__file__).resolve().parents[1])
    print(json.dumps({"stage": receipt["stage"], "price_release_scope": receipt["price_release_scope"],
                      "source_price_values_parsed": 0, "calibration_or_test_release_allowed": False}))


if __name__ == "__main__":
    main()
