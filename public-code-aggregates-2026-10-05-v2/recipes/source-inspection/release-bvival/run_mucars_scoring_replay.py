"""Execute only seven pinned reconstructed scoring stages, then freeze scores.

Never runs the outcome join/evaluator. Explicit --execute-scoring is required.
This is procedural isolation of trusted code, not an OS security sandbox or
unseen-label guarantee: cohort construction reads historical source prices.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mucars_reconstruction import safe_path
from plan_mucars_replay import validate_plan


SCORING_IDS = ["cohort", "pairs", "development-actions", "calibration-actions", "value", "risk-mae", "scores"]
SCORE_COLUMNS = ["score_no_acquisition", "score_random", "score_global_field_prior", "score_uncertainty_only",
                 "score_disagreement_only", "score_mean_value", "score_conservative_lower_value"]
LAUNCHER = "import sys,runpy; from pathlib import Path; p=Path(sys.argv[1]); sys.path.insert(0,str(p.parent)); sys.argv=sys.argv[1:]; runpy.run_path(str(p),run_name='__main__')"
PROBE = '''
import json,sys,platform
from pathlib import Path
from importlib import metadata
import numpy,pandas,catboost,scipy
prefix=Path(sys.prefix).resolve()
inside=lambda p: Path(p).resolve().is_relative_to(prefix)
distributions=list(metadata.distributions())
versions={d.metadata["Name"]:d.version for d in distributions}
model=catboost.CatBoostRegressor(iterations=2,depth=2,thread_count=1,verbose=False,allow_writing_files=False)
model.fit(numpy.arange(16,dtype=float).reshape(8,2),numpy.arange(8,dtype=float))
print(json.dumps({"python":platform.python_version(),"platform":platform.platform(),"machine":platform.machine(),
 "is_virtualenv":sys.prefix!=sys.base_prefix,"all_distribution_metadata_inside_venv":all(inside(d._path) for d in distributions),
 "module_versions":{n:m.__version__ for n,m in [("numpy",numpy),("pandas",pandas),("catboost",catboost),("scipy",scipy)]},
 "all_four_imports_inside_venv":all(inside(m.__file__) for m in [numpy,pandas,catboost,scipy]),
 "installed_versions":versions,"synthetic_native_fit_passed":model.tree_count_==2,"logical_cpu_count":__import__("os").cpu_count(),
 "study_rows_read":False,"study_model_fitted":False}))
'''


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_pins(text):
    pins = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or line == "--only-binary=:all:":
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^ ]+) --hash=sha256:([0-9a-f]{64})", line)
        if not match or canonical(match[1]) in pins:
            raise ValueError("Expected unique exact version/one-wheel hash lock")
        pins[canonical(match[1])] = {"version": match[2], "sha256": match[3]}
    if not pins:
        raise ValueError("Empty runtime wheel lock")
    return pins


def validate_runtime_records(pins, install, probe):
    if any(probe.get(name) is not True for name in ("is_virtualenv", "all_distribution_metadata_inside_venv",
                                                  "all_four_imports_inside_venv", "synthetic_native_fit_passed")):
        raise ValueError("Dedicated non-inherited runtime/native toy check required")
    installed = {canonical(name): version for name, version in probe["installed_versions"].items()}
    if set(installed) - set(pins) != {"pip"}:
        raise ValueError("Only separately recorded bootstrap pip may lie outside the wheel lock")
    if any(installed.get(name) != pin["version"] for name, pin in pins.items()):
        raise ValueError("Installed versions differ from the exact lock")
    records = install["install"]
    by_name = {canonical(r["metadata"]["name"]): r for r in records}
    if len(by_name) != len(records) or set(by_name) != set(pins):
        raise ValueError("Install report does not bind every locked wheel exactly once")
    for name, pin in pins.items():
        record = by_name[name]
        if (record["metadata"]["version"] != pin["version"]
                or record["download_info"]["archive_info"]["hashes"]["sha256"] != pin["sha256"]):
            raise ValueError("Installation wheel archive hash/version mismatch")
    return {"wheel_count": len(pins), "wheel_hashes_and_installed_versions_match": True,
            "bootstrap_pip_version_not_in_wheel_lock": installed["pip"], "probe": probe}


def child_environment(python):
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["MPLCONFIGDIR"] = str(Path(python).absolute().parents[2] / "matplotlib-cache")
    return env


def source_checks(root, plan):
    for relative, expected in plan["static_source_closure"]["local_source_sha256"].items():
        if sha256(safe_path(root, relative)) != expected:
            raise ValueError("Live source changed after the bound plan")


def stage_command(root, plan, stage, python, snapshot):
    if stage["id"] not in SCORING_IDS or stage["phase"] not in {"reconstruction", "scoring"}:
        raise ValueError("Outcome/evaluation stage is forbidden in this runner")
    template = stage["argv_template"]
    relative = template[1]
    if relative not in plan["static_source_closure"]["local_source_sha256"]:
        raise ValueError("Stage source is not in the bound closure")
    source = plan["stages"][0]["argv_template"][3]
    args = []
    for value in template[2:]:
        if not isinstance(value, str):
            raise ValueError("Deferred bindings are forbidden before the outcome gate")
        if value == source or value.startswith(plan["planned_output_root"] + "/"):
            value = str(root / value)
        args.append(value)
    return [str(python), "-I", "-B", "-c", LAUNCHER, str(snapshot / relative), *args]


def check_score_file(path, expected_listings, expected_pairs):
    listings, keys = set(), set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["listing_id", "action_id", *SCORE_COLUMNS]:
            raise ValueError("Frozen score schema must contain only identifiers and the seven original policies")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed frozen score row")
            key = (row["listing_id"], row["action_id"])
            if not all(key) or key in keys:
                raise ValueError("Score keys must be nonempty and unique")
            keys.add(key)
            listings.add(key[0])
            for column in SCORE_COLUMNS:
                value = float(row[column])
                if math.isnan(value) or value == math.inf or (value == -math.inf and column not in {"score_uncertainty_only", "score_disagreement_only"}):
                    raise ValueError("Invalid score value")
    if len(listings) != expected_listings or len(keys) != expected_pairs:
        raise ValueError("Score listing/action universe count differs from the structural contract")
    return {"listing_count": len(listings), "action_pair_count": len(keys), "score_columns": SCORE_COLUMNS}


def cohort_checks(output, contract):
    c = json.loads((output / "cohort/mucars_bvival_audit.json").read_text())
    if c["split_counts"] != contract["split_listing_counts"]:
        raise ValueError("Rebuilt cohort counts differ")
    for relative, expected in contract["recorded_cohort_output_sha256_to_check_after_rebuilding"].items():
        if sha256(safe_path(output / "cohort", relative)) != expected:
            raise ValueError("Rebuilt cohort bytes differ from the recorded historical hash")


def structural_checks(output, contract):
    cohort_checks(output, contract)
    p = json.loads((output / "pairs/prediction_pair_generation_audit.json").read_text())
    for phase, count in contract["phase_counts"].items():
        if (p[phase + "_listing_count"] != count["listings"] or p[phase + "_action_pair_count"] != count["action_pairs"]):
            raise ValueError("Rebuilt pair counts differ")
    if p["feature_columns"] != contract["base_feature_columns"] or p["categorical_columns"] != contract["base_categorical_columns"]:
        raise ValueError("Valuation schema changed")
    for relative in ("value/value_policy_fit_audit.json", "risk-mae/risk_baseline_fit_audit.json"):
        a = json.loads((output / relative).read_text())
        if a["feature_columns"] != contract["policy_feature_columns"] or a["categorical_columns"] != contract["policy_categorical_columns"]:
            raise ValueError("Policy-head schema changed")


def write_new_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def preflight(root, plan_file, python, lock, install_report):
    plan_bytes = plan_file.read_bytes()
    plan = json.loads(plan_bytes)
    validate_plan(root, plan)
    if [s["id"] for s in plan["stages"][:7]] != SCORING_IDS:
        raise ValueError("Seven original scoring stages required")
    pins = lock_pins(lock.read_text())
    probe = subprocess.run([str(python), "-I", "-B", "-c", PROBE], check=True, capture_output=True,
                           text=True, timeout=120, env=child_environment(python))
    runtime = validate_runtime_records(pins, json.loads(install_report.read_text()), json.loads(probe.stdout))
    pip = subprocess.run([str(python), "-I", "-B", "-m", "pip", "--isolated", "check"],
                         check=True, capture_output=True, text=True, timeout=60, env=child_environment(python))
    runtime.update({"pip_check_passed": pip.returncode == 0, "requirements_lock_sha256": sha256(lock),
                    "install_report_sha256": sha256(install_report), "python_executable_sha256": sha256(python)})
    return plan, plan_bytes, runtime


def execute_scoring(root, plan, plan_bytes, python, lock, install_report, runtime):
    source_checks(root, plan)
    source_relative = plan["stages"][0]["argv_template"][3]
    if sha256(safe_path(root, source_relative)) != plan["source_sha256_to_verify_before_execution"]:
        raise ValueError("Original source hash mismatch; no replay output created")
    output = root / plan["planned_output_root"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)  # Never resume, overwrite or delete an existing run.
    snapshot = output / "snapshot-workspace"
    logs = output / "logs"
    logs.mkdir()
    phase = "snapshot"
    completed = []
    try:
        for relative in plan["static_source_closure"]["local_source_sha256"]:
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(root, relative), target)
            if sha256(target) != plan["static_source_closure"]["local_source_sha256"][relative]:
                raise ValueError("Source snapshot hash differs")
        helper_hashes = {}
        for name in ("run_mucars_scoring_replay.py", "plan_mucars_replay.py", "check_mucars_reconstruction.py"):
            helper = Path(__file__).resolve().parent / name
            target = snapshot / "release-bvival" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(helper, target)
            helper_hashes[name] = sha256(target)
        (output / "bound-plan.json").write_bytes(plan_bytes)
        shutil.copyfile(lock, output / "runtime.lock")
        shutil.copyfile(install_report, output / "install-report.json")
        write_new_json(output / "runtime-validation.json", runtime)
        for stage in plan["stages"][:7]:
            phase = stage["id"]
            source_checks(root, plan)
            command = stage_command(root, plan, stage, python, snapshot)
            print(json.dumps({"stage": phase, "status": "started"}), flush=True)
            started = time.monotonic()
            with (logs / (phase + ".txt")).open("x", encoding="utf-8") as log:
                subprocess.run(command, cwd=snapshot, env=child_environment(python), stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=1800)
            completed.append({"stage": phase, "elapsed_seconds": round(time.monotonic() - started, 3)})
            if phase == "cohort":
                phase = "cohort_checks"
                cohort_checks(output, plan["expected_structural_contract"])
                phase = "cohort"
            print(json.dumps({"stage": phase, "status": "completed", "elapsed_seconds": completed[-1]["elapsed_seconds"]}), flush=True)
        phase = "structural_checks"
        source_checks(root, plan)
        structural_checks(output, plan["expected_structural_contract"])
        scores = output / "scores/frozen_policy_scores_absolute_price_error.csv"
        counts = plan["expected_structural_contract"]["phase_counts"]["evaluation"]
        score_check = check_score_file(scores, counts["listings"], counts["action_pairs"])
        if (output / "evaluation-inputs").exists() or (output / "evaluation-mae").exists():
            raise ValueError("Forbidden outcome/evaluation output created")
        artifacts = {}
        for path in sorted(output.rglob("*")):
            if path.is_symlink():
                raise ValueError("Generated symlink rejected")
            if path.is_file():
                artifacts[str(path.relative_to(output))] = sha256(path)
        freeze = {"stage": "RECONSTRUCTED_MUCARS_SCORES_FROZEN_NO_OUTCOME_EVALUATION", "date": "2026-10-01",
                  "replay_name": plan["replay_name"], "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                  "bound_local_source_sha256": plan["static_source_closure"]["local_source_sha256"],
                  "runner_and_helper_sha256": helper_hashes,
                  "runtime_lock_sha256": sha256(lock), "runtime_validation": runtime,
                  "new_score_sha256": sha256(scores), "score_schema_check": score_check,
                  "completed_scoring_stages": completed, "artifact_sha256": artifacts,
                  "source_prices_processed_for_cohort_and_training": True,
                  "historical_test_labels_already_opened": True, "outcome_join_called": False,
                  "test_performance_computed": False, "full_primary_table_concordance_verified": False,
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
    parser.add_argument("--execute-scoring", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    python = args.python.absolute()  # Retain the venv executable path, not its base-interpreter symlink target.
    try:
        plan, data, runtime = preflight(root, args.plan, python, args.requirements_lock, args.install_report)
        if args.execute_scoring:
            freeze = execute_scoring(root, plan, data, python, args.requirements_lock, args.install_report, runtime)
            print(json.dumps({"stage": freeze["stage"], "score_sha256": freeze["new_score_sha256"],
                              "output_root": plan["planned_output_root"], "test_performance_computed": False}), flush=True)
        else:
            print(json.dumps({"stage": "SCORING_RUNTIME_PREFLIGHT_ONLY", "runtime": runtime,
                              "source_rows_read": False, "study_training_started": False, "replay_output_created": False}, indent=2))
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Scoring replay failed: {error}\n")


if __name__ == "__main__":
    main()
