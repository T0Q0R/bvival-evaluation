"""Freeze planned settings and dry-run group/capacity rules without prices.

Does not claim that the future label reader or model integration is implemented.
Read-only verification of nonprice CSVs; no access to XLSX or target tables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from turkey_execution_contract import budget_count, nested_training_roles, validate_plan
from verify_turkey_price_free_bundle import verify_bundle


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_plan(config_path, protocol_path, output_dir, *, project_root, neural_python):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Freeze directory must be empty; preserve previous attempts")
    plan = json.loads(config_path.read_text())
    cohort_dir = project_root / plan["cohort_directory"]
    if sha256(cohort_dir / "cohort_freeze_audit.json") != plan["cohort_audit_sha256"]:
        raise ValueError("Plan does not bind the frozen feature cohort")
    verification = verify_bundle(cohort_dir)
    config = json.loads((cohort_dir / "config_snapshot.json").read_text())
    validate_plan(plan, config)
    with (cohort_dir / "split_manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    train = [row for row in rows if row["split"] == "train"]
    nesting = []
    for outer in range(5):
        roles = nested_training_roles(train, config, outer_fold=outer, namespace=plan["valuation"]["inner_namespace"])
        fit = [r for r in roles if r["role"] == "outer_fit"]
        holdout = [r for r in roles if r["role"] == "outer_holdout"]
        if {r["group_hash"] for r in fit} & {r["group_hash"] for r in holdout}:
            raise AssertionError("Outer holdout leaked into inner fitting")
        nesting.append({"outer_fold": outer, "heldout_listings": len(holdout), "fit_listings": len(fit),
                        "inner_listing_counts": dict(Counter(r["inner_fold"] for r in fit)),
                        "inner_group_counts": {str(i): len({r["group_hash"] for r in fit if r["inner_fold"] == i}) for i in range(3)}})
    packages = ["numpy", "scipy", "pandas", "scikit-learn", "catboost", "pytest"]
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": {p: importlib.metadata.version(p) for p in packages}}
    command = [str(neural_python), "-c", "import json,platform,importlib.metadata as m; print(json.dumps({'python':platform.python_version(),'packages':{p:m.version(p) for p in ['torch','numpy','scikit-learn','catboost']}}))"]
    neural_environment = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    source_names = [
        "experiments/turkey_execution_contract.py", "experiments/freeze_turkey_execution_plan.py",
        "experiments/turkey_valuation_components.py", "experiments/tests/test_turkey_valuation_components.py",
        "experiments/turkey_information_boundary.py", "experiments/verify_turkey_price_free_bundle.py",
        "experiments/build_turkey_price_free_manifest.py", "experiments/audit_turkey_feature_only.py",
        "experiments/archive_turkey_candidate.py", "experiments/tests/test_turkey_execution_contract.py",
        "experiments/tests/test_freeze_turkey_execution_plan.py",
        "experiments/tests/test_turkey_information_boundary.py",
    ]
    snapshot_inputs = {"plan_snapshot.json": config_path, "protocol_snapshot.md": protocol_path,
                       **{Path(name).name: project_root / name for name in source_names}}
    for name, source in snapshot_inputs.items():
        with (output_dir / name).open("xb") as stream:
            stream.write(source.read_bytes())
    result = {
        "stage": "PLANNED_SETTINGS_AND_SYNTHETIC_COMPONENTS_FROZEN_NOT_TRAINING_READY",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha256(config_path), "protocol_sha256": sha256(protocol_path),
        "cohort_audit_sha256": sha256(cohort_dir / "cohort_freeze_audit.json"),
        "cohort_verification": verification, "nested_train_roles": nesting,
        "planned_action_counts_by_partition": {
            split: {str(b): budget_count(n, b) for b in plan["budgets"]}
            for split, n in verification["split_listing_counts"].items()},
        "runtime_environment": environment, "neural_runtime_environment": neural_environment,
        "valuation_candidate_evaluations_per_family": 4,
        "base_fit_count_plan": {"nested_inner_selection": 5 * 3 * 12, "outer_three_seed_oof": 5 * 3,
                                "final_train_cv_selection": 5 * 12, "final_train_three_seed": 3},
        "base_fit_count_excludes_heads_neural_and_seed_diagnostics": True,
        "price_values_parsed": 0, "labels_exported": False, "models_trained": False,
        "test_scores_computed": False, "training_execution_ready": False,
        "development_label_release_allowed_now": False, "external_registration": False,
        "future_runner_integration_required": True, "official_GDFS_reproduction_completed": False,
        "frozen_snapshot_sha256": {name: sha256(output_dir / name) for name in sorted(snapshot_inputs)},
    }
    with (output_dir / "execution_plan_freeze_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    for file in output_dir.iterdir():
        file.chmod(0o400)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neural-python", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_plan(args.config.resolve(), args.protocol.resolve(), args.output_dir.resolve(),
                         project_root=Path(__file__).resolve().parents[1], neural_python=args.neural_python.absolute())
    print(json.dumps({k: result[k] for k in ("stage", "plan_sha256", "price_values_parsed",
                                           "models_trained", "development_label_release_allowed_now",
                                           "planned_action_counts_by_partition")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
