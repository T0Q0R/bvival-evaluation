"""Actual base/head/neural integration on synthetic records only.

The original plan is immutable; reduced training counts are explicitly smoke
settings. This runner never opens the source workbook or issues label readiness.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from run_turkey_synthetic_integration import reduced_synthetic_plan, synthetic_dataset, synthetic_price_check
from train_turkey_nested_valuation import run_nested_valuation
from turkey_development_io import bound_plan, sha256
from turkey_execution_contract import ACTIONS, select_actions
from turkey_policy_components import allocate_policies, fit_policy_heads, post_mae, prepare_policy_material


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


def smoke_plan(original):
    plan = reduced_synthetic_plan(original)
    for candidate in plan["heads"]["candidates"]:
        candidate.update(iterations=5, depth=3)
    plan["neural_comparator"].update(maximum_epochs=3, patience=2)
    return plan


def run_synthetic(plan_freeze, output, neural_python, *, project_root):
    if output.exists() and any(output.iterdir()):
        raise ValueError("New empty integration directory required")
    original, config, _, binding = bound_plan(plan_freeze, project_root=project_root)
    plan, data = smoke_plan(original), synthetic_dataset(config)
    neural_python = neural_python.absolute()  # Preserve virtual-environment identity.
    if not neural_python.is_file():
        raise ValueError("Required neural interpreter missing")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    sources = ["turkey_policy_components.py", "turkey_neural_policy.py", "run_turkey_policy_synthetic_integration.py",
               "turkey_price_cells.py", "turkey_development_io.py", "train_turkey_nested_valuation.py",
               "run_turkey_synthetic_integration.py", "tests/test_turkey_policy_components.py",
               "tests/test_turkey_neural_policy.py"]
    for source in sources:
        with (output / (Path(source).name + ".snapshot")).open("xb") as stream:
            stream.write((project_root / "experiments" / source).read_bytes())
    for name, value in (("synthetic_config.json", plan), ("cohort_config_snapshot.json", config)):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2)
    started = time.perf_counter()
    try:
        access = synthetic_price_check()
        base_directory, head_directory = output / "valuation_models", output / "head_models"
        base_directory.mkdir(mode=0o700)
        head_directory.mkdir(mode=0o700)
        valuation = run_nested_valuation(data, plan, config, model_directory=base_directory)
        train, validation = prepare_policy_material(data, valuation, plan, config)
        heads = fit_policy_heads(train, validation, plan, config, model_directory=head_directory)
        common = select_actions(validation["keys"], ACTIONS, np.repeat(heads["scores"]["risk"][:, None], 3, axis=1),
                                plan["primary_budget"], fixed_action=heads["fixed_field"])
        task = {"stage": "synthetic_only", "scope": ["train", "validation"], "train": jsonable(train),
                "validation": jsonable(validation), "plan": plan, "config": config, "cohort": [k for k, _ in common],
                "expected_environment": binding["neural_runtime_environment"]}
        task_path = output / "synthetic_neural_task.json"
        with task_path.open("x") as stream:
            json.dump(task, stream, indent=2)
        # No source data is sent: only explicitly synthetic contexts/supervision.
        completed = subprocess.run([str(neural_python), str(project_root / "experiments/turkey_neural_policy.py"),
                                    "--task", str(task_path.absolute()), "--output-dir", str((output / "neural").absolute())],
                                   check=True, text=True, capture_output=True, timeout=120)
        neural = json.loads((output / "neural/neural_audit.json").read_text())
        if neural["risk_cohort_keys"] != task["cohort"] or neural["network_fit_count"] != 6:
            raise AssertionError("Neural comparator/common cohort incomplete")
        probability = np.asarray(neural["probability"])
        allocation = allocate_policies(validation["keys"], heads["scores"]["risk"], heads["scores"]["value"],
                                       heads["scores"]["posterror"], probability, plan,
                                       fixed_field=heads["fixed_field"], global_field=heads["global_field"])
        per_seed = {}
        for seed in plan["seeds"]:
            per_seed[str(seed)] = allocate_policies(validation["keys"], heads["per_seed_scores"]["risk"][str(seed)],
                                                     heads["per_seed_scores"]["value"][str(seed)],
                                                     heads["per_seed_scores"]["posterror"][str(seed)],
                                                     neural["per_seed_probability"][str(seed)], plan,
                                                     fixed_field=heads["fixed_field"], global_field=heads["global_field"], seed=seed)
        for allocations in [allocation, *per_seed.values()]:
            for budget, policies in allocations.items():
                reference = {k for k, _ in policies["risk_fixed_validation_best"]}
                for policy in ("risk_global_train_field", "risk_benefit_field", "risk_neural_field"):
                    if {k for k, _ in policies[policy]} != reference:
                        raise AssertionError("Risk field selectors have different listing cohorts")
        if valuation["fit_count"] != 258 or len(heads["fit_audit"]) != 21 or len(heads["candidate_trials"]) != 12:
            raise AssertionError("Required fitting schedule incomplete")
        with (output / "synthetic_allocations.json").open("x") as stream:
            json.dump({"ensemble": allocation, "per_seed": per_seed}, stream, indent=2)
        with (output / "synthetic_head_audit.json").open("x") as stream:
            json.dump(jsonable({k: v for k, v in heads.items() if k != "models"}), stream, indent=2)
        audit = {"stage": "SYNTHETIC_BASE_HEAD_NEURAL_ALLOCATION_INTEGRATION_PASSED_NOT_SOURCE_READY",
                 "source_price_values_parsed": 0, "source_models_trained": False, "source_test_scores_computed": False,
                 "source_workbook_opened": False, "development_label_release_allowed": False, "readiness_receipt_issued": False,
                 "synthetic_train_records": len(train["keys"]), "synthetic_validation_records": len(validation["keys"]),
                 "synthetic_real_valuation_fits": valuation["fit_count"], "synthetic_real_head_fits": len(heads["fit_audit"]),
                 "synthetic_real_neural_fits": neural["network_fit_count"], "head_candidate_trials": len(heads["candidate_trials"]),
                 "neural_learning_rate_trials": len(neural["candidate_trials"]), "budgets": plan["budgets"],
                 "all_eight_policies_and_three_seed_allocations_checked": True, "all_common_risk_cohorts_match": True,
                 "neural_final_fits_use_no_validation_labels": all(not a["validation_labels_used"] for a in neural["final_fit_audit"]),
                 "official_GDFS_reproduction_completed": False, "hyperparameters_reduced_for_synthetic_smoke": True,
                 "elapsed_seconds": time.perf_counter() - started, "synthetic_price_access": access,
                 "plan_sha256": binding["plan_sha256"], "cohort_audit_sha256": binding["cohort_audit_sha256"],
                 "runtime_environment": binding["runtime_environment"], "neural_runtime_environment": neural["runtime_environment"],
                 "synthetic_validation_losses_not_paper_evidence": {budget: {p: post_mae(validation, a) for p, a in policies.items()}
                                                                     for budget, policies in allocation.items()},
                 "artifact_sha256": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}}
        with (output / "synthetic_policy_integration_audit.json").open("x") as stream:
            json.dump(audit, stream, indent=2)
    except Exception as error:
        with (output / "failed_policy_integration_attempt.json").open("x") as stream:
            json.dump({"stage": "FAILED_SYNTHETIC_POLICY_INTEGRATION", "exception_class": type(error).__name__,
                       "source_price_values_parsed": 0, "development_label_release_allowed": False}, stream)
        raise
    finally:
        for path in output.rglob("*"):
            if path.is_file():
                path.chmod(0o400)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neural-python", type=Path, required=True)
    args = parser.parse_args()
    audit = run_synthetic(args.plan_freeze, args.output_dir, args.neural_python, project_root=Path(__file__).resolve().parents[1])
    print(json.dumps({k: audit[k] for k in ("stage", "synthetic_real_valuation_fits", "synthetic_real_head_fits",
                                           "synthetic_real_neural_fits", "source_price_values_parsed", "development_label_release_allowed")}))


if __name__ == "__main__":
    main()
