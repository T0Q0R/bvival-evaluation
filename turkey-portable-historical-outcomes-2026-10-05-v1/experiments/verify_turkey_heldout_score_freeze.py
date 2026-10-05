"""Read-only freeze integrity/role audit. Not a price release or outcome reader."""
from __future__ import annotations

import json
from pathlib import Path

from freeze_turkey_heldout_scores import (compare_saved_inference, load_bound_initial_partitions,
                                         read_inference, validate_partitions)
from run_turkey_development import verify_files
from turkey_development_io import sha256
from turkey_price_free_scoring import verify_completed_package


def verify_score_freeze(directory, development, *, project_root, plan_directory=None):
    path = directory / "heldout_score_freeze_audit.json"
    if not path.is_file():
        raise ValueError("Completed held-out score freeze required; partial attempts cannot authorize anything")
    audit = json.loads(path.read_text())
    stages = {"TURKEY_HELDOUT_SCORES_ACTIONS_FROZEN_NO_PRICE_RELEASE": "authorized_development_only",
              "SYNTHETIC_HELDOUT_FREEZE_CHECK_PASSED_NOT_SOURCE_READY": "synthetic_only"}
    if (audit.get("stage") not in stages or audit.get("evidence_status") != stages[audit["stage"]]
            or audit.get("all_budget_policy_seed_allocations_replayed_exactly") is not True
            or audit.get("refits") != 0 or audit.get("reselection") is not False
            or audit.get("calibration_or_test_labels_read") is not False
            or audit.get("heldout_label_release_allowed") is not False):
        raise ValueError("Complete no-label freeze with exact replay required")
    verify_files(directory, audit["artifact_sha256"], ignored=[path.name])
    for name in ("freeze_turkey_heldout_scores.py", "turkey_price_free_scoring.py", "turkey_price_free_neural_scoring.py"):
        if (directory / (name + ".snapshot")).read_bytes() != (project_root / "experiments" / name).read_bytes():
            raise ValueError("Freeze/inference runtime differs from verified snapshots")
    dev_audit, plan, config = verify_completed_package(development, project_root=project_root)
    if (audit["development_audit_sha256"] != sha256(development / "development_execution_audit.json")
            or Path(audit["input_development_directory"]).resolve() != development.resolve()
            or audit["evidence_status"] != dev_audit["evidence_status"]
            or audit["original_plan_sha256"] != dev_audit["plan_sha256"]
            or audit["cohort_audit_sha256"] != dev_audit["cohort_audit_sha256"]):
        raise ValueError("Freeze bound to another development package")
    partitions = json.loads((directory / "initial_partition_inputs.json").read_text())
    binding = json.loads((directory / "partition_binding.json").read_text())
    if binding["mode"] != audit["evidence_status"]:
        raise ValueError("Freeze input binding evidence role differs")
    saved = json.loads((development / "valuation_results.json").read_text())
    validate_partitions(partitions, binding, list(saved["oof"]) + list(saved["validation"]), config)
    if audit["partition_counts"] != {s: len(m["records"]) for s, m in partitions.items()}:
        raise ValueError("Whole-partition freeze counts differ")
    if audit["evidence_status"] == "authorized_development_only":
        if plan_directory is None:
            raise ValueError("Source freeze must be re-bound to original plan/cohort")
        original_plan, original_config, original_partitions, original_binding = load_bound_initial_partitions(plan_directory, project_root=project_root)
        if (plan != original_plan or config != original_config or partitions != original_partitions or binding != original_binding):
            raise ValueError("Source freeze differs from original scientific input universe")
    scores = {}
    for split in ("calibration", "test"):
        first, replay = (read_inference(directory / (split + suffix)) for suffix in ("_first", "_replay"))
        if first["input_evidence_status"] != audit["evidence_status"]:
            raise ValueError("Partition scoring evidence role differs")
        compare_saved_inference(first, replay, partitions[split]["records"], plan, config)
        scores[split] = first
    return {"audit": audit, "audit_sha256": sha256(path), "partitions": partitions,
            "scores": scores, "plan": plan, "config": config, "price_release_allowed": False,
            "outcomes_read_this_verifier": False}
