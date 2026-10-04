"""Bind whole feature partitions and replay saved pre-action scores, no prices.

Local freeze only: no price-release receipt, outcome evaluation or reselection.
Source execution requires an independently completed authorized development run.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_turkey_price_free_manifest import bucket, fold
from run_turkey_development import allocations_from_scores, file_hashes, jsonable, save_json, verify_files
from turkey_development_io import sha256
from turkey_execution_contract import validate_plan
from turkey_policy_components import context_frame
from turkey_price_free_scoring import initial_records, score_completed_models, verify_completed_package
from verify_turkey_price_free_bundle import table


def load_bound_initial_partitions(plan_directory, *, project_root):
    """Hash all cohort files, parse ONLY config/split/initial. No XLSX or labels.

    Acquisition CSV bytes are hashed for integrity, never decoded as values.
    Group hashes are non-verified attribute groups, not entity IDs.
    """
    path = plan_directory / "execution_plan_freeze_audit.json"
    binding = json.loads(path.read_text())
    verify_files(plan_directory, binding["frozen_snapshot_sha256"], ignored=[path.name])
    if (sha256(plan_directory / "plan_snapshot.json") != binding["plan_sha256"]
            or sha256(plan_directory / "protocol_snapshot.md") != binding["protocol_sha256"]):
        raise ValueError("Original plan/protocol binding changed")
    plan = json.loads((plan_directory / "plan_snapshot.json").read_text())
    relative = Path(plan["cohort_directory"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe cohort path")
    cohort = project_root / relative
    audit_path = cohort / "cohort_freeze_audit.json"
    if sha256(audit_path) != binding["cohort_audit_sha256"] or plan["cohort_audit_sha256"] != binding["cohort_audit_sha256"]:
        raise ValueError("Bound cohort audit differs")
    audit = json.loads(audit_path.read_text())
    verify_files(cohort, audit["output_sha256"], ignored=[audit_path.name])
    config = json.loads((cohort / "config_snapshot.json").read_text())
    validate_plan(plan, config)
    splits = table(cohort / "split_manifest.csv", ["record_key", "group_hash", "profile_hash", "split", "oof_fold"])
    initial = table(cohort / "initial_features.csv", ["record_key", *config["initial_visible_fields"]])
    if ({r["record_key"] for r in splits} != {r["record_key"] for r in initial}
            or dict(Counter(r["split"] for r in splits)) != audit["split_listing_counts"]
            or len(splits) != audit["retained_feature_rows"]
            or len({r["profile_hash"] for r in splits}) != len(splits)):
        raise ValueError("Whole-feature universe/count/profile differs from bound cohort")
    groups = defaultdict(set)
    for row in splits:
        if (row["split"] != bucket(row["group_hash"], config)
                or (row["split"] == "train" and row["oof_fold"] != str(fold(row["group_hash"], config)))
                or (row["split"] != "train" and row["oof_fold"] != "")):
            raise ValueError("Frozen split/fold rule changed")
        groups[row["group_hash"]].add(row["split"])
    if any(len(v) != 1 for v in groups.values()):
        raise ValueError("Attribute group crosses partitions")
    initial_by_key = {r["record_key"]: {f: r[f] for f in config["initial_visible_fields"]} for r in initial}
    partitions = {split: {"records": [{"record_key": r["record_key"], "initial": initial_by_key[r["record_key"]]}
                                      for r in splits if r["split"] == split],
                          "groups": {r["record_key"]: r["group_hash"] for r in splits if r["split"] == split}}
                  for split in ("calibration", "test")}
    expected = {split: sorted(r["record_key"] for r in splits if r["split"] == split) for split in partitions}
    development = {split: sorted(r["record_key"] for r in splits if r["split"] == split) for split in ("train", "validation")}
    receipt = {"mode": "authorized_development_only", "plan_sha256": binding["plan_sha256"],
               "cohort_audit_sha256": binding["cohort_audit_sha256"], "expected_keys": expected,
               "expected_development_keys": development,
               "plan_binding_sha256": sha256(path), "source_workbook_opened": False,
               "acquisition_values_decoded": False, "labels_read": False}
    return plan, config, partitions, receipt


def validate_partitions(partitions, binding, development_keys, config):
    if set(partitions) != {"calibration", "test"} or set(binding["expected_keys"]) != set(partitions):
        raise ValueError("Both complete calibration/test partitions required")
    seen, group_roles = set(development_keys), defaultdict(set)
    for split, material in partitions.items():
        if set(material) != {"records", "groups"}:
            raise ValueError("Exact initial records/group roles required, no outcomes")
        keys, _ = initial_records(material["records"], config)
        expected = binding["expected_keys"][split]
        if not isinstance(expected, list) or expected != sorted(set(expected)) or keys != expected:
            raise ValueError("Exact frozen whole-feature split universe required")
        if seen & set(keys) or set(material["groups"]) != set(keys):
            raise ValueError("Split/development overlap or incomplete groups")
        seen.update(keys)
        for group in material["groups"].values():
            if not isinstance(group, str) or not group:
                raise ValueError("Nonempty attribute groups required")
            group_roles[group].add(split)
    if any(len(roles) != 1 for roles in group_roles.values()):
        raise ValueError("Attribute groups cross held-out partitions")


def read_inference(directory):
    path = directory / "initial_only_scoring_audit.json"
    audit = json.loads(path.read_text())
    if (audit.get("stage") != "INITIAL_ONLY_SAVED_MODEL_INFERENCE_NOT_HELDOUT_FREEZE_RECEIPT"
            or audit.get("calibration_or_test_labels_read") is not False
            or audit.get("heldout_label_release_allowed") is not False):
        raise ValueError("Saved inference role changed")
    verify_files(directory, audit["artifact_sha256"], ignored=[path.name])
    return json.loads((directory / "initial_only_scores.json").read_text())


def compare_saved_inference(first, second, records, plan, config):
    keys, initial = initial_records(records, config)
    for result in (first, second):
        if set(result) != {"before", "heads", "neural", "allocations", "input_evidence_status", "refits", "reselection",
                          "calibration_or_test_labels_read", "heldout_label_release_allowed", "split_universe_not_bound_by_this_component"}:
            raise ValueError("Unexpected scoring output roles")
        before = result["before"]
        if set(before) != {"keys", "contexts", "before_prediction_log", "before_disagreement_log", "per_seed_before_raw_log"}:
            raise ValueError("Unexpected before scoring roles")
        if (before["keys"] != keys or result["refits"] != 0 or result["reselection"] is not False
                or result["input_evidence_status"] not in {"synthetic_only", "authorized_development_only"}
                or result["input_evidence_status"] != first["input_evidence_status"]
                or result["split_universe_not_bound_by_this_component"] is not True
                or result["calibration_or_test_labels_read"] is not False
                or result["heldout_label_release_allowed"] is not False):
            raise ValueError("Replay scope/refit/universe changed")
        context_frame(before["contexts"], plan, config)
        if len(before["contexts"]) != len(keys) or any(
                {f: c[f] for f in config["initial_visible_fields"]} != row
                for c, row in zip(before["contexts"], initial)):
            raise ValueError("Saved contexts do not match frozen initial fields")
        seed_logs = before["per_seed_before_raw_log"]
        if set(seed_logs) != {str(s) for s in plan["seeds"]}:
            raise ValueError("Complete raw seed prediction universe required")
        raw = np.asarray([seed_logs[str(s)] for s in plan["seeds"]], dtype=float)
        if raw.shape != (3, len(keys)) or not np.isfinite(raw).all():
            raise ValueError("Aligned finite raw before predictions required")
        for field, expected in (("before_prediction_log", np.maximum(raw.mean(axis=0), 0)),
                                ("before_disagreement_log", raw.std(axis=0, ddof=0))):
            actual = np.asarray(before[field], dtype=float)
            context_values = np.asarray([c[field] for c in before["contexts"]], dtype=float)
            if (actual.shape != (len(keys),) or not np.isfinite(actual).all()
                    or not np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
                    or not np.array_equal(actual, context_values)):
                raise ValueError("Before-context/ensemble/raw prediction identity changed")
        if jsonable(allocations_from_scores(keys, result["heads"], result["neural"], plan)) != result["allocations"]:
            raise ValueError("Saved allocations disagree with pre-action scores")
    for field in ("heads", "neural", "allocations"):
        if first[field] != second[field]:
            raise ValueError("Head/neural/allocations replay must match exactly")
    for field in ("before_prediction_log", "before_disagreement_log"):
        a, b = [np.asarray(r["before"][field], dtype=float) for r in (first, second)]
        if a.shape != (len(keys),) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, rtol=1e-12, atol=1e-12):
            raise ValueError("Before-log replay differs beyond roundoff")
    for seed in plan["seeds"]:
        a, b = [np.asarray(r["before"]["per_seed_before_raw_log"][str(seed)], dtype=float) for r in (first, second)]
        if a.shape != (len(keys),) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, rtol=1e-12, atol=1e-12):
            raise ValueError("Raw seed-log replay differs")


def freeze_partitions(development, partitions, binding, output, neural_python, *, project_root):
    if output.exists():
        raise ValueError("New nonexistent freeze directory required")
    audit, plan, config = verify_completed_package(development, project_root=project_root)
    if binding["mode"] != audit["evidence_status"]:
        raise ValueError("Synthetic/source binding mismatch")
    saved = json.loads((development / "valuation_results.json").read_text())
    roles = {"train": sorted(saved["oof"]), "validation": sorted(saved["validation"])}
    if binding["mode"] == "authorized_development_only" and (
            binding["plan_sha256"] != audit["plan_sha256"]
            or binding["cohort_audit_sha256"] != audit["cohort_audit_sha256"]
            or binding["expected_development_keys"] != roles):
        raise ValueError("Source plan/cohort/development universe differs")
    validate_partitions(partitions, binding, roles["train"] + roles["validation"], config)
    output.mkdir(parents=True, mode=0o700)
    phase = "created"
    try:
        for name in ("freeze_turkey_heldout_scores.py", "turkey_price_free_scoring.py",
                     "turkey_price_free_neural_scoring.py", "run_turkey_heldout_freeze_check.py",
                     "tests/test_freeze_turkey_heldout_scores.py"):
            with (output / (Path(name).name + ".snapshot")).open("xb") as stream:
                stream.write((project_root / "experiments" / name).read_bytes())
        save_json(output / "initial_partition_inputs.json", partitions)
        save_json(output / "partition_binding.json", binding)
        for split in ("calibration", "test"):
            phase = split + "_first_inference"
            score_completed_models(development, partitions[split]["records"], output / (split + "_first"), neural_python, project_root=project_root)
            phase = split + "_reload_inference"
            score_completed_models(development, partitions[split]["records"], output / (split + "_replay"), neural_python, project_root=project_root)
            compare_saved_inference(read_inference(output / (split + "_first")), read_inference(output / (split + "_replay")),
                                    partitions[split]["records"], plan, config)
        # Source package must remain byte-identical throughout both inference passes.
        verify_completed_package(development, project_root=project_root)
        receipt = {"stage": "TURKEY_HELDOUT_SCORES_ACTIONS_FROZEN_NO_PRICE_RELEASE" if binding["mode"] == "authorized_development_only"
                   else "SYNTHETIC_HELDOUT_FREEZE_CHECK_PASSED_NOT_SOURCE_READY", "evidence_status": binding["mode"],
                   "development_audit_sha256": sha256(development / "development_execution_audit.json"),
                   "input_development_directory": str(development.absolute()),
                   "original_plan_sha256": audit["plan_sha256"], "cohort_audit_sha256": audit["cohort_audit_sha256"],
                   "partition_counts": {s: len(m["records"]) for s, m in partitions.items()},
                   "all_budget_policy_seed_allocations_replayed_exactly": True, "refits": 0, "reselection": False,
                   "calibration_or_test_labels_read": False, "heldout_label_release_allowed": False,
                   "real_request_or_transaction_evidence": False, "local_guard_not_cryptographic_sealing": True,
                   "artifact_sha256": file_hashes(output)}
        save_json(output / "heldout_score_freeze_audit.json", receipt)
        return receipt
    except Exception as error:
        save_json(output / "failed_heldout_score_freeze.json", {"phase": phase, "exception_class": type(error).__name__,
                                                              "heldout_label_release_allowed": False,
                                                              "calibration_or_test_labels_read": False})
        raise
    finally:
        for path in output.rglob("*"):
            if path.is_file():
                path.chmod(0o400)


def run_source_freeze(development, plan_directory, output, neural_python, *, project_root):
    audit, saved_plan, saved_config = verify_completed_package(development, project_root=project_root)
    if audit["evidence_status"] != "authorized_development_only":
        raise ValueError("Source freeze requires completed authorized source development")
    plan, config, partitions, binding = load_bound_initial_partitions(plan_directory, project_root=project_root)
    if plan != saved_plan or config != saved_config:
        raise ValueError("Saved source scientific settings differ from original plan")
    return freeze_partitions(development, partitions, binding, output, neural_python, project_root=project_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-run", type=Path, required=True)
    parser.add_argument("--plan-freeze", type=Path, required=True)
    parser.add_argument("--neural-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_source_freeze(args.development_run, args.plan_freeze, args.output_dir, args.neural_python,
                               project_root=Path(__file__).resolve().parents[1])
    print(json.dumps({k: result[k] for k in ("stage", "partition_counts", "heldout_label_release_allowed")}))


if __name__ == "__main__":
    main()
