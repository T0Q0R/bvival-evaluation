"""Bounded historical Turkey/AutoScout dependency audit, not empirical replay.

Reads explicitly named audit/config JSONs and opaque source-code snapshots only.
Does not read neural_audit.json (it embeds row-level probabilities), source
tables, scores, labels, models or ledger files. Never imports study modules.
Recorded hash links are not verification of the corresponding row/model bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mucars_reconstruction import safe_path, recorded_graph_checks

TURKEY = {
    "plan": "experiments/outputs/turkey_execution_plan_freeze_v1_2026-09-30/execution_plan_freeze_audit.json",
    "config": "experiments/configs/turkey_execution_plan_v1_2026-09-30.json",
    "labels": "experiments/outputs/turkey_source_development_2026-09-30_v1/labels/development_label_release_audit.json",
    "development": "experiments/outputs/turkey_source_development_2026-09-30_v1/training/development_execution_audit.json",
    "freeze": "experiments/outputs/turkey_heldout_score_freeze_source_2026-09-30_v1/heldout_score_freeze_audit.json",
    "calibration_release": "experiments/outputs/turkey_calibration_release_2026-09-30_v1/once_phase_release_audit.json",
    "release": "experiments/outputs/turkey_test_release_2026-10-01_v1/once_phase_release_audit.json",
    "erratum": "experiments/outputs/turkey_test_accounting_erratum_2026-10-01_v1/phase_accounting_erratum.json",
    "evaluation": "experiments/outputs/turkey_test_fixed_evaluation_2026-10-01_v1/released_phase_evaluation_audit.json",
}
AUTO = {
    "cohort": "experiments/data/processed/autoscout24_bvival_v2/autoscout24_bvival_audit.json",
    **{k: "experiments/outputs/autoscout24_bvival_v2_" + v for k, v in {
        "pairs": "full_pairs/prediction_pair_generation_audit.json",
        "development": "full_actions_dev/bvival_action_dataset_audit.json",
        "calibration": "full_actions_cal/bvival_action_dataset_audit.json",
        "value": "full_value/value_policy_fit_audit.json",
        "risk": "full_risk/risk_baseline_fit_audit.json",
        "assembly": "full_scores/policy_score_assembly_audit_absolute_price_error.json",
        "join": "full_evaluation_inputs/evaluation_outcome_join_audit.json",
        "evaluation": "full_mae_test/bvival_evaluation_audit.json",
    }.items()},
}
RIGHTS_NOTE = "notes/analysis/autoscout24-rights-metadata-correction-2026-09-26.md"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_audits(root, paths):
    values, hashes = {}, {}
    for role, relative in paths.items():
        path = safe_path(root, relative)
        values[role] = json.loads(path.read_text())
        hashes[relative] = digest(path)
    return values, hashes


def turkey_checks(a, hashes):
    p, c, l, d, f, cal, r, x, e = (a[k] for k in TURKEY)
    checks = []

    def eq(name, left, right):
        if left is None or right is None or type(left) is not type(right) or left != right:
            raise ValueError("Turkey metadata mismatch: " + name)
        checks.append(name)

    def link(name, left, right):
        for value in (left, right):
            if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError("Malformed dependency hash: " + name)
        eq(name, left, right)

    link("current_config_to_plan", hashes[TURKEY["config"]], p["plan_sha256"])
    for role, obj, field in (("labels", l, "plan_sha256"), ("development", d, "plan_sha256"),
                             ("freeze", f, "original_plan_sha256")):
        link("plan_to_" + role, p["plan_sha256"], obj[field])
        link("cohort_to_" + role, p["cohort_audit_sha256"], obj["cohort_audit_sha256"])
    link("labels_to_development", hashes[TURKEY["labels"]], d["provenance"]["development_label_audit_sha256"])
    link("readiness_provenance", l["readiness_receipt_sha256"], d["provenance"]["readiness_sha256"])
    link("development_to_freeze", hashes[TURKEY["development"]], f["development_audit_sha256"])
    link("calibration_to_test_release", hashes[TURKEY["calibration_release"]], r["prior_calibration_release_audit_sha256"])
    link("release_to_erratum", hashes[TURKEY["release"]], x["input_release_audit_sha256"])
    link("release_to_evaluation", hashes[TURKEY["release"]], e["release_audit_sha256"])
    link("freeze_to_evaluation", hashes[TURKEY["freeze"]], e["score_freeze_audit_sha256"])
    link("erratum_to_evaluation", hashes[TURKEY["erratum"]], e["accounting_erratum"]["audit_sha256"])
    eq("correction_carried_into_evaluation", x["corrected_cell_access"], e["accounting_erratum"]["corrected_cell_access"])
    eq("legacy_counters_preserved", x["original_cell_access"], r["cell_access"])
    eq("historical_development_stage", d["stage"], "SOURCE_DEVELOPMENT_COMPLETED_NOT_TEST_FROZEN")
    eq("historical_freeze_stage", f["stage"], "TURKEY_HELDOUT_SCORES_ACTIONS_FROZEN_NO_PRICE_RELEASE")
    for role, obj in (("calibration", cal), ("test", r)):
        eq(role + "_phase", obj["phase"], role)
        eq(role + "_already_released", obj["release_succeeded"], True)
    eq("test_evaluated", e["stage"], "SOURCE_RELEASED_PHASE_FIXED_EVALUATION_COMPLETE")
    eq("test_records", e["feature_records"], f["partition_counts"]["test"])
    eq("two_primary_comparisons", e["primary_comparisons"], 2)
    eq("base_fit_count", d["valuation_fits"], sum(p["base_fit_count_plan"].values()))
    eq("main_environment_recorded_consistently", d["runtime_environment"], p["runtime_environment"])
    eq("neural_environment_recorded_consistently", d["neural_runtime_environment"], p["neural_runtime_environment"])
    for field in ("source_workbook_opened_here", "cached_target_values_decoded_here", "original_package_modified",
                  "models_or_actions_changed", "phase_reopened"):
        eq("erratum_" + field, x[field], False)
    return checks


def snapshot_checks(root, a):
    """Hash only snapshot Python source; never open another artifact type."""
    records = []
    for role in ("development", "freeze", "evaluation", "erratum"):
        directory = TURKEY[role].rsplit("/", 1)[0]
        for name, expected in a[role]["artifact_sha256"].items():
            if not name.endswith(".py.snapshot"):
                continue
            path = safe_path(root, directory + "/" + name)
            actual = digest(path)
            if actual != expected:
                raise ValueError("Bound code snapshot changed: " + name)
            basename = Path(name).name.removesuffix(".snapshot")
            current_relative = "experiments/" + ("tests/" if basename.startswith("test_") else "") + basename
            current = digest(safe_path(root, current_relative))
            records.append({"role": role, "snapshot_path": directory + "/" + name,
                            "snapshot_sha256": actual, "current_path": current_relative,
                            "current_sha256": current, "current_matches_snapshot": current == actual})
    if not records:
        raise ValueError("No bound source snapshots checked")
    return records


def audit(root):
    root = Path(root).resolve(strict=True)
    t, th = load_audits(root, TURKEY)
    a, ah = load_audits(root, AUTO)
    tc = turkey_checks(t, th)
    ac = recorded_graph_checks(a)
    snapshots = snapshot_checks(root, t)
    note_hash = digest(safe_path(root, RIGHTS_NOTE))
    return {
        "date": "2026-10-01", "stage": "REMAINING_SOURCE_RECORDED_DEPENDENCIES_CHECKED_NOT_REPLAYED",
        "audit_file_sha256": {**th, **ah},
        "Turkey": {
            "passed_dependency_checks": tc, "historical_tests_already_opened": True,
            "historical_fit_counts": {"valuation": t["development"]["valuation_fits"],
                                      "heads": t["development"]["head_fits"], "neural": t["development"]["neural_fits"]},
            "historical_partition_counts": t["freeze"]["partition_counts"],
            "recorded_main_environment": t["development"]["runtime_environment"],
            "recorded_neural_environment": t["development"]["neural_runtime_environment"],
            "code_snapshot_checks": snapshots,
            "current_code_matches_all_checked_snapshots": all(r["current_matches_snapshot"] for r in snapshots),
            "original_accounting_count": t["erratum"]["original_cell_access"]["source_data_rows_scanned"],
            "corrected_accounting_count": t["erratum"]["corrected_cell_access"]["source_data_rows_scanned"],
            "next_gate": "bind existing release/ledger and saved-model bytes; use cached labels only; validate main/neural runtimes separately",
            "saved_model_or_label_bytes_verified": False,
            "fresh_training_or_outcome_reconstruction_completed": False,
            "unopened_test_or_new_confirmation": False,
        },
        "AutoScout24": {
            "passed_dependency_checks": ac, "historical_tests_already_opened": True,
            "historical_split_counts": a["cohort"]["split_counts"],
            "recorded_base_parameters": a["pairs"]["model_parameters"],
            "recorded_folds": a["pairs"]["folds"], "recorded_seeds": a["pairs"]["seeds"],
            "recorded_primary_budget": a["evaluation"]["budgets"],
            "recorded_fixed_field": a["assembly"]["resolved_fixed_action"],
            "rights_correction_note": RIGHTS_NOTE, "rights_correction_note_sha256": note_hash,
            "legacy_cohort_rights_text_is_not_current_permission_decision": True,
            "underlying_data_or_derivative_publication_rights_adjudicated_here": False,
            "saved_model_or_label_bytes_verified": False,
            "fresh_training_or_outcome_reconstruction_completed": False,
            "next_gate": "recover exact final-head/field-selection and source-version contracts; keep rights decision separate from local technical checks",
        },
        "read_scope": {
            "explicit_metadata_json_files": len(TURKEY) + len(AUTO),
            "neural_audit_with_embedded_record_probabilities_read": False,
            "raw_source_or_row_level_table_opened": False, "saved_models_deserialized": False,
            "ledger_read_or_modified": False, "study_modules_imported": False,
            "training_prediction_or_outcome_evaluation_called": False,
            "source_release_called": False, "public_archive_or_license_created": False,
        },
        "all_four_sources_reproduced": False, "independent_reproduction": False,
        "submission_ready": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        # Explicit caller-selected file, no parent creation or overwrite.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        print(json.dumps({"stage": report["stage"], "output_sha256": digest(args.output),
                          "Turkey_checks": len(report["Turkey"]["passed_dependency_checks"]),
                          "AutoScout24_checks": len(report["AutoScout24"]["passed_dependency_checks"])}))
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
