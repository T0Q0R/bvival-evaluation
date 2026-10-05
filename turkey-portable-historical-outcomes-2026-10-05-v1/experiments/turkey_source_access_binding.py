"""Price-free original-row binding and source prerequisites, NOT release authority.

Never opens XLSX or reads candidate values/targets. A successful preflight is
not a readiness receipt: a controlled source release controller remains needed.
The local ledger location is fixed by this module, not a caller/CLI parameter.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from archive_turkey_candidate import EXPECTED_BLOB, EXPECTED_SHA256, EXPECTED_SIZE
from build_turkey_price_free_manifest import digest
from freeze_turkey_heldout_scores import load_bound_initial_partitions
from run_turkey_development import file_hashes, save_json
from turkey_development_io import sha256
from verify_turkey_heldout_score_freeze import verify_score_freeze
from verify_turkey_price_free_bundle import table


RAW_ROWS = 52051
SPLIT_COUNTS = {"train": 28024, "validation": 7767, "calibration": 7814, "test": 7802}
SOURCE_VERSION = {"bytes": EXPECTED_SIZE, "sha256": EXPECTED_SHA256, "git_blob_sha1": EXPECTED_BLOB}


def canonical_source_ledger():
    """Resolve only; do not create a ledger, reserve a phase or read outcomes.

    Fail on existing symlink/non-directory ancestors instead of permitting a
    redirected ledger. Not a defence against concurrent filesystem mutation,
    copying the project, or manual deletion; not cryptographic/global sealing.
    """
    project = Path(__file__).absolute().parents[1]
    ledger = project / "experiments/access_ledgers/turkey_heldout_v1"
    for path in (ledger, *ledger.parents):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError("Canonical source ledger cannot traverse symlinks or files")
    return ledger


def validate_source_coordinates(locators, splits, excluded, *, raw_rows, split_counts):
    """Bind full retained + excluded ordinal universe, never action-selected rows."""
    if type(raw_rows) is not int or raw_rows != RAW_ROWS or split_counts != SPLIT_COUNTS:
        raise ValueError("Original pinned row/split inventory required")
    if not all(isinstance(rows, list) for rows in (locators, splits, excluded)):
        raise ValueError("Exact coordinate, role and exclusion lists required")
    roles = {}
    for row in splits:
        if (not isinstance(row, dict) or set(row) != {"record_key", "group_hash", "profile_hash", "split", "oof_fold"}
                or not isinstance(row["record_key"], str) or row["record_key"] in roles
                or row["split"] not in SPLIT_COUNTS):
            raise ValueError("Unique frozen record roles required")
        roles[row["record_key"]] = row["split"]
    if dict(Counter(roles.values())) != SPLIT_COUNTS:
        raise ValueError("Complete original split counts required, not action cohort")
    manifest, seen_keys, seen_ordinals = [], set(), set()
    for row in locators:
        if (not isinstance(row, dict) or set(row) != {"record_key", "source_data_ordinal"}
                or not isinstance(row["record_key"], str)
                or not isinstance(row["source_data_ordinal"], str)
                or not re.fullmatch(r"[1-9][0-9]*", row["source_data_ordinal"])):
            raise ValueError("Canonical positive integer ordinal CSV tokens required")
        key, ordinal = row["record_key"], int(row["source_data_ordinal"])
        if (ordinal > raw_rows or ordinal in seen_ordinals or key in seen_keys
                or key not in roles or key != digest([EXPECTED_SHA256, "source_data_ordinal", ordinal])):
            raise ValueError("Unique original ordinal-to-key binding required")
        seen_keys.add(key)
        seen_ordinals.add(ordinal)
        manifest.append({"record_key": key, "source_data_ordinal": ordinal, "split": roles[key]})
    if seen_keys != set(roles):
        raise ValueError("Locators must cover every frozen retained record")
    excluded_keys = set()
    for row in excluded:
        if (not isinstance(row, dict) or set(row) != {"record_key", "reason"}
                or not isinstance(row["record_key"], str) or row["record_key"] in excluded_keys
                or not isinstance(row["reason"], str) or not row["reason"]):
            raise ValueError("Unique original nonprice exclusions required")
        excluded_keys.add(row["record_key"])
    raw_keys = {digest([EXPECTED_SHA256, "source_data_ordinal", i]) for i in range(1, raw_rows + 1)}
    if seen_keys & excluded_keys or seen_keys | excluded_keys != raw_keys:
        raise ValueError("Retained/excluded universe must exhaust original source ordinals")
    return sorted(manifest, key=lambda row: row["record_key"])


def bind_original_source_coordinates(plan_directory, *, project_root):
    # Existing loader verifies original plan/protocol + ALL cohort file hashes
    # before any new locator/exclusion CSV is decoded. XLSX is never opened.
    plan, config, partitions, binding = load_bound_initial_partitions(plan_directory, project_root=project_root)
    cohort = project_root / plan["cohort_directory"]
    audit = json.loads((cohort / "cohort_freeze_audit.json").read_text())
    if audit.get("source_verified") != SOURCE_VERSION:
        raise ValueError("Original pinned archive provenance differs")
    manifest = validate_source_coordinates(
        table(cohort / "source_locators.csv", ["record_key", "source_data_ordinal"]),
        table(cohort / "split_manifest.csv", ["record_key", "group_hash", "profile_hash", "split", "oof_fold"]),
        table(cohort / "excluded_records.csv", ["record_key", "reason"]),
        raw_rows=audit["raw_feature_rows"], split_counts=audit["split_listing_counts"])
    expected = {**binding["expected_development_keys"], **binding["expected_keys"]}
    actual = {s: [r["record_key"] for r in manifest if r["split"] == s] for s in SPLIT_COUNTS}
    if expected != actual or any(sorted(partitions[s]["groups"]) != actual[s] for s in partitions):
        raise ValueError("Original complete partition/coordinate universe differs")
    return {"stage": "SOURCE_PRICE_FREE_COORDINATES_BOUND_NOT_RELEASE_READY",
            "evidence_status": "source_coordinate_integrity_only_not_empirical_result",
            "source_version": SOURCE_VERSION, "raw_data_rows": RAW_ROWS,
            "retained_rows": len(manifest), "excluded_rows": RAW_ROWS - len(manifest),
            "split_counts": SPLIT_COUNTS, "plan_sha256": binding["plan_sha256"],
            "cohort_audit_sha256": binding["cohort_audit_sha256"],
            "plan_binding_sha256": binding["plan_binding_sha256"],
            "locator_csv_sha256": sha256(cohort / "source_locators.csv"),
            "coordinate_manifest_sha256": digest(manifest), "manifest": manifest,
            "source_workbook_opened": False, "acquisition_values_decoded": False,
            "calibration_or_test_labels_read": False, "price_release_allowed": False,
            "source_release_ready": False, "ledger_reserved": False,
            "verified_vehicle_entity_independence": False,
            "train_validation_targets_already_open_in_separate_development_run": True}


def source_prerequisite_preflight(freeze_directory, development, plan_directory, *, project_root):
    """Reject partial/synthetic freeze BEFORE its artifacts/models/labels load.

    Even success authorizes no target access. The future release controller
    must independently verify readiness, source bytes and once-only reservation.
    """
    path = freeze_directory / "heldout_score_freeze_audit.json"
    if not path.is_file():
        raise ValueError("Completed real-source held-out score freeze required")
    header = json.loads(path.read_text())
    if (header.get("stage") != "TURKEY_HELDOUT_SCORES_ACTIONS_FROZEN_NO_PRICE_RELEASE"
            or header.get("evidence_status") != "authorized_development_only"):
        raise ValueError("Synthetic/partial evidence cannot satisfy source prerequisites")
    verified = verify_score_freeze(freeze_directory, development, project_root=project_root, plan_directory=plan_directory)
    coordinates = bind_original_source_coordinates(plan_directory, project_root=project_root)
    if (verified["audit"]["original_plan_sha256"] != coordinates["plan_sha256"]
            or verified["audit"]["cohort_audit_sha256"] != coordinates["cohort_audit_sha256"]
            or any(sorted(verified["partitions"][s]["groups"]) !=
                   [r["record_key"] for r in coordinates["manifest"] if r["split"] == s]
                   for s in ("calibration", "test"))):
        raise ValueError("Verified freeze and original source coordinates differ")
    return {"stage": "SOURCE_PREREQUISITES_CHECKED_NOT_RELEASE_AUTHORIZATION",
            "score_freeze_audit_sha256": verified["audit_sha256"],
            "coordinate_manifest_sha256": coordinates["coordinate_manifest_sha256"],
            "canonical_ledger_directory": str(canonical_source_ledger()),
            "source_workbook_opened": False, "calibration_or_test_labels_read": False,
            "ledger_reserved": False, "source_release_ready": False, "price_release_allowed": False}


def write_coordinate_audit(plan_directory, output, *, project_root):
    if output.exists():
        raise ValueError("Preserve previous attempts; new nonexistent output required")
    result = bind_original_source_coordinates(plan_directory, project_root=project_root)
    result["canonical_ledger_directory"] = str(canonical_source_ledger())
    output.mkdir(parents=True, mode=0o700)
    try:
        manifest = result.pop("manifest")
        save_json(output / "coordinate_manifest.json", manifest)
        for name in ("turkey_source_access_binding.py", "tests/test_turkey_source_access_binding.py"):
            with (output / (Path(name).name + ".snapshot")).open("xb") as stream:
                stream.write((project_root / "experiments" / name).read_bytes())
        result["artifact_sha256"] = file_hashes(output)
        save_json(output / "source_coordinate_binding_audit.json", result)
    finally:
        for path in output.iterdir():
            if path.is_file():
                path.chmod(0o400)
    return result


def main():
    parser = argparse.ArgumentParser(description="Price-free coordinate audit only; no source price release")
    parser.add_argument("--plan-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).absolute().parents[1]
    print(json.dumps(write_coordinate_audit(args.plan_freeze, args.output_dir, project_root=root), ensure_ascii=False))


if __name__ == "__main__":
    main()
