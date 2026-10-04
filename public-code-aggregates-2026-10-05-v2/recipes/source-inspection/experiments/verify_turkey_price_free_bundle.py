"""Independent audit of saved CSV boundaries/hashes; no XLSX label access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from archive_turkey_candidate import EXPECTED_SHA256
from build_turkey_price_free_manifest import bucket, digest, fold, validate_config


def table(path, expected):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected:
            raise ValueError("Unexpected table columns; information-boundary violation")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed CSV record")
    keys = [row["record_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate record key within saved table")
    return rows


def verify_bundle(directory: Path):
    audit = json.loads((directory / "cohort_freeze_audit.json").read_text())
    config = json.loads((directory / "config_snapshot.json").read_text())
    validate_config(config)
    required = {"split_manifest.csv", "initial_features.csv", "acquisition_fields.csv", "source_locators.csv",
                "excluded_records.csv", "cohort_builder_snapshot.py", "reader_snapshot.py",
                "archive_helper_snapshot.py", "config_snapshot.json", "prelabel_protocol_snapshot.md"}
    if set(audit["output_sha256"]) != required or {p.name for p in directory.iterdir()} != required | {"cohort_freeze_audit.json"}:
        raise ValueError("Unexpected files or missing frozen outputs")
    for name, expected in audit["output_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen output hash mismatch")
    splits = table(directory / "split_manifest.csv", ["record_key", "group_hash", "profile_hash", "split", "oof_fold"])
    initial = table(directory / "initial_features.csv", ["record_key", *config["initial_visible_fields"]])
    acquisition = table(directory / "acquisition_fields.csv", ["record_key", *config["action_fields"]])
    locators = table(directory / "source_locators.csv", ["record_key", "source_data_ordinal"])
    excluded = table(directory / "excluded_records.csv", ["record_key", "reason"])
    retained_keys = {r["record_key"] for r in splits}
    if any({r["record_key"] for r in rows} != retained_keys for rows in (initial, acquisition, locators)):
        raise ValueError("Initial/acquisition/locator universes differ")
    excluded_keys = {r["record_key"] for r in excluded}
    if retained_keys & excluded_keys or len({r["profile_hash"] for r in splits}) != len(splits):
        raise ValueError("Overlap in retained/excluded records or repeated retained profile")
    expected_keys = {digest([EXPECTED_SHA256, "source_data_ordinal", i]) for i in range(1, audit["raw_feature_rows"] + 1)}
    if retained_keys | excluded_keys != expected_keys:
        raise ValueError("Raw record accounting does not reconcile")
    if any(r["record_key"] != digest([EXPECTED_SHA256, "source_data_ordinal", int(r["source_data_ordinal"])]) for r in locators):
        raise ValueError("Source locator binding changed")
    groups, folds = defaultdict(set), defaultdict(set)
    for row in splits:
        if row["split"] != bucket(row["group_hash"], config):
            raise ValueError("Saved group split differs from frozen hash rule")
        groups[row["group_hash"]].add(row["split"])
        if row["split"] == "train":
            if row["oof_fold"] not in {"0", "1", "2", "3", "4"}:
                raise ValueError("Invalid train OOF fold")
            if int(row["oof_fold"]) != fold(row["group_hash"], config):
                raise ValueError("Saved OOF fold differs from frozen hash rule")
            folds[row["group_hash"]].add(row["oof_fold"])
        elif row["oof_fold"]:
            raise ValueError("Nontrain rows must not have an OOF assignment")
    if any(len(v) > 1 for v in groups.values()) or any(len(v) > 1 for v in folds.values()):
        raise ValueError("Profile group crosses split or OOF boundaries")
    split_counts = dict(Counter(r["split"] for r in splits))
    if split_counts != audit["split_listing_counts"] or len(splits) != audit["retained_feature_rows"]:
        raise ValueError("Saved count differs from audit")
    split_groups = {s: len({r["group_hash"] for r in splits if r["split"] == s}) for s in ("train", "validation", "calibration", "test")}
    if split_groups != audit["split_group_counts"]:
        raise ValueError("Saved group counts differ from audit")
    if (audit["price_cell_values_decoded"] != 0 or audit["labels_exported"]
            or audit["models_trained"] or audit["test_scores_computed"] or audit["confirmatory_test_allowed_now"]):
        raise ValueError("Bundle is not in the declared prelabel stage")
    return {"stage": "SAVED_PRICE_FREE_COHORT_VERIFIED", "hashes_checked": len(required),
            "retained_feature_rows": len(splits), "retained_profile_groups": len(groups),
            "split_listing_counts": split_counts, "cross_split_profile_groups": 0,
            "cross_oof_train_groups": 0, "initial_acquisition_columns_disjoint": True,
            "record_universes_and_source_locators_match": True, "labels_exported": False,
            "verified_vehicle_entity_independence": False, "complete_model_protocol_frozen": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = verify_bundle(args.cohort_dir)
    result["cohort_audit_sha256"] = hashlib.sha256((args.cohort_dir / "cohort_freeze_audit.json").read_bytes()).hexdigest()
    result["verifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    snapshot = args.report.with_name(args.report.stem + "_verifier_snapshot.py")
    with snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
