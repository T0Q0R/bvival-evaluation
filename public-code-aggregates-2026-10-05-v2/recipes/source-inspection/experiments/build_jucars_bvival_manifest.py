"""Build a leakage-conscious external JUCars-2024 manifest for BVI-Val.

The source has no advertisement identifier. We therefore use a SHA-256
fingerprint of all non-target listing attributes as a conservative entity key,
drop duplicate fingerprints before splitting, and document the possibility of
over-collapsing distinct vehicles with identical observed attributes.

This builder writes no model predictions or performance metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATASET_DOI = "10.17632/ddcz486x5t.2"
DATASET_LICENSE = "CC BY 4.0"
TARGET_COLUMN = "price_value_JOD"
ACTION_FIELDS = ("mileage_km", "transmission", "fuel_type", "feat_body_condition")
SPLIT_RULES = (
    ("train", 0, 54),
    ("validation", 55, 69),
    ("calibration", 70, 84),
    ("test", 85, 99),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-dictionary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-record-sha256", required=True)
    parser.add_argument("--dictionary-record-sha256", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_value(value: object) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value).strip()


def row_fingerprint(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    columns = list(columns)
    if not columns:
        raise ValueError("Fingerprint requires at least one column")
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Fingerprint columns are missing: {missing}")

    def digest_row(row: pd.Series) -> str:
        payload = "\x1f".join(_normalized_value(row[column]) for column in columns)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return frame.apply(digest_row, axis=1)


def split_from_listing_id(listing_id: str) -> str:
    bucket = int(str(listing_id)[:8], 16) % 100
    for name, lower, upper in SPLIT_RULES:
        if lower <= bucket <= upper:
            return name
    raise AssertionError(f"No split rule covers bucket {bucket}")


def build_manifest(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {
        TARGET_COLUMN,
        "condition",
        "year",
        "make",
        "model",
        *ACTION_FIELDS,
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"JUCars source is missing required columns: {missing}")
    raw_count = int(len(source))
    working = source.copy()
    working["source_row_number"] = np.arange(len(working), dtype=int)
    condition = working["condition"].astype("string").str.strip().str.casefold()
    target = pd.to_numeric(working[TARGET_COLUMN], errors="coerce")
    year = pd.to_numeric(working["year"], errors="coerce")
    valid_used = condition.eq("used")
    valid_target = target.notna() & np.isfinite(target) & target.gt(0)
    valid_year = year.notna() & np.isfinite(year) & year.between(1980, 2026)
    valid = valid_used & valid_target & valid_year
    exclusions = {
        "not_used_or_missing_condition": int((~valid_used).sum()),
        "invalid_or_missing_positive_price": int((~valid_target).sum()),
        "implausible_or_missing_year": int((~valid_year).sum()),
    }
    working = working.loc[valid].copy()
    working[TARGET_COLUMN] = target.loc[valid].astype(float)
    working["year"] = year.loc[valid].astype(int)

    non_target_columns = sorted(
        column
        for column in source.columns
        if column != TARGET_COLUMN
    )
    working["listing_id"] = row_fingerprint(working, non_target_columns)
    duplicate_entity_rows = int(working["listing_id"].duplicated(keep="first").sum())
    working = working.drop_duplicates("listing_id", keep="first").copy()
    working["target_log"] = np.log1p(working[TARGET_COLUMN].to_numpy(dtype=float))
    working["split"] = working["listing_id"].map(split_from_listing_id)

    if working["listing_id"].duplicated().any():
        raise AssertionError("Listing identifiers must be unique after conservative deduplication")
    split_sets = {
        split: set(group["listing_id"])
        for split, group in working.groupby("split", sort=True)
    }
    overlap = 0
    split_names = sorted(split_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap += len(split_sets[left].intersection(split_sets[right]))
    if overlap:
        raise AssertionError("Listing identifier overlap detected across splits")

    action_eligibility = {
        field: int(working[field].notna().sum()) for field in ACTION_FIELDS
    }
    audit = {
        "dataset": "JUCars-2024",
        "dataset_version": 2,
        "dataset_doi": DATASET_DOI,
        "dataset_license": DATASET_LICENSE,
        "raw_row_count": raw_count,
        "conservative_non_target_duplicate_rows_removed": duplicate_entity_rows,
        "retained_used_listing_count": int(len(working)),
        "exclusion_flag_counts_before_joint_filter": exclusions,
        "split_counts": {
            str(key): int(value)
            for key, value in working["split"].value_counts().sort_index().items()
        },
        "cross_split_listing_id_overlap": overlap,
        "action_fields": list(ACTION_FIELDS),
        "action_eligibility_counts": action_eligibility,
        "split_rule": "sha256_non_target_listing_fingerprint_first8_mod100",
        "split_buckets": {
            name: [lower, upper] for name, lower, upper in SPLIT_RULES
        },
        "target": "log1p advertised price in Jordanian dinar",
        "performance_metrics_computed": False,
        "known_limitation": (
            "No source advert ID is available. Fingerprinting all non-target attributes may "
            "over-collapse distinct vehicles with identical observed attributes, but prevents "
            "exact observed-attribute duplicates from crossing splits."
        ),
    }
    return working, audit


def write_manifest_outputs(
    manifest: pd.DataFrame,
    audit: dict[str, object],
    *,
    output_dir: Path,
    source: Path,
    dictionary: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / "splits"
    label_dir = output_dir / "sealed_labels"
    split_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "jucars_bvival_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    target_columns = ["listing_id", TARGET_COLUMN, "target_log"]
    feature_columns = [
        column
        for column in manifest.columns
        if column not in {TARGET_COLUMN, "target_log", "split"}
    ]
    output_hashes: dict[str, str] = {manifest_path.name: file_sha256(manifest_path)}
    for split in ["train", "validation", "calibration"]:
        path = split_dir / f"{split}.csv"
        manifest.loc[manifest["split"].eq(split)].drop(columns="split").to_csv(path, index=False)
        output_hashes[str(path.relative_to(output_dir))] = file_sha256(path)
    test = manifest.loc[manifest["split"].eq("test")].copy()
    test_features_path = split_dir / "test_features.csv"
    test_labels_path = label_dir / "test_labels.csv"
    test[feature_columns].to_csv(test_features_path, index=False)
    test[target_columns].to_csv(test_labels_path, index=False)
    test_labels_path.chmod(0o444)
    output_hashes[str(test_features_path.relative_to(output_dir))] = file_sha256(test_features_path)
    output_hashes[str(test_labels_path.relative_to(output_dir))] = file_sha256(test_labels_path)

    audit.update(
        {
            "source_file": str(source),
            "source_file_sha256": file_sha256(source),
            "data_dictionary_file": str(dictionary),
            "data_dictionary_sha256": file_sha256(dictionary),
            "outputs": output_hashes,
            "test_target_absent_from_test_features": TARGET_COLUMN not in feature_columns
            and "target_log" not in feature_columns,
            "test_label_permissions": oct(test_labels_path.stat().st_mode & 0o777),
            "evidence_status": "prospective_external_replication_manifest_before_model_evaluation",
        }
    )
    audit_path = output_dir / "jucars_bvival_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    actual_source_hash = file_sha256(args.source)
    actual_dictionary_hash = file_sha256(args.data_dictionary)
    if actual_source_hash != args.source_record_sha256:
        raise ValueError("Source CSV hash does not match the recorded repository file hash")
    if actual_dictionary_hash != args.dictionary_record_sha256:
        raise ValueError("Data dictionary hash does not match the recorded repository file hash")
    source = pd.read_csv(args.source)
    manifest, audit = build_manifest(source)
    write_manifest_outputs(
        manifest,
        audit,
        output_dir=args.output_dir,
        source=args.source,
        dictionary=args.data_dictionary,
    )


if __name__ == "__main__":
    main()
