"""Build a sealed MUCars-2024 cohort for structured acquisition research.

The source has no public advertisement identifier. Exact non-price attribute
fingerprints are collapsed before deterministic splitting. This script writes
no model predictions or performance metrics.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from build_jucars_bvival_manifest import row_fingerprint, split_from_listing_id


SOURCE_DOI = "10.17632/vjrbcb2rrt.2"
SOURCE_SHA256 = "cb3574e38f34adfd70ee08d18c7748899fe86dc01338b50562bf91bcd145e749"
ACTION_FIELDS = ("mileage_km", "transmission", "fuel_type", "feat_body_condition")
PRICE_MIN_MAD = 5000.0
PRICE_MAX_MAD = 2000000.0
SOURCE_COLUMNS = (
    "Brand", "Model", "Year", "Condition", "Mileage", "Gearbox",
    "Fiscal Power", "Fuel", "Equipment", "Number of Doors", "Origin",
    "First Owner", "Location", "Sector", "Price",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mileage_band_numeric(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    numbers = [int(token.replace(" ", "")) for token in re.findall(r"\d[\d ]*", str(value))]
    if not numbers:
        return float("nan")
    if len(numbers) == 1:
        return float(numbers[0])
    return float((numbers[0] + numbers[1]) / 2.0)


def numeric_first_token(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    match = re.search(r"\d+", str(value))
    return float(match.group()) if match else float("nan")


def equipment_count(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return float("nan")
    return float(len(parsed)) if isinstance(parsed, list) else float("nan")


def build_manifest(source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = sorted(set(SOURCE_COLUMNS).difference(source.columns))
    if missing:
        raise ValueError(f"MUCars source lacks expected columns: {missing}")
    working = source[list(SOURCE_COLUMNS)].copy()
    working["source_row_number"] = np.arange(len(working), dtype=int)
    price = pd.to_numeric(working["Price"], errors="coerce")
    year = working["Year"].map(numeric_first_token)
    used = ~working["Condition"].fillna("").astype(str).str.strip().str.casefold().eq("new")
    valid_price = price.between(PRICE_MIN_MAD, PRICE_MAX_MAD)
    valid_year = year.between(1980, 2025)
    valid = used & valid_price & valid_year
    exclusions = {
        "new_condition": int((~used).sum()),
        "missing_or_outside_price_range": int((~valid_price).sum()),
        "missing_or_outside_model_year_range": int((~valid_year).sum()),
    }
    working = working.loc[valid].copy()
    working["source_row_number"] = working["source_row_number"].astype(int)
    working["listing_id"] = row_fingerprint(
        working, [column for column in SOURCE_COLUMNS if column != "Price"]
    )
    duplicates = int(working["listing_id"].duplicated(keep="first").sum())
    working = working.drop_duplicates("listing_id", keep="first")
    mapped = pd.DataFrame({
        "listing_id": working["listing_id"],
        "source_row_number": working["source_row_number"],
        "make": working["Brand"],
        "model": working["Model"],
        "year": year.loc[working.index].astype(int),
        "mileage_km": working["Mileage"].map(mileage_band_numeric),
        "transmission": working["Gearbox"],
        "fuel_type": working["Fuel"],
        "feat_body_condition": working["Condition"],
        "fiscal_power": working["Fiscal Power"].map(numeric_first_token),
        "number_of_doors": pd.to_numeric(working["Number of Doors"], errors="coerce"),
        "origin": working["Origin"],
        "first_owner": working["First Owner"],
        "location": working["Location"],
        "sector": working["Sector"],
        "equipment_count": working["Equipment"].map(equipment_count),
        "price_value_MAD": price.loc[working.index].astype(float),
    })
    mapped["target_log"] = np.log1p(mapped["price_value_MAD"])
    mapped["split"] = mapped["listing_id"].map(split_from_listing_id)
    if mapped["listing_id"].duplicated().any():
        raise AssertionError("Deduplicated listing IDs must be unique")
    audit = {
        "dataset": "MUCars-2024",
        "version": 2,
        "doi": SOURCE_DOI,
        "license": "CC BY 4.0",
        "raw_row_count": int(len(source)),
        "retained_used_labeled_listing_count": int(len(mapped)),
        "duplicate_nonprice_fingerprints_removed": duplicates,
        "pre_joint_filter_exclusion_flags": exclusions,
        "price_filter_MAD": [PRICE_MIN_MAD, PRICE_MAX_MAD],
        "price_filter_interpretation": "prespecified listing-quality filter; full-source sensitivity remains required",
        "split_counts": {
            str(key): int(value) for key, value in mapped["split"].value_counts().sort_index().items()
        },
        "action_eligibility_counts": {
            action: int(mapped[action].notna().sum()) for action in ACTION_FIELDS
        },
        "cross_split_listing_id_overlap": 0,
        "split_rule": "sha256_nonprice_fingerprint_first8_mod100",
        "mileage_interpretation": "numeric midpoint of listed band; single-ended band uses stated lower limit",
        "target": "log1p advertised price in Moroccan dirham",
        "test_performance_opened": False,
        "known_limitations": [
            "no_original_advertisement_id", "no_images", "no_transaction_prices",
            "no_within_year_listing_timestamp", "seller_reported_condition",
            "possible_overcollapse_of_identical_nonprice_listings",
        ],
    }
    return mapped.reset_index(drop=True), audit


def write_outputs(
    manifest: pd.DataFrame, audit: dict, source: Path, output_dir: Path
) -> None:
    split_dir = output_dir / "splits"
    sealed_dir = output_dir / "sealed_labels"
    split_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)
    target_columns = {"price_value_MAD", "target_log"}
    public = manifest.drop(columns=list(target_columns))
    manifest_path = output_dir / "mucars_bvival_manifest_features_only.csv"
    public.to_csv(manifest_path, index=False)
    hashes = {str(manifest_path.relative_to(output_dir)): file_sha256(manifest_path)}
    for split in ("train", "validation", "calibration"):
        path = split_dir / f"{split}.csv"
        manifest.loc[manifest["split"].eq(split)].drop(columns="split").to_csv(
            path, index=False
        )
        hashes[str(path.relative_to(output_dir))] = file_sha256(path)
    test = manifest.loc[manifest["split"].eq("test")]
    features_path = split_dir / "test_features.csv"
    labels_path = sealed_dir / "test_labels.csv"
    test.drop(columns=["split", *target_columns]).to_csv(features_path, index=False)
    test[["listing_id", "price_value_MAD", "target_log"]].to_csv(
        labels_path, index=False
    )
    labels_path.chmod(0o444)
    hashes[str(features_path.relative_to(output_dir))] = file_sha256(features_path)
    hashes[str(labels_path.relative_to(output_dir))] = file_sha256(labels_path)
    audit.update({
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "output_hashes": hashes,
        "test_features_exclude_target": not target_columns.intersection(
            pd.read_csv(features_path, nrows=0).columns
        ),
        "test_label_permissions": oct(labels_path.stat().st_mode & 0o777),
        "performance_metrics_computed": False,
    })
    (output_dir / "mucars_bvival_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if file_sha256(args.source) != SOURCE_SHA256:
        raise ValueError("MUCars source hash differs from the recorded version-2 file")
    source = pd.read_csv(args.source)
    manifest, audit = build_manifest(source)
    write_outputs(manifest, audit, args.source, args.output_dir)


if __name__ == "__main__":
    main()
