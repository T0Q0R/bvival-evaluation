"""Build a price-sealed AutoScout24 candidate cohort for BVI-Val.

This script performs source filtering and a VIN-aware hash split only. It
fits no model and computes no predictive or policy performance. The 2025
Zenodo API/DataCite metadata label the record MIT, but marketplace-derived
redistribution rights remain unverified; do not redistribute raw rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_autoscout24_candidate import EXPECTED_BYTES, EXPECTED_MD5, is_plausible_vin, source_md5
from build_jucars_bvival_manifest import split_from_listing_id


SOURCE_URL = "https://zenodo.org/records/17643343"
SNAPSHOT_DATE = pd.Timestamp("2025-11-08")
MIN_PRICE_EUR = 1_000.0
MAX_PRICE_EUR = 1_000_000.0
ACTION_FIELDS = ("mileage_km", "transmission", "fuel_type", "service_history")
SOURCE_COLUMNS = (
    "id", "vin", "price", "price_currency", "is_used", "make", "model",
    "registration_date", "vehicle_type", "body_type", "power_hp",
    "nr_seats", "nr_doors", "country_code", "seller_is_dealer",
    "nr_prev_owners", "mileage_km_raw", "transmission", "fuel_category",
    "has_full_service_history",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_check(path: Path) -> None:
    if path.stat().st_size != EXPECTED_BYTES or source_md5(path) != EXPECTED_MD5:
        raise ValueError("AutoScout24 CSV does not match the audited Zenodo file")


def build_manifest(source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = sorted(set(SOURCE_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError(f"Source lacks required columns: {missing}")
    working = source[list(SOURCE_COLUMNS)].copy()
    listing_id = working["id"].astype("string").str.strip()
    if listing_id.isna().any() or listing_id.eq("").any() or listing_id.duplicated().any():
        raise ValueError("Source listing IDs must be present and unique")

    price = pd.to_numeric(working["price"], errors="coerce")
    registration = pd.to_datetime(working["registration_date"], errors="coerce")
    used = working["is_used"].eq(True)
    currency = working["price_currency"].eq("EUR")
    plausible_price = price.between(MIN_PRICE_EUR, MAX_PRICE_EUR)
    plausible_registration = registration.between(pd.Timestamp("1950-01-01"), SNAPSHOT_DATE)
    keep = used & currency & plausible_price & plausible_registration
    exclusion_flags = {
        "not_marked_used": int((~used).sum()),
        "not_eur": int((~currency).sum()),
        "price_missing_or_outside_range": int((~plausible_price).sum()),
        "registration_missing_or_outside_1950_to_snapshot": int((~plausible_registration).sum()),
    }
    working = working.loc[keep].copy()
    working["listing_id"] = listing_id.loc[working.index]
    working["price_value_EUR"] = price.loc[working.index].astype(float)
    working["registration_year"] = registration.loc[working.index].dt.year.astype(int)

    vin = working["vin"].fillna("").astype(str).str.strip().str.upper()
    valid_vin = vin.map(is_plausible_vin)
    group_key = pd.Series(
        np.where(valid_vin, "vin:" + vin, "id:" + working["listing_id"].astype(str)),
        index=working.index,
    )
    working["group_hash"] = group_key.map(sha256_text)
    # Select a representative without using source order or price.
    working["representative_rank"] = working["listing_id"].map(sha256_text)
    working = working.sort_values("representative_rank", kind="stable")
    duplicate_group_rows = int(working["group_hash"].duplicated(keep="first").sum())
    working = working.drop_duplicates("group_hash", keep="first").copy()
    working["split"] = working["group_hash"].map(split_from_listing_id)

    mapped = pd.DataFrame({
        "listing_id": working["listing_id"].astype(str),
        "split": working["split"],
        "make": working["make"],
        "model": working["model"],
        "registration_year": working["registration_year"],
        "vehicle_type": working["vehicle_type"],
        "body_type": working["body_type"],
        "power_hp": pd.to_numeric(working["power_hp"], errors="coerce"),
        "nr_seats": pd.to_numeric(working["nr_seats"], errors="coerce"),
        "nr_doors": pd.to_numeric(working["nr_doors"], errors="coerce"),
        "country_code": working["country_code"],
        "seller_is_dealer": working["seller_is_dealer"],
        "nr_prev_owners": pd.to_numeric(working["nr_prev_owners"], errors="coerce"),
        # In this CSV mileage_km is a formatted string (e.g. "10,500 km");
        # mileage_km_raw is the numeric source column. Never expose both.
        "mileage_km": pd.to_numeric(working["mileage_km_raw"], errors="coerce"),
        "transmission": working["transmission"],
        "fuel_type": working["fuel_category"],
        "service_history": working["has_full_service_history"].astype("string"),
        "price_value_EUR": working["price_value_EUR"],
    }).reset_index(drop=True)
    mapped["target_log"] = np.log1p(mapped["price_value_EUR"])
    if mapped["listing_id"].duplicated().any():
        raise AssertionError("Listing IDs must remain unique")
    audit = {
        "source": SOURCE_URL,
        "snapshot_date": str(SNAPSHOT_DATE.date()),
        "raw_rows": int(len(source)),
        "pre_dedup_filter_count": int(keep.sum()),
        "representatives_retained": int(len(mapped)),
        "duplicate_group_rows_removed": duplicate_group_rows,
        "pre_joint_filter_exclusion_flags": exclusion_flags,
        "price_filter_eur": [MIN_PRICE_EUR, MAX_PRICE_EUR],
        "split_counts": mapped["split"].value_counts().sort_index().astype(int).to_dict(),
        "action_eligibility_counts": {
            action: int(mapped[action].notna().sum()) for action in ACTION_FIELDS
        },
        "split_rule": "sha256 of plausible VIN else listing ID, then first8 mod100; JUCars 55/15/15/15 buckets",
        "dedup_rule": "minimum SHA256 listing ID per VIN/ID group; no price used",
        "target": "log1p advertised price EUR",
        "test_performance_opened": False,
        "source_rights": "Zenodo API/DataCite label record MIT; underlying marketplace-derived redistribution rights not independently verified; do not redistribute raw records without author-side review",
        "known_limitations": [
            "no_images", "no_sale_prices", "no_listing_date", "registration_year_is_not_listing_year",
            "seller_reported_service_history", "VIN_missing_for_majority", "VIN_structural_screen_only",
        ],
    }
    return mapped, audit


def write_outputs(mapped: pd.DataFrame, audit: dict, source: Path, output_dir: Path) -> None:
    split_dir = output_dir / "splits"
    sealed_dir = output_dir / "sealed_labels"
    split_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)
    target_columns = ["price_value_EUR", "target_log"]
    features = mapped.drop(columns=target_columns)
    feature_manifest_path = output_dir / "autoscout24_bvival_manifest_features_only.csv"
    features.to_csv(feature_manifest_path, index=False)
    hashes = {str(feature_manifest_path.relative_to(output_dir)): file_sha256(feature_manifest_path)}
    for split in ("train", "validation", "calibration"):
        path = split_dir / f"{split}.csv"
        mapped.loc[mapped["split"].eq(split)].drop(columns="split").to_csv(path, index=False)
        hashes[str(path.relative_to(output_dir))] = file_sha256(path)
    test = mapped.loc[mapped["split"].eq("test")]
    test_features_path = split_dir / "test_features.csv"
    test_labels_path = sealed_dir / "test_labels.csv"
    test.drop(columns=["split", *target_columns]).to_csv(test_features_path, index=False)
    test[["listing_id", *target_columns]].to_csv(test_labels_path, index=False)
    test_labels_path.chmod(0o444)
    hashes[str(test_features_path.relative_to(output_dir))] = file_sha256(test_features_path)
    hashes[str(test_labels_path.relative_to(output_dir))] = file_sha256(test_labels_path)
    audit.update({
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "output_hashes": hashes,
        "test_features_exclude_target": not set(target_columns).intersection(
            pd.read_csv(test_features_path, nrows=0).columns
        ),
        "test_label_permissions": oct(test_labels_path.stat().st_mode & 0o777),
        "performance_metrics_computed": False,
    })
    (output_dir / "autoscout24_bvival_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_check(args.source)
    source = pd.read_csv(args.source, usecols=list(SOURCE_COLUMNS), low_memory=False)
    mapped, audit = build_manifest(source)
    write_outputs(mapped, audit, args.source, args.output_dir)


if __name__ == "__main__":
    main()
