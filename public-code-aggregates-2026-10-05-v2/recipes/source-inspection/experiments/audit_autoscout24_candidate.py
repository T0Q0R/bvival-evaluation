"""Source-quality audit only; never fits a model or opens a held-out test.

This is a candidate third-source audit, not a claim of external replication.
Zenodo's API and DataCite metadata label the record MIT, although the landing
page Rights field is blank. Third-party marketplace redistribution rights
remain unverified; do not republish raw rows without author-side review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


SOURCE_URL = "https://zenodo.org/records/17643343"
EXPECTED_MD5 = "b23a122cc51baf7de39f449193ff0d28"
EXPECTED_BYTES = 548_610_318
AUDIT_COLUMNS = (
    "id", "vin", "price", "price_currency", "price_net", "price_tax_deductible",
    "make", "model", "mileage_km", "mileage_km_raw", "production_year",
    "registration_date", "transmission", "gears", "fuel_category", "primary_fuel",
    "is_used", "is_new", "had_accident", "has_full_service_history",
    "nr_prev_owners", "co2_emission_grper_km", "country_code", "seller_is_dealer",
)
ACTION_CANDIDATES = (
    "mileage_km", "transmission", "fuel_category", "has_full_service_history"
)
VIN_PATTERN = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")


def is_plausible_vin(value: str) -> bool:
    """Structural screen only; not a manufacturer/check-digit validation."""
    return bool(VIN_PATTERN.fullmatch(value) and len(set(value)) >= 5)


def source_md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324: official checksum, not cryptographic security
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(path: Path, chunk_size: int = 20_000) -> dict:
    actual_bytes = path.stat().st_size
    if actual_bytes != EXPECTED_BYTES:
        raise ValueError(f"Incomplete or changed source: {actual_bytes} != {EXPECTED_BYTES}")
    checksum = source_md5(path)
    if checksum != EXPECTED_MD5:
        raise ValueError(f"Official MD5 mismatch: {checksum}")

    available = set(pd.read_csv(path, nrows=0).columns)
    missing_columns = sorted(set(AUDIT_COLUMNS) - available)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    total = 0
    nonmissing = Counter()
    categorical = {name: Counter() for name in (
        "price_currency", "country_code", "is_used", "is_new", "had_accident",
        "has_full_service_history", "transmission", "fuel_category",
    )}
    price_filter = Counter()
    seen_ids: set[str] = set()
    duplicate_ids = 0
    seen_vins: set[str] = set()
    duplicate_nonnull_vins = 0
    seen_plausible_vins: set[str] = set()
    plausible_vin_rows = 0
    duplicate_plausible_vins = 0
    for frame in pd.read_csv(path, usecols=list(AUDIT_COLUMNS), chunksize=chunk_size,
                             low_memory=False):
        total += len(frame)
        for column in AUDIT_COLUMNS:
            nonmissing[column] += int(frame[column].notna().sum())
        for column, counts in categorical.items():
            counts.update(frame[column].astype("string").fillna("<NA>").tolist())
        price = pd.to_numeric(frame["price"], errors="coerce")
        used = frame["is_used"].eq(True)
        euro = frame["price_currency"].eq("EUR")
        plausible_price = price.between(1_000, 1_000_000)
        price_filter["used_eur_plausible_price"] += int((used & euro & plausible_price).sum())
        price_filter["used_eur_plausible_price_with_mileage"] += int(
            (used & euro & plausible_price & frame["mileage_km"].notna()).sum()
        )
        for listing_id in frame["id"].dropna().astype(str):
            if listing_id in seen_ids:
                duplicate_ids += 1
            seen_ids.add(listing_id)
        for vin in frame["vin"].dropna().astype(str).str.strip():
            if not vin:
                continue
            if vin in seen_vins:
                duplicate_nonnull_vins += 1
            seen_vins.add(vin)
            if is_plausible_vin(vin):
                plausible_vin_rows += 1
                if vin in seen_plausible_vins:
                    duplicate_plausible_vins += 1
                seen_plausible_vins.add(vin)

    return {
        "source": SOURCE_URL,
        "source_bytes": actual_bytes,
        "source_md5": checksum,
        "source_rights_status": "Zenodo API metadata.license.id=mit-license and DataCite rightsList=MIT; landing-page Rights field blank. Marketplace-derived redistribution rights not independently verified. Do not redistribute raw rows without author-side review.",
        "rows": total,
        "nonmissing_counts": dict(nonmissing),
        "categorical_counts": {key: dict(value) for key, value in categorical.items()},
        "candidate_cohort_counts": dict(price_filter),
        "duplicate_listing_ids": duplicate_ids,
        "duplicate_nonnull_vins": duplicate_nonnull_vins,
        "plausible_vin_rows": plausible_vin_rows,
        "duplicate_plausible_vin_rows": duplicate_plausible_vins,
        "candidate_action_fields": list(ACTION_CANDIDATES),
        "known_leakage_fields_to_exclude_or_mask": [
            "description", "price_net", "price_tax_deductible", "price_vat_rate",
            "mileage_km_raw (when mileage_km hidden)",
            "primary_fuel (when fuel_category hidden)",
        ],
        "protocol_status": "source audit only; no split, training, or held-out scoring",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                     sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "rows", "candidate_cohort_counts", "duplicate_listing_ids",
        "duplicate_nonnull_vins", "plausible_vin_rows",
        "duplicate_plausible_vin_rows")}, indent=2))


if __name__ == "__main__":
    main()
