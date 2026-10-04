"""Pretest exact-spec overlap screen; uses feature manifest and no labels.

This deliberately over-flags common vehicle configurations. The resulting
"novel key" subset is a conservative sensitivity, not a verified VIN dedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


QUASI_KEY = (
    "make", "model", "registration_year", "mileage_km",
    "power_hp", "body_type", "country_code",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_mask(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"listing_id", "split", *QUASI_KEY}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Feature manifest lacks {missing}")
    if manifest["listing_id"].duplicated().any():
        raise ValueError("Listing IDs must be unique")
    non_test_keys = manifest.loc[manifest["split"].ne("test"), list(QUASI_KEY)].drop_duplicates()
    test = manifest.loc[manifest["split"].eq("test"), ["listing_id", *QUASI_KEY]]
    joined = test.merge(non_test_keys, on=list(QUASI_KEY), how="left", indicator=True,
                        validate="many_to_one")
    return pd.DataFrame({
        "listing_id": joined["listing_id"],
        "quasi_key_seen_in_non_test": joined["_merge"].eq("both"),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.feature_manifest)
    mask = build_mask(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.output_dir / "quasi_duplicate_test_mask.csv"
    mask.to_csv(mask_path, index=False)
    audit = {
        "protocol": "pretest-exact-spec-overlap-sensitivity",
        "key": list(QUASI_KEY),
        "feature_manifest_sha256": file_sha256(args.feature_manifest),
        "test_listing_count": int(len(mask)),
        "key_seen_in_non_test_count": int(mask["quasi_key_seen_in_non_test"].sum()),
        "novel_key_test_count": int((~mask["quasi_key_seen_in_non_test"]).sum()),
        "mask_sha256": file_sha256(mask_path),
        "test_labels_read": False,
        "interpretation": "Conservative exact-spec overlap, not confirmed same vehicle",
    }
    (args.output_dir / "quasi_duplicate_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({key: audit[key] for key in (
        "test_listing_count", "key_seen_in_non_test_count", "novel_key_test_count"
    )}))


if __name__ == "__main__":
    main()
