"""Project NEW reconstructed targets to the two recorded legacy headers.

Removes only risk_target_absolute_price_error_before. Never recomputes numbers,
sorts/selects rows or reads historical target rows. Exclusive new output only;
the resulting bytes must match the recorded legacy hash before fitting proceeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

REMOVED = "risk_target_absolute_price_error_before"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project_csv(source, destination, input_header, legacy_header):
    if len(input_header) != len(set(input_header)) or len(legacy_header) != len(set(legacy_header)):
        raise ValueError("Duplicate contract column")
    if REMOVED not in input_header or legacy_header != [c for c in input_header if c != REMOVED]:
        raise ValueError("Only the single MAE-risk column may be removed; order must be preserved")
    if not {"listing_id", "action_id"}.issubset(legacy_header):
        raise ValueError("Listing/action identifiers required")
    indexes = [input_header.index(c) for c in legacy_header]
    key_indexes = [input_header.index(c) for c in ("listing_id", "action_id")]
    keys, listings = set(), set()
    with source.open(newline="", encoding="utf-8") as incoming, destination.open("x", newline="", encoding="utf-8") as outgoing:
        reader = csv.reader(incoming)
        if next(reader, None) != input_header:
            raise ValueError("Rebuilt input header differs from contract")
        writer = csv.writer(outgoing, lineterminator="\n")
        writer.writerow(legacy_header)
        for row in reader:
            if len(row) != len(input_header):
                raise ValueError("Malformed target row")
            key = tuple(row[i] for i in key_indexes)
            if not all(key) or key in keys:
                raise ValueError("Nonempty unique listing/action keys required")
            keys.add(key)
            listings.add(key[0])
            writer.writerow([row[i] for i in indexes])
    return {"listing_count": len(listings), "action_pair_count": len(keys),
            "retained_cell_text_and_row_order_preserved": True,
            "removed_column": REMOVED, "numeric_values_recomputed": False}


def project_bound(source, output, contract_path, expected_contract, role, expected_input, expected_output, expected_listings, expected_pairs):
    for path in (source, contract_path, output):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Symlink target/input component rejected")
    if expected_listings <= 0 or expected_pairs < expected_listings:
        raise ValueError("Positive listing/pair counts required")
    if sha256(contract_path) != expected_contract or sha256(source) != expected_input:
        raise ValueError("Contract or generated input hash mismatch; no projection output created")
    contract = json.loads(contract_path.read_text())
    if (contract.get("stage") != "HISTORICAL_JUCARS_METADATA_RECOVERY_NOT_EMPIRICAL_REPLAY"
            or contract.get("empirical_replay_completed") is not False
            or contract.get("historical_test_labels_already_opened") is not True):
        raise ValueError("Bound recovery receipt required")
    records = contract["target_schema_inspection"]["records"]
    schemas = {r["role"]: r for r in records}
    if len(schemas) != 3 or len(records) != 3 or set(schemas) != {"development_value", "development_risk", "calibration"}:
        raise ValueError("All three target schema records required")
    if role not in {"development_value", "calibration"} or schemas[role]["sha256"] != expected_output:
        raise ValueError("Projection output hash/role differs from historical contract")
    input_header, legacy_header = schemas["development_risk"]["header"], schemas[role]["header"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    audit = {"role": role, "contract_sha256": expected_contract, "input_sha256": expected_input,
             "historical_legacy_output_sha256": expected_output, "fitting_called": False,
             "historical_target_rows_read": False, "new_reconstructed_target_rows_parsed": True}
    try:
        path = output / "bvival_action_dataset.csv"
        audit.update(project_csv(source, path, input_header, legacy_header))
        audit["output_sha256"] = sha256(path)
        if audit["listing_count"] != expected_listings or audit["action_pair_count"] != expected_pairs:
            raise ValueError("Projected listing/pair counts differ")
        if audit["output_sha256"] != expected_output:
            raise ValueError("Projected legacy bytes differ; retain discrepancy, do not adjust values")
        audit["stage"] = "LEGACY_SCHEMA_PROJECTED_WITH_RECORDED_BYTE_HASH_MATCH"
    except Exception as error:
        audit.update({"stage": "LEGACY_PROJECTION_FAILED_RETAINED", "exception_class": type(error).__name__, "fitting_must_not_proceed": True})
        with (output / "legacy_projection_audit.json").open("x") as stream:
            json.dump(audit, stream, indent=2)
        raise
    with (output / "legacy_projection_audit.json").open("x") as stream:
        json.dump(audit, stream, indent=2)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--role", choices=("development_value", "calibration"), required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-output-sha256", required=True)
    parser.add_argument("--expected-listings", type=int, required=True)
    parser.add_argument("--expected-pairs", type=int, required=True)
    args = parser.parse_args()
    try:
        result = project_bound(args.input, args.output_dir, args.contract, args.expected_contract_sha256,
                               args.role, args.expected_input_sha256, args.expected_output_sha256,
                               args.expected_listings, args.expected_pairs)
        print(json.dumps({"stage": result["stage"], "output_sha256": result["output_sha256"]}))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Legacy projection failed: {error}\n")


if __name__ == "__main__":
    main()
