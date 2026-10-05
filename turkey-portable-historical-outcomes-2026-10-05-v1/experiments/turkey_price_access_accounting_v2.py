"""Correct a known legacy counter shadowing bug, without rewriting old packages.

Legacy parsing checks full row inventory BEFORE reusing `ordinal` for selected
targets. Labels/authorization are unchanged; only its last returned count is
wrong when max(selected) < total rows. Requires the exact audited legacy code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import turkey_price_cells as legacy
from run_turkey_development import file_hashes, save_json
from turkey_development_io import sha256
from turkey_source_access_binding import canonical_source_ledger
from turkey_source_price_release import selected_coordinates, verify_release_package


LEGACY_READER_SHA256 = "722440cef119fa53084047bc412db442eb7aaffdca995be7f2145818887aa740"


def corrected_accounting(access, expected_rows, ordinals, *, legacy_reader_sha256, successful_parse):
    if legacy_reader_sha256 != LEGACY_READER_SHA256 or successful_parse is not True:
        raise ValueError("Exact known legacy reader and successful full inventory check required")
    if (type(expected_rows) is not int or expected_rows < 1 or not isinstance(ordinals, set) or not ordinals
            or any(type(i) is not int or not 1 <= i <= expected_rows for i in ordinals)):
        raise ValueError("Exact bounded original selected ordinal set required")
    count_fields = ("source_data_rows_scanned", "selected_rows", "selected_price_cells_xml_parsed",
                    "selected_rows_without_price_cell", "selected_valid_targets", "selected_invalid_targets",
                    "unselected_price_cells_skipped_before_xml_parser", "unselected_price_payloads_xml_parsed",
                    "unselected_price_values_decoded")
    if any(type(access.get(f, 0)) is not int or access.get(f, 0) < 0 for f in count_fields):
        raise ValueError("Nonnegative integer access counters required")
    if (access.get("source_data_rows_scanned") != max(ordinals) or access.get("selected_rows") != len(ordinals)
            or access.get("selected_price_cells_xml_parsed", 0) + access.get("selected_rows_without_price_cell", 0) != len(ordinals)
            or access.get("selected_valid_targets", 0) + access.get("selected_invalid_targets", 0) != len(ordinals)
            or access.get("selected_price_cells_xml_parsed", 0) + access.get("unselected_price_cells_skipped_before_xml_parser", 0) > expected_rows
            or access.get("unselected_price_payloads_xml_parsed") != 0 or access.get("unselected_price_values_decoded") != 0):
        raise ValueError("Legacy selected/skip/validity counter identity differs")
    return {**access, "legacy_returned_last_selected_ordinal": access["source_data_rows_scanned"],
            "source_data_rows_scanned": expected_rows, "raw_row_inventory_checked_before_selected_target_loop": True,
            "accounting_correction_only_targets_and_selection_unchanged": True}


def read_selected_prices(archive, header, allowed_ordinals, *, expected_data_rows):
    """Corrected future reader adapter, same cells/labels and original check."""
    if sha256(Path(legacy.__file__)) != LEGACY_READER_SHA256:
        raise ValueError("Known legacy reader changed; stop before price parser")
    allowed = set(allowed_ordinals)
    labels, access = legacy.read_selected_prices(archive, header, allowed, expected_data_rows=expected_data_rows)
    return labels, corrected_accounting(access, expected_data_rows, allowed,
                                         legacy_reader_sha256=LEGACY_READER_SHA256, successful_parse=True)


def write_erratum(release_directory, output):
    if output.exists():
        raise ValueError("New nonexistent erratum output required; preserve original releases")
    binding = json.loads((release_directory / "phase_binding.json").read_text())
    original = json.loads((release_directory / "once_phase_release_audit.json").read_text())
    ledger = canonical_source_ledger() if binding["evidence_status"] == "authorized_heldout_only" else Path(original["ledger_claim_directory"]).parent
    audit = verify_release_package(release_directory, binding, ledger)  # hashes only, no target CSV decode
    snapshot_hash = sha256(release_directory / "turkey_price_cells.py.snapshot")
    selected = selected_coordinates(binding)
    corrected = corrected_accounting(audit["cell_access"], binding["expected_data_rows"], set(selected),
                                     legacy_reader_sha256=snapshot_hash, successful_parse=audit["release_succeeded"])
    result = {"stage": "APPEND_ONLY_PHASE_ACCOUNTING_ERRATUM_NO_TARGET_REREAD", "phase": binding["phase"],
              "evidence_status": binding["evidence_status"], "input_release_directory": str(release_directory.absolute()),
              "input_release_audit_sha256": sha256(release_directory / "once_phase_release_audit.json"),
              "input_phase_binding_sha256": sha256(release_directory / "phase_binding.json"),
              "legacy_reader_sha256": snapshot_hash, "original_cell_access": audit["cell_access"],
              "corrected_cell_access": corrected, "source_workbook_opened_here": False,
              "cached_target_values_decoded_here": False, "original_package_modified": False,
              "models_or_actions_changed": False, "phase_reopened": False}
    output.mkdir(parents=True, mode=0o700)
    try:
        for name in ("turkey_price_access_accounting_v2.py", "tests/test_turkey_price_access_accounting_v2.py"):
            path = Path(__file__).parent / name
            with (output / (path.name + ".snapshot")).open("xb") as stream:
                stream.write(path.read_bytes())
        result["artifact_sha256"] = file_hashes(output)
        save_json(output / "phase_accounting_erratum.json", result)
    finally:
        for path in output.iterdir():
            if path.is_file():
                path.chmod(0o400)
    return result


def main():
    parser = argparse.ArgumentParser(description="Append separate verified metadata correction; never reread targets")
    parser.add_argument("--phase-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = write_erratum(args.phase_release, args.output_dir)
    print(json.dumps({"stage": result["stage"], "phase": result["phase"], "reported_legacy_value": result["original_cell_access"]["source_data_rows_scanned"],
                      "correct_full_rows": result["corrected_cell_access"]["source_data_rows_scanned"], "phase_reopened": False}))


if __name__ == "__main__":
    main()
