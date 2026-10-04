"""Selected-cell header probe for the AT 2023 workbook; never decode data labels.

Uses the already-tested XLSX byte-fragment reader. Unknown schemas fail closed;
neither an exception nor the receipt emits unrecognized cell payloads. This
probe does not establish vehicle IDs, actual observation provenance or a study GO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from audit_turkey_feature_only import (
    CELL, MAX_ENTRY, MAX_TOTAL, NS, RID, cell_reference,
    decode_approved_cell, fragments, resolve_shared_strings, row_number,
)

# Exact first-row labels observed during the metadata-only audit, not aliases
# inferred from row values. The publisher dictionary omits these schema changes.
MONTH_HEADERS = {
    "01": {"A": "name", "B": "price", "C": "year", "D": "mileage",
           "E": "power", "F": "fuel, gearbox"},
    **{m: {"A": "name", "B": "currency", "C": "price", "D": "fuel",
           "E": "transmition", "F": "additional info", "G": "power, ps",
           "H": "power, kw", "I": "milage, km", "J": "year"}
       for m in ("02", "03")},
    **{f"{m:02}": {"A": "name", "B": "price", "C": "transmition",
                    "D": "additional", "E": "power, ps", "F": "power, kw",
                    "G": "milage", "H": "year", "I": "currency", "J": "fuel",
                    "K": "engine volume"} for m in range(4, 13)},
}
HEADER_FIELDS = {
    "name": "name", "price": "price", "year": "year",
    "mileage": "mileage_km", "milage, km": "mileage_km", "milage": "mileage_km",
    "power": "power_combined", "power, ps": "power_ps", "power, kw": "power_kw",
    "fuel, gearbox": "fuel_gearbox_combined", "fuel": "fuel",
    "transmition": "transmission", "currency": "currency",
    "additional": "additional", "additional info": "additional",
    "engine volume": "engine_volume_unit_unverified",
}
AT_BYTES = 5238560
AT_SHA256 = "b0eeff0ad4f91c4dda2190cfb31d4831ff50003da6fe9c325ddf47ac556d54c7"
SAFE_FAILURE_REASONS = {
    "Duplicate ZIP members", "Inflated archive exceeds bound",
    "Unsupported macro or external link", "External relationship",
    "Unexpected monthly worksheet metadata", "Worksheet relationship is not unique",
    "Unsupported worksheet path", "Expected twelve distinct monthly worksheets",
    "Header must be first row; no row-value fallback", "Header coordinates are invalid",
    "Header not recognized; unrecognized values withheld",
    "Source does not match fixed publisher metadata",
}


def normalized(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def workbook_sheets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate ZIP members")
    if any(i.file_size > MAX_ENTRY for i in archive.infolist()) or sum(i.file_size for i in archive.infolist()) > MAX_TOTAL:
        raise ValueError("Inflated archive exceeds bound")
    if any("vbaProject" in name or "externalLinks" in name for name in names):
        raise ValueError("Unsupported macro or external link")
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    if any(r.attrib.get("TargetMode") == "External" for r in relationships):
        raise ValueError("External relationship")
    output = []
    for sheet in workbook.findall(NS + "sheets/" + NS + "sheet"):
        name = sheet.attrib["name"]
        if not re.fullmatch(r"0[1-9]|1[0-2]", name) or sheet.attrib.get("state", "visible") != "visible":
            raise ValueError("Unexpected monthly worksheet metadata")
        links = [r for r in relationships if r.attrib["Id"] == sheet.attrib[RID]]
        if len(links) != 1:
            raise ValueError("Worksheet relationship is not unique")
        target = links[0].attrib["Target"]
        member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        if not re.fullmatch(r"xl/worksheets/sheet[0-9]+\.xml", member) or member not in names:
            raise ValueError("Unsupported worksheet path")
        output.append((name, member))
    if len(output) != 12 or len({name for name, _ in output}) != 12:
        raise ValueError("Expected twelve distinct monthly worksheets")
    return sorted(output)


def recognized_header(archive: zipfile.ZipFile, member: str, month: str) -> dict[str, str]:
    with archive.open(member) as stream:
        block = next(fragments(stream, b"row"))
    if row_number(block) != 1:
        raise ValueError("Header must be first row; no row-value fallback")
    raw = {}
    for cell in CELL.findall(block):
        column, number = cell_reference(cell)
        if number != 1 or column in raw:
            raise ValueError("Header coordinates are invalid")
        raw[column] = decode_approved_cell(cell, column, {column})
    needed = {value for kind, value in raw.values() if kind == "shared"}
    strings = resolve_shared_strings(archive, needed)
    header = {
        col: normalized(strings[value] if kind == "shared" else value)
        for col, (kind, value) in raw.items()
    }
    header = {col: value for col, value in header.items() if value}
    if header != MONTH_HEADERS.get(month):
        raise ValueError("Header not recognized; unrecognized values withheld")
    return header


def probe(source: Path) -> dict[str, object]:
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != AT_BYTES or digest != AT_SHA256:
        raise ValueError("Source does not match fixed publisher metadata")
    with zipfile.ZipFile(source) as archive:
        sheets = [
            {"month": name, "header_row": 1,
             "recognized_columns": recognized_header(archive, member, name),
             "canonical_columns": {col: HEADER_FIELDS[label]
                                   for col, label in MONTH_HEADERS[name].items()}}
            for name, member in workbook_sheets(archive)
        ]
    return {
        "stage": "DOCUMENTED_HEADERS_VERIFIED_NO_DATA_VALUES_DECODED",
        "source_bytes": len(payload), "source_sha256": digest,
        "source_matches_publisher_metadata": True, "sheets": sheets,
        "raw_price_bytes_machine_received": True,
        "source_archive_accessible_to_project_author": True,
        "data_row_price_cells_decoded": False, "data_row_values_displayed": False,
        "data_rows_read_for_feature_audit": False,
        "independent_label_custody_or_blinding_claimed": False,
        "source_workbook_modified_or_exported": False,
        "model_fit_or_outcome_evaluation_started": False, "experiment_GO": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = probe(args.source)
    except Exception as error:
        # Do not expose arbitrary exception payloads that could contain cell values.
        receipt = {"stage": "HOLD_UNRECOGNIZED_SCHEMA_NO_VALUE_FALLBACK", "error_type": type(error).__name__,
                   "safe_reason": str(error) if str(error) in SAFE_FAILURE_REASONS else "Unsupported helper representation; details withheld",
                   "data_row_values_displayed": False, "experiment_GO": False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))
    return 0 if receipt["stage"].startswith("DOCUMENTED_HEADERS_VERIFIED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
