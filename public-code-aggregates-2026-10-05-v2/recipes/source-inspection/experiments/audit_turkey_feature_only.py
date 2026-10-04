"""Strict selected-cell XLSX audit, not whole-table loading followed by drop.

Raw compressed/XML bytes necessarily contain labels. Data-row cells in blocked
columns are never sent to the value decoder or XML cell parser. Shared-string
entries are resolved only when referenced by headers or approved feature cells.
No price distribution, row dump, filtering by price, modelling or splitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from xml.etree import ElementTree as ET

from archive_turkey_candidate import verify_archive

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
CELL = re.compile(rb"<c\b[^>]*(?:/>|>.*?</c\s*>)", re.S)
CELL_REF = re.compile(rb'\br="([A-Z]+)([1-9][0-9]*)"')
BLOCKED_FIELD_NAME = re.compile(r"fiyat|price|bedel|tutar|ucret|ücret|target|label", re.I)
MAX_FRAGMENT = 1_000_000
MAX_ENTRY = 100_000_000
MAX_TOTAL = 250_000_000


def fragments(stream, tag: bytes):
    """Extract unprefixed row/si byte blocks without decoding their payloads.

    This deliberately supports a bounded, audited subset of XLSX XML. Unknown
    representations fail closed instead of falling back to a whole-table reader.
    """
    opening = re.compile(rb"<" + tag + rb"(?=[\s>/])")
    ending = re.compile(rb"</" + tag + rb"\s*>")
    buffer, ended = b"", False
    while True:
        start = opening.search(buffer)
        if start is not None:
            head_end = buffer.find(b">", start.start())
            finish = ending.search(buffer, head_end + 1) if head_end >= 0 else None
            if head_end >= 0 and buffer[head_end - 1:head_end + 1] == b"/>":
                end = head_end + 1
            elif finish is not None:
                end = finish.end()
            else:
                end = None
            if end is not None:
                block = buffer[start.start():end]
                if len(block) > MAX_FRAGMENT or b"<!" in block:
                    raise ValueError("Unsupported or oversized XML fragment")
                yield block
                buffer = buffer[end:]
                continue
            if len(buffer) - start.start() > MAX_FRAGMENT:
                raise ValueError("XML fragment exceeds byte bound")
        if ended:
            if start is not None:
                raise ValueError("Incomplete XML fragment")
            return
        chunk = stream.read(65536)
        if not chunk:
            ended = True
        else:
            if b"<!DOCTYPE" in chunk or b"<!ENTITY" in chunk:
                raise ValueError("DTD/entity declarations are forbidden")
            if start is None:
                buffer = buffer[-64:] + chunk
            else:
                buffer += chunk


def row_number(block: bytes) -> int:
    header = block[:block.index(b">") + 1]
    # Row metadata can carry inherited x14ac:* attributes. Extract only the
    # coordinate, without parsing a detached tag with unresolved prefixes.
    matches = re.findall(rb'\sr="([1-9][0-9]*)"', header)
    if len(matches) != 1:
        raise ValueError("One explicit row coordinate is required")
    return int(matches[0])


def cell_reference(block: bytes) -> tuple[str, int]:
    header = block[:block.index(b">") + 1]
    match = CELL_REF.search(header)
    if not match:
        raise ValueError("Explicit cell coordinate required")
    return match[1].decode("ascii"), int(match[2])


def decode_approved_cell(block: bytes, column: str, approved: set[str]):
    # The check is intentionally before XML parsing or payload conversion.
    if column not in approved:
        raise ValueError("Cell value decoder cannot access blocked column")
    root = ET.fromstring(block)
    if root.find("f") is not None:
        raise ValueError("Formula feature/header cells are unsupported")
    kind = root.attrib.get("t", "n")
    if kind == "inlineStr":
        return "text", "".join(node.text or "" for node in root.findall(".//t"))
    value = root.findtext("v")
    if value is None:
        return "text", ""
    if kind == "s":
        return "shared", int(value)
    if kind in {"n", "b", "str", "d"}:
        return "text", value
    if kind == "e":
        raise ValueError("Excel error cell in approved feature/header")
    raise ValueError("Unsupported XLSX cell type")


def resolve_shared_strings(archive: zipfile.ZipFile, indices: set[int]) -> dict[int, str]:
    if not indices:
        return {}
    if min(indices) < 0 or "xl/sharedStrings.xml" not in archive.namelist():
        raise ValueError("Shared-string references cannot be resolved")
    resolved = {}
    with archive.open("xl/sharedStrings.xml") as stream:
        for index, block in enumerate(fragments(stream, b"si")):
            if index in indices:
                # Unrequested entries, including any price strings, remain raw bytes.
                root = ET.fromstring(block)
                resolved[index] = "".join(node.text or "" for node in root.findall(".//t"))
    if set(resolved) != indices:
        raise ValueError("Invalid or missing shared-string index")
    return resolved


def sheet_path(archive: zipfile.ZipFile) -> str:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate ZIP members forbidden")
    if any(info.file_size > MAX_ENTRY for info in archive.infolist()) or sum(i.file_size for i in archive.infolist()) > MAX_TOTAL:
        raise ValueError("Inflated archive exceeds size bounds")
    if any("vbaProject" in name for name in names):
        raise ValueError("Macros are unsupported")
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheets = workbook.findall(NS + "sheets/" + NS + "sheet")
    if len(sheets) != 1 or sheets[0].attrib.get("state", "visible") != "visible":
        raise ValueError("Exactly one visible worksheet required")
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    if any(r.attrib.get("TargetMode") == "External" for r in relationships):
        raise ValueError("External workbook relationships forbidden")
    matches = [r for r in relationships if r.attrib["Id"] == sheets[0].attrib[RID]]
    if len(matches) != 1:
        raise ValueError("Worksheet relationship is not unique")
    target = matches[0].attrib["Target"]
    path = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
    if not re.fullmatch(r"xl/worksheets/sheet[0-9]+\.xml", path):
        raise ValueError("Unsupported worksheet member path")
    return path


def read_schema(archive: zipfile.ZipFile, path: str) -> dict[str, str]:
    with archive.open(path) as stream:
        block = next(fragments(stream, b"row"))
    if row_number(block) != 1:
        raise ValueError("Header must be row 1")
    raw = {}
    for cell in CELL.findall(block):
        column, row = cell_reference(cell)
        if row != 1 or column in raw:
            raise ValueError("Header coordinates must be unique and in row 1")
        raw[column] = decode_approved_cell(cell, column, {column})
    strings = resolve_shared_strings(archive, {value for kind, value in raw.values() if kind == "shared"})
    header = {column: (strings[value] if kind == "shared" else value).strip() for column, (kind, value) in raw.items()}
    if (not header or len(set(header.values())) != len(header)
            or any(not re.fullmatch(r"[^\W\d]+(?:_[^\W\d]+)*", name, flags=re.U) for name in header.values())
            or not {"marka", "seri", "model", "fiyat"}.issubset(header.values())):
        raise ValueError("Unexpected or duplicate header; do not print cell values")
    return header


def selected_rows(archive, path, header, fields, counters):
    approved = {column for column, name in header.items() if name in fields}
    last_row = 1
    with archive.open(path) as stream:
        for block in fragments(stream, b"row"):
            row = row_number(block)
            if row == 1:
                continue
            if row <= last_row:
                raise ValueError("Worksheet rows must be strictly increasing")
            last_row = row
            record, seen = {}, set()
            for cell in CELL.findall(block):
                column, cell_row = cell_reference(cell)
                if cell_row != row or column in seen or column not in header:
                    raise ValueError("Unexpected or duplicate data-cell coordinates")
                seen.add(column)
                if column not in approved:
                    counters["blocked_data_cells_skipped"] += 1
                    continue
                record[header[column]] = decode_approved_cell(cell, column, approved)
                counters["approved_data_cells_decoded"] += 1
            yield record


def audit_features(archive, path, header, config):
    fields = config["feature_fields"]
    if (set(header.values()) != set(config["expected_headers"]) or len(fields) != len(set(fields))
            or any(BLOCKED_FIELD_NAME.search(field) for field in fields)
            or set(fields) != set(header.values()) - {"fiyat"}):
        raise ValueError("Feature allowlist/schema must match exactly and exclude price")
    counters, needed = Counter(), set()
    for record in selected_rows(archive, path, header, fields, counters):
        needed.update(value for kind, value in record.values() if kind == "shared")
    strings = resolve_shared_strings(archive, needed)
    counts = {field: Counter() for field in fields}
    numeric = {field: [] for field in config["numeric_fields"]}
    fingerprint_fields = config.get("nonprice_fingerprint_fields", fields)
    if (not set(fingerprint_fields).issubset(fields) or not set(config["numeric_fields"]).issubset(fields)
            or not set(config["coarse_fingerprint_fields"]).issubset(fields)):
        raise ValueError("All statistic and fingerprint fields must be approved nonprice fields")
    fingerprints, coarse, alias, rows = Counter(), Counter(), Counter(), 0
    cohort_flags = Counter()
    availability_groups = config.get("nonblank_candidate_groups", {})
    if any(not set(group).issubset(fields) for group in availability_groups.values()):
        raise ValueError("Candidate availability groups must contain only approved feature fields")
    id_offset_matches = Counter()
    # New counters avoid counting the two selected-row passes as independent rows.
    counters = Counter()
    for record in selected_rows(archive, path, header, fields, counters):
        values = {field: (strings[record[field][1]] if record[field][0] == "shared" else record[field][1]).strip()
                  if field in record else "" for field in fields}
        rows += 1
        cohort_flags["missing_brand_series_or_model_rows"] += int(any(not values.get(f, "") for f in ("marka", "seri", "model")))
        cohort_flags["all_vehicle_profile_fields_blank_rows"] += int(all(not values[f] for f in fingerprint_fields))
        for name, group in availability_groups.items():
            cohort_flags["nonblank_group_" + name] += int(all(values[f] for f in group))
        for field, value in values.items():
            counts[field][value] += 1
        for field in numeric:
            if values[field]:
                try:
                    number = Decimal(values[field])
                except InvalidOperation:
                    continue
                if number.is_finite():
                    numeric[field].append(number)
        fp_values = {field: values[field] for field in fingerprint_fields}
        fp = hashlib.sha256(json.dumps(fp_values, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        fingerprints[fp] += 1
        key = tuple(values[field] for field in config["coarse_fingerprint_fields"])
        coarse[key] += 1
        model_text = values.get("model", "").casefold()
        for name, pattern in config["model_alias_patterns"].items():
            alias[name] += int(bool(re.search(pattern, model_text)))
        if "id" in values:
            id_offset_matches["one_based_order_matches"] += int(values["id"] == str(rows))
            id_offset_matches["zero_based_order_matches"] += int(values["id"] == str(rows - 1))
    field_summary = {}
    for field, values in counts.items():
        blank = values.get("", 0)
        summary = {"blank_cells": blank, "blank_rate": blank / rows if rows else None,
                   "distinct_nonblank": len(values) - int("" in values)}
        if field in config["categorical_summary_fields"]:
            summary["most_common_nonblank"] = [[v, n] for v, n in values.most_common() if v][:12]
        if field in numeric:
            numbers = numeric[field]
            summary["finite_numeric_cells"] = len(numbers)
            summary["minimum_raw_numeric"] = str(min(numbers)) if numbers else None
            summary["maximum_raw_numeric"] = str(max(numbers)) if numbers else None
        field_summary[field] = summary
    id_counts = counts.get("id", Counter())
    return {"feature_rows": rows, "fields": field_summary, "cell_access": dict(counters),
            "price_cells_decoded": 0, "unrequested_shared_strings_decoded": 0,
            "duplicate_nonprice_fingerprint_extra_rows": sum(n - 1 for n in fingerprints.values()),
            "duplicate_nonprice_fingerprint_groups": sum(n > 1 for n in fingerprints.values()),
            "largest_nonprice_fingerprint_group": max(fingerprints.values(), default=0),
            "coarse_fingerprint_extra_rows": sum(n - 1 for n in coarse.values()),
            "nonprice_cohort_counts": dict(cohort_flags),
            "brand_sampling_diagnostic": {
                "max_nonblank_brand_count": max((n for value, n in counts.get("marka", {}).items() if value), default=0),
                "brand_categories_with_counts_2400_to_2500": sum(2400 <= n <= 2500 for value, n in counts.get("marka", {}).items() if value),
                "platform_sampling_cap_not_verified": True},
            "nonprice_fingerprint_fields": fingerprint_fields,
            "id_duplicate_nonblank_extra_rows": sum(n - 1 for value, n in id_counts.items() if value),
            "id_order_checks": dict(id_offset_matches), "id_is_marketplace_identifier_verified": False,
            "model_alias_pattern_match_rows": dict(alias),
            "fingerprints_are_not_vehicle_entity_ids": True, "price_validity_not_checked": True,
            "observed_or_imputed_status_not_proven": True, "confirmatory_test_allowed_now": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-config", type=Path)
    args = parser.parse_args()
    verified = verify_archive(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError("Audit output directory must be empty")
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "source_verified": verified,
              "mode": "features" if args.feature_config else "schema_only", "price_cells_decoded": 0,
              "raw_bytes_include_labels": True, "models_trained": False, "test_scores_computed": False,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "archive_helper_sha256": hashlib.sha256(Path(__file__).with_name("archive_turkey_candidate.py").read_bytes()).hexdigest()}
    with zipfile.ZipFile(args.source) as archive:
        path = sheet_path(archive)
        header = read_schema(archive, path)
        report["worksheet"] = path
        report["headers"] = header
        if args.feature_config:
            config = json.loads(args.feature_config.read_text())
            report["feature_config_sha256"] = hashlib.sha256(args.feature_config.read_bytes()).hexdigest()
            report.update(audit_features(archive, path, header, config))
            (args.output_dir / "feature_config_snapshot.json").write_bytes(args.feature_config.read_bytes())
    (args.output_dir / "feature_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "audit_runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (args.output_dir / "archive_helper_snapshot.py").write_bytes(Path(__file__).with_name("archive_turkey_candidate.py").read_bytes())
    print(json.dumps({"mode": report["mode"], "headers": report["headers"], "feature_rows": report.get("feature_rows"),
                      "price_cells_decoded": report["price_cells_decoded"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
