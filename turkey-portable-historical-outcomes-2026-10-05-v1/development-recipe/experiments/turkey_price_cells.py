"""Row/column-authorized price extraction, never whole-table XLSX import.

Unselected row price cells are skipped before XML payload parsing. Raw ZIP/XML
buffers still physically contain labels; this is not a cryptographic enclave.
The caller must establish immutable cohort and phase authorization first.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
from xml.etree import ElementTree as ET

import audit_turkey_feature_only as reader
from turkey_execution_contract import price_is_valid


def decode_price_cell(block: bytes, column: str, price_column: str, *, ordinal: int, allowed: set[int]):
    # Both coordinates are checked before handing payload bytes to the parser.
    if column != price_column or ordinal not in allowed:
        raise ValueError("Price decoder refuses an unselected row or column")
    try:
        root = ET.fromstring(block)
    except ET.ParseError:
        raise ValueError("Malformed selected price cell; payload is not logged") from None
    if root.find("f") is not None:
        return "invalid", "formula_target"
    kind = root.attrib.get("t", "n")
    if kind == "inlineStr":
        return "text", "".join(node.text or "" for node in root.findall(".//t"))
    value = root.findtext("v")
    if value is None:
        return "text", ""
    if kind == "s":
        try:
            return "shared", int(value)
        except ValueError:
            raise ValueError("Invalid selected shared-string reference") from None
    if kind in {"n", "str"}:
        return "text", value
    # Boolean/date/error are not advertised prices, even if their v is numeric.
    return "invalid", "unsupported_target_cell_type"


def read_selected_prices(archive, header, allowed_ordinals, *, expected_data_rows):
    allowed = set(allowed_ordinals)
    if (not allowed or any(type(n) is not int or n < 1 or n > expected_data_rows for n in allowed)
            or len([name for name in header.values() if name == "fiyat"]) != 1):
        raise ValueError("Bounded selected source ordinals and one price column required")
    column = next(c for c, name in header.items() if name == "fiyat")
    decoded, needed, counts = {}, set(), Counter()
    path, ordinal, last_row = reader.sheet_path(archive), 0, 1
    with archive.open(path) as stream:
        for block in reader.fragments(stream, b"row"):
            row = reader.row_number(block)
            if row == 1:
                continue
            if row <= last_row:
                raise ValueError("Worksheet source rows must be strictly increasing")
            last_row, ordinal = row, ordinal + 1
            selected, seen = ordinal in allowed, set()
            for cell in reader.CELL.findall(block):
                cell_column, cell_row = reader.cell_reference(cell)
                if cell_row != row or cell_column in seen or cell_column not in header:
                    raise ValueError("Unexpected or duplicate source data-cell coordinates")
                seen.add(cell_column)
                if cell_column != column:
                    counts["nonprice_payloads_skipped"] += 1
                    continue
                if not selected:
                    counts["unselected_price_cells_skipped_before_xml_parser"] += 1
                    continue
                value = decode_price_cell(cell, cell_column, column, ordinal=ordinal, allowed=allowed)
                decoded[ordinal] = value
                counts["selected_price_cells_xml_parsed"] += 1
                if value[0] == "shared":
                    needed.add(value[1])
            if selected and ordinal not in decoded:
                decoded[ordinal] = "text", ""
                counts["selected_rows_without_price_cell"] += 1
    if ordinal != expected_data_rows or set(decoded) != allowed:
        raise ValueError("Source row inventory or selected row universe changed")
    strings = reader.resolve_shared_strings(archive, needed)
    results = {}
    for ordinal in sorted(allowed):
        kind, payload = decoded[ordinal]
        token = strings[payload] if kind == "shared" else payload if kind == "text" else ""
        valid = kind != "invalid" and price_is_valid(token)
        results[ordinal] = {"price": format(Decimal(token.strip()).normalize(), "f") if valid else "",
                            "price_valid": valid}
        counts["selected_valid_targets" if valid else "selected_invalid_targets"] += 1
    return results, {**dict(counts), "source_data_rows_scanned": ordinal,
                     "selected_rows": len(allowed), "requested_price_shared_string_entries": len(needed),
                     "unselected_price_payloads_xml_parsed": 0, "unselected_price_values_decoded": 0,
                     "raw_buffers_physically_contain_labels": True}
