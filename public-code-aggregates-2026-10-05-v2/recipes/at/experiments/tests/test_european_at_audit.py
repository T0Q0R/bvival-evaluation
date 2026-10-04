"""Synthetic guards only; these tests do not reproduce any empirical result."""
import json
import zipfile
from collections import Counter
from io import BytesIO
from xml.sax.saxutils import escape

import pytest

import audit_european_at_features as audit
import audit_european_at_schema as schema
import build_european_at_cohort as cohort


def cell(col, row, value):
    return f'<c r="{col}{row}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'


def fixture(*, month_rows=None, changed_header=None, shared=None):
    buf = BytesIO()
    month_rows = month_rows or {}
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/workbook.xml", '<workbook xmlns="' + schema.NS[1:-1] + '" xmlns:r="' +
                   schema.RID[1:schema.RID.index('}')] + '"><sheets>' + ''.join(
                       f'<sheet name="{m}" sheetId="{i}" r:id="r{i}"/>'
                       for i, m in enumerate(schema.MONTH_HEADERS, 1)) + '</sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels", '<Relationships>' + ''.join(
            f'<Relationship Id="r{i}" Target="worksheets/sheet{i}.xml"/>' for i in range(1, 13)) + '</Relationships>')
        for i, (month, header) in enumerate(schema.MONTH_HEADERS.items(), 1):
            header = changed_header if changed_header and month == "02" else header
            z.writestr(f"xl/worksheets/sheet{i}.xml", '<worksheet><sheetData><row r="1">' + ''.join(
                cell(c, 1, v) for c, v in header.items()) + '</row>' + ''.join(month_rows.get(month, [])) + '</sheetData></worksheet>')
        if shared is not None:
            z.writestr("xl/sharedStrings.xml", '<sst>' + shared + '</sst>')
    buf.seek(0)
    return zipfile.ZipFile(buf)


def eligible_row(month, row=2, name="Synthetic Vehicle", km="12.000 km"):
    vals = {"name": name, "year": "2020", "mileage_km": km, "power_ps": "100",
            "power_kw": "74", "fuel": "Diesel", "transmission": "Schaltgetriebe", "currency": "EUR"}
    body = ''
    for col, label in schema.MONTH_HEADERS[month].items():
        field = schema.HEADER_FIELDS[label]
        if field in audit.BLOCKED_FIELDS:
            body += f'<c r="{col}{row}"><v>BLOCKED_PAYLOAD &invalid;</v></c>'
        else:
            body += cell(col, row, vals.get(field, ""))
    return f'<row r="{row}">{body}</row>'


def test_exact_three_schemas_map_without_reading_data(monkeypatch):
    with fixture(month_rows={"02": [eligible_row("02")]}) as z:
        original = schema.decode_approved_cell
        def guard(block, col, approved):
            assert schema.cell_reference(block)[1] == 1
            return original(block, col, approved)
        monkeypatch.setattr(schema, "decode_approved_cell", guard)
        for month, member in schema.workbook_sheets(z):
            assert schema.recognized_header(z, member, month) == schema.MONTH_HEADERS[month]


def test_unknown_header_never_echoed():
    headers = {**schema.MONTH_HEADERS["02"], "G": "UNKNOWN_PRIVATE_PAYLOAD"}
    with fixture(changed_header=headers) as z:
        with pytest.raises(ValueError, match="unrecognized values withheld") as err:
            schema.recognized_header(z, "xl/worksheets/sheet2.xml", "02")
    assert "UNKNOWN_PRIVATE_PAYLOAD" not in str(err.value)


def test_data_price_and_additional_never_enter_decoder(monkeypatch):
    seen, original = [], audit.decode_approved_cell
    def guard(block, col, approved):
        assert b"BLOCKED_PAYLOAD" not in block
        seen.append(col)
        return original(block, col, approved)
    monkeypatch.setattr(audit, "decode_approved_cell", guard)
    with fixture(month_rows={"02": [eligible_row("02")]}) as z:
        result = audit.audit_archive(z)
    assert result["total_data_rows"] == 1
    assert result["months"][1]["joint_four_action_nonprice_eligible_rows"] == 1
    assert result["price_cells_decoded"] == result["additional_cells_decoded"] == 0
    assert result["cell_access_second_pass"]["blocked_data_cells_skipped"] == 2
    assert "C" not in seen and "F" not in seen
    assert "Synthetic Vehicle" not in json.dumps(result)


def test_unrequested_shared_price_remains_unparsed():
    data = '<row r="2"><c r="A2" t="s"><v>0</v></c><c r="C2" t="s"><v>1</v></c></row>'
    with fixture(month_rows={"02": [data]}, shared='<si><t>NAME</t></si><si><t>PRIVATE_PRICE_SENTINEL &invalid;</t></si>') as z:
        result = audit.audit_archive(z)
    assert result["price_cells_decoded"] == 0
    assert "PRIVATE_PRICE_SENTINEL" not in json.dumps(result)


def test_title_year_group_disjointness_is_conservative_not_vehicle_identity():
    with fixture(month_rows={m: [eligible_row(m)] for m in ("02", "08", "09", "11")}) as z:
        result = audit.audit_archive(z)
    counts = result["earliest_role_unique_group_feasibility"]["title_year"]
    assert counts["development"]["earliest_role_disjoint_groups"] == 1
    assert all(counts[r]["earliest_role_disjoint_groups"] == 0 for r in ("validation", "calibration", "evaluation"))
    assert result["fingerprints_are_not_verified_vehicle_ids"]


@pytest.mark.parametrize("value,field,expected", [
    ("12.000 km", "mileage_km", "12000"), ("12,000", "mileage_km", "12000"),
    ("12 000", "mileage_km", "12000"), ("74 kW", "power_kw", "74"),
    ("100 PS", "power_ps", "100"), ("74,5", "power_kw", "74.5"),
    ("unknown", "mileage_km", None), ("-1", "mileage_km", None),
    ("NaN", "power_kw", None), ("2020", "year", "2020"),
])
def test_numeric_grammar(value, field, expected):
    result = audit.number(value, field)
    assert (str(result) if result is not None else None) == expected


def test_unit_conflict_is_not_hidden_by_fallback():
    assert audit.power_pair({"power_ps": "100", "power_kw": "74"}) == (True, False)
    assert audit.power_pair({"power_ps": "100", "power_kw": "100"}) == (True, True)
    assert audit.power_pair({"power_ps": "", "power_kw": "74"}) == (True, False)


def test_january_combined_fuel_gearbox_not_guessed_as_separate_actions():
    data = '<row r="2">' + cell("A", 2, "Car") + cell("C", 2, "2020") + cell("D", 2, "1000") + cell("E", 2, "100 PS") + cell("F", 2, "Diesel Automatic") + '</row>'
    with fixture(month_rows={"01": [data]}) as z:
        result = audit.audit_archive(z)
    assert result["months"][0]["data_rows"] == 1
    assert result["months"][0]["joint_four_action_nonprice_eligible_rows"] == 0
    assert "fuel_gearbox_combined" in result["months"][0]["fields"]


def test_unlabelled_data_column_is_quarantined_before_decoder():
    data = '<row r="2">' + cell("A", 2, "Car") + '<c r="G2"><v>UNKNOWN_SENTINEL &invalid;</v></c></row>'
    with fixture(month_rows={"01": [data]}) as z:
        result = audit.audit_archive(z)
    assert result["cell_access_second_pass"]["unlabelled_data_cells_skipped"] == 1
    assert result["unlabelled_data_cells_decoded"] == 0
    assert "UNKNOWN_SENTINEL" not in json.dumps(result)


def test_price_change_does_not_change_audit():
    base = eligible_row("02")
    with fixture(month_rows={"02": [base.replace("BLOCKED_PAYLOAD &invalid;", "100")]}) as z:
        a = audit.audit_archive(z)
    with fixture(month_rows={"02": [base.replace("BLOCKED_PAYLOAD &invalid;", "999999999")]}) as z:
        b = audit.audit_archive(z)
    assert a == b


def test_schema_verified_before_any_data_decode(tmp_path, monkeypatch):
    bad = tmp_path / "bad.xlsx"
    bad.write_bytes(b"not the fixed workbook")
    monkeypatch.setattr(schema, "recognized_header", lambda *a: pytest.fail("Should reject hash first"))
    with pytest.raises(ValueError, match="fixed publisher metadata"):
        schema.probe(bad)


def test_price_free_cohort_keeps_one_earliest_group_without_titles():
    with fixture(month_rows={m: [eligible_row(m)] for m in ("02", "08", "11")}) as z:
        records, counts = cohort.build(z)
    assert counts["development"] == 1
    assert len(records) == 1 and records[0]["role"] == "development"
    assert counts["repeated_eligible_title_year_records_dropped"] == 2
    assert "Synthetic Vehicle" not in json.dumps(records)
    assert set(records[0]["actions"]) == set(cohort.ACTIONS)
    assert "price" not in json.dumps(records)


def test_context_does_not_keep_action_aliases_or_arbitrary_title_tokens():
    a = cohort.summary_context("Volkswagen Crafter 2.0 TDI DSG 150 PS", 2020)
    b = cohort.summary_context("Volkswagen Crafter Elektro Automatik 100 kW", 2020)
    assert a == b == {"brand": "vw", "family": "crafter", "year": 2020.0}
    assert cohort.summary_context("Unknown PRIVATE_TITLE 300 PS", 2020) == {
        "brand": "unknown", "family": "unknown", "year": 2020.0}
