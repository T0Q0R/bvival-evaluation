import json
from collections import Counter
from io import BytesIO
from types import SimpleNamespace
from xml.sax.saxutils import escape
import zipfile

import pytest

import archive_turkey_candidate as downloader
import audit_turkey_feature_only as audit


def inline_cell(column, row, text):
    return f'<c r="{column}{row}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'


def workbook(data_rows, *, shared=None, headers=None, extra_sheet=False, external=False):
    headers = headers or {"A": "marka", "B": "seri", "C": "model", "D": "fiyat"}
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        second_sheet = '<sheet name="Extra" sheetId="2" r:id="rId2"/>' if extra_sheet else ""
        archive.writestr("xl/workbook.xml", '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                         'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
                         '<sheet name="Sheet1" sheetId="1" r:id="rId1"/>' + second_sheet + '</sheets></workbook>')
        target_mode = ' TargetMode="External"' if external else ""
        archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="rId1" '
                         'Target="worksheets/sheet1.xml"' + target_mode + '/></Relationships>')
        header_row = '<row r="1">' + ''.join(inline_cell(c, 1, name) for c, name in headers.items()) + '</row>'
        archive.writestr("xl/worksheets/sheet1.xml", '<worksheet><sheetData>' + header_row + ''.join(data_rows) + '</sheetData></worksheet>')
        if shared is not None:
            archive.writestr("xl/sharedStrings.xml", '<sst>' + shared + '</sst>')
    stream.seek(0)
    return zipfile.ZipFile(stream)


def config(headers=None):
    headers = headers or ["marka", "seri", "model", "fiyat"]
    return {"expected_headers": headers, "feature_fields": [h for h in headers if h != "fiyat"],
            "numeric_fields": [], "categorical_summary_fields": ["marka"],
            "coarse_fingerprint_fields": ["marka", "seri"],
            "model_alias_patterns": {"transmission": r"\bdsg\b"}}


def test_price_cell_never_reaches_value_decoder_or_cell_xml_parser(monkeypatch):
    data = '<row r="2">' + inline_cell("A", 2, "Brand") + inline_cell("B", 2, "Series") + inline_cell("C", 2, "DSG")
    data += '<c r="D2"><v>PRICE_DO_NOT_PARSE &invalid;</v></c></row>'
    with workbook([data]) as archive:
        path = audit.sheet_path(archive)
        header = audit.read_schema(archive, path)
        original, columns = audit.decode_approved_cell, []

        def guarded_decoder(block, column, approved):
            assert column != "D"
            assert b"PRICE_DO_NOT_PARSE" not in block
            columns.append(column)
            return original(block, column, approved)

        monkeypatch.setattr(audit, "decode_approved_cell", guarded_decoder)
        result = audit.audit_features(archive, path, header, config())
    assert result["feature_rows"] == 1
    assert result["cell_access"]["blocked_data_cells_skipped"] == 1
    assert result["price_cells_decoded"] == 0
    assert result["model_alias_pattern_match_rows"]["transmission"] == 1
    assert columns == ["A", "B", "C", "A", "B", "C"]
    assert "PRICE_DO_NOT_PARSE" not in json.dumps(result)


def test_unrequested_price_shared_string_is_not_parsed():
    strings = '<si><t>RequestedBrand</t></si><si><t>PRIVATE_PRICE &invalid;</t></si>'
    data = '<row r="2"><c r="A2" t="s"><v>0</v></c><c r="D2" t="s"><v>1</v></c></row>'
    with workbook([data], shared=strings) as archive:
        header = audit.read_schema(archive, audit.sheet_path(archive))
        result = audit.audit_features(archive, "xl/worksheets/sheet1.xml", header, config())
    assert result["fields"]["marka"]["most_common_nonblank"] == [["RequestedBrand", 1]]
    assert "PRIVATE_PRICE" not in json.dumps(result)
    assert result["unrequested_shared_strings_decoded"] == 0


def test_guard_rejects_price_before_parsing_even_malformed_payload():
    with pytest.raises(ValueError, match="blocked column"):
        audit.decode_approved_cell(b"not XML containing a price", "D", {"A", "B"})


@pytest.mark.parametrize("changed", [
    {"feature_fields": ["marka", "seri", "model", "fiyat"]},
    {"feature_fields": ["marka", "seri", "seri", "model"]},
    {"expected_headers": ["marka", "seri", "model", "other"]},
])
def test_schema_and_allowlist_mismatch_fail_before_data_decoding(changed, monkeypatch):
    with workbook([]) as archive:
        header = audit.read_schema(archive, audit.sheet_path(archive))
        monkeypatch.setattr(audit, "selected_rows", lambda *a: pytest.fail("Should fail before rows"))
        with pytest.raises(ValueError, match="allowlist"):
            audit.audit_features(archive, "xl/worksheets/sheet1.xml", header, {**config(), **changed})


def test_additional_price_alias_is_not_approvable():
    header = {"A": "marka", "B": "seri", "C": "model", "D": "fiyat", "E": "price_net"}
    with workbook([], headers=header) as archive:
        with pytest.raises(ValueError, match="allowlist"):
            audit.audit_features(archive, "xl/worksheets/sheet1.xml", header, config(list(header.values())))


def test_nonprice_duplicates_do_not_depend_on_price_values():
    prefix = lambda r: '<row r="' + str(r) + '">' + inline_cell("A", r, "Brand") + inline_cell("B", r, "S") + inline_cell("C", r, "Model")
    data = [prefix(2) + '<c r="D2"><v>1</v></c></row>', prefix(3) + '<c r="D3"><v>99999999</v></c></row>']
    with workbook(data) as archive:
        header = audit.read_schema(archive, audit.sheet_path(archive))
        result = audit.audit_features(archive, "xl/worksheets/sheet1.xml", header, config())
    assert result["duplicate_nonprice_fingerprint_extra_rows"] == 1
    assert result["duplicate_nonprice_fingerprint_groups"] == 1
    assert result["largest_nonprice_fingerprint_group"] == 2
    assert result["fingerprints_are_not_vehicle_entity_ids"]
    assert result["price_validity_not_checked"]


def test_unique_positional_ids_do_not_hide_duplicate_vehicle_profiles():
    headers = {"A": "marka", "B": "seri", "C": "model", "D": "fiyat", "E": "id"}
    rows = ['<row r="' + str(r) + '">' + inline_cell("A", r, "Brand") + inline_cell("E", r, str(r - 1)) + '</row>' for r in (2, 3)]
    settings = {**config(list(headers.values())), "nonprice_fingerprint_fields": ["marka", "seri", "model"]}
    with workbook(rows, headers=headers) as archive:
        result = audit.audit_features(archive, "xl/worksheets/sheet1.xml", headers, settings)
    assert result["id_duplicate_nonblank_extra_rows"] == 0
    assert result["id_order_checks"]["one_based_order_matches"] == 2
    assert result["duplicate_nonprice_fingerprint_extra_rows"] == 1
    assert not result["id_is_marketplace_identifier_verified"]


def test_nonblank_groups_and_empty_vehicle_rows_use_features_only():
    rows = ['<row r="2">' + inline_cell("A", 2, "Brand") + inline_cell("B", 2, "Series") + '</row>',
            '<row r="3"><c r="D3"><v>PRICE &invalid;</v></c></row>']
    settings = {**config(), "nonblank_candidate_groups": {"brand_series": ["marka", "seri"]}}
    with workbook(rows) as archive:
        header = audit.read_schema(archive, audit.sheet_path(archive))
        result = audit.audit_features(archive, "xl/worksheets/sheet1.xml", header, settings)
    assert result["nonprice_cohort_counts"] == {
        "missing_brand_series_or_model_rows": 2, "all_vehicle_profile_fields_blank_rows": 1,
        "nonblank_group_brand_series": 1}
    assert result["price_cells_decoded"] == 0


def test_unapproved_field_in_candidate_group_fails_closed():
    settings = {**config(), "nonblank_candidate_groups": {"forbidden": ["fiyat"]}}
    with workbook([]) as archive:
        header = audit.read_schema(archive, audit.sheet_path(archive))
        with pytest.raises(ValueError, match="only approved"):
            audit.audit_features(archive, "xl/worksheets/sheet1.xml", header, settings)


@pytest.mark.parametrize("headers", [
    {"A": "marka", "B": "seri", "C": "model", "D": "123456"},
    {"A": "marka", "B": "seri", "C": "marka", "D": "fiyat"},
])
def test_unexpected_header_is_rejected_without_echoing_values(headers):
    with workbook([], headers=headers) as archive:
        with pytest.raises(ValueError, match="Unexpected or duplicate header") as error:
            audit.read_schema(archive, audit.sheet_path(archive))
    assert "123456" not in str(error.value)


@pytest.mark.parametrize("kwargs", [{"extra_sheet": True}, {"external": True}])
def test_unknown_sheet_or_external_workbook_relationship_is_rejected(kwargs):
    with workbook([], **kwargs) as archive:
        with pytest.raises(ValueError):
            audit.sheet_path(archive)


def test_formula_feature_cells_are_not_evaluated_or_accepted():
    with pytest.raises(ValueError, match="Formula"):
        audit.decode_approved_cell(b'<c r="A2"><f>D2</f><v>1</v></c>', "A", {"A"})


def test_header_only_does_not_call_data_decoder(monkeypatch):
    data = '<row r="2"><c r="D2"><v>UNREAD_PRICE &invalid;</v></c></row>'
    original = audit.decode_approved_cell

    def header_decoder(block, column, approved):
        assert audit.cell_reference(block)[1] == 1
        return original(block, column, approved)

    monkeypatch.setattr(audit, "decode_approved_cell", header_decoder)
    with workbook([data]) as archive:
        assert audit.read_schema(archive, audit.sheet_path(archive))["D"] == "fiyat"


def test_missing_shared_string_reference_fails_closed():
    with workbook([], shared='<si><t>OnlyOne</t></si>') as archive:
        with pytest.raises(ValueError, match="missing shared-string"):
            audit.resolve_shared_strings(archive, {2})


def test_inherited_row_attribute_namespace_does_not_require_payload_parsing():
    assert audit.row_number(b'<row r="1" x14ac:dyDescent="0.25"><c r="D1"/></row>') == 1
    with pytest.raises(ValueError, match="coordinate"):
        audit.row_number(b'<row><c r="D1"/></row>')


def test_partial_raw_archive_is_rejected_before_cell_parsing(tmp_path):
    path = tmp_path / "candidate.opaque"
    path.write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="Archive size"):
        downloader.verify_archive(path)


class RangeResponse(BytesIO):
    def __init__(self, data, start=0, end=3, code=206):
        super().__init__(data)
        self.code = code
        self.headers = {"Content-Range": f"bytes {start}-{end}/{downloader.EXPECTED_SIZE}"}

    def geturl(self):
        return downloader.URL

    def getcode(self):
        return self.code


def test_http_range_truncation_retries_with_exact_coordinates(monkeypatch):
    responses = iter([RangeResponse(b"ab"), RangeResponse(b"abcd")])
    monkeypatch.setattr(downloader, "build_opener", lambda *a: SimpleNamespace(open=lambda *a, **k: next(responses)))
    logs = []
    assert downloader.fetch_range(0, 3, logs) == b"abcd"
    assert len(logs) == 2 and not logs[0]["success"] and logs[1]["success"]


@pytest.mark.parametrize("kwargs", [{"start": 1, "end": 4}, {"code": 200}])
def test_wrong_range_or_ignored_range_is_rejected(monkeypatch, kwargs):
    monkeypatch.setattr(downloader, "build_opener", lambda *a: SimpleNamespace(open=lambda *a, **k: RangeResponse(b"abcd", **kwargs)))
    with pytest.raises(ValueError, match="coordinates"):
        downloader.fetch_range(0, 3, [])
