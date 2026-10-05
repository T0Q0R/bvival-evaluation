import json

import pytest

import turkey_price_cells as cells
from test_turkey_feature_only import inline_cell, workbook


HEADERS = {"A": "marka", "B": "seri", "C": "model", "D": "fiyat"}


def test_train_validation_selection_skips_malformed_test_payload_before_parser(monkeypatch):
    rows = ['<row r="2"><c r="D2"><v>1000</v></c></row>',
            '<row r="3"><c r="D3"><v>2000</v></c></row>',
            '<row r="4"><c r="D4"><v>UNSELECTED_PRICE &invalid;</v></c></row>']
    original = cells.decode_price_cell
    accessed = []
    def guarded(block, column, price_column, *, ordinal, allowed):
        assert ordinal in {1, 2} and b"UNSELECTED_PRICE" not in block
        accessed.append(ordinal)
        return original(block, column, price_column, ordinal=ordinal, allowed=allowed)
    monkeypatch.setattr(cells, "decode_price_cell", guarded)
    with workbook(rows) as archive:
        labels, access = cells.read_selected_prices(archive, HEADERS, {1, 2}, expected_data_rows=3)
    assert accessed == [1, 2]
    assert labels[1] == {"price": "1000", "price_valid": True}
    assert access["unselected_price_cells_skipped_before_xml_parser"] == 1
    assert access["unselected_price_values_decoded"] == 0
    assert "UNSELECTED_PRICE" not in json.dumps(labels)


def test_only_selected_price_shared_string_entries_are_resolved():
    rows = ['<row r="2"><c r="D2" t="s"><v>0</v></c></row>',
            '<row r="3"><c r="D3" t="s"><v>1</v></c></row>']
    with workbook(rows, shared='<si><t>1234</t></si><si><t>UNREAD_SHARED_PRICE &invalid;</t></si>') as archive:
        labels, access = cells.read_selected_prices(archive, HEADERS, {1}, expected_data_rows=2)
    assert labels[1]["price"] == "1234" and access["requested_price_shared_string_entries"] == 1


@pytest.mark.parametrize("cell", ['<c r="D2" t="b"><v>1</v></c>', '<c r="D2" t="d"><v>2026-01-01</v></c>',
                                 '<c r="D2" t="e"><v>#N/A</v></c>', '<c r="D2"><f>1+1</f><v>2</v></c>',
                                 '<c r="D2"><v>0</v></c>', '<c r="D2"><v>-1</v></c>',
                                 inline_cell("D", 2, "1,234"), inline_cell("D", 2, "₺1234")])
def test_invalid_selected_target_is_flagged_not_guessed(cell):
    with workbook(['<row r="2">' + cell + '</row>']) as archive:
        labels, access = cells.read_selected_prices(archive, HEADERS, {1}, expected_data_rows=1)
    assert labels[1] == {"price": "", "price_valid": False}
    assert access["selected_invalid_targets"] == 1


@pytest.mark.parametrize("column,ordinal", [("A", 1), ("D", 2)])
def test_unauthorized_decoder_rejects_before_malformed_xml_is_parsed(column, ordinal, monkeypatch):
    monkeypatch.setattr(cells.ET, "fromstring", lambda *a: pytest.fail("Unselected parser access"))
    with pytest.raises(ValueError, match="refuses"):
        cells.decode_price_cell(b"not XML", column, "D", ordinal=ordinal, allowed={1})


def test_source_ordinal_is_not_assumed_to_equal_physical_row_minus_one():
    with workbook(['<row r="5"><c r="D5"><v>1000</v></c></row>', '<row r="8"><c r="D8"><v>2000</v></c></row>']) as archive:
        labels, _ = cells.read_selected_prices(archive, HEADERS, {2}, expected_data_rows=2)
    assert labels == {2: {"price": "2000", "price_valid": True}}


def test_missing_selected_cell_is_retained_as_invalid_target_not_removed_row():
    with workbook(['<row r="2">' + inline_cell("A", 2, "Synthetic") + '</row>']) as archive:
        labels, access = cells.read_selected_prices(archive, HEADERS, {1}, expected_data_rows=1)
    assert labels[1]["price_valid"] is False and access["selected_rows_without_price_cell"] == 1


def test_changed_source_row_inventory_is_rejected():
    with workbook(['<row r="2"><c r="D2"><v>1000</v></c></row>']) as archive:
        with pytest.raises(ValueError, match="inventory"):
            cells.read_selected_prices(archive, HEADERS, {1}, expected_data_rows=2)


@pytest.mark.parametrize("selection", [set(), {0}, {True}, {3}])
def test_unbounded_ordinals_are_rejected(selection):
    with workbook([]) as archive:
        with pytest.raises(ValueError, match="Bounded"):
            cells.read_selected_prices(archive, HEADERS, selection, expected_data_rows=2)
