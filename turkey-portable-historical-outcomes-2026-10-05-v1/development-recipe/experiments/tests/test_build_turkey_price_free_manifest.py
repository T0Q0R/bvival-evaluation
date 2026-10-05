import copy
import json
from collections import Counter
from pathlib import Path

import pytest

import audit_turkey_feature_only as reader
import build_turkey_price_free_manifest as cohort
from test_turkey_feature_only import inline_cell, workbook


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "configs/turkey_price_free_cohort_2026-09-30.json").read_text())


def record(ordinal=1, **changes):
    row = {"id": str(ordinal), "marka": "BMW", "seri": "Series", "model": "320 D",
           "yil": "2015", "kilometre": "123000", "vites_tipi": "Otomatik", "yakit_tipi": "Dizel",
           "kasa_tipi": "Sedan", "renk": "White", "motor_hacmi": "1995", "motor_gucu": "184",
           "degisen_sayisi": "0", "boyali_sayisi": "0", "kimden": "Galeriden"}
    row.update(changes)
    return ordinal, row


def test_dedup_representative_and_splits_are_input_order_invariant(config):
    records = [record(1), record(2, kilometre="123000.0", marka=" bmw "), record(3, renk="Black")]
    kept, excluded, summary = cohort.build_records(records, config)
    reverse_kept, reverse_excluded, reverse_summary = cohort.build_records(list(reversed(records)), config)
    assert kept == reverse_kept and excluded == reverse_excluded and summary == reverse_summary
    assert len(kept) == 2 and len(excluded) == 1
    expected = min(cohort.digest([cohort.EXPECTED_SHA256, "source_data_ordinal", i]) for i in (1, 2))
    assert any(r["record_key"] == expected for r in kept)
    assert len({r["group_hash"] for r in kept}) == len({r["split"] for r in kept}) == 1
    assert summary["removed_duplicate_profile_rows"] == 1


@pytest.mark.parametrize("changes,reason", [
    ({"model": ""}, "missing_core_identity"),
    ({"yil": "2027"}, "invalid_year"),
    ({"yil": "2015.5"}, "invalid_year"),
    ({"kilometre": "99999999"}, "invalid_mileage"),
    ({"kilometre": "-1"}, "invalid_mileage"),
    ({"kilometre": "NaN"}, "invalid_mileage"),
    ({"vites_tipi": "unknown"}, "missing_or_unsupported_transmission"),
    ({"yakit_tipi": "NA"}, "missing_or_unsupported_fuel"),
])
def test_nonprice_exclusion_rules(changes, reason, config):
    kept, excluded, summary = cohort.build_records([record(**changes)], config)
    assert not kept and excluded[0]["reason"] == reason
    assert summary["raw_feature_rows"] == len(excluded)
    assert summary["prices_unparsed"]


def test_range_boundaries_and_zero_vs_missing_parts(config):
    rows = [record(1, yil="1900", kilometre="0", boyali_sayisi="0"),
            record(2, yil="2026", kilometre="1000000", boyali_sayisi="31", degisen_sayisi="-"),
            record(3, boyali_sayisi="2.5")]
    kept, excluded, summary = cohort.build_records(rows, config)
    by_ordinal = {r["source_data_ordinal"]: r for r in kept}
    assert not excluded and len(kept) == 3
    assert by_ordinal[1]["values"]["boyali_sayisi"] == "0"
    assert by_ordinal[2]["values"]["boyali_sayisi"] == by_ordinal[2]["values"]["degisen_sayisi"] == ""
    assert by_ordinal[3]["values"]["boyali_sayisi"] == ""


def test_price_payload_is_rejected_without_touching_value(config):
    class UnreadPrice:
        def __str__(self):
            pytest.fail("Price payload must never be accessed")

    ordinal, values = record()
    values["fiyat"] = UnreadPrice()
    with pytest.raises(ValueError, match="nonprice contract"):
        cohort.build_records([(ordinal, values)], config)


@pytest.mark.parametrize("field", ["model", "motor_hacmi", "motor_gucu", "id"])
def test_excluded_predictor_fields_cannot_be_moved_into_initial_context(field, config):
    changed = copy.deepcopy(config)
    changed["excluded_predictor_fields"].remove(field)
    changed["initial_visible_fields"].append(field)
    with pytest.raises(ValueError, match="feature roles"):
        cohort.validate_config(changed)


def test_acquisition_field_cannot_also_be_initial_visible(config):
    config["initial_visible_fields"].append("kilometre")
    with pytest.raises(ValueError, match="feature roles"):
        cohort.validate_config(config)


def test_groups_and_training_oof_do_not_cross_boundaries(config):
    rows = [record(i, model=f"Model {i // 2}", kilometre=str(1000 + i // 2), renk=str(i)) for i in range(1, 501)]
    kept, _, _ = cohort.build_records(rows, config)
    cohort.assert_group_boundaries(kept)
    assert set(r["split"] for r in kept) == {"train", "validation", "calibration", "test"}
    assert set(r["oof_fold"] for r in kept if r["split"] == "train") == set(range(5))
    first = next(r for r in kept if sum(x["group_hash"] == r["group_hash"] for x in kept) > 1)
    altered = copy.deepcopy(kept)
    next(r for r in altered if r["record_key"] == first["record_key"])["split"] = "inconsistent"
    with pytest.raises(AssertionError, match="crosses"):
        cohort.assert_group_boundaries(altered)


def test_duplicate_source_locator_is_rejected(config):
    with pytest.raises(ValueError, match="unique positive ordinal"):
        cohort.build_records([record(1), record(1, renk="Blue")], config)


def test_xlsx_reader_rejects_price_cell_decoder_access(config, monkeypatch):
    headers = {chr(65 + i): f for i, f in enumerate(config["expected_headers"])}
    _, values = record()
    cells = ''.join(inline_cell(column, 2, values[field]) for column, field in headers.items() if field != "fiyat")
    cells += '<c r="P2"><v>UNPARSEABLE_PRICE &invalid;</v></c>'
    with workbook(['<row r="2">' + cells + '</row>'], headers=headers) as archive:
        header = reader.read_schema(archive, reader.sheet_path(archive))
        original = reader.decode_approved_cell

        def guard(block, column, approved):
            assert column != "P" and b"UNPARSEABLE_PRICE" not in block
            return original(block, column, approved)

        monkeypatch.setattr(reader, "decode_approved_cell", guard)
        records, counters = cohort.read_records(archive, header, config)
    kept, _, _ = cohort.build_records(records, config)
    assert len(kept) == 1 and "fiyat" not in kept[0]["values"]
    assert counters["blocked_data_cells_skipped"] == 1


def test_initial_and_acquisition_csv_headers_remain_disjoint(config, tmp_path):
    kept, _, _ = cohort.build_records([record()], config)
    row = kept[0]
    for table, role in [("initial.csv", "initial_visible_fields"), ("acquisition.csv", "action_fields")]:
        columns = ["record_key", *config[role]]
        cohort.write_csv(tmp_path / table, columns, [{"record_key": row["record_key"], **{f: row["values"][f] for f in config[role]}}])
        assert (tmp_path / table).stat().st_mode & 0o777 == 0o600
    assert not set(config["action_fields"]) & set((tmp_path / "initial.csv").read_text().splitlines()[0].split(","))
    assert not any(f in (tmp_path / "initial.csv").read_text().splitlines()[0].split(",") for f in config["excluded_predictor_fields"])


def test_exclusion_counts_reconcile_and_no_labels_are_exported(config):
    kept, excluded, summary = cohort.build_records([record(1), record(2), record(3, yil="bad", kilometre="")], config)
    assert len(kept) + len(excluded) == 3
    assert sum(summary["exclusive_exclusion_counts"].values()) == 2
    assert summary["pre_joint_filter_flags"]["invalid_year"] == summary["pre_joint_filter_flags"]["invalid_mileage"] == 1
    assert not summary["confirmatory_test_allowed_now"] and summary["final_scored_sample_count_unknown"]
