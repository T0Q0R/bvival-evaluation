import json
from pathlib import Path

import pytest

import turkey_development_io as io
from build_turkey_price_free_manifest import write_csv
from test_turkey_feature_only import workbook


def test_current_unready_receipt_blocks_workbook_opening(tmp_path, monkeypatch):
    audit = {"plan_sha256": "plan", "cohort_audit_sha256": "cohort"}
    monkeypatch.setattr(io, "bound_plan", lambda *a, **k: ({}, {}, tmp_path, audit))
    monkeypatch.setattr(io, "verify_archive", lambda *a: pytest.fail("No source byte/ZIP access before readiness"))
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps({"stage": "VALUATION_COMPONENT_ONLY_NOT_READY", "development_label_release_allowed": False}))
    with pytest.raises(ValueError, match="Full development"):
        io.export_development_prices(tmp_path / "not_opened.xlsx", tmp_path, path, tmp_path / "out", project_root=tmp_path)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("scope", [["test"], ["train", "validation", "calibration"], ["train", "validation", "test"]])
def test_other_scopes_are_never_authorized_by_development_receipt(tmp_path, scope):
    receipt = {"stage": "TURKEY_FULL_DEVELOPMENT_INTEGRATION_READY", "development_label_release_allowed": True,
               "price_release_scope": scope, "policy_and_neural_integration_passed": True,
               "nested_valuation_integration_passed": True, "plan_sha256": "p", "cohort_audit_sha256": "c"}
    with pytest.raises(ValueError, match="calibration/test"):
        io.require_development_receipt(receipt, {"plan_sha256": "p", "cohort_audit_sha256": "c"}, project_root=tmp_path)


def test_ready_claim_without_current_source_bindings_is_refused(tmp_path):
    receipt = {"stage": "TURKEY_FULL_DEVELOPMENT_INTEGRATION_READY", "development_label_release_allowed": True,
               "price_release_scope": ["train", "validation"], "policy_and_neural_integration_passed": True,
               "nested_valuation_integration_passed": True, "plan_sha256": "p", "cohort_audit_sha256": "c"}
    with pytest.raises(ValueError, match="source bindings"):
        io.require_development_receipt(receipt, {"plan_sha256": "p", "cohort_audit_sha256": "c"}, project_root=tmp_path)


@pytest.fixture
def synthetic_release(tmp_path):
    root = Path(__file__).parents[2]
    config = json.loads((root / "experiments/configs/turkey_price_free_cohort_2026-09-30.json").read_text())
    cohort, labels = tmp_path / "cohort", tmp_path / "labels"
    cohort.mkdir()
    labels.mkdir()
    rows = [{"record_key": str(i), "group_hash": "group" + str(i), "profile_hash": "p" + str(i),
             "split": "train" if i < 500 else "validation" if i < 600 else "test",
             "oof_fold": str(i % 5) if i < 500 else ""} for i in range(601)]
    write_csv(cohort / "split_manifest.csv", ["record_key", "group_hash", "profile_hash", "split", "oof_fold"], rows)
    for name, fields in [("initial_features.csv", config["initial_visible_fields"]), ("acquisition_fields.csv", config["action_fields"])]:
        write_csv(cohort / name, ["record_key", *fields], [{"record_key": row["record_key"], **dict.fromkeys(fields, "synthetic")} for row in rows])
    (cohort / "cohort_freeze_audit.json").write_text('{"synthetic_only":true}')
    write_csv(labels / "development_prices.csv", ["record_key", "price", "price_valid"],
              [{"record_key": row["record_key"], "price": "1000", "price_valid": "true"} for row in rows[:-1]])
    audit = {"release_scope": ["train", "validation"], "release_succeeded": True,
             "calibration_or_test_prices_decoded": False, "labels_sha256": io.sha256(labels / "development_prices.csv"),
             "cohort_audit_sha256": io.sha256(cohort / "cohort_freeze_audit.json")}
    (labels / "development_label_release_audit.json").write_text(json.dumps(audit))
    return cohort, labels, config


def test_loader_returns_development_records_and_targets_only(synthetic_release):
    cohort, labels, config = synthetic_release
    data = io.load_development_dataset(cohort, labels, config)
    assert len(data["rows"]) == 600 and set(data["prices"]) == set(data["initial"]) == set(data["actions"])
    assert "600" not in data["prices"]


def test_loader_rejects_test_target_even_when_untrusted_label_hash_is_updated(synthetic_release):
    cohort, labels, config = synthetic_release
    path = labels / "development_prices.csv"
    with path.open("a") as stream:
        stream.write("600,DO_NOT_CONVERT_TEST_PRICE,true\n")
    audit_path = labels / "development_label_release_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["labels_sha256"] = io.sha256(path)
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="no test keys"):
        io.load_development_dataset(cohort, labels, config)


def test_loader_refuses_failed_quality_release(synthetic_release):
    cohort, labels, config = synthetic_release
    path = labels / "development_label_release_audit.json"
    audit = json.loads(path.read_text())
    audit["release_succeeded"] = False
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="successful development-only"):
        io.load_development_dataset(cohort, labels, config)


@pytest.mark.parametrize("payload,partial_counts_unknown", [("0", False), ("MALFORMED_SELECTED_TARGET &invalid;", True)])
def test_failed_authorized_synthetic_export_preserves_access_audit_without_raw_payload(tmp_path, monkeypatch, payload, partial_counts_unknown):
    cohort = tmp_path / "synthetic_cohort"
    cohort.mkdir()
    (cohort / "cohort_freeze_audit.json").write_text('{"raw_feature_rows":2}')
    splits = [{"record_key": "a", "group_hash": "ga", "profile_hash": "pa", "split": "train", "oof_fold": "0"},
              {"record_key": "b", "group_hash": "gb", "profile_hash": "pb", "split": "validation", "oof_fold": ""}]
    write_csv(cohort / "split_manifest.csv", ["record_key", "group_hash", "profile_hash", "split", "oof_fold"], splits)
    write_csv(cohort / "source_locators.csv", ["record_key", "source_data_ordinal"], [{"record_key": "a", "source_data_ordinal": 1}, {"record_key": "b", "source_data_ordinal": 2}])
    audit = {"plan_sha256": "synthetic", "cohort_audit_sha256": "synthetic"}
    monkeypatch.setattr(io, "bound_plan", lambda *a, **k: ({}, {"expected_headers": ["marka", "seri", "model", "fiyat"]}, cohort, audit))
    monkeypatch.setattr(io, "require_development_receipt", lambda *a, **k: None)
    monkeypatch.setattr(io, "verify_archive", lambda *a: {"synthetic_only": True})
    archive = workbook(['<row r="2"><c r="D2"><v>' + payload + '</v></c></row>', '<row r="3"><c r="D3"><v>1000</v></c></row>'])
    monkeypatch.setattr(io.zipfile, "ZipFile", lambda *a, **k: archive)
    receipt = tmp_path / "synthetic_readiness.json"
    receipt.write_text('{"synthetic_fixture_only":true}')
    output = tmp_path / "failed_release"
    with pytest.raises(ValueError):
        io.export_development_prices(tmp_path / "not_a_source.xlsx", cohort, receipt, output, project_root=tmp_path)
    saved_text = (output / "development_label_release_audit.json").read_text()
    saved = json.loads(saved_text)
    assert saved["stage"] == "FAILED_DEVELOPMENT_PRICE_RELEASE"
    assert saved["price_access_started"] and not saved["release_succeeded"]
    assert saved["partial_cell_access_counts_unknown"] is partial_counts_unknown
    assert "MALFORMED_SELECTED_TARGET" not in saved_text
    assert all(p.stat().st_mode & 0o777 == 0o400 for p in output.iterdir())
