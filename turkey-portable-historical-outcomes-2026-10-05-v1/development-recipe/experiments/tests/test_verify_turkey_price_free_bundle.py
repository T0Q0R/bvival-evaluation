import hashlib
import json
from pathlib import Path

import pytest

from build_turkey_price_free_manifest import bucket, digest, fold, write_csv
from archive_turkey_candidate import EXPECTED_SHA256
from verify_turkey_price_free_bundle import table, verify_bundle


@pytest.fixture
def bundle(tmp_path):
    config = json.loads((Path(__file__).parents[1] / "configs/turkey_price_free_cohort_2026-09-30.json").read_text())
    key = digest([EXPECTED_SHA256, "source_data_ordinal", 1])
    group = digest(["synthetic", 1])
    split = bucket(group, config)
    write_csv(tmp_path / "split_manifest.csv", ["record_key", "group_hash", "profile_hash", "split", "oof_fold"],
              [{"record_key": key, "group_hash": group, "profile_hash": "profile", "split": split,
                "oof_fold": str(fold(group, config)) if split == "train" else ""}])
    for name, role in [("initial_features.csv", "initial_visible_fields"), ("acquisition_fields.csv", "action_fields")]:
        columns = ["record_key", *config[role]]
        write_csv(tmp_path / name, columns, [{"record_key": key, **{f: "" for f in config[role]}}])
    write_csv(tmp_path / "source_locators.csv", ["record_key", "source_data_ordinal"], [{"record_key": key, "source_data_ordinal": 1}])
    write_csv(tmp_path / "excluded_records.csv", ["record_key", "reason"], [])
    (tmp_path / "config_snapshot.json").write_text(json.dumps(config))
    for name in ("cohort_builder_snapshot.py", "reader_snapshot.py", "archive_helper_snapshot.py", "prelabel_protocol_snapshot.md"):
        (tmp_path / name).write_text("synthetic snapshot")
    audit = {"raw_feature_rows": 1, "retained_feature_rows": 1, "split_listing_counts": {split: 1},
             "split_group_counts": {s: int(s == split) for s in ("train", "validation", "calibration", "test")},
             "price_cell_values_decoded": 0, "labels_exported": False, "models_trained": False,
             "test_scores_computed": False, "confirmatory_test_allowed_now": False,
             "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}}
    (tmp_path / "cohort_freeze_audit.json").write_text(json.dumps(audit))
    return tmp_path


def test_saved_bundle_can_be_verified_without_loading_workbook(bundle):
    result = verify_bundle(bundle)
    assert result["retained_feature_rows"] == 1 and result["hashes_checked"] == 10
    assert not result["labels_exported"] and not result["complete_model_protocol_frozen"]


def test_hash_mutation_is_detected(bundle):
    with (bundle / "initial_features.csv").open("a") as stream:
        stream.write("corruption\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(bundle)


def test_extra_label_file_is_rejected(bundle):
    (bundle / "labels.csv").write_text("unexpected")
    with pytest.raises(ValueError, match="Unexpected files"):
        verify_bundle(bundle)


def test_unexpected_initial_columns_are_rejected_even_if_rehashed(bundle):
    path = bundle / "initial_features.csv"
    path.write_text(path.read_text().replace("record_key,", "record_key,kilometre,", 1))
    audit_path = bundle / "cohort_freeze_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["output_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="table columns"):
        verify_bundle(bundle)


def test_nonprelabel_stage_is_rejected(bundle):
    path = bundle / "cohort_freeze_audit.json"
    audit = json.loads(path.read_text())
    audit["labels_exported"] = True
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="prelabel stage"):
        verify_bundle(bundle)


def test_duplicate_table_key_is_rejected(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("record_key,value\nx,1\nx,2\n")
    with pytest.raises(ValueError, match="Duplicate record"):
        table(path, ["record_key", "value"])


def test_saved_group_assignment_must_obey_frozen_namespace_rule(bundle):
    path = bundle / "split_manifest.csv"
    rows = table(path, ["record_key", "group_hash", "profile_hash", "split", "oof_fold"])
    content = path.read_text()
    changed = "test" if rows[0]["split"] != "test" else "validation"
    path.write_text(content.replace("," + rows[0]["split"] + ",", "," + changed + ","))
    audit_path = bundle / "cohort_freeze_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["output_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="hash rule"):
        verify_bundle(bundle)
