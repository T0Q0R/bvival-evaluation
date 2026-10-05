import copy
import json
from pathlib import Path

import pytest

import freeze_turkey_heldout_scores as freezer
from turkey_price_free_scoring import verify_completed_package


ROOT = Path(__file__).parents[2]
SYNTHETIC = ROOT / "experiments/outputs/turkey_development_readiness_2026-09-30_v2/synthetic_execution"
CHECK = ROOT / "experiments/outputs/turkey_initial_only_scoring_synthetic_2026-09-30_v1/inference"


@pytest.fixture
def bound():
    _, _, config = verify_completed_package(SYNTHETIC, project_root=ROOT)
    result = json.loads((CHECK / "initial_only_scores.json").read_text())
    initial = {f: result["before"]["contexts"][0][f] for f in config["initial_visible_fields"]}
    partitions = {s: {"records": [{"record_key": s + str(i), "initial": initial.copy()} for i in range(3)],
                      "groups": {s + str(i): s + "group" + str(i) for i in range(3)}} for s in ("calibration", "test")}
    binding = {"mode": "synthetic_only", "expected_keys": {s: sorted(m["groups"]) for s, m in partitions.items()}}
    return config, partitions, binding


def test_exact_partition_universe_validates_without_outcomes(bound):
    config, partitions, binding = bound
    freezer.validate_partitions(partitions, binding, ["unrelated_development"], config)


@pytest.mark.parametrize("mutation", ["missing_split", "missing_record", "extra_record", "missing_group",
                                    "group_crossing", "outcome_role", "hidden_initial", "duplicate_expected", "unsorted_expected"])
def test_partition_boundary_and_capacity_mutations_rejected(bound, mutation):
    config, partitions, binding = bound
    if mutation == "missing_split":
        del partitions["test"]
    elif mutation == "missing_record":
        partitions["test"]["records"].pop()
    elif mutation == "extra_record":
        partitions["test"]["records"].append({"record_key": "testextra", "initial": partitions["test"]["records"][0]["initial"]})
    elif mutation == "missing_group":
        del partitions["test"]["groups"]["test0"]
    elif mutation == "group_crossing":
        partitions["test"]["groups"]["test0"] = partitions["calibration"]["groups"]["calibration0"]
    elif mutation == "outcome_role":
        partitions["test"]["price_valid"] = [True] * 3
    elif mutation == "hidden_initial":
        partitions["test"]["records"][0]["initial"]["kilometre"] = "1000"
    elif mutation == "duplicate_expected":
        binding["expected_keys"]["test"].append("test0")
    else:
        binding["expected_keys"]["test"].reverse()
    with pytest.raises(ValueError):
        freezer.validate_partitions(partitions, binding, [], config)


def test_development_listing_overlap_rejected(bound):
    config, partitions, binding = bound
    with pytest.raises(ValueError, match="overlap"):
        freezer.validate_partitions(partitions, binding, ["test0"], config)


@pytest.fixture
def replay():
    _, plan, config = verify_completed_package(SYNTHETIC, project_root=ROOT)
    result = freezer.read_inference(CHECK)
    records = [{"record_key": key, "initial": {f: c[f] for f in config["initial_visible_fields"]}}
               for key, c in zip(result["before"]["keys"], result["before"]["contexts"])]
    return result, records, plan, config


def test_saved_inference_reconstructs_all_policies(replay):
    result, records, plan, config = replay
    freezer.compare_saved_inference(result, copy.deepcopy(result), records, plan, config)


@pytest.mark.parametrize("mutation", ["refit", "release", "price_read", "wrong_key", "missing_seed", "before_identity",
                                    "context_initial", "target_added", "negative_disagreement", "action_change", "head_change"])
def test_cached_replay_mutations_cannot_freeze(replay, mutation):
    result, records, plan, config = replay
    changed = copy.deepcopy(result)
    if mutation == "refit":
        changed["refits"] = 1
    elif mutation == "release":
        changed["heldout_label_release_allowed"] = True
    elif mutation == "price_read":
        changed["calibration_or_test_labels_read"] = True
    elif mutation == "wrong_key":
        changed["before"]["keys"][0] = "wrong"
    elif mutation == "missing_seed":
        del changed["before"]["per_seed_before_raw_log"]["13"]
    elif mutation == "before_identity":
        changed["before"]["before_prediction_log"][0] += .1
    elif mutation == "context_initial":
        changed["before"]["contexts"][0]["marka"] = "changed"
    elif mutation == "target_added":
        changed["targets"] = [1000.]
    elif mutation == "negative_disagreement":
        changed["before"]["contexts"][0]["before_disagreement_log"] = -1.
    elif mutation == "action_change":
        changed["allocations"]["ensemble"]["0.10"]["benefit"][0][1] = "wrong_action"
    else:
        changed["heads"]["scores"]["value"][0][0] += 100.
    with pytest.raises(ValueError):
        freezer.compare_saved_inference(result, changed, records, plan, config)


def test_source_loader_parses_only_config_split_initial(monkeypatch):
    real_table = freezer.table
    observed = []
    def limited_table(path, columns):
        assert path.name in {"split_manifest.csv", "initial_features.csv"}
        observed.append(path.name)
        return real_table(path, columns)
    monkeypatch.setattr(freezer, "table", limited_table)
    _, config, partitions, binding = freezer.load_bound_initial_partitions(
        ROOT / "experiments/outputs/turkey_execution_plan_freeze_v1_2026-09-30", project_root=ROOT)
    freezer.validate_partitions(partitions, binding, [], config)
    assert observed == ["split_manifest.csv", "initial_features.csv"]
    assert len(partitions["calibration"]["records"]) == 7814
    assert len(partitions["test"]["records"]) == 7802
    assert binding["acquisition_values_decoded"] is False and binding["labels_read"] is False


def test_source_freeze_refuses_synthetic_package_before_cohort_access(tmp_path, monkeypatch):
    monkeypatch.setattr(freezer, "load_bound_initial_partitions", lambda *args, **kwargs: pytest.fail("Source loader reached from synthetic"))
    with pytest.raises(ValueError, match="authorized source"):
        freezer.run_source_freeze(SYNTHETIC, Path("unused"), tmp_path / "out", Path("unused"), project_root=ROOT)


def test_incomplete_source_refused_before_cohort_or_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(freezer, "load_bound_initial_partitions", lambda *args, **kwargs: pytest.fail("Partial source reached cohort"))
    with pytest.raises(ValueError, match="Completed"):
        freezer.run_source_freeze(tmp_path, Path("unused"), tmp_path / "out", Path("unused"), project_root=ROOT)


def test_overlapping_partition_refused_before_model_scoring(bound, tmp_path, monkeypatch):
    _, partitions, binding = bound
    saved = json.loads((SYNTHETIC / "valuation_results.json").read_text())
    key = next(iter(saved["oof"]))
    partitions["test"]["records"][0]["record_key"] = key
    del partitions["test"]["groups"]["test0"]
    partitions["test"]["groups"][key] = "synthetic_group"
    binding["expected_keys"]["test"] = sorted(partitions["test"]["groups"])
    monkeypatch.setattr(freezer, "score_completed_models", lambda *args, **kwargs: pytest.fail("Overlap reached models"))
    with pytest.raises(ValueError, match="overlap"):
        freezer.freeze_partitions(SYNTHETIC, partitions, binding, tmp_path / "out", Path("unused"), project_root=ROOT)
    assert not (tmp_path / "out").exists()


def test_existing_freeze_preserved(bound, tmp_path):
    _, partitions, binding = bound
    with pytest.raises(ValueError, match="nonexistent"):
        freezer.freeze_partitions(SYNTHETIC, partitions, binding, tmp_path, Path("unused"), project_root=ROOT)


def test_failed_inference_preserves_attempt_and_never_issues_freeze(bound, tmp_path, monkeypatch):
    _, partitions, binding = bound
    def fail(*args, **kwargs):
        raise ValueError("synthetic inference failure")
    monkeypatch.setattr(freezer, "score_completed_models", fail)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="inference failure"):
        freezer.freeze_partitions(SYNTHETIC, partitions, binding, output, Path("unused"), project_root=ROOT)
    failure = json.loads((output / "failed_heldout_score_freeze.json").read_text())
    assert failure["phase"] == "calibration_first_inference"
    assert failure["heldout_label_release_allowed"] is False
    assert not (output / "heldout_score_freeze_audit.json").exists()
    assert (output / "initial_partition_inputs.json").is_file()
    assert all(p.stat().st_mode & 0o777 == 0o400 for p in output.rglob("*") if p.is_file())


def test_tampered_saved_inference_rejected_before_score_decoding(tmp_path):
    (tmp_path / "initial_only_scores.json").write_text("not valid JSON and must not be parsed")
    audit = {"stage": "INITIAL_ONLY_SAVED_MODEL_INFERENCE_NOT_HELDOUT_FREEZE_RECEIPT",
             "calibration_or_test_labels_read": False, "heldout_label_release_allowed": False,
             "artifact_sha256": {"initial_only_scores.json": "wrong_hash"}}
    (tmp_path / "initial_only_scoring_audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="hash"):
        freezer.read_inference(tmp_path)


def test_synthetic_driver_refuses_source_package_before_scoring(tmp_path, monkeypatch):
    import run_turkey_heldout_freeze_check as driver
    monkeypatch.setattr(driver, "verify_completed_package", lambda *args, **kwargs: ({"evidence_status": "authorized_development_only"}, {}, {}))
    monkeypatch.setattr(driver, "freeze_partitions", lambda *args, **kwargs: pytest.fail("Source entered synthetic scoring"))
    with pytest.raises(ValueError, match="refuses"):
        driver.run_check(Path("unused"), tmp_path / "out", Path("unused"), project_root=ROOT)
