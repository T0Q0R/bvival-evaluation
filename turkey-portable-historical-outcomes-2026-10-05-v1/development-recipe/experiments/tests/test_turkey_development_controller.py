import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import run_turkey_development as runner
from run_turkey_policy_synthetic_integration import smoke_plan
from run_turkey_synthetic_integration import synthetic_dataset


ROOT = Path(__file__).parents[2]


@pytest.fixture
def setup():
    plan = json.loads((ROOT / "experiments/configs/turkey_execution_plan_v1_2026-09-30.json").read_text())
    config = json.loads((ROOT / "experiments/configs/turkey_price_free_cohort_2026-09-30.json").read_text())
    binding = json.loads((ROOT / "experiments/outputs/turkey_execution_plan_freeze_v1_2026-09-30/execution_plan_freeze_audit.json").read_text())
    return synthetic_dataset(config), smoke_plan(plan), config, binding


def test_artifact_verification_requires_exact_bytes_and_inventory(tmp_path):
    (tmp_path / "model.json").write_text("synthetic")
    hashes = runner.file_hashes(tmp_path)
    runner.verify_files(tmp_path, hashes)
    (tmp_path / "model.json").write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        runner.verify_files(tmp_path, hashes)
    (tmp_path / "extra.json").write_text("extra")
    with pytest.raises(ValueError, match="inventory"):
        runner.verify_files(tmp_path, hashes)


def test_changed_hash_is_rejected_before_model_deserialization(tmp_path, monkeypatch):
    (tmp_path / "model.joblib").write_bytes(b"synthetic_not_a_model")
    hashes = runner.file_hashes(tmp_path)
    (tmp_path / "model.joblib").write_bytes(b"changed")
    monkeypatch.setattr(runner.joblib, "load", lambda *args: pytest.fail("Deserialization reached before hash validation"))
    with pytest.raises(ValueError, match="hash"):
        runner.reload_check(tmp_path, {}, {}, {}, Path("unused"), project_root=ROOT, hashes=hashes)


@pytest.mark.parametrize("mode", ["test", "calibration", "any_source"])
def test_no_test_or_calibration_execution_mode(setup, tmp_path, mode):
    data, plan, config, binding = setup
    with pytest.raises(ValueError, match="No calibration/test"):
        runner.execute_pipeline(data, plan, config, binding, tmp_path / "out", Path("unused"),
                                project_root=ROOT, evidence_status=mode, provenance={})


def test_previous_attempt_cannot_be_overwritten(setup, tmp_path):
    data, plan, config, binding = setup
    (tmp_path / "previous.json").write_text("preserve")
    with pytest.raises(ValueError, match="preserve"):
        runner.execute_pipeline(data, plan, config, binding, tmp_path, Path("unused"),
                                project_root=ROOT, evidence_status="synthetic_only", provenance={})
    assert (tmp_path / "previous.json").read_text() == "preserve"


def test_bad_readiness_stops_before_workbook_or_label_access(setup, tmp_path, monkeypatch):
    _, plan, config, binding = setup
    monkeypatch.setattr(runner, "bound_plan", lambda *args, **kwargs: (plan, config, Path("unused"), binding))
    monkeypatch.setattr(runner, "verify_readiness_package", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("not ready")))
    monkeypatch.setattr(runner, "export_development_prices", lambda *args, **kwargs: pytest.fail("No workbook price access allowed"))
    monkeypatch.setattr(runner, "load_development_dataset", lambda *args, **kwargs: pytest.fail("No label loading allowed"))
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="not ready"):
        runner.run_authorized(Path("unopened_source.xlsx"), None, Path("plan"), Path("receipt"), output,
                              Path("python"), project_root=ROOT)
    assert not output.exists()


def test_neural_preflight_stops_before_prices(setup, tmp_path, monkeypatch):
    _, plan, config, binding = setup
    monkeypatch.setattr(runner, "bound_plan", lambda *args, **kwargs: (plan, config, Path("unused"), binding))
    monkeypatch.setattr(runner, "verify_readiness_package", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "probe_neural_environment", lambda *args: (_ for _ in ()).throw(ValueError("wrong environment")))
    monkeypatch.setattr(runner, "export_development_prices", lambda *args, **kwargs: pytest.fail("Prices reached before environment gate"))
    with pytest.raises(ValueError, match="environment"):
        runner.run_authorized(Path("unopened_source.xlsx"), None, Path("plan"), Path("receipt"), tmp_path / "out",
                              Path("python"), project_root=ROOT)


def test_missing_mandatory_neural_comparator_preserves_failed_attempt(setup, tmp_path, monkeypatch):
    data, plan, config, binding = setup
    records = {r["record_key"]: {"before_prediction_log": 7., "before_disagreement_log": .01,
                                 "after_prediction_log": [7., 7., 7.], "per_seed_state_logs": [[7.] * 4] * 3} for r in data["rows"]}
    valuation = {"oof": {r["record_key"]: records[r["record_key"]] for r in data["rows"] if r["split"] == "train"},
                 "validation": {r["record_key"]: records[r["record_key"]] for r in data["rows"] if r["split"] == "validation"},
                 "validation_targets_used_for_base_selection": False, "fit_count": 258}
    monkeypatch.setattr(runner, "run_nested_valuation", lambda *args, **kwargs: valuation)
    monkeypatch.setattr(runner, "fit_policy_heads", lambda *args, **kwargs: {"models": {}, "scores": {"risk": np.ones(20)}, "fixed_field": "kilometre"})
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(99, ["synthetic-neural"])))
    output = tmp_path / "out"
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute_pipeline(data, plan, config, binding, output, Path("python"),
                                project_root=ROOT, evidence_status="synthetic_only", provenance={"source_prices_parsed": 0})
    failure = json.loads((output / "failed_development_execution.json").read_text())
    assert failure["last_phase"] == "neural_started" and failure["no_silent_resume_or_comparator_deletion"]
    assert not (output / "development_execution_audit.json").exists()
    assert (output / "valuation_results.json").exists()
    assert all(p.stat().st_mode & 0o777 == 0o400 for p in output.rglob("*") if p.is_file())


@pytest.mark.parametrize("change", ["controller_flag", "test_scope", "source_inventory", "neural_environment"])
def test_partial_receipt_is_insufficient(setup, monkeypatch, change):
    _, _, _, binding = setup
    receipt = {"source_sha256": {name: "synthetic_hash" for name in runner.RUNTIME_SOURCES},
               "full_controller_and_reload_integration_passed": True, "calibration_or_test_release_allowed": False,
               "runtime_environment": binding["runtime_environment"], "neural_runtime_environment": binding["neural_runtime_environment"]}
    monkeypatch.setattr(runner, "require_development_receipt", lambda *args, **kwargs: None)
    if change == "controller_flag":
        receipt["full_controller_and_reload_integration_passed"] = False
    elif change == "test_scope":
        receipt["calibration_or_test_release_allowed"] = True
    elif change == "source_inventory":
        del receipt["source_sha256"]["turkey_neural_policy.py"]
    else:
        receipt["neural_runtime_environment"] = {}
    with pytest.raises(ValueError, match="Complete controller"):
        runner.require_controller_receipt(receipt, binding, project_root=ROOT)


def test_failed_test_run_never_counts_as_readiness(tmp_path, monkeypatch):
    import freeze_turkey_development_readiness as module
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "1 failed, 10 passed", ""))
    with pytest.raises(ValueError, match="regression tests failed"):
        module.run_tests(Path("python"), ["tests"], tmp_path / "test_report.txt", project_root=ROOT)
    assert "1 failed" in (tmp_path / "test_report.txt").read_text()


def test_existing_label_release_is_not_source_workbook_argument(setup, tmp_path):
    _, _, _, _ = setup
    with pytest.raises(ValueError, match="Exactly one"):
        runner.run_authorized(Path("source"), Path("labels"), Path("plan"), Path("ready"), tmp_path / "out",
                              Path("python"), project_root=ROOT)


def test_replay_roundoff_tolerance_does_not_hide_material_prediction_changes():
    first = {"synthetic": {"before_prediction_log": 7., "before_disagreement_log": .01,
                           "after_prediction_log": [7.] * 3, "per_seed_state_logs": [[7.] * 4] * 3}}
    second = copy.deepcopy(first)
    second["synthetic"]["before_prediction_log"] += 1e-14
    assert runner.prediction_records_match(first, second)
    second["synthetic"]["before_prediction_log"] += 1e-3
    assert not runner.prediction_records_match(first, second)


@pytest.mark.parametrize("with_platform", [False, True])
def test_environment_probe_handles_both_frozen_environment_schemas(monkeypatch, with_platform):
    expected = {"python": "synthetic_version", "packages": {"synthetic_package": "synthetic_version"}}
    if with_platform:
        expected["platform"] = "synthetic_platform"
    def fake_command(command, **kwargs):
        assert ("observed['platform']" in command[-1]) == with_platform
        return subprocess.CompletedProcess(command, 0, json.dumps(expected), "")
    monkeypatch.setattr(runner.subprocess, "run", fake_command)
    runner.probe_neural_environment(Path("synthetic_python"), expected)
