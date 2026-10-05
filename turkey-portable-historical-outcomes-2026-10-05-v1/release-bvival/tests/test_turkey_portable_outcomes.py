"""Synthetic contract checks, not historical source reconstruction evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_turkey_portable_outcomes as checker
import run_turkey_portable_outcomes as runner
from turkey_portable_development_contract import SOURCE_SHA
from turkey_portable_outcome_contract import (
    DEVELOPMENT_MANIFEST_SHA, FALSE_CAPABILITIES, SCOPE, VERSION, closure_roots,
)


def manifest():
    return {"version": VERSION, "scope": SCOPE, "source_sha256": SOURCE_SHA,
        "historical_tests_already_opened": True, "development_manifest_sha256": DEVELOPMENT_MANIFEST_SHA,
        "closure_roots": closure_roots(), **{k: False for k in FALSE_CAPABILITIES},
        "files": [{"file": "README.md"}]}


def test_explicitly_historical_not_once_release_scope():
    assert len(checker.validate_manifest(manifest())) == 1


@pytest.mark.parametrize("flag", FALSE_CAPABILITIES)
def test_no_scope_upgrades(flag):
    value = manifest()
    value[flag] = True
    with pytest.raises(ValueError, match="Unsupported companion"):
        checker.validate_manifest(value)


@pytest.mark.parametrize("key,value", [
    ("historical_tests_already_opened", False), ("source_sha256", "other"),
    ("development_manifest_sha256", "other"), ("scope", "new_confirmation"),
])
def test_cannot_substitute_fresh_test_or_other_development(key, value):
    data = manifest()
    data[key] = value
    with pytest.raises(ValueError, match="historical scope"):
        checker.validate_manifest(data)


def test_required_path_snapshot_cannot_be_omitted():
    data = manifest()
    data["closure_roots"].remove("experiments/tests/test_freeze_turkey_heldout_scores.py")
    with pytest.raises(ValueError, match="historical scope"):
        checker.validate_manifest(data)


@pytest.mark.parametrize("name", ["rows.csv", "prices.csv", "models/seed.joblib", "models/seed.pt",
    "experiments/access_ledgers/ledger.json", "paper-bvival/main.tex", "reference-metadata/phase_prices.csv"])
def test_no_rows_models_ledger_or_manuscript_payload(name):
    assert not checker.permitted(name)


def test_private_output_must_not_edit_development_or_package(tmp_path):
    package, development = tmp_path / "package", tmp_path / "development"
    package.mkdir()
    development.mkdir()
    for path in (package, package / "new", development, development / "new"):
        with pytest.raises(ValueError, match="separate"):
            runner.new_output(path, package, development)


def test_completed_fit_required_before_copy_or_source_parse(tmp_path, monkeypatch):
    package, development = tmp_path / "package", tmp_path / "development"
    package.mkdir()
    development.mkdir()
    monkeypatch.setattr(runner, "verify", lambda *args: {})
    monkeypatch.setattr(runner, "bound_workspace", lambda *args: (development, {}))
    monkeypatch.setattr(runner, "safe_file", lambda *args: development / "synthetic")
    monkeypatch.setattr(runner, "sha", lambda *args: DEVELOPMENT_MANIFEST_SHA)
    def require_complete(*args):
        if args[-1] == "fit":
            raise ValueError("Completed bound phase required: fit")
        return {}
    monkeypatch.setattr(runner, "completed_phase", require_complete)
    output = tmp_path / "new-private"
    with pytest.raises(ValueError, match="Completed bound phase"):
        runner.prepare(package, development, output, True)
    assert not output.exists()


@pytest.mark.parametrize("actual", [{"x": 1.1}, {"x": True}, {"x": float("nan")},
                                  {"y": 1}, {"x": "1"}, {"x": 1, "selected_only": 1}])
def test_no_changed_numerics_bool_nan_or_selection_scope(actual):
    with pytest.raises(ValueError):
        runner.compare_aggregates({"x": 1}, actual)


def test_fixed_roundoff_bound_and_whole_structure():
    reference = {"budgets": [0.05, 0.10, 0.20], "scope": "historical", "flags": [False, None], "count": 7814}
    assert runner.compare_aggregates(reference, json.loads(json.dumps(reference))) == 7
    assert runner.compare_aggregates(1.0, 1.0 + 1e-13) == 1
    with pytest.raises(ValueError):
        runner.compare_aggregates(1.0, 1.0 + 1e-8)
    with pytest.raises(ValueError):
        runner.compare_aggregates(reference, {**reference, "budgets": [0.1]})


def test_outcomes_stop_before_any_worker_if_score_not_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "bound_output", lambda *args: (tmp_path, {}, tmp_path, {}))
    def not_complete(*args):
        raise ValueError("Completed price-free companion score required")
    monkeypatch.setattr(runner, "completed_score", not_complete)
    monkeypatch.setattr(runner, "run_phase", lambda *a: pytest.fail("Worker reached without scoring"))
    with pytest.raises(ValueError, match="price-free"):
        runner.execute_phase(tmp_path, "outcomes")
    assert not list(tmp_path.iterdir())


def test_existing_outcome_attempt_is_never_resumed(tmp_path, monkeypatch):
    started = tmp_path / "historical-outcomes-started.json"
    started.write_text("{}")
    monkeypatch.setattr(runner, "bound_output", lambda *args: (tmp_path, {}, tmp_path, {}))
    with pytest.raises(ValueError, match="resumed or overwritten"):
        runner.execute_phase(tmp_path, "outcomes")
    assert started.read_text() == "{}"


@pytest.mark.parametrize("existing", ["historical-outcomes-receipt.json", "failed-historical-outcomes.json",
    "historical-prediction-freeze.json", "test-complete-predictions.json"])
def test_worker_cannot_restart_failed_or_partial_targets(existing, tmp_path):
    prepared = tmp_path / "historical-companion-preparation.json"
    prepared.write_text("{}")
    started = {"version": VERSION, "phase": "outcomes", "scope": SCOPE,
        "new_independent_confirmation": False, "preparation_sha256": runner.sha(prepared)}
    (tmp_path / "historical-outcomes-started.json").write_text(json.dumps(started))
    (tmp_path / existing).write_text("{}")
    with pytest.raises(ValueError):
        runner.require_worker_entry(tmp_path, "outcomes")


@pytest.mark.parametrize("phase", ["release", "reset-ledger", "resume", "new-test"])
def test_no_authority_or_retry_modes(phase, tmp_path):
    with pytest.raises(ValueError, match="Only companion"):
        runner.execute_phase(tmp_path, phase)
