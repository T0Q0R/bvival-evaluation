import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import freeze_turkey_execution_plan as runner
from build_turkey_price_free_manifest import digest, fold, write_csv


@pytest.fixture
def synthetic_plan(tmp_path, monkeypatch):
    root = Path(__file__).parents[2]
    cohort = tmp_path / "synthetic_cohort"
    cohort.mkdir()
    config = json.loads((root / "experiments/configs/turkey_price_free_cohort_2026-09-30.json").read_text())
    (cohort / "config_snapshot.json").write_text(json.dumps(config))
    (cohort / "cohort_freeze_audit.json").write_text('{"synthetic_only":true}\n')
    rows = []
    for i in range(20):
        group = digest(["synthetic group", i])
        rows.append({"record_key": str(i), "group_hash": group, "profile_hash": group,
                     "split": "train", "oof_fold": str(fold(group, config))})
    rows.append({"record_key": "test", "group_hash": "test-group", "profile_hash": "test-profile", "split": "test", "oof_fold": ""})
    write_csv(cohort / "split_manifest.csv", ["record_key", "group_hash", "profile_hash", "split", "oof_fold"], rows)
    # Bundle verification has its own independent tests and a real read-only run.
    monkeypatch.setattr(runner, "verify_bundle", lambda path: {"split_listing_counts": {"train": 20, "test": 1}, "labels_exported": False})
    original_run = runner.subprocess.run
    def synthetic_neural_probe(command, *args, **kwargs):
        if command[0] == "synthetic-python":
            return SimpleNamespace(stdout='{"synthetic_only":true}')
        return original_run(command, *args, **kwargs)
    monkeypatch.setattr(runner.subprocess, "run", synthetic_neural_probe)
    plan = json.loads((root / "experiments/configs/turkey_execution_plan_v1_2026-09-30.json").read_text())
    plan["cohort_directory"] = str(cohort)
    plan["cohort_audit_sha256"] = runner.sha256(cohort / "cohort_freeze_audit.json")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    protocol = root / "notes/design/bvival-turkey-execution-plan-v1-2026-09-30.md"
    return path, protocol, tmp_path / "freeze", root


def test_freeze_records_snapshots_nesting_capacities_without_authorizing_prices(synthetic_plan):
    config, protocol, output, root = synthetic_plan
    result = runner.freeze_plan(config, protocol, output, project_root=root, neural_python=Path("synthetic-python"))
    assert result["price_values_parsed"] == 0 and not result["models_trained"]
    assert not result["development_label_release_allowed_now"] and not result["training_execution_ready"]
    assert not result["external_registration"] and not result["official_GDFS_reproduction_completed"]
    assert result["planned_action_counts_by_partition"]["train"] == {"0.05": 1, "0.1": 2, "0.2": 4}
    assert len(result["nested_train_roles"]) == 5
    assert all(row["fit_listings"] + row["heldout_listings"] == 20 for row in result["nested_train_roles"])
    assert sum(result["base_fit_count_plan"].values()) == 258
    assert len(result["frozen_snapshot_sha256"]) == 14
    for name, expected in result["frozen_snapshot_sha256"].items():
        path = output / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        assert path.stat().st_mode & 0o777 == 0o400


def test_freeze_never_overwrites_existing_attempt(synthetic_plan):
    config, protocol, output, root = synthetic_plan
    output.mkdir()
    (output / "preserve.txt").write_text("old attempt")
    with pytest.raises(ValueError, match="must be empty"):
        runner.freeze_plan(config, protocol, output, project_root=root, neural_python=Path("synthetic-python"))
    assert (output / "preserve.txt").read_text() == "old attempt"


def test_wrong_cohort_binding_stops_before_output(synthetic_plan):
    config, protocol, output, root = synthetic_plan
    plan = json.loads(config.read_text())
    plan["cohort_audit_sha256"] = "0" * 64
    config.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="bind"):
        runner.freeze_plan(config, protocol, output, project_root=root, neural_python=Path("synthetic-python"))
    assert not output.exists()


def test_freeze_has_no_workbook_or_label_table_access(synthetic_plan, monkeypatch):
    config, protocol, output, root = synthetic_plan
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.suffix == ".xlsx" or path.name in {"prices.csv", "labels.csv"}:
            raise AssertionError("Unexpected label source access")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    runner.freeze_plan(config, protocol, output, project_root=root, neural_python=Path("synthetic-python"))


def test_cli_keeps_venv_interpreter_symlink_path_instead_of_system_target(synthetic_plan, monkeypatch, tmp_path):
    config, protocol, output, root = synthetic_plan
    system = tmp_path / "system-python"
    system.write_text("synthetic executable")
    venv = tmp_path / "venv" / "bin" / "python"
    venv.parent.mkdir(parents=True)
    venv.symlink_to(system)
    seen = {}
    def synthetic_freeze(*args, **kwargs):
        seen["interpreter"] = kwargs["neural_python"]
        return {"stage": "synthetic", "plan_sha256": "synthetic", "price_values_parsed": 0,
                "models_trained": False, "development_label_release_allowed_now": False,
                "planned_action_counts_by_partition": {}}
    monkeypatch.setattr(runner, "freeze_plan", synthetic_freeze)
    monkeypatch.setattr(runner.sys, "argv", ["freeze", "--config", str(config), "--protocol", str(protocol),
                                            "--output-dir", str(output), "--neural-python", str(venv)])
    runner.main()
    assert seen["interpreter"] == venv.absolute()
    assert seen["interpreter"] != system.resolve()
