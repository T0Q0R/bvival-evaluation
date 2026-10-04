"""Controlled source entry + shared once-only fixture engine; no model selection.

Production entry requires a complete local readiness package, original source
coordinates and saved score freeze. Local discipline, not cryptographic sealing
or third-party permission. Fixture entry explicitly rejects the pinned source.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import zipfile
from collections import Counter
from pathlib import Path

import audit_turkey_feature_only as reader
from archive_turkey_candidate import EXPECTED_SHA256, verify_archive
from build_turkey_price_free_manifest import digest, write_csv
from run_turkey_development import RUNTIME_SOURCES, file_hashes, save_json, verify_files
from turkey_development_io import sha256
from turkey_execution_contract import price_is_valid, target_validity_inventory
from turkey_price_cells import read_selected_prices
from turkey_source_access_binding import (RAW_ROWS, SOURCE_VERSION, SPLIT_COUNTS, bind_original_source_coordinates,
                                          canonical_source_ledger, source_prerequisite_preflight)
from verify_turkey_price_free_bundle import table


RELEASE_RUNTIME = sorted(set(RUNTIME_SOURCES + [
    "freeze_turkey_heldout_scores.py", "turkey_price_free_scoring.py", "turkey_price_free_neural_scoring.py",
    "verify_turkey_heldout_score_freeze.py", "turkey_source_access_binding.py", "turkey_frozen_outcome_evaluation.py",
    "turkey_source_price_release.py", "run_turkey_source_release_fixture_check.py",
    "freeze_turkey_source_release_readiness.py", "run_turkey_once_release_check.py", "turkey_once_price_release.py",
    "tests/test_turkey_source_price_release.py", "tests/test_turkey_source_release_readiness.py",
    "tests/test_turkey_source_access_binding.py", "tests/test_turkey_price_cells.py",
    "tests/test_verify_turkey_heldout_score_freeze.py", "tests/test_turkey_frozen_outcome_evaluation.py",
]))
HASH = re.compile(r"[0-9a-f]{64}\Z")


def observed_environment(expected):
    result = {"python": platform.python_version(),
              "packages": {p: importlib.metadata.version(p) for p in expected["packages"]}}
    if "platform" in expected:
        result["platform"] = platform.platform()
    return result


def selected_coordinates(binding):
    fields = {"stage", "evidence_status", "phase", "source_sha256", "plan_sha256", "score_freeze_audit_sha256",
              "coordinate_manifest_sha256", "source_readiness_audit_sha256", "manifest", "frozen_partition_keys",
              "expected_headers", "expected_data_rows"}
    if (not isinstance(binding, dict) or set(binding) != fields
            or binding["stage"] != "BOUND_ONCE_PHASE_REQUIRES_VERIFIED_ENTRY"
            or binding["evidence_status"] not in {"synthetic_only", "authorized_heldout_only"}
            or binding["phase"] not in {"calibration", "test"}):
        raise ValueError("Exact bound phase/evidence roles required")
    for field in ("source_sha256", "plan_sha256", "score_freeze_audit_sha256", "coordinate_manifest_sha256", "source_readiness_audit_sha256"):
        if not isinstance(binding[field], str) or not HASH.fullmatch(binding[field]):
            raise ValueError("Complete source/plan/freeze/coordinate/readiness hashes required")
    source = binding["evidence_status"] == "authorized_heldout_only"
    if source != (binding["source_sha256"] == EXPECTED_SHA256):
        raise ValueError("Source vs fixture archive roles differ")
    n, headers, rows = binding["expected_data_rows"], binding["expected_headers"], binding["manifest"]
    if (type(n) is not int or n < 1 or not isinstance(headers, list) or not headers
            or any(not isinstance(h, str) or not h for h in headers)
            or len(headers) != len(set(headers)) or headers.count("fiyat") != 1
            or not isinstance(rows, list) or not rows or digest(rows) != binding["coordinate_manifest_sha256"]):
        raise ValueError("Exact source inventory and coordinate content required")
    keys, ordinals, selected = set(), set(), {}
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"record_key", "source_data_ordinal", "split"}
                or not isinstance(row["record_key"], str) or not row["record_key"] or row["record_key"] in keys
                or type(row["source_data_ordinal"]) is not int or not 1 <= row["source_data_ordinal"] <= n
                or row["source_data_ordinal"] in ordinals or row["split"] not in SPLIT_COUNTS):
            raise ValueError("Unique original keys/ordinals and frozen roles required")
        key, ordinal = row["record_key"], row["source_data_ordinal"]
        if source and key != digest([EXPECTED_SHA256, "source_data_ordinal", ordinal]):
            raise ValueError("Original source ordinal-to-key relation differs")
        keys.add(key)
        ordinals.add(ordinal)
        if row["split"] == binding["phase"]:
            selected[ordinal] = key
    frozen = binding["frozen_partition_keys"]
    if (not isinstance(frozen, list) or not frozen or any(not isinstance(k, str) for k in frozen)
            or frozen != sorted(set(frozen)) or set(selected.values()) != set(frozen)):
        raise ValueError("Complete frozen phase required, not action-selected subset")
    if source and (n != RAW_ROWS or dict(Counter(r["split"] for r in rows)) != SPLIT_COUNTS):
        raise ValueError("Original complete source cohort required")
    return selected


def _safe_ledger(ledger):
    for path in (ledger, *ledger.parents):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError("Ledger cannot traverse symlink/file ancestors")


def _reservation(binding, output):
    return {"stage": "RESERVED_BEFORE_PRICE_PARSER", "source_sha256": binding["source_sha256"],
            "phase": binding["phase"], "evidence_status": binding["evidence_status"],
            "binding_sha256": digest(binding), "output_directory": str(output.absolute()),
            "local_guard_not_cryptographic_sealing": True}


def verify_release_package(output, binding, ledger):
    """Verify phase/ledger/hash identity WITHOUT decoding cached target values."""
    selected_coordinates(binding)
    _safe_ledger(ledger)
    path = output / "once_phase_release_audit.json"
    audit = json.loads(path.read_text())
    if (audit.get("stage") != "BOUND_PHASE_PRICES_RELEASED_ONCE"
            or audit.get("release_succeeded") is not True or audit.get("price_access_started") is not True
            or audit.get("evidence_status") != binding["evidence_status"]
            or audit.get("phase") != binding["phase"]
            or audit.get("binding_sha256") != digest(binding)
            or audit.get("source_release_ready") is not (binding["evidence_status"] == "authorized_heldout_only")):
        raise ValueError("Matching completed phase release required")
    verify_files(output, audit["artifact_sha256"], ignored=[path.name])
    if json.loads((output / "phase_binding.json").read_text()) != binding:
        raise ValueError("Saved phase binding differs")
    for name in ("turkey_source_price_release.py", "turkey_price_cells.py", "audit_turkey_feature_only.py"):
        if (output / (name + ".snapshot")).read_bytes() != Path(__file__).with_name(name).read_bytes():
            raise ValueError("Saved phase reader/controller code differs")
    claim = ledger / (binding["source_sha256"] + "." + binding["phase"])
    if (claim.is_symlink() or Path(audit["ledger_claim_directory"]).absolute() != claim.absolute()
            or sha256(claim / "reservation.json") != audit["reservation_sha256"]
            or json.loads((claim / "reservation.json").read_text()) != _reservation(binding, output)
            or sha256(claim / "terminal.json") != audit["terminal_sha256"]
            or json.loads((claim / "terminal.json").read_text()) != {
                "stage": audit["stage"], "binding_sha256": digest(binding), "price_access_started": True,
                "release_succeeded": True, "no_silent_reopen": True}):
        raise ValueError("Canonical phase reservation/terminal identity differs")
    return audit


def load_released_labels(output, binding, ledger):
    audit = verify_release_package(output, binding, ledger)
    selected = selected_coordinates(binding)
    rows = table(output / "phase_prices.csv", ["record_key", "price", "price_valid"])
    if {r["record_key"] for r in rows} != set(selected.values()):
        raise ValueError("Released targets must cover complete phase")
    labels = {}
    for row in rows:
        token = row["price_valid"]
        if (token not in {"true", "false"} or (token == "true" and not price_is_valid(row["price"]))
                or (token == "false" and row["price"] != "")):
            raise ValueError("Saved target value/validity differ")
        labels[row["record_key"]] = {"price": row["price"], "price_valid": token == "true"}
    inventory = target_validity_inventory(binding["manifest"], {k: v["price_valid"] for k, v in labels.items()},
                                          released_splits=[binding["phase"]])
    if inventory != audit["target_validity_inventory"]:
        raise ValueError("Saved target inventory differs")
    return labels


def _verify_phase_order(binding, ledger, prior_calibration):
    if binding["phase"] == "calibration":
        if prior_calibration is not None:
            raise ValueError("Calibration cannot take a prior calibration release")
        return None
    if prior_calibration is None:
        raise ValueError("Successful bound calibration phase required before test")
    prior = {**binding, "phase": "calibration", "frozen_partition_keys": sorted(
        r["record_key"] for r in binding["manifest"] if r["split"] == "calibration")}
    verify_release_package(prior_calibration, prior, ledger)
    return sha256(prior_calibration / "once_phase_release_audit.json")


def _release_once_engine(source, binding, ledger, output, *, prior_calibration=None):
    """Shared tested engine; production callers MUST use release_source_phase."""
    if output.exists():
        raise ValueError("New nonexistent release output required")
    selected = selected_coordinates(binding)
    if source.is_symlink() or sha256(source) != binding["source_sha256"]:
        raise ValueError("Bound archive bytes/path differ; stop before reservation")
    _safe_ledger(ledger)
    prior_hash = _verify_phase_order(binding, ledger, prior_calibration)
    ledger.mkdir(parents=True, exist_ok=True, mode=0o700)
    ledger.chmod(0o700)
    claim = ledger / (binding["source_sha256"] + "." + binding["phase"])
    try:
        claim.mkdir(mode=0o700)
    except FileExistsError:
        raise ValueError("Source/phase already reserved, including failed or crashed attempts") from None
    save_json(claim / "reservation.json", _reservation(binding, output))
    (claim / "reservation.json").chmod(0o400)
    audit = {"stage": "RESERVED_PHASE_NOT_COMPLETED", "evidence_status": binding["evidence_status"],
             "phase": binding["phase"], "binding_sha256": digest(binding),
             "ledger_claim_directory": str(claim.absolute()), "reservation_sha256": sha256(claim / "reservation.json"),
             "prior_calibration_release_audit_sha256": prior_hash,
             "price_access_started": False, "release_succeeded": False,
             "source_release_ready": binding["evidence_status"] == "authorized_heldout_only",
             "local_guard_not_cryptographic_sealing": True, "models_refitted": 0, "policies_reselected": False}
    try:
        output.mkdir(parents=True, mode=0o700)
        save_json(output / "phase_binding.json", binding)
        for name in ("turkey_source_price_release.py", "turkey_price_cells.py", "audit_turkey_feature_only.py"):
            with (output / (name + ".snapshot")).open("xb") as stream:
                stream.write(Path(__file__).with_name(name).read_bytes())
        audit["price_access_started"] = True
        with zipfile.ZipFile(source) as archive:
            header = reader.read_schema(archive, reader.sheet_path(archive))
            if len(header) != len(binding["expected_headers"]) or set(header.values()) != set(binding["expected_headers"]):
                raise ValueError("Bound workbook header schema differs")
            labels, access = read_selected_prices(archive, header, set(selected), expected_data_rows=binding["expected_data_rows"])
        audit["cell_access"] = access
        validity = {selected[i]: v["price_valid"] for i, v in labels.items()}
        audit["target_validity_inventory"] = target_validity_inventory(binding["manifest"], validity,
                                                                       released_splits=[binding["phase"]])
        write_csv(output / "phase_prices.csv", ["record_key", "price", "price_valid"], [
            {"record_key": selected[i], "price": v["price"], "price_valid": str(v["price_valid"]).lower()}
            for i, v in sorted(labels.items(), key=lambda pair: selected[pair[0]])])
        audit.update(stage="BOUND_PHASE_PRICES_RELEASED_ONCE", release_succeeded=True)
    except Exception as error:
        audit.update(stage="FAILED_BOUND_ONCE_PHASE", exception_class=type(error).__name__,
                     partial_cell_access_counts_unknown="cell_access" not in audit)
        raise
    finally:
        save_json(claim / "terminal.json", {"stage": audit["stage"], "binding_sha256": digest(binding),
                  "price_access_started": audit["price_access_started"], "release_succeeded": audit["release_succeeded"],
                  "no_silent_reopen": True})
        (claim / "terminal.json").chmod(0o400)
        audit["terminal_sha256"] = sha256(claim / "terminal.json")
        if output.is_dir():
            audit["artifact_sha256"] = file_hashes(output)
            save_json(output / "once_phase_release_audit.json", audit)
            for path in output.iterdir():
                if path.is_file():
                    path.chmod(0o400)
    return audit


def release_fixture_phase(source, binding, ledger, output, *, prior_calibration=None):
    if binding.get("evidence_status") != "synthetic_only" or binding.get("source_sha256") == EXPECTED_SHA256:
        raise ValueError("Fixture entry refuses actual source role/archive")
    return _release_once_engine(source, binding, ledger, output, prior_calibration=prior_calibration)


def verify_source_readiness(directory, prerequisites, *, project_root):
    path = directory / "source_release_readiness_audit.json"
    audit = json.loads(path.read_text())
    if (audit.get("stage") != "LOCAL_SOURCE_ONCE_RELEASE_READY_NO_PRICES_OPENED"
            or audit.get("source_release_ready") is not True or type(audit.get("source_prices_parsed_here")) is not int
            or audit.get("source_prices_parsed_here") != 0 or audit.get("verified_opaque_archive") != SOURCE_VERSION
            or audit.get("prerequisites") != prerequisites or audit.get("external_registration") is not False
            or audit.get("main_tests", {}).get("exit_code") != 0
            or audit.get("neural_tests", {}).get("exit_code") != 0
            or any(type(audit.get(kind, {}).get("passed")) is not int or audit[kind]["passed"] < 1
                   for kind in ("main_tests", "neural_tests"))
            or audit.get("calibration_release_allowed_after_binding_checks") is not True
            or audit.get("test_release_requires_successful_bound_calibration_release") is not True
            or audit.get("source_ledger_reserved_here") is not False
            or audit.get("frozen_models_or_policies_changed") is not False
            or audit.get("shared_engine_fixture_integration_passed") is not True
            or set(audit.get("runtime_sha256", {})) != set(RELEASE_RUNTIME)):
        raise ValueError("Complete original local source readiness required")
    verify_files(directory, audit["artifact_sha256"], ignored=[path.name])
    for kind in ("main_tests", "neural_tests"):
        if sha256(directory / (kind + ".txt")) != audit[kind]["report_sha256"]:
            raise ValueError("Source readiness test report binding differs")
    for name in RELEASE_RUNTIME:
        if (sha256(project_root / "experiments" / name) != audit["runtime_sha256"][name]
                or sha256(directory / "runtime_snapshots" / name) != audit["runtime_sha256"][name]):
            raise ValueError("Source release runtime/test snapshot differs")
    if observed_environment(audit["runtime_environment"]) != audit["runtime_environment"]:
        raise ValueError("Source release runtime environment differs")
    fixture = json.loads((directory / "fixture_integration/shared_release_fixture_audit.json").read_text())
    if (fixture.get("stage") != "SHARED_ONCE_ENGINE_FIXTURE_EVALUATION_PASSED_NOT_SOURCE_RESULT"
            or fixture.get("actual_source_prices_read") is not False
            or fixture.get("both_phase_reopens_refused") is not True
            or fixture.get("partition_checks") != {"calibration": {"records": 17, "primary_comparisons": 0},
                                                   "test": {"records": 23, "primary_comparisons": 2}}):
        raise ValueError("Complete two-phase shared-engine fixture integration required")
    return audit


def release_source_phase(source, freeze, development, plan, readiness, output, *, phase, prior_calibration=None):
    """Only production entry: caller cannot override root or canonical ledger."""
    if phase not in {"calibration", "test"} or output.exists():
        raise ValueError("Known single phase and new nonexistent output required")
    root = Path(__file__).absolute().parents[1]
    # Missing/wrong readiness stops before any workbook or price parser.
    if not (readiness / "source_release_readiness_audit.json").is_file():
        raise ValueError("Complete source-release readiness required before price access")
    prerequisites = source_prerequisite_preflight(freeze, development, plan, project_root=root)
    verify_source_readiness(readiness, prerequisites, project_root=root)
    verify_archive(source)  # opaque byte hashes only, no XLSX/schema payload parsing
    coordinates = bind_original_source_coordinates(plan, project_root=root)
    config = json.loads((development / "cohort_config.json").read_text())
    binding = {"stage": "BOUND_ONCE_PHASE_REQUIRES_VERIFIED_ENTRY", "evidence_status": "authorized_heldout_only",
               "phase": phase, "source_sha256": EXPECTED_SHA256, "plan_sha256": coordinates["plan_sha256"],
               "score_freeze_audit_sha256": prerequisites["score_freeze_audit_sha256"],
               "coordinate_manifest_sha256": coordinates["coordinate_manifest_sha256"],
               "source_readiness_audit_sha256": sha256(readiness / "source_release_readiness_audit.json"),
               "manifest": coordinates["manifest"], "expected_headers": config["expected_headers"],
               "expected_data_rows": RAW_ROWS,
               "frozen_partition_keys": sorted(r["record_key"] for r in coordinates["manifest"] if r["split"] == phase)}
    return _release_once_engine(source, binding, canonical_source_ledger(), output, prior_calibration=prior_calibration)


def main():
    parser = argparse.ArgumentParser(description="Read one bound source phase once, with local readiness")
    for flag in ("source", "score-freeze", "development-run", "plan-freeze", "readiness", "output-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--phase", choices=["calibration", "test"], required=True)
    parser.add_argument("--prior-calibration-release", type=Path)
    args = parser.parse_args()
    result = release_source_phase(args.source, args.score_freeze, args.development_run, args.plan_freeze,
                                  args.readiness, args.output_dir, phase=args.phase,
                                  prior_calibration=args.prior_calibration_release)
    print(json.dumps({k: result[k] for k in ("stage", "phase", "release_succeeded", "cell_access")}))


if __name__ == "__main__":
    main()
