"""Frozen-plan bindings and development-only input/release guards.

Local provenance/access discipline, not authorization signed by a third party.
No readiness receipt is issued here; full policy/neural integration is required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import zipfile
from pathlib import Path

import audit_turkey_feature_only as reader
from archive_turkey_candidate import verify_archive
from build_turkey_price_free_manifest import write_csv
from turkey_execution_contract import price_is_valid, target_validity_inventory, validate_plan
from turkey_price_cells import read_selected_prices
from verify_turkey_price_free_bundle import table, verify_bundle


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound_plan(plan_directory, *, project_root):
    audit_path = plan_directory / "execution_plan_freeze_audit.json"
    audit = json.loads(audit_path.read_text())
    expected_environment = audit["runtime_environment"]
    if platform.python_version() != expected_environment["python"] or any(
            importlib.metadata.version(package) != version for package, version in expected_environment["packages"].items()):
        raise ValueError("Runtime versions differ from the frozen main environment")
    required = set(audit["frozen_snapshot_sha256"])
    if {p.name for p in plan_directory.iterdir()} != required | {audit_path.name}:
        raise ValueError("Unexpected frozen-plan files")
    for name, expected in audit["frozen_snapshot_sha256"].items():
        if Path(name).name != name or sha256(plan_directory / name) != expected:
            raise ValueError("Frozen-plan snapshot hash mismatch")
    if (sha256(plan_directory / "plan_snapshot.json") != audit["plan_sha256"]
            or sha256(plan_directory / "protocol_snapshot.md") != audit["protocol_sha256"]):
        raise ValueError("Plan/protocol binding changed")
    plan = json.loads((plan_directory / "plan_snapshot.json").read_text())
    cohort_directory = project_root / plan["cohort_directory"]
    if sha256(cohort_directory / "cohort_freeze_audit.json") != plan["cohort_audit_sha256"]:
        raise ValueError("Different price-free cohort")
    verify_bundle(cohort_directory)
    cohort_config = json.loads((cohort_directory / "config_snapshot.json").read_text())
    validate_plan(plan, cohort_config)
    # Frozen legacy dependencies must still be the exact versions used by this runner.
    for name in ("audit_turkey_feature_only.py", "archive_turkey_candidate.py",
                 "build_turkey_price_free_manifest.py", "verify_turkey_price_free_bundle.py",
                 "turkey_information_boundary.py", "turkey_execution_contract.py", "turkey_valuation_components.py"):
        if sha256(project_root / "experiments" / name) != audit["frozen_snapshot_sha256"][name]:
            raise ValueError("Current dependency differs from frozen plan; preserve then re-freeze")
    return plan, cohort_config, cohort_directory, audit


def require_development_receipt(receipt, plan_audit, *, project_root):
    if (receipt.get("stage") != "TURKEY_FULL_DEVELOPMENT_INTEGRATION_READY"
            or receipt.get("development_label_release_allowed") is not True
            or receipt.get("price_release_scope") != ["train", "validation"]
            or receipt.get("policy_and_neural_integration_passed") is not True
            or receipt.get("nested_valuation_integration_passed") is not True
            or receipt.get("plan_sha256") != plan_audit["plan_sha256"]
            or receipt.get("cohort_audit_sha256") != plan_audit["cohort_audit_sha256"]):
        raise ValueError("Full development integration receipt required; calibration/test release refused")
    required = {"turkey_price_cells.py", "turkey_development_io.py", "train_turkey_nested_valuation.py"}
    hashes = receipt.get("source_sha256", {})
    if not required.issubset(hashes):
        raise ValueError("Receipt lacks current reader/loader/runner source bindings")
    for name, expected in hashes.items():
        if Path(name).name != name or sha256(project_root / "experiments" / name) != expected:
            raise ValueError("Readiness receipt source hash changed")


def export_development_prices(source, plan_directory, readiness_path, output_directory, *, project_root):
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError("Output directory must be empty; never overwrite a label release")
    plan, config, cohort, audit = bound_plan(plan_directory, project_root=project_root)
    receipt = json.loads(readiness_path.read_text())
    # Crucially, no workbook ZIP/schema/price access before this readiness gate.
    require_development_receipt(receipt, audit, project_root=project_root)
    verified = verify_archive(source)
    splits = table(cohort / "split_manifest.csv", ["record_key", "group_hash", "profile_hash", "split", "oof_fold"])
    locators = table(cohort / "source_locators.csv", ["record_key", "source_data_ordinal"])
    selected = {r["record_key"] for r in splits if r["split"] in {"train", "validation"}}
    ordinals = {int(r["source_data_ordinal"]): r["record_key"] for r in locators if r["record_key"] in selected}
    raw_audit = json.loads((cohort / "cohort_freeze_audit.json").read_text())
    output_directory.mkdir(parents=True, exist_ok=True)
    output_directory.chmod(0o700)
    result = {"stage": "AUTHORIZED_DEVELOPMENT_PRICE_ACCESS_NOT_COMPLETED",
              "source_verified": verified, "plan_sha256": audit["plan_sha256"],
              "cohort_audit_sha256": audit["cohort_audit_sha256"], "release_scope": ["train", "validation"],
              "readiness_receipt_sha256": sha256(readiness_path), "calibration_or_test_prices_decoded": False,
              "release_succeeded": False, "price_access_started": False,
              "local_guard_not_cryptographic_sealing": True}
    snapshots = {"price_cells_snapshot.py": Path(__file__).with_name("turkey_price_cells.py"),
                 "development_io_snapshot.py": Path(__file__), "readiness_receipt_snapshot.json": readiness_path}
    for name, path in snapshots.items():
        with (output_directory / name).open("xb") as stream:
            stream.write(path.read_bytes())
    result["snapshot_sha256"] = {name: sha256(output_directory / name) for name in snapshots}
    try:
        result["price_access_started"] = True
        with zipfile.ZipFile(source) as archive:
            header = reader.read_schema(archive, reader.sheet_path(archive))
            if set(header.values()) != set(config["expected_headers"]):
                raise ValueError("Source schema differs from frozen cohort")
            labels, access = read_selected_prices(archive, header, set(ordinals), expected_data_rows=raw_audit["raw_feature_rows"])
        result["cell_access"] = access
        validity = {ordinals[i]: row["price_valid"] for i, row in labels.items()}
        label_path = output_directory / "development_prices.csv"
        write_csv(label_path, ["record_key", "price", "price_valid"],
                  [{"record_key": ordinals[i], "price": row["price"], "price_valid": str(row["price_valid"]).lower()}
                   for i, row in sorted(labels.items(), key=lambda pair: ordinals[pair[0]])])
        result["labels_sha256"] = sha256(label_path)
        result["target_validity_inventory"] = target_validity_inventory(splits, validity, released_splits=["train", "validation"])
        result.update(stage="DEVELOPMENT_LABELS_RELEASED_AND_VALIDATED", release_succeeded=True)
    except Exception as error:
        result.update(stage="FAILED_DEVELOPMENT_PRICE_RELEASE", exception_class=type(error).__name__,
                      partial_cell_access_counts_unknown="cell_access" not in result)
        raise
    finally:
        with (output_directory / "development_label_release_audit.json").open("x") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        for file in output_directory.iterdir():
            file.chmod(0o400)
    return result


def load_development_dataset(cohort, label_directory, config):
    splits = table(cohort / "split_manifest.csv", ["record_key", "group_hash", "profile_hash", "split", "oof_fold"])
    development = [row for row in splits if row["split"] in {"train", "validation"}]
    keys = {row["record_key"] for row in development}
    audit = json.loads((label_directory / "development_label_release_audit.json").read_text())
    label_path = label_directory / "development_prices.csv"
    if (audit.get("release_scope") != ["train", "validation"] or audit.get("release_succeeded") is not True
            or audit.get("calibration_or_test_prices_decoded") is not False or sha256(label_path) != audit["labels_sha256"]
            or sha256(cohort / "cohort_freeze_audit.json") != audit["cohort_audit_sha256"]):
        raise ValueError("Matching successful development-only label release required")
    labels = table(label_path, ["record_key", "price", "price_valid"])
    if {r["record_key"] for r in labels} != keys:
        raise ValueError("Development labels must match the exact train/validation universe, no test keys")
    validity, prices = {}, {}
    for row in labels:
        if row["price_valid"] not in {"true", "false"}:
            raise ValueError("Invalid validity token")
        valid = row["price_valid"] == "true"
        if (valid and not price_is_valid(row["price"])) or (not valid and row["price"]):
            raise ValueError("Invalid price payload/validity agreement")
        validity[row["record_key"]] = valid
        if valid:
            prices[row["record_key"]] = float(row["price"])
    target_validity_inventory(splits, validity, released_splits=["train", "validation"])
    initial = table(cohort / "initial_features.csv", ["record_key", *config["initial_visible_fields"]])
    acquisition = table(cohort / "acquisition_fields.csv", ["record_key", *config["action_fields"]])
    if not keys.issubset({r["record_key"] for r in initial}) or not keys.issubset({r["record_key"] for r in acquisition}):
        raise ValueError("Missing development feature records")
    return {"rows": development, "prices": prices, "validity": validity,
            "initial": {r["record_key"]: {f: r[f] for f in config["initial_visible_fields"]} for r in initial if r["record_key"] in keys},
            "actions": {r["record_key"]: {f: r[f] for f in config["action_fields"]} for r in acquisition if r["record_key"] in keys}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan-freeze", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = export_development_prices(args.source, args.plan_freeze, args.readiness, args.output_dir,
                                       project_root=Path(__file__).resolve().parents[1])
    print(json.dumps({"stage": result["stage"], "cell_access": result["cell_access"],
                      "calibration_or_test_prices_decoded": False}))


if __name__ == "__main__":
    main()
