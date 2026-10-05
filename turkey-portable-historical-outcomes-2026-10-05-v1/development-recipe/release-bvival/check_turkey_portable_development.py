"""Stdlib-only integrity check; no workbook, study imports or training."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_source_bundle import MARKERS, safe_file, source_closure
from turkey_portable_development_contract import (
    COHORT_AUDIT_SHA, COHORT_REFERENCE, FALSE_CAPABILITIES, PLAN,
    PLAN_AUDIT_SHA, SCOPE, SOURCE_SHA, VERSION, closure_roots, sha,
)


def validate_manifest(manifest):
    if (manifest.get("version") != VERSION or manifest.get("scope") != SCOPE
            or manifest.get("historical_tests_already_opened") is not True
            or manifest.get("source_sha256") != SOURCE_SHA
            or manifest.get("original_reference_metadata_not_new_run_receipts") is not True):
        raise ValueError("Wrong reconstruction contract or evidence scope")
    for name in FALSE_CAPABILITIES:
        if manifest.get(name) is not False:
            raise ValueError("Unsupported capability: " + name)
    if manifest.get("closure_roots") != closure_roots():
        raise ValueError("Explicit script/test/subprocess roots differ")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Nonempty exact payload inventory required")
    names = [record["file"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate payload")
    return records


def permitted(relative):
    path = Path(relative)
    if path.suffix == ".py":
        return (len(path.parts) == 2 and path.parts[0] in {"experiments", "release-bvival"}
                or len(path.parts) == 3 and path.parts[:2] == ("experiments", "tests")
                or path.parent.as_posix() == PLAN)
    if relative in {"README.md", "pytest.ini", "LICENSE", "LICENSING_SCOPE.md",
                    "THIRD_PARTY_NOTICES.md", "environment/resolved-main-macos-arm64.lock",
                    "environment/resolved-neural-macos-arm64.lock", COHORT_REFERENCE,
                    "reference-metadata/prelabel_protocol_snapshot.md",
                    "notes/design/bvival-turkey-execution-plan-v1-2026-09-30.md",
                    "experiments/configs/turkey_execution_plan_v1_2026-09-30.json",
                    "experiments/configs/turkey_price_free_cohort_2026-09-30.json"}:
        return True
    return relative in {PLAN + "/" + n for n in (
        "execution_plan_freeze_audit.json", "plan_snapshot.json", "protocol_snapshot.md")}


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads(safe_file(root, "manifest.json").read_text())
    records = validate_manifest(manifest)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink payload")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != {record["file"] for record in records} | {"manifest.json"}:
        raise ValueError("Unexpected or missing payload")
    for record in records:
        path = safe_file(root, record["file"])
        if not permitted(record["file"]):
            raise ValueError("Disallowed payload: " + record["file"])
        data = path.read_bytes()
        if len(data) != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("Payload hash/size changed: " + record["file"])
        if any(marker in data for marker in MARKERS):
            raise ValueError("Private content marker")
        if path.suffix == ".py":
            ast.parse(data, filename=record["file"])
    if (sha(safe_file(root, COHORT_REFERENCE)) != COHORT_AUDIT_SHA
            or sha(safe_file(root, PLAN + "/execution_plan_freeze_audit.json")) != PLAN_AUDIT_SHA):
        raise ValueError("Original reference metadata changed")
    closure = source_closure(root, closure_roots())
    if closure != manifest["static_import_closure"] or closure["unresolved_dynamic_imports"]:
        raise ValueError("Import closure changed or unresolved")
    return {"stage": "PORTABLE_DEVELOPMENT_PACKAGE_INTEGRITY_ONLY_NOT_EMPIRICAL_RECONSTRUCTION",
            "payload_files": len(records), "source_rows_read": False,
            "study_modules_imported": False, "heldout_price_access_allowed": False,
            "full_study_empirical_pipeline_portable": False,
            "new_independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().package_root), indent=2))
