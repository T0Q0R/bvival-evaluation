"""Check an exact code/metadata/aggregate package without reading source rows."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_source_bundle import MARKERS, safe_file, source_closure
from historical_reconstruction_contract import VERSION, LOCK, LOCK_SHA, closure_entrypoints, bindings, sha256

FALSE_CAPABILITIES = ("publicly_archived", "project_license_selected", "raw_or_row_level_data_included",
                      "full_empirical_pipeline_portable", "new_independent_confirmation",
                      "fresh_external_install_or_reconstruction_verified")


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads(safe_file(root, "manifest.json").read_text())
    if manifest.get("version") != VERSION or manifest.get("public_archive_or_reviewer_link") is not None:
        raise ValueError("Version/access boundary differs")
    if any(manifest.get(key) is not False for key in FALSE_CAPABILITIES):
        raise ValueError("Unsupported capability claim")
    if manifest.get("historical_tests_already_opened") is not True:
        raise ValueError("Opened-test disclosure required")
    bound = bindings(root)
    closure = source_closure(root, closure_entrypoints(bound))
    if closure != manifest.get("static_import_closure") or closure["unresolved_dynamic_imports"]:
        raise ValueError("Code import closure differs")
    if set(closure["external_import_roots"]) - {"catboost", "numpy", "pandas", "scipy"}:
        raise ValueError("Unreviewed dependency")
    if bound["sources"] != manifest.get("sources"):
        raise ValueError("Input-to-result map differs")
    expected = set(closure["local_source_sha256"]) | set(bound["metadata"]) | {LOCK, "README.md"}
    recorded = {}
    for record in manifest["files"]:
        relative = record["file"]
        if relative in recorded:
            raise ValueError("Duplicate payload record")
        recorded[relative] = record
    if set(recorded) != expected:
        raise ValueError("Exact payload allowlist differs")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink payload")
        if path.is_file():
            actual.add(str(path.relative_to(root)))
    if actual != expected | {"manifest.json"}:
        raise ValueError("Unexpected or missing file")
    for relative, record in recorded.items():
        path = safe_file(root, relative)
        data = path.read_bytes()
        if len(data) != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("Payload hash/size differs")
        if any(marker in data for marker in MARKERS):
            raise ValueError("Private content marker")
        if relative.endswith(".py"):
            ast.parse(data)
    if sha256(safe_file(root, LOCK)) != LOCK_SHA:
        raise ValueError("Runtime lock drift")
    return {"stage": "HISTORICAL_PRIMARY_PACKAGE_VERIFIED_NOT_EMPIRICAL_RECONSTRUCTION",
            "payload_files": len(recorded), "historical_sources": list(bound["sources"]),
            "source_rows_read": False, "study_fitting_called": False, "new_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.package_root), indent=2))
