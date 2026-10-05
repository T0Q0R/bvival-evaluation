"""Stdlib-only companion integrity; never import scientific modules or rows."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_source_bundle import MARKERS, safe_file, source_closure
from check_turkey_portable_development import verify as verify_development
from turkey_portable_development_contract import SOURCE_SHA, sha
from turkey_portable_outcome_contract import (
    DEVELOPMENT_MANIFEST_SHA, DEVELOPMENT_PACKAGE, FALSE_CAPABILITIES,
    PRIOR_RELEASES, REFERENCE_FILES, SCIENTIFIC_SOURCE_SHA, SCOPE, VERSION, closure_roots,
)


def permitted(relative):
    path = Path(relative)
    if path.suffix == ".py":
        return (len(path.parts) == 2 and path.parts[0] in {"experiments", "release-bvival"}
                or len(path.parts) == 3 and path.parts[0] in {"experiments", "release-bvival"}
                and path.parts[1] == "tests")
    return relative in {"README.md", "LICENSE", "LICENSING_SCOPE.md", "THIRD_PARTY_NOTICES.md",
                        "reference-metadata/already-opened-history.json",
                        "aggregates/calibration-fixed-evaluation.json", "aggregates/test-fixed-evaluation.json"}


def validate_manifest(value):
    if (value.get("version") != VERSION or value.get("scope") != SCOPE
            or value.get("source_sha256") != SOURCE_SHA
            or value.get("historical_tests_already_opened") is not True
            or value.get("development_manifest_sha256") != DEVELOPMENT_MANIFEST_SHA
            or value.get("closure_roots") != closure_roots()):
        raise ValueError("Wrong companion, source or historical scope")
    for name in FALSE_CAPABILITIES:
        if value.get(name) is not False:
            raise ValueError("Unsupported companion capability: " + name)
    files = value.get("files")
    if not isinstance(files, list) or not files or len({r["file"] for r in files}) != len(files):
        raise ValueError("Nonempty unique companion inventory required")
    return files


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads(safe_file(root, "manifest.json").read_text())
    files = validate_manifest(manifest)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink companion payload")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != {r["file"] for r in files} | {"manifest.json"}:
        raise ValueError("Unexpected or missing companion file")
    for record in files:
        name = record["file"]
        path = safe_file(root, name)
        if not (name.startswith(DEVELOPMENT_PACKAGE + "/") or permitted(name)):
            raise ValueError("Disallowed companion payload: " + name)
        data = path.read_bytes()
        if len(data) != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("Companion hash/size mismatch: " + name)
        if any(marker in data for marker in MARKERS):
            raise ValueError("Private content marker")
        if path.suffix == ".py":
            ast.parse(data, filename=name)
    development = root / DEVELOPMENT_PACKAGE
    if sha(safe_file(development, "manifest.json")) != DEVELOPMENT_MANIFEST_SHA:
        raise ValueError("Frozen development package changed")
    verify_development(development)
    closure = source_closure(root, closure_roots())
    if closure != manifest.get("static_import_closure") or closure["unresolved_dynamic_imports"]:
        raise ValueError("Companion closure changed or unresolved")
    for name, expected in SCIENTIFIC_SOURCE_SHA.items():
        if sha(safe_file(root, name)) != expected:
            raise ValueError("Original scientific helper changed")
    for phase, (_, expected) in REFERENCE_FILES.items():
        if sha(safe_file(root, "aggregates/" + phase + "-fixed-evaluation.json")) != expected:
            raise ValueError("Historical aggregate changed")
    history = json.loads(safe_file(root, "reference-metadata/already-opened-history.json").read_text())
    expected_history = {"source_sha256": SOURCE_SHA, "historical_tests_already_opened": True,
        "scope": SCOPE, "original_ledger_or_price_cache_not_required": True,
        "phases": {phase: {"historical_release_audit_sha256": PRIOR_RELEASES[phase][1],
            "feature_records": count, "aggregate_sha256": REFERENCE_FILES[phase][1]}
            for phase, count in (("calibration", 7814), ("test", 7802))}}
    if history != expected_history:
        raise ValueError("Known already-opened history binding changed")
    return {"stage": "HISTORICAL_COMPANION_INTEGRITY_ONLY_NOT_EMPIRICAL_RECONSTRUCTION",
            "payload_files": len(files), "source_rows_read": False,
            "historical_tests_already_opened": True, "new_independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().package_root), indent=2))
