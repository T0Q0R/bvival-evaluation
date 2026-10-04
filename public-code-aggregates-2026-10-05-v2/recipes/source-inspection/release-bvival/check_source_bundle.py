"""Verify a source-only local package, without importing study code or data."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

VERSION = "bvival-local-source-inspection-2026-10-02-v2"
MARKERS = (b"/Users/" + b"tqr/", b"BEGIN " + b"PRIVATE KEY", b"BEGIN " + b"OPENSSH PRIVATE KEY")


def safe_file(root, relative):
    if not isinstance(relative, str):
        raise ValueError("Unsafe package path")
    p = PurePosixPath(relative)
    if (str(p) != relative or p.is_absolute()
            or any(x in (".", "..") for x in p.parts) or not p.parts):
        raise ValueError("Unsafe package path")
    current = root
    for part in p.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlink package path")
    if not current.is_file():
        raise ValueError("Missing package file: " + relative)
    return current


def permitted_payload(relative):
    p = PurePosixPath(relative)
    if relative in ("README.md", "environment/resolved-main-macos-arm64.lock"):
        return True
    if p.suffix == ".py" and (
        len(p.parts) == 2 and p.parts[0] in ("experiments", "release-bvival")
        or len(p.parts) == 3 and p.parts[0] in ("experiments", "release-bvival") and p.parts[1] == "tests"
    ):
        return True
    return relative in {
        "experiments/configs/autoscout24_bvival_v2_full.json",
        "experiments/configs/jucars_bvival_subgroups.json",
        "experiments/configs/mucars_bvival_v5_subgroups.json",
        *{"inspection-plans/" + name + "-reconstructed-primary-plan-2026-10-01.json"
          for name in ("mucars", "jucars", "autoscout24")},
    }


def source_closure(root, entrypoints):
    """AST imports, including parseable embedded Python workers; never execute."""
    pending, sources, external, embedded, dynamic = list(entrypoints), {}, set(), [], []
    while pending:
        relative = pending.pop()
        if relative in sources:
            continue
        path = safe_file(root, relative)
        data = path.read_bytes()
        tree = ast.parse(data, filename=relative)
        sources[relative] = hashlib.sha256(data).hexdigest()
        trees = [(tree, None)]
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and "\n" in n.value:
                try:
                    worker = ast.parse(n.value)
                except SyntaxError:
                    continue
                if any(isinstance(x, (ast.Import, ast.ImportFrom)) for x in ast.walk(worker)):
                    trees.append((worker, n.lineno))
                    embedded.append({"file": relative, "string_line": n.lineno})
        for sub, string_line in trees:
            for n in ast.walk(sub):
                names = []
                if isinstance(n, ast.Import):
                    names = [a.name for a in n.names]
                elif isinstance(n, ast.ImportFrom):
                    if n.level:
                        raise ValueError("Relative import needs explicit review: " + relative)
                    names = [n.module] if n.module else []
                elif isinstance(n, ast.Call) and (
                    isinstance(n.func, ast.Name) and n.func.id == "__import__"
                    or isinstance(n.func, ast.Attribute) and n.func.attr == "import_module"
                ):
                    if n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                        names = [n.args[0].value]
                    else:
                        dynamic.append({"file": relative, "line": n.lineno, "string_line": string_line})
                for name in names:
                    module = name.split(".")[0]
                    if module in sys.stdlib_module_names:
                        continue
                    candidates = list(dict.fromkeys([
                        str(PurePosixPath(relative).parent / (module + ".py")),
                        "experiments/" + module + ".py",
                        "release-bvival/" + module + ".py",
                    ]))
                    found = [p for p in candidates if (root / p).is_file()]
                    if len(found) > 1:
                        raise ValueError("Ambiguous local module: " + module)
                    if found:
                        safe_file(root, found[0])
                        pending.append(found[0])
                    else:
                        # Packages require an explicit review, not a false external label.
                        if any((root / d / module).exists() for d in ("experiments", "release-bvival")):
                            raise ValueError("Unreviewed local package: " + module)
                        external.add(module)
    return {"local_source_sha256": dict(sorted(sources.items())),
            "external_import_roots": sorted(external),
            "embedded_workers": sorted(embedded, key=lambda x: (x["file"], x["string_line"])),
            "unresolved_dynamic_imports": sorted(dynamic, key=lambda x: (x["file"], x["line"])),
            "not_a_runtime_or_full_file_dependency_closure": True}


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads(safe_file(root, "manifest.json").read_text())
    if manifest.get("version") != VERSION:
        raise ValueError("Wrong package version")
    for flag in ("publicly_archived", "project_license_selected", "raw_or_row_level_data_included",
                 "full_empirical_pipeline_portable", "new_independent_confirmation"):
        if manifest.get(flag) is not False:
            raise ValueError("Unsupported capability: " + flag)
    if manifest.get("public_archive_or_reviewer_link") is not None:
        raise ValueError("Local package must not claim a public route")
    records = manifest["files"]
    paths = [r["file"] for r in records]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate file record")
    actual = set()
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError("Symlink package content")
        if p.is_file():
            actual.add(p.relative_to(root).as_posix())
    if actual != set(paths) | {"manifest.json"}:
        raise ValueError("Unexpected or missing package content")
    for r in records:
        p = safe_file(root, r["file"])
        if not permitted_payload(r["file"]):
            raise ValueError("Disallowed source-only payload path")
        data = p.read_bytes()
        if len(data) != r["bytes"] or hashlib.sha256(data).hexdigest() != r["sha256"]:
            raise ValueError("File hash/size mismatch: " + r["file"])
        if any(m in data for m in MARKERS):
            raise ValueError("Private content marker: " + r["file"])
        if p.suffix == ".py":
            ast.parse(data, filename=r["file"])
    closure = source_closure(root, manifest["closure_roots"])
    if closure != manifest["static_import_closure"]:
        raise ValueError("Static import closure changed")
    if closure["unresolved_dynamic_imports"]:
        raise ValueError("Unresolved dynamic import needs review")
    return {"status": "LOCAL_SOURCE_PACKAGE_VERIFIED_NOT_EMPIRICAL_RETRAINING",
            "payload_files": len(records), "python_sources_in_closure": len(closure["local_source_sha256"]),
            "external_import_roots": closure["external_import_roots"],
            "study_modules_imported": False, "study_rows_read": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.package_root), indent=2))


if __name__ == "__main__":
    main()
