"""Check pinned source/CLI declarations without importing study modules.

This is a deliberately limited static index, not an argparse interpreter,
dependency check, training runner, artifact authenticator or rights audit.
Only Python source bytes beneath experiments/ are read by check_manifest().
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath


SOURCES = {"MUCars", "JUCars", "AutoScout24", "Turkey"}
SCOPE = "static_code_interface_only"


def _literal_bool(call, name, default=False):
    matches = [kw.value for kw in call.keywords if kw.arg == name]
    if not matches:
        return default
    value = matches[0]
    if not isinstance(value, ast.Constant) or type(value.value) is not bool:
        raise ValueError(f"Nonliteral {name} declaration at line {call.lineno}")
    return value.value


def inspect_interface(source):
    """Recognize literal flags and literal-string loops; fail on other forms.

    Captures declared per-flag required values and mutually exclusive groups.
    It does NOT prove runtime reachability, parser behavior, input schemas,
    defaults, or data-access safety. No AST expression is evaluated or compiled.
    """
    tree = ast.parse(source)
    flags = {}
    groups = []
    main_guard = False

    def flag_text(node, bindings):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in bindings:
            return bindings[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return flag_text(node.left, bindings) + flag_text(node.right, bindings)
        raise ValueError(f"Unsupported CLI flag expression: {ast.unparse(node)}")

    def walk(node, bindings, receivers):
        nonlocal main_guard
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            receivers = {}  # Do not conflate same-named local parser groups.
        if isinstance(node, ast.If):
            test = node.test
            if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__" and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Eq) and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value == "__main__"):
                main_guard = True
        if isinstance(node, ast.For):
            has_cli = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                          and n.func.attr == "add_argument" for n in ast.walk(node))
            if has_cli:
                if (not isinstance(node.target, ast.Name)
                        or not isinstance(node.iter, (ast.Tuple, ast.List))
                        or not all(isinstance(n, ast.Constant) and isinstance(n.value, str)
                                   for n in node.iter.elts) or node.orelse):
                    raise ValueError(f"Unsupported CLI loop at line {node.lineno}")
                for item in node.iter.elts:
                    local = {**bindings, node.target.id: item.value}
                    for child in node.body:
                        walk(child, local, receivers)
                return
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "add_mutually_exclusive_group":
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    raise ValueError("Unsupported group receiver")
                group = {"required": _literal_bool(call, "required"), "flags": []}
                groups.append(group)
                receivers[node.targets[0].id] = group
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if not node.args:
                raise ValueError("Empty argument declaration")
            names = [flag_text(arg, bindings) for arg in node.args]
            if any(not name.startswith("--") for name in names):
                raise ValueError("This index supports long CLI flags only")
            receiver = node.func.value
            group = receivers.get(receiver.id) if isinstance(receiver, ast.Name) else None
            for name in names:
                if name in flags:
                    raise ValueError(f"Duplicate CLI declaration: {name}")
                flags[name] = _literal_bool(node, "required")
                if group is not None:
                    group["flags"].append(name)
        for child in ast.iter_child_nodes(node):
            walk(child, bindings, receivers)

    walk(tree, {}, {})
    return {
        "main_guard_present": main_guard,
        "flags": [{"flag": name, "required": flags[name]} for name in sorted(flags)],
        "mutually_exclusive_groups": sorted(
            [{"required": g["required"], "flags": sorted(g["flags"])} for g in groups],
            key=lambda g: tuple(g["flags"])),
    }


def safe_source_path(root, relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Source path must be a POSIX relative string")
    parts = PurePosixPath(relative).parts
    if (len(parts) != 2 or parts[0] != "experiments" or parts[1] in {".", ".."}
            or PurePosixPath(relative).is_absolute() or not parts[1].endswith(".py")
            or relative != "/".join(parts)):
        raise ValueError("Only direct experiments/*.py source paths are allowed")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink source/component rejected: {relative}")
    if not current.is_file():
        raise ValueError(f"Missing source: {relative}")
    return current


def check_manifest(root, manifest):
    if manifest.get("schema_version") != 1 or manifest.get("scope") != SCOPE:
        raise ValueError("Unexpected static-index schema/scope")
    for name in ("empirical_rerun_verified", "submission_ready", "study_modules_executed"):
        if manifest.get(name) is not False:
            raise ValueError(f"Static-only boundary must be literal false: {name}")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Nonempty source index required")
    root = Path(root).resolve(strict=True)
    seen, covered = set(), set()
    cli_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Entry must be an object")
        relative = entry.get("entrypoint")
        path = safe_source_path(root, relative)
        if relative in seen:
            raise ValueError(f"Duplicate source: {relative}")
        seen.add(relative)
        sources = entry.get("sources")
        if (not isinstance(sources, list) or not sources
                or any(not isinstance(s, str) or s not in SOURCES for s in sources)
                or len(sources) != len(set(sources))):
            raise ValueError(f"Invalid source associations: {relative}")
        covered.update(sources)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            raise ValueError(f"Source hash mismatch: {relative}")
        actual = inspect_interface(data.decode("utf-8"))
        expected = entry.get("interface")
        if not isinstance(expected, dict):
            raise ValueError("Interface object required")
        # JSON considers 0 == False. Reject nonboolean metadata before comparison.
        if type(expected.get("main_guard_present")) is not bool:
            raise ValueError("Main-guard metadata must be boolean")
        for items in (expected.get("flags"), expected.get("mutually_exclusive_groups")):
            if not isinstance(items, list) or any(not isinstance(i, dict) or type(i.get("required")) is not bool for i in items):
                raise ValueError("Required metadata must be boolean")
        if actual != expected:
            raise ValueError(f"CLI/group declaration mismatch: {relative}")
        kind = entry.get("kind")
        if kind == "cli":
            if not actual["main_guard_present"] or not actual["flags"]:
                raise ValueError(f"Indexed CLI has no supported CLI declaration: {relative}")
            cli_count += 1
        elif kind == "library":
            functions = {n.name for n in ast.walk(ast.parse(data)) if isinstance(n, ast.FunctionDef)}
            if actual["main_guard_present"] or actual["flags"] or entry.get("library_function") not in functions:
                raise ValueError(f"Library/CLI role mismatch: {relative}")
        else:
            raise ValueError("Kind must be cli or library")
        if entry.get("evidence_role") not in {"historical_reproduction", "post_test_diagnostic"}:
            raise ValueError("Evidence role required; no fresh confirmation claimed")
    if covered != SOURCES:
        raise ValueError("The index must cover all four named sources")
    return {
        "scope": SCOPE, "indexed_source_files": len(seen), "cli_files": cli_count,
        "library_files": len(seen) - cli_count, "sources": sorted(covered),
        "source_hashes_and_declared_interfaces_match": True,
        "study_modules_executed": False, "row_level_files_read": False,
        "empirical_rerun_verified": False, "submission_ready": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check_manifest(args.project_root, json.loads(args.manifest.read_text(encoding="utf-8")))
    except (ValueError, OSError, SyntaxError) as error:
        parser.exit(1, f"Static entrypoint check failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
