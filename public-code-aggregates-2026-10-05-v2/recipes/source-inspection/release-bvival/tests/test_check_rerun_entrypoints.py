"""Synthetic-only tests of the static index checker; no study imports/data."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_rerun_entrypoints import check_manifest, inspect_interface, SCOPE


CLI = '''
raise RuntimeError("This module must never be executed")
def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source")
    group.add_argument("--development-labels")
    for flag in ("output-dir", "plan-freeze"):
        parser.add_argument("--" + flag, required=True)
if __name__ == "__main__":
    main()
'''
LIBRARY = '''
raise RuntimeError("Library must not be imported either")
def verify_score_freeze(directory):
    return directory
'''


class StaticIndexTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bvival-static-index-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "experiments").mkdir()
        self.manifest = {
            "schema_version": 1, "scope": SCOPE,
            "empirical_rerun_verified": False, "submission_ready": False,
            "study_modules_executed": False,
            "entries": [self.add_source("cli.py", CLI, "cli"),
                        self.add_source("library.py", LIBRARY, "library")],
        }

    def add_source(self, name, text, kind):
        (self.root / "experiments" / name).write_text(text, encoding="utf-8")
        entry = {
            "entrypoint": "experiments/" + name,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "kind": kind, "sources": ["MUCars", "JUCars", "AutoScout24", "Turkey"],
            "evidence_role": "historical_reproduction",
            "interface": inspect_interface(text),
        }
        if kind == "library":
            entry["library_function"] = "verify_score_freeze"
        return entry

    def check(self):
        return check_manifest(self.root, self.manifest)

    def test_positive_checks_without_executing_top_level_code(self):
        result = self.check()  # Both source bodies would raise if imported/run.
        self.assertEqual((result["indexed_source_files"], result["cli_files"], result["library_files"]), (2, 1, 1))
        self.assertIs(result["row_level_files_read"], False)
        self.assertIs(result["empirical_rerun_verified"], False)

    def test_literal_loop_flags_are_expanded(self):
        flags = self.manifest["entries"][0]["interface"]["flags"]
        self.assertIn({"flag": "--output-dir", "required": True}, flags)
        self.assertEqual(len(flags), 4)

    def test_required_exclusive_group_preserved(self):
        groups = self.manifest["entries"][0]["interface"]["mutually_exclusive_groups"]
        self.assertEqual(groups, [{"required": True, "flags": ["--development-labels", "--source"]}])

    def test_source_hash_drift_rejected(self):
        with (self.root / "experiments/cli.py").open("a") as handle:
            handle.write("\n# drift\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.check()

    def test_missing_documented_flag_rejected(self):
        self.manifest["entries"][0]["interface"]["flags"].pop()
        with self.assertRaisesRegex(ValueError, "declaration mismatch"):
            self.check()

    def test_extra_documented_flag_rejected(self):
        self.manifest["entries"][0]["interface"]["flags"].append({"flag": "--invented", "required": True})
        with self.assertRaisesRegex(ValueError, "declaration mismatch"):
            self.check()

    def test_missing_exclusive_group_rejected(self):
        self.manifest["entries"][0]["interface"]["mutually_exclusive_groups"] = []
        with self.assertRaisesRegex(ValueError, "declaration mismatch"):
            self.check()

    def test_wrong_library_role_rejected(self):
        self.manifest["entries"][1]["kind"] = "cli"
        with self.assertRaisesRegex(ValueError, "no supported CLI"):
            self.check()

    def test_invented_library_function_rejected(self):
        self.manifest["entries"][1]["library_function"] = "invented"
        with self.assertRaisesRegex(ValueError, "role mismatch"):
            self.check()

    def test_truthy_false_boundaries_rejected(self):
        for key in ("empirical_rerun_verified", "submission_ready", "study_modules_executed"):
            for bad in ("false", 0, None, True):
                with self.subTest(key=key, value=bad):
                    saved = self.manifest[key]
                    self.manifest[key] = bad
                    with self.assertRaisesRegex(ValueError, "literal false"):
                        self.check()
                    self.manifest[key] = saved

    def test_nonboolean_required_metadata_rejected(self):
        self.manifest["entries"][0]["interface"]["flags"][0]["required"] = 0
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            self.check()

    def test_missing_source_association_rejected(self):
        for entry in self.manifest["entries"]:
            entry["sources"] = ["MUCars"]
        with self.assertRaisesRegex(ValueError, "all four"):
            self.check()

    def test_unknown_source_rejected(self):
        self.manifest["entries"][0]["sources"].append("FifthSource")
        with self.assertRaisesRegex(ValueError, "Invalid source"):
            self.check()

    def test_duplicate_index_entry_rejected(self):
        self.manifest["entries"].append(copy.deepcopy(self.manifest["entries"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate source"):
            self.check()

    def test_missing_source_file_rejected(self):
        self.manifest["entries"][0]["entrypoint"] = "experiments/missing.py"
        with self.assertRaisesRegex(ValueError, "Missing source"):
            self.check()

    def test_paths_cannot_open_data_or_escape(self):
        for bad in ("../outside.py", "/tmp/outside.py", "experiments/../outside.py",
                    "experiments/data/raw.csv", "experiments\\cli.py", "experiments//cli.py"):
            with self.subTest(path=bad):
                self.manifest["entries"][0]["entrypoint"] = bad
                with self.assertRaises(ValueError):
                    self.check()

    def test_source_symlink_rejected(self):
        (self.root / "experiments/link.py").symlink_to(self.root / "experiments/cli.py")
        self.manifest["entries"][0]["entrypoint"] = "experiments/link.py"
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.check()

    def test_parent_directory_symlink_rejected(self):
        alias = self.root / "alias"
        alias.mkdir()
        (alias / "experiments").symlink_to(self.root / "experiments", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            check_manifest(alias, self.manifest)

    def test_dynamic_loop_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "Unsupported CLI loop"):
            inspect_interface('for flag in configured_flags:\n    parser.add_argument("--" + flag)')

    def test_dynamic_flag_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "Unsupported CLI flag"):
            inspect_interface('parser.add_argument(make_flag())')

    def test_duplicate_cli_flag_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate CLI"):
            inspect_interface('parser.add_argument("--x")\nparser.add_argument("--x")')

    def test_nonliteral_required_declaration_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nonliteral required"):
            inspect_interface('parser.add_argument("--x", required=configuration)')


if __name__ == "__main__":
    unittest.main()
