"""Only synthetic package/source fixtures; no study inputs or fitting."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import check_source_bundle as c


class SourceBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "experiments").mkdir()
        (self.root / "experiments/toy.py").write_text("import os\nVALUE = 1\n")
        self.roots = ["experiments/toy.py"]

    def fixture(self, **changes):
        data = (self.root / self.roots[0]).read_bytes()
        manifest = {"version": c.VERSION, "closure_roots": self.roots,
                    "static_import_closure": c.source_closure(self.root, self.roots),
                    "public_archive_or_reviewer_link": None,
                    **{k: False for k in ("publicly_archived", "project_license_selected",
                                         "raw_or_row_level_data_included", "full_empirical_pipeline_portable",
                                         "new_independent_confirmation")},
                    "files": [{"file": self.roots[0], "bytes": len(data),
                               "sha256": hashlib.sha256(data).hexdigest()}]}
        manifest.update(changes)
        (self.root / "manifest.json").write_text(json.dumps(manifest))

    def test_valid_source_fixture(self):
        self.fixture()
        self.assertEqual(c.verify(self.root)["payload_files"], 1)

    def test_code_change_rejected(self):
        self.fixture()
        (self.root / self.roots[0]).write_text("VALUE = 2\n")
        with self.assertRaisesRegex(ValueError, "hash/size"):
            c.verify(self.root)

    def test_unexpected_row_file_rejected(self):
        self.fixture()
        (self.root / "rows.csv").write_text("synthetic_only\n")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            c.verify(self.root)

    def test_no_public_release_claim(self):
        self.fixture(publicly_archived=True)
        with self.assertRaisesRegex(ValueError, "Unsupported capability"):
            c.verify(self.root)

    def test_duplicate_record_rejected(self):
        self.fixture()
        p = self.root / "manifest.json"
        value = json.loads(p.read_text())
        value["files"] *= 2
        p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            c.verify(self.root)

    def test_traversal_and_absolute_paths_rejected(self):
        for p in ("../outside.py", "/absolute.py", "experiments//toy.py", "experiments/./toy.py"):
            with self.subTest(path=p), self.assertRaises(ValueError):
                c.safe_file(self.root, p)

    def test_symlink_rejected(self):
        (self.root / "experiments/link.py").symlink_to(self.root / "experiments/toy.py")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            c.source_closure(self.root, ["experiments/link.py"])

    def test_embedded_worker_imports_closed_without_execution(self):
        (self.root / "experiments/helper.py").write_text("raise RuntimeError('never import')\n")
        (self.root / self.roots[0]).write_text("WORKER = '''\nimport helper\nraise RuntimeError('never execute')\n'''\n")
        result = c.source_closure(self.root, self.roots)
        self.assertEqual(len(result["local_source_sha256"]), 2)
        self.assertEqual(len(result["embedded_workers"]), 1)

    def test_unresolved_dynamic_import_recorded(self):
        (self.root / self.roots[0]).write_text("module = 'unknown'\nx = __import__(module)\n")
        self.assertEqual(len(c.source_closure(self.root, self.roots)["unresolved_dynamic_imports"]), 1)

    def test_literal_stdlib_dynamic_import_resolved(self):
        (self.root / self.roots[0]).write_text("x = __import__('os')\n")
        self.assertEqual(c.source_closure(self.root, self.roots)["external_import_roots"], [])

    def test_arbitrary_json_is_not_allowed_as_source_data(self):
        self.assertFalse(c.permitted_payload("experiments/data/records.json"))
        self.assertFalse(c.permitted_payload("inspection-plans/records.json"))
        self.assertTrue(c.permitted_payload("inspection-plans/mucars-reconstructed-primary-plan-2026-10-01.json"))

    def test_nonstring_path_rejected(self):
        with self.assertRaises(ValueError):
            c.safe_file(self.root, None)

    def test_private_marker_rejected_even_with_matching_hash(self):
        (self.root / self.roots[0]).write_text("x = 'BEGIN " + "PRIVATE KEY'\n")
        self.fixture()
        with self.assertRaisesRegex(ValueError, "Private content"):
            c.verify(self.root)


if __name__ == "__main__":
    unittest.main()
