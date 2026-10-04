"""Synthetic-only cached Turkey replay guard/concordance checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_turkey_cached_outcome_replay as r


class CachedReplayTests(unittest.TestCase):
    def test_exact_nested_values_and_counts(self):
        x = {"n": 7802, "ci": [-.7, .4], "flag": False, "optional": None}
        a = r.compare_aggregates(x, copy.deepcopy(x))
        self.assertEqual(a["finite_numeric_cells"], 3)
        self.assertEqual(a["other_leaf_cells"], 2)
        self.assertEqual(a["numerical_tolerance"], 0)

    def test_zero_tolerance_for_small_change(self):
        with self.assertRaisesRegex(ValueError, "value mismatch"):
            r.compare_aggregates({"gain": .1}, {"gain": .100000000000001})

    def test_boolean_is_not_numeric_count(self):
        with self.assertRaisesRegex(ValueError, "type mismatch"):
            r.compare_aggregates({"flag": False}, {"flag": 0})

    def test_missing_strategy_fails(self):
        with self.assertRaisesRegex(ValueError, "key mismatch"):
            r.compare_aggregates({"risk": 1, "benefit": 2}, {"benefit": 2})

    def test_missing_seed_fails(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            r.compare_aggregates([13, 42, 2026], [13, 42])

    def test_nonfinite_result_fails(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            r.compare_aggregates({"ci": float("inf")}, {"ci": float("inf")})

    def test_valid_new_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(r.output_path(root, "experiments/replays/new-v1"), root.resolve() / "experiments/replays/new-v1")

    def test_existing_output_preserved_before_metadata_access(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "experiments/replays/new-v1"
            p.mkdir(parents=True)
            with patch.object(r, "protected_files", side_effect=AssertionError("Premature inputs")):
                with self.assertRaisesRegex(ValueError, "Existing replay"):
                    r.execute(root, Path(sys.executable), "experiments/replays/new-v1")

    def test_collection_root_traversal_and_absolute_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("experiments/replays", "experiments/replays/../x", "/tmp/replay", "other/replay", "experiments//replays/x"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    r.output_path(Path(d), name)

    def test_symlink_output_parent_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "experiments").mkdir()
            (root / "target").mkdir()
            (root / "experiments/replays").symlink_to(root / "target")
            with self.assertRaisesRegex(ValueError, "Symlink"):
                r.output_path(root, "experiments/replays/new-v1")

    def test_protected_byte_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "input.json").write_text('{}')
            digest = r.digest(root / "input.json")
            r.verify_hashes(root, {"input.json": digest})
            (root / "input.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "Protected historical"):
                r.verify_hashes(root, {"input.json": digest})

    def test_receipt_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "receipt.json"
            r.save_new(p, {"old": True})
            with self.assertRaises(FileExistsError):
                r.save_new(p, {"new": True})
            self.assertEqual(json.loads(p.read_text()), {"old": True})

    def test_worker_guards_with_fake_study_module_only(self):
        # Fake modules in a new fixture; never imports real study/data/models.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            exp = root / "experiments"
            exp.mkdir()
            fake = '''
def release_source_phase(*a,**k): raise AssertionError("unguarded")
def _release_once_engine(*a,**k): raise AssertionError("unguarded")
def read_selected_prices(*a,**k): raise AssertionError("unguarded")
def verify_archive(*a,**k): raise AssertionError("unguarded")
'''
            (exp / "turkey_source_price_release.py").write_text(fake)
            (exp / "turkey_price_cells.py").write_text(fake)
            evaluator = '''
import zipfile
import turkey_source_price_release as p
import turkey_price_cells as c
def evaluate_source_phase(*a,**k):
    assert k["phase"] == "test"
    for f in (zipfile.ZipFile,p.release_source_phase,p._release_once_engine,p.read_selected_prices,p.verify_archive,c.read_selected_prices):
        try: f()
        except RuntimeError as e: assert "FORBIDDEN" in str(e)
        else: raise AssertionError("Access guard absent")
    return {"stage":"synthetic_only", "feature_records":2,"primary_comparisons":2}
'''
            (exp / "evaluate_turkey_released_phase.py").write_text(evaluator)
            child = subprocess.run([sys.executable, "-I", "-B", "-S", "-c", r.WORKER, str(root), str(root / "new")],
                                   check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(child.stdout)["stage"], "synthetic_only")
            self.assertFalse((root / "new").exists())


if __name__ == "__main__":
    unittest.main()
