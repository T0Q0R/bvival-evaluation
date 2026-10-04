"""Synthetic parser guards for cached validation field selection."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import replay_autoscout24_cached_field_selection as r


class CachedSelectionTests(unittest.TestCase):
    def test_existing_output_stops_before_cached_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "experiments/replays/existing").mkdir(parents=True)
            with patch.object(r, "bindings", side_effect=AssertionError("Premature inputs")):
                with self.assertRaisesRegex(ValueError, "Existing replay"):
                    r.execute(root, Path(sys.executable), "experiments/replays/existing")

    def test_modified_recovery_receipt_stops(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            p = root / r.REPORT
            p.parent.mkdir(parents=True)
            p.write_text('{}')
            with self.assertRaisesRegex(ValueError, "recovery report changed"):
                r.bindings(root)

    def test_parser_reads_only_two_bound_paths(self):
        # Fake pandas + fake selector, stdlib only; no real study modules/data.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "pandas.py").write_text('def read_csv(path,*a,**k): return str(path)\n')
            selector = root / "selector.py"
            selector.write_text('''
import json,sys
from pathlib import Path
import pandas as pd
a=Path(sys.argv[sys.argv.index("--development-actions")+1])
r=Path(sys.argv[sys.argv.index("--development-risk-scores")+1])
assert pd.read_csv(a)==str(a)
assert pd.read_csv(r)==str(r)
for path in (a.parent/"test_labels.csv",a.parent/"raw_source.csv"):
    try: pd.read_csv(path)
    except RuntimeError as e: assert "ONLY_BOUND" in str(e)
    else: raise AssertionError("Unbounded source read")
print(json.dumps({"synthetic_parser_guards":True}))
''')
            launcher = 'import sys; sys.path.insert(0,sys.argv.pop(1)); exec(' + repr(r.WORKER) + ')'
            child = subprocess.run([sys.executable, "-I", "-B", "-S", "-c", launcher, str(root), str(selector),
                                    str(root / "actions.csv"), str(root / "risk.csv"), str(root / "out.json")],
                                   check=True, capture_output=True, text=True)
            self.assertTrue(json.loads(child.stdout)["synthetic_parser_guards"])
            self.assertFalse((root / "out.json").exists())

    def test_receipts_never_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "receipt.json"
            r.save_new(p, {"original": True})
            with self.assertRaises(FileExistsError):
                r.save_new(p, {"replacement": True})
            self.assertEqual(json.loads(p.read_text()), {"original": True})


if __name__ == "__main__":
    unittest.main()
