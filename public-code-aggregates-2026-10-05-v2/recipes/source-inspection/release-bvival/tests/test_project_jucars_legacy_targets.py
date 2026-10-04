"""Synthetic CSVs only; never use study targets or outcome labels."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import project_jucars_legacy_targets as adapter


class LegacyProjectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="jucars-projection-toy-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.header = ["listing_id", "action_id", adapter.REMOVED, "value", "text"]
        self.old = [c for c in self.header if c != adapter.REMOVED]
        self.rows = [["b", "fuel", "9.000", "1.2300", 'comma,quote"'],
                     ["a", "fuel", "8", "", "two\nlines"],
                     ["a", "gear", "7", "-0.00", ""]]
        self.source = self.root / "new.csv"
        self.source.write_bytes(self.csv_bytes(self.header, self.rows))
        self.expected = hashlib.sha256(self.csv_bytes(self.old, [[r[i] for i in (0, 1, 3, 4)] for r in self.rows])).hexdigest()
        self.contract = self.root / "contract.json"
        self.receipt = {"stage": "HISTORICAL_JUCARS_METADATA_RECOVERY_NOT_EMPIRICAL_REPLAY",
                        "empirical_replay_completed": False, "historical_test_labels_already_opened": True,
                        "target_schema_inspection": {"records": [
                            {"role": role, "header": self.header if role == "development_risk" else self.old,
                             "sha256": adapter.sha256(self.source) if role == "development_risk" else self.expected}
                            for role in ("development_value", "development_risk", "calibration")]}}
        self.save_contract()

    def csv_bytes(self, header, rows):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
        return stream.getvalue().encode()

    def save_contract(self):
        self.contract.write_text(json.dumps(self.receipt))

    def bound(self, **changes):
        args = dict(source=self.source, output=self.root / "output", contract_path=self.contract,
                    expected_contract=adapter.sha256(self.contract), role="development_value",
                    expected_input=adapter.sha256(self.source), expected_output=self.expected,
                    expected_listings=2, expected_pairs=3)
        args.update(changes)
        return adapter.project_bound(**args)

    def test_text_order_and_quotes_preserved_without_numeric_recompute(self):
        result = self.bound()
        self.assertEqual(result["output_sha256"], self.expected)
        self.assertFalse(result["numeric_values_recomputed"])
        self.assertFalse(result["historical_target_rows_read"])
        with (self.root / "output/bvival_action_dataset.csv").open(newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows, [self.old, *[[r[i] for i in (0, 1, 3, 4)] for r in self.rows]])

    def test_calibration_role_supported_separately(self):
        self.assertEqual(self.bound(role="calibration")["role"], "calibration")

    def test_wrong_input_or_contract_hash_creates_no_output(self):
        for key in ("expected_input", "expected_contract"):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.bound(**{key: "0" * 64})
            self.assertFalse((self.root / "output").exists())

    def test_wrong_legacy_hash_retains_failed_output_without_adjustment(self):
        self.receipt["target_schema_inspection"]["records"][0]["sha256"] = "0" * 64
        self.save_contract()
        with self.assertRaisesRegex(ValueError, "legacy bytes differ"):
            self.bound(expected_output="0" * 64)
        result = json.loads((self.root / "output/legacy_projection_audit.json").read_text())
        self.assertTrue(result["fitting_must_not_proceed"])
        self.assertEqual(adapter.sha256(self.root / "output/bvival_action_dataset.csv"), self.expected)

    def test_counts_mismatch_retained_and_no_overwrite(self):
        with self.assertRaisesRegex(ValueError, "counts differ"):
            self.bound(expected_pairs=4)
        self.assertTrue((self.root / "output/legacy_projection_audit.json").exists())
        with self.assertRaises(FileExistsError):
            self.bound()

    def test_only_one_column_can_be_removed_and_order_cannot_change(self):
        for old in (self.old[:-1], list(reversed(self.old)), self.old + ["value"]):
            with self.assertRaises(ValueError):
                adapter.project_csv(self.source, self.root / "never.csv", self.header, old)
        self.assertFalse((self.root / "never.csv").exists())

    def test_bad_rows_duplicate_keys_empty_keys_and_width_rejected(self):
        for i, rows in enumerate(([self.rows[0], self.rows[0]], [["", *self.rows[0][1:]]], [self.rows[0][:-1]])):
            self.source.write_bytes(self.csv_bytes(self.header, rows))
            with self.assertRaises(ValueError):
                adapter.project_csv(self.source, self.root / f"partial-{i}.csv", self.header, self.old)

    def test_incoming_header_must_equal_contract(self):
        self.source.write_bytes(self.csv_bytes(list(reversed(self.header)), self.rows))
        with self.assertRaisesRegex(ValueError, "header differs"):
            adapter.project_csv(self.source, self.root / "partial.csv", self.header, self.old)

    def test_invalid_role_or_missing_record_rejected_before_output(self):
        with self.assertRaises(ValueError):
            self.bound(role="development_risk")
        self.receipt["target_schema_inspection"]["records"].pop()
        self.save_contract()
        with self.assertRaises(ValueError):
            self.bound()
        self.assertFalse((self.root / "output").exists())

    def test_symlink_parent_rejected(self):
        (self.root / "linked").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.bound(output=self.root / "linked/output")

    def test_boolean_boundary_and_nonpositive_counts_rejected(self):
        self.receipt["empirical_replay_completed"] = 0
        self.save_contract()
        with self.assertRaises(ValueError):
            self.bound()
        with self.assertRaises(ValueError):
            self.bound(expected_listings=0)
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
