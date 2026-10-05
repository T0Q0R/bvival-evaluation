"""Actual saved-estimator freeze/replay on new synthetic partitions only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_turkey_price_free_manifest import digest
from freeze_turkey_heldout_scores import freeze_partitions
from run_turkey_development import verify_files
from run_turkey_synthetic_integration import synthetic_dataset
from turkey_price_free_scoring import verify_completed_package


def run_check(development, output, neural_python, *, project_root):
    audit, _, config = verify_completed_package(development, project_root=project_root)
    if audit["evidence_status"] != "synthetic_only":
        raise ValueError("Synthetic freeze check refuses actual source development")
    # Synthetic generated values only. No source cohort or source targets here.
    synthetic = synthetic_dataset(config)
    examples = list(synthetic["initial"].values())
    partitions = {}
    for split, n in (("calibration", 17), ("test", 23)):
        records = [{"record_key": digest(["synthetic_new_heldout", split, i]), "initial": examples[i % len(examples)]}
                   for i in range(n)]
        partitions[split] = {"records": records,
                             "groups": {row["record_key"]: digest(["synthetic_new_group", split, i // 2])
                                        for i, row in enumerate(records)}}
    binding = {"mode": "synthetic_only", "expected_keys": {s: sorted(m["groups"]) for s, m in partitions.items()},
               "source_cohort_used": False, "source_price_values_decoded": False,
               "new_synthetic_keys_not_development_or_source_records": True}
    result = freeze_partitions(development, partitions, binding, output, neural_python, project_root=project_root)
    verify_files(output, result["artifact_sha256"], ignored=["heldout_score_freeze_audit.json"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-development-run", type=Path, required=True)
    parser.add_argument("--neural-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_check(args.synthetic_development_run, args.output_dir, args.neural_python,
                       project_root=Path(__file__).resolve().parents[1])
    print(json.dumps({k: result[k] for k in ("stage", "partition_counts", "refits", "heldout_label_release_allowed")}))


if __name__ == "__main__":
    main()
