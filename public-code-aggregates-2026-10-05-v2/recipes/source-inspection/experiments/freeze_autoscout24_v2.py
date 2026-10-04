"""Hash protocol, code and feature/label files before full test evaluation.

Computes file hashes only; never parses or scores sealed test labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RELATIVE_FILES = (
    "experiments/configs/autoscout24_bvival_v2_full.json",
    "notes/analysis/autoscout24-v2-policy-protocol-before-model-2026-09-25.md",
    "notes/analysis/autoscout24-v2-full-validation-field-decision-2026-09-25.md",
    "notes/analysis/autoscout24-v2-pretest-score-audit-2026-09-25.md",
    "experiments/data/processed/autoscout24_bvival_v2/autoscout24_bvival_audit.json",
    "experiments/data/processed/autoscout24_bvival_v2/splits/train.csv",
    "experiments/data/processed/autoscout24_bvival_v2/splits/validation.csv",
    "experiments/data/processed/autoscout24_bvival_v2/splits/calibration.csv",
    "experiments/data/processed/autoscout24_bvival_v2/splits/test_features.csv",
    "experiments/data/processed/autoscout24_bvival_v2/sealed_labels/test_labels.csv",
    "experiments/outputs/autoscout24_bvival_v2_full_fixed_field/selection.json",
    "experiments/outputs/autoscout24_bvival_v2_full_selection_pairs/field_selection_pair_audit.json",
    "experiments/outputs/autoscout24_bvival_v2_full_quasi_duplicate/quasi_duplicate_test_mask.csv",
    "experiments/outputs/autoscout24_bvival_v2_full_quasi_duplicate/quasi_duplicate_audit.json",
    "experiments/outputs/autoscout24_bvival_v2_full_pairs/prediction_pair_generation_audit.json",
    "experiments/outputs/autoscout24_bvival_v2_full_pairs/evaluation_prediction_pairs.csv",
    "experiments/outputs/autoscout24_bvival_v2_full_pairs/evaluation_pre_action_features.csv",
    "experiments/outputs/autoscout24_bvival_v2_full_actions_dev/bvival_action_dataset_audit.json",
    "experiments/outputs/autoscout24_bvival_v2_full_actions_cal/bvival_action_dataset_audit.json",
    "experiments/outputs/autoscout24_bvival_v2_full_value/value_policy_fit_audit.json",
    "experiments/outputs/autoscout24_bvival_v2_full_value/evaluation_action_scores.csv",
    "experiments/outputs/autoscout24_bvival_v2_full_risk/risk_baseline_fit_audit.json",
    "experiments/outputs/autoscout24_bvival_v2_full_risk/evaluation_risk_scores.csv",
    "experiments/outputs/autoscout24_bvival_v2_full_scores/policy_score_assembly_audit_absolute_price_error.json",
    "experiments/outputs/autoscout24_bvival_v2_full_scores/frozen_policy_scores_absolute_price_error.csv",
    "experiments/generate_bvival_prediction_pairs.py",
    "experiments/generate_bvival_field_selection_pairs.py",
    "experiments/build_bvival_action_dataset.py",
    "experiments/fit_bvival_value_policy.py",
    "experiments/fit_bvival_risk_baseline.py",
    "experiments/select_bvival_fixed_field_development.py",
    "experiments/assemble_bvival_policy_scores.py",
    "experiments/bvi_val_baselines.py",
    "experiments/join_bvival_evaluation_outcomes.py",
    "experiments/evaluate_bvival_policies.py",
    "experiments/build_autoscout24_quasi_duplicate_mask.py",
    "experiments/freeze_autoscout24_v2.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    missing = [name for name in RELATIVE_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Cannot freeze missing files: {missing}")
    hashes = {name: file_sha256(root / name) for name in RELATIVE_FILES}
    record = {
        "protocol": "autoscout24-bvival-v2-full-pretest-input-and-code-freeze",
        "files_sha256": hashes,
        "test_labels": "raw-byte SHA-256 only; no CSV parse, metric, or policy selection",
        "test_performance_opened": False,
        "modification_rule": "Any later code/config change requires a new freeze before test scoring",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"file_count": len(hashes), "test_performance_opened": False}))


if __name__ == "__main__":
    main()
