"""Open sealed external labels only after frozen policy scores are verified.

This boundary program performs no fitting or policy selection.  It verifies the
declared SHA-256 of the score table before reading labels, joins the frozen
prediction-pair universe to labels, and writes the only outcome table accepted
by the locked evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-prediction-pairs", type=Path, required=True)
    parser.add_argument("--sealed-labels", type=Path, required=True)
    parser.add_argument("--frozen-policy-scores", type=Path, required=True)
    parser.add_argument("--expected-policy-scores-sha256", required=True)
    parser.add_argument("--additional-frozen-policy-scores", type=Path)
    parser.add_argument("--expected-additional-policy-scores-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze-id", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_score_file(path: Path, expected_sha256: str) -> tuple[pd.DataFrame, str]:
    actual = file_sha256(path)
    if actual != expected_sha256.strip().lower():
        raise ValueError("Frozen policy score hash mismatch; labels remain unopened by this program")
    return pd.read_csv(path), actual


def verify_score_universe(primary: pd.DataFrame, additional: pd.DataFrame) -> None:
    keys = ["listing_id", "action_id"]
    for name, frame in [("primary", primary), ("additional", additional)]:
        if not set(keys).issubset(frame.columns) or frame.duplicated(keys).any():
            raise ValueError(f"{name} frozen scores have invalid listing-action keys")
    comparison = primary[keys].merge(
        additional[keys], on=keys, how="outer", validate="one_to_one", indicator=True
    )
    if not comparison["_merge"].eq("both").all():
        raise ValueError("Primary and additional frozen-score universes differ")


def build_evaluation_outcomes(
    *,
    prediction_pairs: pd.DataFrame,
    labels: pd.DataFrame,
    policy_scores: pd.DataFrame,
) -> pd.DataFrame:
    pair_columns = {
        "listing_id",
        "action_id",
        "before_prediction_log",
        "after_prediction_log",
    }
    label_columns = {"listing_id", "target_log"}
    key_columns = {"listing_id", "action_id"}
    for name, frame, required in [
        ("prediction_pairs", prediction_pairs, pair_columns),
        ("labels", labels, label_columns),
        ("policy_scores", policy_scores, key_columns),
    ]:
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
    if prediction_pairs.duplicated(["listing_id", "action_id"]).any():
        raise ValueError("prediction_pairs contains duplicate listing-action pairs")
    if labels["listing_id"].duplicated().any():
        raise ValueError("labels contains duplicate listing IDs")
    if policy_scores.duplicated(["listing_id", "action_id"]).any():
        raise ValueError("policy_scores contains duplicate listing-action pairs")

    pair_keys = prediction_pairs[["listing_id", "action_id"]]
    score_keys = policy_scores[["listing_id", "action_id"]]
    universe = pair_keys.merge(
        score_keys,
        on=["listing_id", "action_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not universe["_merge"].eq("both").all():
        raise ValueError(
            "Prediction and frozen-score universes differ: "
            + str(universe["_merge"].value_counts().to_dict())
        )

    output = prediction_pairs[
        [
            "listing_id",
            "action_id",
            "before_prediction_log",
            "after_prediction_log",
        ]
    ].merge(labels[["listing_id", "target_log"]], on="listing_id", how="left", validate="many_to_one")
    if output["target_log"].isna().any():
        raise ValueError("Some evaluation listings have no sealed target")
    output = output[
        [
            "listing_id",
            "action_id",
            "target_log",
            "before_prediction_log",
            "after_prediction_log",
        ]
    ]
    numeric = output[["target_log", "before_prediction_log", "after_prediction_log"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Evaluation outcomes contain non-finite numeric values")
    return output


def main() -> None:
    args = parse_args()
    if bool(args.additional_frozen_policy_scores) != bool(
        args.expected_additional_policy_scores_sha256
    ):
        raise ValueError("Additional frozen scores require both path and expected SHA-256")
    scores, actual_score_hash = verify_frozen_score_file(
        args.frozen_policy_scores, args.expected_policy_scores_sha256
    )
    additional_hash = None
    if args.additional_frozen_policy_scores:
        additional_scores, additional_hash = verify_frozen_score_file(
            args.additional_frozen_policy_scores,
            args.expected_additional_policy_scores_sha256,
        )
        verify_score_universe(scores, additional_scores)

    # Labels are deliberately read only after the policy-score hash passes.
    pairs = pd.read_csv(args.evaluation_prediction_pairs)
    verify_score_universe(scores, pairs)
    labels = pd.read_csv(args.sealed_labels)
    outcomes = build_evaluation_outcomes(
        prediction_pairs=pairs,
        labels=labels,
        policy_scores=scores,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "evaluation_outcomes.csv"
    outcomes.to_csv(output_path, index=False)
    audit = {
        "protocol": "v4-bvival-prospective",
        "freeze_id": args.freeze_id,
        "labels_opened": True,
        "policy_score_hash_verified_before_label_read": True,
        "inputs": {
            "prediction_pairs_sha256": file_sha256(args.evaluation_prediction_pairs),
            "sealed_labels_sha256": file_sha256(args.sealed_labels),
            "frozen_policy_scores_sha256": actual_score_hash,
            "additional_frozen_policy_scores_sha256": additional_hash,
        },
        "output_sha256": file_sha256(output_path),
        "listing_count": int(outcomes["listing_id"].nunique()),
        "listing_action_count": int(len(outcomes)),
    }
    (args.output_dir / "evaluation_outcome_join_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
