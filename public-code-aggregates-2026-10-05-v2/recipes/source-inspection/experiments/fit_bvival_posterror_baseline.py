"""Fit a two-head, one-step post-error baseline without test outcomes.

The frozen pre-action risk head predicts absolute advertised-price error.
This independent head predicts the absolute error *after* each field reveal
from the same pre-acquisition features used by BVI-Val. Their difference is
the acquisition score. This is a transparent local baseline, not an official
reproduction of a published AFA implementation. Since the three source test
sets were already opened before this baseline was devised, comparisons on
those sets are exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_bvival_action_dataset import RISK_TARGET_COLUMNS
from fit_bvival_value_policy import (
    FORBIDDEN_EVALUATION_COLUMNS,
    fit_seed_ensemble,
    prepare_model_matrix,
)


TARGET = "post_action_absolute_price_error"
RISK_COLUMN = "predicted_pre_action_absolute_price_error"
SCORE_COLUMN = "score_risk_minus_posterror_full_budget"
KEYS = ["listing_id", "action_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-value-fit-audit", type=Path, required=True)
    parser.add_argument("--train-actions", type=Path, required=True)
    parser.add_argument("--calibration-actions", type=Path, required=True)
    parser.add_argument("--train-price-targets", type=Path)
    parser.add_argument("--calibration-labels", type=Path)
    parser.add_argument("--evaluation-features", type=Path, required=True)
    parser.add_argument("--frozen-risk-scores", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_post_error(actions: pd.DataFrame) -> pd.DataFrame:
    required = {
        *KEYS,
        "prediction_is_oof",
        "risk_target_absolute_price_error_before",
        "value_absolute_price_error",
    }
    if missing := sorted(required.difference(actions.columns)):
        raise ValueError(f"Action table is missing required columns: {missing}")
    if actions.duplicated(KEYS).any():
        raise ValueError("Action table contains duplicate listing-action pairs")
    if not actions["prediction_is_oof"].astype(str).str.lower().eq("true").all():
        raise ValueError("Post-error targets require out-of-fold prediction pairs")
    before = pd.to_numeric(
        actions["risk_target_absolute_price_error_before"], errors="raise"
    ).to_numpy(dtype=float)
    value = pd.to_numeric(actions["value_absolute_price_error"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(before).all() or not np.isfinite(value).all():
        raise ValueError("Before-error and action-value targets must be finite")
    after = before - value
    if (after < -1e-7).any():
        raise ValueError("Derived post-action absolute error is negative")
    output = actions.copy()
    output[TARGET] = np.maximum(after, 0.0)
    return output


def add_before_error_from_matched_actions(
    original: pd.DataFrame, enriched: pd.DataFrame
) -> pd.DataFrame:
    """Use a later action-table version only for an absent training target."""
    target = "risk_target_absolute_price_error_before"
    if target in original.columns:
        raise ValueError("Original action table already has the price-error target")
    required = {*original.columns, target}
    if missing := sorted(required.difference(enriched.columns)):
        raise ValueError(f"Enriched action table is missing columns: {missing}")
    if original.duplicated(KEYS).any() or enriched.duplicated(KEYS).any():
        raise ValueError("Duplicate listing-action pairs in action table")
    matched = original.merge(
        enriched[list(original.columns) + [target]],
        on=KEYS,
        how="outer",
        suffixes=("_original", "_enriched"),
        validate="one_to_one",
        indicator=True,
    )
    if len(matched) != len(original) or not matched["_merge"].eq("both").all():
        raise ValueError("Original and enriched action universes differ")
    for column in original.columns:
        if column in KEYS:
            continue
        left = matched[f"{column}_original"]
        right = matched[f"{column}_enriched"]
        if not left.equals(right):
            raise ValueError(f"Enriched action table changed frozen column: {column}")
    return original.merge(enriched[KEYS + [target]], on=KEYS, validate="one_to_one")


def add_before_error_from_labels(actions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Derive a calibration target from calibration-only labels, never test labels."""
    target = "risk_target_absolute_price_error_before"
    if target in actions.columns:
        raise ValueError("Action table already has the price-error target")
    if missing := sorted({"listing_id", "target_log"}.difference(labels.columns)):
        raise ValueError(f"Calibration labels lack columns: {missing}")
    if labels.duplicated("listing_id").any():
        raise ValueError("Duplicate calibration listing labels")
    merged = actions.merge(labels[["listing_id", "target_log"]], on="listing_id", validate="many_to_one")
    if len(merged) != len(actions):
        raise ValueError("Calibration labels do not cover every action")
    truth = np.maximum(np.expm1(pd.to_numeric(merged["target_log"], errors="raise")), 0.0)
    before = np.maximum(np.expm1(pd.to_numeric(merged["before_prediction_log"], errors="raise")), 0.0)
    merged[target] = np.abs(truth - before)
    return merged.drop(columns="target_log")


def validate_evaluation_features(
    evaluation: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> None:
    forbidden = set(FORBIDDEN_EVALUATION_COLUMNS) | set(RISK_TARGET_COLUMNS) | {TARGET}
    leaked = sorted(
        column for column in evaluation.columns
        if column in forbidden or column.startswith("price_value_")
    )
    if leaked:
        raise ValueError(f"Evaluation feature file contains outcome columns: {leaked}")
    if missing := sorted(set(KEYS).difference(evaluation.columns)):
        raise ValueError(f"Evaluation features lack identifiers: {missing}")
    if evaluation.duplicated(KEYS).any():
        raise ValueError("Evaluation features contain duplicate listing-action pairs")
    prepare_model_matrix(
        evaluation,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
    )


def combine_risk_and_posterror(
    evaluation: pd.DataFrame,
    frozen_risk: pd.DataFrame,
    predicted_post_error: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    if missing := sorted({*KEYS, RISK_COLUMN}.difference(frozen_risk.columns)):
        raise ValueError(f"Frozen risk scores are missing columns: {missing}")
    if frozen_risk.duplicated(KEYS).any():
        raise ValueError("Frozen risk scores contain duplicate listing-action pairs")
    if len(predicted_post_error) != len(evaluation):
        raise ValueError("Predicted post-error count differs from evaluation rows")
    post = np.asarray(predicted_post_error, dtype=float)
    if not np.isfinite(post).all():
        raise ValueError("Predicted post-errors must be finite")
    merged = evaluation[KEYS].merge(
        frozen_risk[KEYS + [RISK_COLUMN]],
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all() or len(merged) != len(evaluation):
        raise ValueError("Frozen risk and evaluation listing-action universes differ")
    score_frame = evaluation[KEYS].copy()
    score_frame["predicted_post_action_absolute_price_error"] = np.maximum(post, 0.0)
    score_frame = score_frame.merge(
        frozen_risk[KEYS + [RISK_COLUMN]], on=KEYS, how="left", validate="one_to_one"
    )
    risk = pd.to_numeric(score_frame[RISK_COLUMN], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(risk).all():
        raise ValueError("Frozen pre-action risk scores must be finite")
    raw = risk - score_frame["predicted_post_action_absolute_price_error"].to_numpy(dtype=float)
    shift = max(0.0, 1.0 - float(raw.min()))
    score_frame["risk_minus_posterror_raw"] = raw
    score_frame[SCORE_COLUMN] = raw + shift
    if not (score_frame[SCORE_COLUMN] > 0).all():
        raise AssertionError("Full-budget score shift did not make every action eligible")
    return score_frame, shift


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.depth < 1 or args.learning_rate <= 0 or args.l2_leaf_reg <= 0:
        raise ValueError("Training hyperparameters must be positive")
    reference = json.loads(args.reference_value_fit_audit.read_text(encoding="utf-8"))
    if reference.get("evaluation_outcomes_read") is not False:
        raise ValueError("Reference value fit must document no test-outcome access")
    feature_columns = list(reference["feature_columns"])
    categorical_columns = list(reference["categorical_columns"])
    seeds = list(reference["seeds"])
    paths = {
        "train_actions": args.train_actions,
        "calibration_actions": args.calibration_actions,
        "evaluation_features": args.evaluation_features,
    }
    for key, path in paths.items():
        if file_sha256(path) != reference["inputs"][f"{key}_sha256"]:
            raise ValueError(f"{key} does not match the frozen reference value-fit input hash")
    train_actions = pd.read_csv(args.train_actions)
    calibration_actions = pd.read_csv(args.calibration_actions)
    if args.train_price_targets:
        train_actions = add_before_error_from_matched_actions(
            train_actions, pd.read_csv(args.train_price_targets)
        )
    if args.calibration_labels:
        calibration_actions = add_before_error_from_labels(
            calibration_actions, pd.read_csv(args.calibration_labels)
        )
    train = derive_post_error(train_actions)
    calibration = derive_post_error(calibration_actions)
    evaluation = pd.read_csv(args.evaluation_features)
    validate_evaluation_features(evaluation, feature_columns, categorical_columns)
    frozen_risk = pd.read_csv(args.frozen_risk_scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_prediction, evaluation_prediction, models = fit_seed_ensemble(
        train_frame=train,
        calibration_frame=calibration,
        evaluation_frame=evaluation,
        target_column=TARGET,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        seeds=seeds,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        model_dir=args.output_dir / "models",
    )
    scores, shift = combine_risk_and_posterror(
        evaluation, frozen_risk, evaluation_prediction
    )
    scores_path = args.output_dir / "evaluation_posterror_scores.csv"
    scores.to_csv(scores_path, index=False)
    calibration_truth = calibration[TARGET].to_numpy(dtype=float)
    audit = {
        "status": "post-test exploratory local baseline; not official published AFA code",
        "evaluation_outcomes_read": False,
        "target": TARGET,
        "score_column": SCORE_COLUMN,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "seeds": seeds,
        "hyperparameters": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
        },
        "input_hashes": {
            **{key: file_sha256(path) for key, path in paths.items()},
            "frozen_risk_scores": file_sha256(args.frozen_risk_scores),
            "reference_value_fit_audit": file_sha256(args.reference_value_fit_audit),
            **({"train_price_targets": file_sha256(args.train_price_targets)} if args.train_price_targets else {}),
            **({"calibration_labels": file_sha256(args.calibration_labels)} if args.calibration_labels else {}),
        },
        "calibration_posterror_mae": float(
            np.mean(np.abs(calibration_truth - np.maximum(calibration_prediction, 0.0)))
        ),
        "calibration_listing_count": int(calibration["listing_id"].nunique()),
        "score_shift_for_full_budget": shift,
        "model_audit": models,
        "evaluation_posterror_scores_sha256": file_sha256(scores_path),
        "interpretation": (
            "The shared score shift preserves candidate ordering and forces a full unit-cost "
            "capacity when enough listings exist. It is not an estimate of monetary profit."
        ),
    }
    (args.output_dir / "posterror_fit_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
