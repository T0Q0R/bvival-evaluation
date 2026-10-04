"""Fit an objective-matched pre-acquisition risk baseline without test outcomes.

The model predicts the masked valuation state's error under the requested
objective. It is a routing baseline, not an action-value model: predicted
listing risk determines *which listing* to inspect, while a development-only
global action prior determines *which field* to acquire.
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
    fit_seed_ensemble,
    parse_csv_list,
    validate_feature_columns,
)
from generate_bvival_prediction_pairs import stable_fold


TARGET_COLUMNS = {
    "squared_log_error": "risk_target_squared_log_error_before",
    "absolute_price_error": "risk_target_absolute_price_error_before",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-actions", type=Path, required=True)
    parser.add_argument("--evaluation-features", type=Path, required=True)
    parser.add_argument(
        "--development-features", type=Path,
        help="Optional label-free development features for listing-level OOF risk scores",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--feature-columns", required=True)
    parser.add_argument("--categorical-columns", default="action_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--objective", choices=tuple(TARGET_COLUMNS), default="squared_log_error"
    )
    parser.add_argument("--seeds", default="13,42,2026")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_evaluation_frame(frame: pd.DataFrame) -> None:
    leaked = sorted(
        column for column in frame.columns
        if column in RISK_TARGET_COLUMNS
        or column in {"target_log", "after_prediction_log"}
        or column.startswith("price_value_")
    )
    if leaked:
        raise ValueError(
            "Evaluation feature file contains risk outcomes and is refused: "
            + ", ".join(leaked)
        )
    required = {"listing_id", "action_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Evaluation feature file is missing identifiers: {missing}")
    if frame.duplicated(["listing_id", "action_id"]).any():
        raise ValueError("Evaluation feature file contains duplicate listing-action pairs")


def crossfit_development_risk(
    *,
    train_actions: pd.DataFrame,
    development_features: pd.DataFrame,
    target_column: str,
    score_column: str,
    feature_columns: list[str],
    categorical_columns: list[str],
    seeds: list[int],
    folds: int,
    iterations: int,
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    model_dir: Path,
) -> tuple[pd.DataFrame, list[dict]]:
    if folds < 2:
        raise ValueError("OOF risk folds must be at least two")
    validate_evaluation_frame(development_features)
    keys = ["listing_id", "action_id"]
    universe = train_actions[keys].merge(
        development_features[keys], on=keys, how="outer",
        validate="one_to_one", indicator=True,
    )
    if not universe["_merge"].eq("both").all():
        raise ValueError("Development actions/features listing-action universes differ")
    assignments = train_actions["listing_id"].map(lambda value: stable_fold(value, folds))
    feature_assignments = development_features["listing_id"].map(
        lambda value: stable_fold(value, folds)
    )
    outputs = []
    fold_audits = []
    for fold in range(folds):
        fit_rows = train_actions.loc[assignments.ne(fold)]
        heldout = development_features.loc[feature_assignments.eq(fold)]
        if fit_rows.empty or heldout.empty:
            raise ValueError(f"Risk fold {fold} has empty fit or held-out partition")
        fit_ids = set(fit_rows["listing_id"])
        heldout_ids = set(heldout["listing_id"])
        if fit_ids.intersection(heldout_ids):
            raise AssertionError("Risk fold trains and predicts the same listing")
        calibration_stub = fit_rows.iloc[: min(8, len(fit_rows))]
        _, prediction, model_audit = fit_seed_ensemble(
            train_frame=fit_rows,
            calibration_frame=calibration_stub,
            evaluation_frame=heldout,
            target_column=target_column,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            seeds=seeds,
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            model_dir=model_dir / f"fold_{fold}",
        )
        block = heldout[keys].copy()
        block[score_column] = np.maximum(prediction, 0.0)
        block["risk_fold"] = fold
        outputs.append(block)
        fold_audits.append({
            "fold": fold,
            "fit_listing_count": len(fit_ids),
            "heldout_listing_count": len(heldout_ids),
            "model_audit": model_audit,
        })
    scores = pd.concat(outputs, ignore_index=True)
    matched = development_features[keys].merge(
        scores[keys], on=keys, how="outer", validate="one_to_one", indicator=True
    )
    if not matched["_merge"].eq("both").all():
        raise AssertionError("OOF risk scores do not cover development features")
    return scores, fold_audits


def main() -> None:
    args = parse_args()
    feature_columns = validate_feature_columns(parse_csv_list(args.feature_columns))
    categorical_columns = parse_csv_list(args.categorical_columns)
    seeds = [int(value) for value in parse_csv_list(args.seeds)]
    if not seeds:
        raise ValueError("At least one seed is required")

    train = pd.read_csv(args.train_actions)
    evaluation = pd.read_csv(args.evaluation_features)
    validate_evaluation_frame(evaluation)
    target_column = TARGET_COLUMNS[args.objective]
    if target_column not in train:
        raise ValueError(f"Training action table requires {target_column}")
    if train.duplicated(["listing_id", "action_id"]).any():
        raise ValueError("Training action table contains duplicate listing-action pairs")

    # fit_seed_ensemble requires a calibration frame only to emit predictions;
    # using a deterministic development subset here does not calibrate or tune
    # the model and no calibration metric is reported.
    calibration_stub = train.iloc[: min(8, len(train))].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, evaluation_mean, model_audit = fit_seed_ensemble(
        train_frame=train,
        calibration_frame=calibration_stub,
        evaluation_frame=evaluation,
        target_column=target_column,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        seeds=seeds,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        model_dir=args.output_dir / "models",
    )
    scores = evaluation[["listing_id", "action_id"]].copy()
    risk_score_column = f"predicted_pre_action_{args.objective}"
    scores[risk_score_column] = np.maximum(
        evaluation_mean, 0.0
    )
    score_path = args.output_dir / "evaluation_risk_scores.csv"
    scores.to_csv(score_path, index=False)
    development_oof_audit = None
    if args.development_features is not None:
        development = pd.read_csv(args.development_features)
        oof_scores, fold_audits = crossfit_development_risk(
            train_actions=train,
            development_features=development,
            target_column=target_column,
            score_column=risk_score_column,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            seeds=seeds,
            folds=args.folds,
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            l2_leaf_reg=args.l2_leaf_reg,
            model_dir=args.output_dir / "oof_models",
        )
        oof_path = args.output_dir / "development_oof_risk_scores.csv"
        oof_scores.to_csv(oof_path, index=False)
        development_oof_audit = {
            "development_features_sha256": file_sha256(args.development_features),
            "development_oof_scores_sha256": file_sha256(oof_path),
            "folds": args.folds,
            "fold_audits": fold_audits,
            "listing_level_oof": True,
        }
    audit = {
        "protocol": "v4-bvival-prospective",
        "evaluation_outcomes_read": False,
        "target_column": target_column,
        "objective": args.objective,
        "risk_score_column": risk_score_column,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "seeds": seeds,
        "inputs": {
            "train_actions_sha256": file_sha256(args.train_actions),
            "evaluation_features_sha256": file_sha256(args.evaluation_features),
        },
        "models": model_audit,
        "evaluation_risk_scores_sha256": file_sha256(score_path),
        "development_oof": development_oof_audit,
        "interpretation": (
            "Predicted pre-acquisition error is a risk-routing baseline and is not "
            "an estimate of action value."
        ),
    }
    (args.output_dir / "risk_baseline_fit_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
