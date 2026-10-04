"""Fit BVI-Val action-value models without reading evaluation outcomes.

Training and calibration tables contain cross-fitted realized action values.
The evaluation table must contain pre-acquisition features only. A separate
locked evaluator may later join selected actions to post-reveal predictions and
labels after the policy artifacts have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from build_bvival_action_dataset import FORBIDDEN_FEATURE_TOKENS, RISK_TARGET_COLUMNS
from bvi_val import OneSidedActionValueCalibrator


OBJECTIVE_TARGETS = {
    "squared_log_error": "value_squared_log_error",
    "absolute_price_error": "value_absolute_price_error",
}
FORBIDDEN_EVALUATION_COLUMNS = set(OBJECTIVE_TARGETS.values()) | {
    "target_log",
    "after_prediction_log",
} | set(RISK_TARGET_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-actions", type=Path, required=True)
    parser.add_argument("--calibration-actions", type=Path, required=True)
    parser.add_argument("--evaluation-features", type=Path, required=True)
    parser.add_argument("--feature-columns", required=True)
    parser.add_argument("--categorical-columns", default="action_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="13,42,2026")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--minimum-action-group-size", type=int, default=100)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_csv_list(value: str) -> list[str]:
    output = [item.strip() for item in value.split(",") if item.strip()]
    if len(output) != len(set(output)):
        raise ValueError("Column or seed list contains duplicates")
    return output


def validate_feature_columns(feature_columns: Iterable[str]) -> list[str]:
    columns = list(feature_columns)
    if "action_id" not in columns:
        columns = ["action_id", *columns]
    forbidden = [
        column
        for column in columns
        if column in FORBIDDEN_EVALUATION_COLUMNS
        or column.startswith("price_value_")
        or any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"Value-model features contain forbidden outcome fields: {forbidden}")
    if len(columns) != len(set(columns)):
        raise ValueError("Feature columns contain duplicates")
    return columns


def prepare_model_matrix(
    frame: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    categorical_columns: Iterable[str],
) -> tuple[pd.DataFrame, list[int]]:
    features = validate_feature_columns(feature_columns)
    categorical = list(categorical_columns)
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"Input table is missing model features: {missing}")
    unknown_categorical = sorted(set(categorical).difference(features))
    if unknown_categorical:
        raise ValueError(f"Categorical columns are not model features: {unknown_categorical}")
    matrix = frame[features].copy()
    for column in categorical:
        matrix[column] = matrix[column].fillna("__MISSING__").astype(str)
    for column in set(features).difference(categorical):
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    categorical_indices = [features.index(column) for column in categorical]
    return matrix, categorical_indices


def fit_seed_ensemble(
    *,
    train_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    categorical_columns: list[str],
    seeds: list[int],
    iterations: int,
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    train_x, categorical_indices = prepare_model_matrix(
        train_frame,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
    )
    calibration_x, _ = prepare_model_matrix(
        calibration_frame,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
    )
    evaluation_x, _ = prepare_model_matrix(
        evaluation_frame,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
    )
    if target_column not in train_frame or target_column not in calibration_frame:
        raise ValueError(f"Training and calibration tables require {target_column}")
    train_y = pd.to_numeric(train_frame[target_column], errors="raise").to_numpy(dtype=float)
    calibration_y = pd.to_numeric(
        calibration_frame[target_column], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(train_y).all() or not np.isfinite(calibration_y).all():
        raise ValueError(f"{target_column} contains non-finite values")

    model_dir.mkdir(parents=True, exist_ok=True)
    calibration_predictions = []
    evaluation_predictions = []
    model_audit: list[dict[str, object]] = []
    for seed in seeds:
        model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(train_x, train_y, cat_features=categorical_indices)
        calibration_predictions.append(model.predict(calibration_x))
        evaluation_predictions.append(model.predict(evaluation_x))
        model_path = model_dir / f"seed_{seed}.cbm"
        model.save_model(model_path)
        model_audit.append(
            {"seed": seed, "model_path": str(model_path), "sha256": file_sha256(model_path)}
        )
    calibration_mean = np.mean(np.vstack(calibration_predictions), axis=0)
    evaluation_mean = np.mean(np.vstack(evaluation_predictions), axis=0)
    return calibration_mean, evaluation_mean, model_audit


def main() -> None:
    args = parse_args()
    feature_columns = validate_feature_columns(parse_csv_list(args.feature_columns))
    categorical_columns = parse_csv_list(args.categorical_columns)
    seeds = [int(value) for value in parse_csv_list(args.seeds)]
    if not seeds:
        raise ValueError("At least one seed is required")

    train = pd.read_csv(args.train_actions)
    calibration = pd.read_csv(args.calibration_actions)
    evaluation = pd.read_csv(args.evaluation_features)
    leaked = sorted(
        column for column in evaluation.columns
        if column in FORBIDDEN_EVALUATION_COLUMNS or column.startswith("price_value_")
    )
    if leaked:
        raise ValueError(
            "Evaluation feature file contains outcome columns and is refused: " + ", ".join(leaked)
        )
    for frame_name, frame in [
        ("train", train),
        ("calibration", calibration),
        ("evaluation", evaluation),
    ]:
        required_ids = {"listing_id", "action_id"}
        missing_ids = sorted(required_ids.difference(frame.columns))
        if missing_ids:
            raise ValueError(f"{frame_name} is missing identifiers: {missing_ids}")
        if frame.duplicated(["listing_id", "action_id"]).any():
            raise ValueError(f"{frame_name} contains duplicate listing-action pairs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = evaluation[["listing_id", "action_id"]].copy()
    full_audit: dict[str, object] = {
        "protocol": "v4-bvival-prospective",
        "evaluation_outcomes_read": False,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "seeds": seeds,
        "inputs": {
            "train_actions_sha256": file_sha256(args.train_actions),
            "calibration_actions_sha256": file_sha256(args.calibration_actions),
            "evaluation_features_sha256": file_sha256(args.evaluation_features),
        },
        "objectives": {},
    }
    for objective, target_column in OBJECTIVE_TARGETS.items():
        calibration_mean, evaluation_mean, model_audit = fit_seed_ensemble(
            train_frame=train,
            calibration_frame=calibration,
            evaluation_frame=evaluation,
            target_column=target_column,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            seeds=seeds,
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            l2_leaf_reg=args.l2_leaf_reg,
            model_dir=args.output_dir / "models" / objective,
        )
        realized_calibration = pd.to_numeric(
            calibration[target_column], errors="raise"
        ).to_numpy(dtype=float)
        calibrator = OneSidedActionValueCalibrator(
            alpha=args.alpha,
            minimum_group_size=args.minimum_action_group_size,
            nonnegative_correction=True,
        ).fit(
            calibration_mean,
            realized_calibration,
            calibration["action_id"].astype(str),
        )
        lower = calibrator.lower_bound(evaluation_mean, evaluation["action_id"].astype(str))
        scores[f"mean_value_{objective}"] = evaluation_mean
        scores[f"lower_value_{objective}"] = lower
        calibration_lower = calibrator.lower_bound(
            calibration_mean, calibration["action_id"].astype(str)
        )
        full_audit["objectives"][objective] = {
            "target_column": target_column,
            "calibration_marginal_lower_coverage": float(
                np.mean(realized_calibration >= calibration_lower)
            ),
            "calibrator": calibrator.audit(),
            "models": model_audit,
        }

    score_path = args.output_dir / "evaluation_action_scores.csv"
    scores.to_csv(score_path, index=False)
    full_audit["evaluation_action_scores_sha256"] = file_sha256(score_path)
    (args.output_dir / "value_policy_fit_audit.json").write_text(
        json.dumps(full_audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
