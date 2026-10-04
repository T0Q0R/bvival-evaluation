"""Generate cross-fitted before/after acquisition predictions for BVI-Val.

The development rows are cross-fitted. A final universal mask-aware CatBoost
ensemble is then fitted on all development rows and used for calibration rows
and label-free evaluation features. This script never accepts evaluation labels
and computes no evaluation loss or policy metric.
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


ACTION_FIELDS = ("mileage_km", "transmission", "fuel_type", "feat_body_condition")
IDENTIFIER_COLUMNS = {"listing_id", "source_row_number"}
TARGET_COLUMNS = {"target_log"}
# These source fields reproduce candidate actions and would invalidate the
# common hidden-information state if retained as model inputs.
ACTION_PROXY_COLUMNS = {"mileage_text", "engine_type"}


def is_target_column(column: str) -> bool:
    return column in TARGET_COLUMNS or column.startswith("price_value_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-fields", default=",".join(ACTION_FIELDS))
    parser.add_argument("--seeds", default="13,42,2026")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fold(identifier: object, folds: int) -> int:
    if folds < 2:
        raise ValueError("folds must be at least two")
    digest = hashlib.sha256(str(identifier).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def infer_feature_columns(
    frame: pd.DataFrame, *, action_fields: Iterable[str] = ACTION_FIELDS
) -> tuple[list[str], list[str]]:
    action_fields = tuple(action_fields)
    features = [
        column
        for column in frame.columns
        if column not in IDENTIFIER_COLUMNS | ACTION_PROXY_COLUMNS
        and not is_target_column(column)
    ]
    missing_actions = sorted(set(action_fields).difference(features))
    if missing_actions:
        raise ValueError(f"Input is missing action fields: {missing_actions}")
    categorical = [
        column
        for column in features
        if pd.api.types.is_object_dtype(frame[column])
        or pd.api.types.is_string_dtype(frame[column])
    ]
    return features, categorical


def prepare_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    categorical_columns: Iterable[str],
) -> pd.DataFrame:
    features = list(feature_columns)
    categorical = set(categorical_columns)
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"Feature frame is missing columns: {missing}")
    output = frame[features].copy()
    for column in features:
        if column in categorical:
            output[column] = output[column].fillna("__MISSING__").astype(str)
        else:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def build_acquisition_state(
    frame: pd.DataFrame,
    *,
    revealed_action: str | None,
    action_fields: Iterable[str] = ACTION_FIELDS,
) -> pd.DataFrame:
    """Mask all acquirable fields, optionally revealing exactly one action."""

    action_fields = tuple(action_fields)
    if revealed_action is not None and revealed_action not in action_fields:
        raise ValueError(f"Unsupported action: {revealed_action}")
    output = frame.copy()
    for action in action_fields:
        output[action] = np.nan
    if revealed_action is None:
        output["action_mask"] = "all_hidden"
    else:
        output[revealed_action] = frame[revealed_action]
        output["action_mask"] = f"reveal:{revealed_action}"
    return output


def augment_mask_aware_training(
    frame: pd.DataFrame,
    *,
    action_fields: Iterable[str] = ACTION_FIELDS,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return clean plus masked training rows with balanced listing weights."""

    if "target_log" not in frame:
        raise ValueError("Training frame requires target_log")
    clean = frame.copy()
    clean["action_mask"] = "none"
    blocks = [clean]
    weights = [np.ones(len(clean), dtype=float)]
    action_fields = tuple(action_fields)
    eligible_counts_raw = frame[list(action_fields)].notna().sum(axis=1)
    eligible_counts = eligible_counts_raw.clip(lower=1)
    all_hidden = build_acquisition_state(
        frame, revealed_action=None, action_fields=action_fields
    )
    blocks.append(all_hidden)
    weights.append(
        np.where(eligible_counts_raw.to_numpy() > 0, 0.5, 1.0).astype(float)
    )
    for action in action_fields:
        eligible = frame[action].notna()
        if not eligible.any():
            continue
        revealed = build_acquisition_state(
            frame.loc[eligible], revealed_action=action, action_fields=action_fields
        )
        blocks.append(revealed)
        weights.append((0.5 / eligible_counts.loc[eligible]).to_numpy(dtype=float))
    augmented = pd.concat(blocks, ignore_index=True)
    sample_weight = np.concatenate(weights)
    target = pd.to_numeric(augmented["target_log"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(target).all() or not np.isfinite(sample_weight).all():
        raise ValueError("Training target or sample weight contains non-finite values")
    return augmented, target, sample_weight


def fit_ensemble(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    categorical_columns: list[str],
    seeds: list[int],
    iterations: int,
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    action_fields: Iterable[str] = ACTION_FIELDS,
) -> list[CatBoostRegressor]:
    augmented, target, sample_weight = augment_mask_aware_training(
        frame, action_fields=action_fields
    )
    model_features = [*feature_columns, "action_mask"]
    model_categorical = [*categorical_columns, "action_mask"]
    x = prepare_feature_frame(
        augmented,
        feature_columns=model_features,
        categorical_columns=model_categorical,
    )
    categorical_indices = [model_features.index(column) for column in model_categorical]
    models = []
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
        model.fit(x, target, cat_features=categorical_indices, sample_weight=sample_weight)
        models.append(model)
    return models


def ensemble_predict(
    models: list[CatBoostRegressor],
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    if not models:
        raise ValueError("At least one fitted model is required")
    model_features = [*feature_columns, "action_mask"]
    model_categorical = [*categorical_columns, "action_mask"]
    x = prepare_feature_frame(
        frame,
        feature_columns=model_features,
        categorical_columns=model_categorical,
    )
    predictions = np.vstack([model.predict(x) for model in models])
    return predictions.mean(axis=0), predictions.std(axis=0, ddof=0)


def predict_listing_actions(
    models: list[CatBoostRegressor],
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    categorical_columns: list[str],
    fold_id: int | pd.Series,
    prediction_is_oof: bool,
    action_fields: Iterable[str] = ACTION_FIELDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create long prediction-pair and pre-action feature tables."""

    if "listing_id" not in frame:
        raise ValueError("Input frame requires listing_id")
    if frame["listing_id"].duplicated().any():
        raise ValueError("Input frame contains duplicate listing IDs")
    action_fields = tuple(action_fields)
    before_raw = build_acquisition_state(
        frame, revealed_action=None, action_fields=action_fields
    )
    before_mean, before_std = ensemble_predict(
        models,
        before_raw,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
    )
    before_prediction = pd.Series(before_mean, index=frame.index)
    before_disagreement = pd.Series(before_std, index=frame.index)
    pair_blocks = []
    feature_blocks = []
    if isinstance(fold_id, pd.Series):
        fold_lookup = fold_id
    else:
        fold_lookup = pd.Series(int(fold_id), index=frame.index)

    for action in action_fields:
        eligible = frame[action].notna()
        if not eligible.any():
            continue
        after_raw = build_acquisition_state(
            frame.loc[eligible], revealed_action=action, action_fields=action_fields
        )
        after_mean, _ = ensemble_predict(
            models,
            after_raw,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
        )
        eligible_index = frame.index[eligible]
        pairs = pd.DataFrame(
            {
                "listing_id": frame.loc[eligible_index, "listing_id"].to_numpy(),
                "action_id": action,
                "fold_id": fold_lookup.loc[eligible_index].to_numpy(),
                "prediction_is_oof": prediction_is_oof,
                "before_prediction_log": before_prediction.loc[eligible_index].to_numpy(),
                "after_prediction_log": after_mean,
            }
        )
        pair_blocks.append(pairs)

        pre_features = before_raw.loc[eligible_index, feature_columns].copy().reset_index(drop=True)
        pre_features.insert(0, "action_id", action)
        pre_features.insert(
            0,
            "listing_id",
            frame.loc[eligible_index, "listing_id"].to_numpy(),
        )
        pre_features["before_prediction_log"] = before_prediction.loc[
            eligible_index
        ].to_numpy()
        pre_features["before_prediction_std"] = before_disagreement.loc[
            eligible_index
        ].to_numpy()
        pre_features["missing_action_field_count"] = before_raw.loc[eligible_index,
            list(action_fields)
        ].isna().sum(axis=1).to_numpy(dtype=int)
        feature_blocks.append(pre_features)

    if not pair_blocks:
        raise ValueError("No eligible listing-action pairs were generated")
    pairs = pd.concat(pair_blocks, ignore_index=True)
    pre_features = pd.concat(feature_blocks, ignore_index=True)
    if pairs.duplicated(["listing_id", "action_id"]).any():
        raise AssertionError("Generated prediction pairs are not unique")
    if pre_features.duplicated(["listing_id", "action_id"]).any():
        raise AssertionError("Generated pre-action features are not unique")
    return pairs, pre_features


def crossfit_development(
    development: pd.DataFrame,
    *,
    feature_columns: list[str],
    categorical_columns: list[str],
    folds: int,
    seeds: list[int],
    iterations: int,
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    action_fields: Iterable[str] = ACTION_FIELDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = development["listing_id"].map(lambda value: stable_fold(value, folds))
    pair_blocks = []
    feature_blocks = []
    for fold in range(folds):
        train = assignments.ne(fold)
        holdout = assignments.eq(fold)
        if not train.any() or not holdout.any():
            raise ValueError(f"Fold {fold} has an empty train or holdout partition")
        models = fit_ensemble(
            development.loc[train],
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            seeds=seeds,
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            action_fields=action_fields,
        )
        pairs, features = predict_listing_actions(
            models,
            development.loc[holdout],
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            fold_id=assignments.loc[holdout],
            prediction_is_oof=True,
            action_fields=action_fields,
        )
        pair_blocks.append(pairs)
        feature_blocks.append(features)
    pairs = pd.concat(pair_blocks, ignore_index=True)
    features = pd.concat(feature_blocks, ignore_index=True)
    return pairs, features


def write_pair_bundle(
    *,
    name: str,
    pairs: pd.DataFrame,
    pre_features: pd.DataFrame,
    labels: pd.DataFrame | None,
    output_dir: Path,
) -> dict[str, str]:
    paths = {
        "pairs": output_dir / f"{name}_prediction_pairs.csv",
        "features": output_dir / f"{name}_pre_action_features.csv",
    }
    pairs.to_csv(paths["pairs"], index=False)
    pre_features.to_csv(paths["features"], index=False)
    if labels is not None:
        paths["labels"] = output_dir / f"{name}_labels.csv"
        labels[["listing_id", "target_log"]].to_csv(paths["labels"], index=False)
    return {key: file_sha256(path) for key, path in paths.items()}


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least two")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    action_fields = tuple(
        value.strip() for value in args.action_fields.split(",") if value.strip()
    )
    if not action_fields or len(action_fields) != len(set(action_fields)):
        raise ValueError("Action fields must be nonempty and unique")
    train = pd.read_csv(args.train)
    validation = pd.read_csv(args.validation)
    calibration = pd.read_csv(args.calibration)
    evaluation = pd.read_csv(args.evaluation_features)
    leaked = sorted(column for column in evaluation if is_target_column(column))
    if leaked:
        raise ValueError(f"Evaluation feature input contains forbidden target columns: {leaked}")
    for name, frame in [("train", train), ("validation", validation), ("calibration", calibration)]:
        if "target_log" not in frame:
            raise ValueError(f"{name} requires target_log")
    development = pd.concat([train, validation], ignore_index=True)
    if development["listing_id"].duplicated().any():
        raise ValueError("Development train and validation listing IDs overlap")
    feature_columns, categorical_columns = infer_feature_columns(
        development, action_fields=action_fields
    )
    evaluation_features, evaluation_categorical = infer_feature_columns(
        evaluation, action_fields=action_fields
    )
    if feature_columns != evaluation_features or categorical_columns != evaluation_categorical:
        raise ValueError("Development and evaluation feature schemas differ")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    development_pairs, development_pre = crossfit_development(
        development,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        folds=args.folds,
        seeds=seeds,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        action_fields=action_fields,
    )
    output_hashes = {
        "development": write_pair_bundle(
            name="development",
            pairs=development_pairs,
            pre_features=development_pre,
            labels=development,
            output_dir=args.output_dir,
        )
    }

    final_models = fit_ensemble(
        development,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        seeds=seeds,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        action_fields=action_fields,
    )
    calibration_pairs, calibration_pre = predict_listing_actions(
        final_models,
        calibration,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        fold_id=-1,
        prediction_is_oof=True,
        action_fields=action_fields,
    )
    output_hashes["calibration"] = write_pair_bundle(
        name="calibration",
        pairs=calibration_pairs,
        pre_features=calibration_pre,
        labels=calibration,
        output_dir=args.output_dir,
    )
    evaluation_pairs, evaluation_pre = predict_listing_actions(
        final_models,
        evaluation,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        fold_id=-1,
        prediction_is_oof=True,
        action_fields=action_fields,
    )
    output_hashes["evaluation"] = write_pair_bundle(
        name="evaluation",
        pairs=evaluation_pairs,
        pre_features=evaluation_pre,
        labels=None,
        output_dir=args.output_dir,
    )

    model_dir = args.output_dir / "base_models"
    model_dir.mkdir(exist_ok=True)
    model_hashes = []
    for seed, model in zip(seeds, final_models):
        path = model_dir / f"mask_aware_catboost_seed_{seed}.cbm"
        model.save_model(path)
        model_hashes.append({"seed": seed, "path": str(path), "sha256": file_sha256(path)})

    audit = {
        "protocol": "v4-bvival-prospective-external-replication",
        "development_listing_count": int(development["listing_id"].nunique()),
        "calibration_listing_count": int(calibration["listing_id"].nunique()),
        "evaluation_listing_count": int(evaluation["listing_id"].nunique()),
        "development_action_pair_count": int(len(development_pairs)),
        "calibration_action_pair_count": int(len(calibration_pairs)),
        "evaluation_action_pair_count": int(len(evaluation_pairs)),
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "action_fields": list(action_fields),
        "seeds": seeds,
        "folds": args.folds,
        "model_parameters": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
        },
        "inputs": {
            "train_sha256": file_sha256(args.train),
            "validation_sha256": file_sha256(args.validation),
            "calibration_sha256": file_sha256(args.calibration),
            "evaluation_features_sha256": file_sha256(args.evaluation_features),
        },
        "outputs": output_hashes,
        "models": model_hashes,
        "evaluation_labels_read": False,
        "performance_metrics_computed": False,
    }
    (args.output_dir / "prediction_pair_generation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
