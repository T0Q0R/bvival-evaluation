"""Select a risk penalty using calibration-only held-out listings.

This script never accepts external test labels or predictions. The original
calibration partition is split by listing hash: folds 0--2 fit one-sided
corrections; folds 3--4 select a fixed penalty for the final test policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from bvi_val import OneSidedActionValueCalibrator
from bvi_val_baselines import build_baseline_score_table, fit_global_action_prior
from evaluate_bvival_policies import evaluate_score_policies, validate_and_join
from fit_bvival_value_policy import prepare_model_matrix
from generate_bvival_prediction_pairs import stable_fold
from join_bvival_evaluation_outcomes import build_evaluation_outcomes


PENALTIES = (0.25, 0.5, 1.0)
SEEDS = (13, 42, 2026)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-actions", type=Path, required=True)
    parser.add_argument("--calibration-actions", type=Path, required=True)
    parser.add_argument("--calibration-pairs", type=Path, required=True)
    parser.add_argument("--calibration-labels", type=Path, required=True)
    parser.add_argument("--calibration-features", type=Path, required=True)
    parser.add_argument("--value-model-dir", type=Path, required=True)
    parser.add_argument("--risk-model-dir", type=Path, required=True)
    parser.add_argument("--feature-columns", required=True)
    parser.add_argument("--categorical-columns", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_ensemble_prediction(
    model_dir: Path, frame: pd.DataFrame,
    feature_columns: list[str], categorical_columns: list[str],
) -> np.ndarray:
    matrix, _ = prepare_model_matrix(
        frame, feature_columns=feature_columns,
        categorical_columns=categorical_columns,
    )
    predictions = []
    for seed in SEEDS:
        model = CatBoostRegressor()
        model.load_model(model_dir / f"seed_{seed}.cbm")
        predictions.append(model.predict(matrix))
    return np.mean(np.vstack(predictions), axis=0)


def choose_penalty(curve: pd.DataFrame) -> dict[str, float | str | bool]:
    primary = curve[np.isclose(curve["budget"], 0.10)].set_index("policy")
    mean_harm = float(primary.loc["score_mean_value", "harmful_action_rate"])
    candidates = []
    for penalty in PENALTIES:
        policy = f"score_penalty_{str(penalty).replace('.', 'p')}"
        row = primary.loc[policy]
        if row["action_count"] > 0 and row["harmful_action_rate"] <= mean_harm:
            candidates.append((float(row["post_rmsle"]), -penalty, policy))
    if not candidates:
        return {
            "selected_penalty": 1.0,
            "selected_policy": "score_penalty_1p0",
            "safety_constraint_satisfied": False,
            "selection_rule": "fallback_to_maximum_penalty_when_no_candidate_satisfies_harm_gate",
        }
    _, negative_penalty, policy = min(candidates)
    return {
        "selected_penalty": float(-negative_penalty),
        "selected_policy": policy,
        "safety_constraint_satisfied": True,
        "selection_rule": "minimum_heldout_RMSLE_at_10pct_subject_to_harm_rate_no_higher_than_mean_value",
    }


def main() -> None:
    args = parse_args()
    inputs = {
        "development_actions": args.development_actions,
        "calibration_actions": args.calibration_actions,
        "calibration_pairs": args.calibration_pairs,
        "calibration_labels": args.calibration_labels,
        "calibration_features": args.calibration_features,
    }
    frames = {name: pd.read_csv(path) for name, path in inputs.items()}
    features = [value.strip() for value in args.feature_columns.split(",") if value.strip()]
    categorical = [
        value.strip() for value in args.categorical_columns.split(",") if value.strip()
    ]
    actions = frames["calibration_actions"]
    features_frame = frames["calibration_features"]
    if actions.duplicated(["listing_id", "action_id"]).any():
        raise ValueError("Calibration action keys are not unique")
    if features_frame.duplicated(["listing_id", "action_id"]).any():
        raise ValueError("Calibration feature keys are not unique")
    score_rows = features_frame[["listing_id", "action_id", "before_prediction_std"]].copy()
    score_rows["mean_value_squared_log_error"] = model_ensemble_prediction(
        args.value_model_dir, features_frame, features, categorical
    )
    score_rows["predicted_pre_action_squared_log_error"] = np.maximum(
        model_ensemble_prediction(args.risk_model_dir, features_frame, features, categorical),
        0.0,
    )
    action_targets = actions[["listing_id", "action_id", "value_squared_log_error"]]
    score_rows = score_rows.merge(
        action_targets, on=["listing_id", "action_id"], how="inner", validate="one_to_one"
    )
    if len(score_rows) != len(features_frame) or len(score_rows) != len(actions):
        raise ValueError("Calibration score and action universes differ")
    fold = score_rows["listing_id"].map(lambda value: stable_fold(value, 5))
    fit_mask = fold.isin([0, 1, 2]).to_numpy()
    selection_mask = ~fit_mask
    calibrator = OneSidedActionValueCalibrator(
        alpha=0.10, minimum_group_size=100, nonnegative_correction=True
    ).fit(
        score_rows.loc[fit_mask, "mean_value_squared_log_error"].to_numpy(),
        score_rows.loc[fit_mask, "value_squared_log_error"].to_numpy(),
        score_rows.loc[fit_mask, "action_id"].astype(str),
    )
    validation = score_rows.loc[selection_mask].copy().reset_index(drop=True)
    mean_value = validation["mean_value_squared_log_error"].to_numpy(dtype=float)
    lower = calibrator.lower_bound(mean_value, validation["action_id"].astype(str))
    correction = mean_value - lower
    validation["lower_value_squared_log_error"] = lower
    prior = fit_global_action_prior(
        frames["development_actions"], target_column="value_squared_log_error"
    )
    scores = build_baseline_score_table(
        validation,
        action_prior=prior,
        mean_value_column="mean_value_squared_log_error",
        lower_value_column="lower_value_squared_log_error",
        uncertainty_column="predicted_pre_action_squared_log_error",
        disagreement_column="before_prediction_std",
        random_seed=2026,
    )
    for penalty in PENALTIES:
        scores[f"score_penalty_{str(penalty).replace('.', 'p')}"] = (
            mean_value - penalty * correction
        )
    pairs = frames["calibration_pairs"]
    selected_ids = set(validation["listing_id"])
    outcomes = build_evaluation_outcomes(
        prediction_pairs=pairs[pairs["listing_id"].isin(selected_ids)],
        labels=frames["calibration_labels"][
            frames["calibration_labels"]["listing_id"].isin(selected_ids)
        ],
        policy_scores=scores,
    )
    score_columns = [
        "score_uncertainty_only", "score_mean_value",
        *[f"score_penalty_{str(p).replace('.', 'p')}" for p in PENALTIES],
    ]
    joined = validate_and_join(outcomes, scores, score_columns)
    curve, _, _ = evaluate_score_policies(
        joined, score_columns=score_columns, budgets=[0.05, 0.10, 0.20],
        objective="squared_log_error",
    )
    selection = choose_penalty(curve)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.output_dir / "heldout_development_penalty_curve.csv"
    curve.to_csv(curve_path, index=False)
    audit = {
        "evidence_status": "development_only",
        "test_labels_read": False,
        "calibration_fit_listing_count": int(score_rows.loc[fit_mask, "listing_id"].nunique()),
        "policy_selection_listing_count": int(validation["listing_id"].nunique()),
        "policy_selection_action_count": int(len(validation)),
        "penalty_candidates": list(PENALTIES),
        "selection": selection,
        "calibrator": calibrator.audit(),
        "input_hashes": {name: file_sha256(path) for name, path in inputs.items()},
        "model_files": {
            "value": [
                file_sha256(args.value_model_dir / f"seed_{seed}.cbm") for seed in SEEDS
            ],
            "risk": [
                file_sha256(args.risk_model_dir / f"seed_{seed}.cbm") for seed in SEEDS
            ],
        },
        "curve_sha256": file_sha256(curve_path),
    }
    (args.output_dir / "development_penalty_selection.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
