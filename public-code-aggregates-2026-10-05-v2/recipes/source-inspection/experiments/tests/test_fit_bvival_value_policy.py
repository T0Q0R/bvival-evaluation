import numpy as np
import pandas as pd
import pytest

from fit_bvival_value_policy import (
    fit_seed_ensemble,
    prepare_model_matrix,
    validate_feature_columns,
)


def test_prepare_matrix_encodes_only_declared_pre_action_features():
    frame = pd.DataFrame(
        {
            "listing_id": ["a", "b"],
            "action_id": ["fuel", "mileage"],
            "maker": ["x", None],
            "before_prediction_log": [8.0, 9.0],
            "unused": [1, 2],
        }
    )
    matrix, categorical_indices = prepare_model_matrix(
        frame,
        feature_columns=["action_id", "maker", "before_prediction_log"],
        categorical_columns=["action_id", "maker"],
    )
    assert matrix.columns.tolist() == ["action_id", "maker", "before_prediction_log"]
    assert categorical_indices == [0, 1]
    assert matrix.loc[1, "maker"] == "__MISSING__"


def test_action_id_is_automatically_included():
    assert validate_feature_columns(["maker"]) == ["action_id", "maker"]


@pytest.mark.parametrize(
    "forbidden", ["target_log", "value_squared_log_error", "after_prediction_log", "Price", "price_value_EUR"]
)
def test_value_model_rejects_outcome_features(forbidden):
    with pytest.raises(ValueError, match="forbidden"):
        validate_feature_columns(["action_id", forbidden])


def test_tiny_value_model_runs_end_to_end(tmp_path):
    train = pd.DataFrame(
        {
            "action_id": ["fuel", "mileage"] * 6,
            "maker": ["x", "y", "x", "y"] * 3,
            "before_prediction_log": np.linspace(8.0, 9.1, 12),
            "value_squared_log_error": np.linspace(-0.1, 0.3, 12),
        }
    )
    calibration = train.iloc[:4].copy()
    evaluation = train.iloc[4:7].drop(columns="value_squared_log_error").copy()
    calibration_prediction, evaluation_prediction, audit = fit_seed_ensemble(
        train_frame=train,
        calibration_frame=calibration,
        evaluation_frame=evaluation,
        target_column="value_squared_log_error",
        feature_columns=["action_id", "maker", "before_prediction_log"],
        categorical_columns=["action_id", "maker"],
        seeds=[13],
        iterations=5,
        depth=2,
        learning_rate=0.1,
        l2_leaf_reg=1.0,
        model_dir=tmp_path / "models",
    )
    assert calibration_prediction.shape == (4,)
    assert evaluation_prediction.shape == (3,)
    assert np.isfinite(calibration_prediction).all()
    assert np.isfinite(evaluation_prediction).all()
    assert len(audit) == 1
    assert (tmp_path / "models" / "seed_13.cbm").exists()
