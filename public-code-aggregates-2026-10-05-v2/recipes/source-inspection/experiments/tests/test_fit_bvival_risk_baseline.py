import pandas as pd
import pytest

from fit_bvival_risk_baseline import TARGET_COLUMNS, validate_evaluation_frame
from fit_bvival_value_policy import validate_feature_columns


def test_risk_baseline_refuses_evaluation_outcomes():
    frame = pd.DataFrame(
        {
            "listing_id": ["a"],
            "action_id": ["fuel_type"],
            "risk_target_squared_log_error_before": [0.2],
        }
    )
    with pytest.raises(ValueError, match="risk outcomes"):
        validate_evaluation_frame(frame)


def test_risk_baseline_accepts_unique_label_free_pairs():
    frame = pd.DataFrame(
        {
            "listing_id": ["a", "a"],
            "action_id": ["fuel_type", "mileage_km"],
            "maker": ["x", "x"],
        }
    )
    validate_evaluation_frame(frame)


def test_risk_baseline_refuses_mae_target_in_evaluation_features():
    frame = pd.DataFrame(
        {
            "listing_id": ["a"],
            "action_id": ["fuel_type"],
            "risk_target_absolute_price_error_before": [100.0],
        }
    )
    with pytest.raises(ValueError, match="risk outcomes"):
        validate_evaluation_frame(frame)


def test_risk_baseline_refuses_eur_price_proxy():
    frame = pd.DataFrame({
        "listing_id": ["a"], "action_id": ["fuel_type"],
        "price_value_EUR": [20000.0],
    })
    with pytest.raises(ValueError, match="risk outcomes"):
        validate_evaluation_frame(frame)


def test_risk_objectives_have_distinct_matched_targets():
    assert TARGET_COLUMNS["absolute_price_error"] == "risk_target_absolute_price_error_before"
    assert TARGET_COLUMNS["squared_log_error"] == "risk_target_squared_log_error_before"


def test_value_feature_validator_rejects_risk_target():
    with pytest.raises(ValueError, match="forbidden"):
        validate_feature_columns(["risk_target_squared_log_error_before"])
