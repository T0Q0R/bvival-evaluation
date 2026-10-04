import numpy as np
import pandas as pd
import pytest

from build_bvival_action_dataset import build_action_dataset, validate_feature_allowlist


def fixtures():
    pairs = pd.DataFrame(
        {
            "listing_id": ["a", "a", "b"],
            "action_id": ["mileage", "fuel", "mileage"],
            "fold_id": [0, 0, 1],
            "prediction_is_oof": [True, True, True],
            "before_prediction_log": np.log1p([80.0, 90.0, 240.0]),
            "after_prediction_log": np.log1p([100.0, 95.0, 210.0]),
        }
    )
    labels = pd.DataFrame(
        {"listing_id": ["a", "b"], "target_log": np.log1p([100.0, 200.0])}
    )
    features = pd.DataFrame(
        {
            "listing_id": ["a", "a", "b"],
            "action_id": ["mileage", "fuel", "mileage"],
            "maker": ["x", "x", "y"],
            "uncertainty_width": [0.2, 0.3, 0.5],
        }
    )
    return pairs, labels, features


def test_builder_emits_only_pre_action_features_and_value_targets():
    pairs, labels, features = fixtures()
    output, audit = build_action_dataset(
        prediction_pairs=pairs,
        labels=labels,
        pre_action_features=features,
        feature_columns=["maker", "uncertainty_width"],
    )
    assert "target_log" not in output
    assert "after_prediction_log" not in output
    assert "value_squared_log_error" in output
    assert "value_absolute_price_error" in output
    assert "risk_target_squared_log_error_before" in output
    assert "risk_target_absolute_log_error_before" in output
    assert "risk_target_absolute_price_error_before" in output
    expected = np.square(output["before_prediction_log"] - np.log1p([100.0, 100.0, 200.0]))
    np.testing.assert_allclose(output["risk_target_squared_log_error_before"], expected)
    expected_price_error = np.abs(
        np.array([100.0, 100.0, 200.0]) - np.array([80.0, 90.0, 240.0])
    )
    np.testing.assert_allclose(output["risk_target_absolute_price_error_before"], expected_price_error)
    assert audit["all_predictions_oof"] is True
    assert audit["listing_action_count"] == 3


def test_builder_emits_pre_action_prediction_once_when_allowlisted():
    pairs, labels, features = fixtures()
    output, _ = build_action_dataset(
        prediction_pairs=pairs,
        labels=labels,
        pre_action_features=features,
        feature_columns=["before_prediction_log", "maker"],
    )
    assert output.columns.tolist().count("before_prediction_log") == 1


def test_builder_rejects_non_oof_prediction_pair():
    pairs, labels, features = fixtures()
    pairs.loc[0, "prediction_is_oof"] = False
    with pytest.raises(ValueError, match="out of fold"):
        build_action_dataset(
            prediction_pairs=pairs,
            labels=labels,
            pre_action_features=features,
            feature_columns=["maker"],
        )


def test_feature_allowlist_rejects_outcome_proxies():
    for forbidden in [
        "target_price",
        "realized_gain",
        "after_prediction_log",
        "risk_target_squared_log_error_before",
    ]:
        with pytest.raises(ValueError, match="forbidden"):
            validate_feature_allowlist(["maker", forbidden])


def test_builder_rejects_duplicate_listing_action_pairs():
    pairs, labels, features = fixtures()
    pairs = pd.concat([pairs, pairs.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_action_dataset(
            prediction_pairs=pairs,
            labels=labels,
            pre_action_features=features,
            feature_columns=["maker"],
        )
