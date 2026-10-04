import numpy as np
import pandas as pd
import pytest

from fit_bvival_posterror_baseline import (
    SCORE_COLUMN,
    add_before_error_from_labels,
    add_before_error_from_matched_actions,
    combine_risk_and_posterror,
    derive_post_error,
    validate_evaluation_features,
)


def test_posterror_target_is_before_error_minus_realized_value():
    actions = pd.DataFrame(
        {
            "listing_id": ["a", "b"],
            "action_id": ["fuel", "fuel"],
            "prediction_is_oof": [True, True],
            "risk_target_absolute_price_error_before": [10.0, 15.0],
            "value_absolute_price_error": [3.0, -5.0],
        }
    )
    derived = derive_post_error(actions)
    assert derived["post_action_absolute_price_error"].tolist() == [7.0, 20.0]


def test_rejects_non_oof_targets_and_label_bearing_evaluation():
    actions = pd.DataFrame(
        {
            "listing_id": ["a"],
            "action_id": ["fuel"],
            "prediction_is_oof": [False],
            "risk_target_absolute_price_error_before": [10.0],
            "value_absolute_price_error": [1.0],
        }
    )
    with pytest.raises(ValueError, match="out-of-fold"):
        derive_post_error(actions)
    evaluation = pd.DataFrame(
        {"listing_id": ["a"], "action_id": ["fuel"], "target_log": [2.0]}
    )
    with pytest.raises(ValueError, match="outcome columns"):
        validate_evaluation_features(evaluation, ["action_id"], ["action_id"])


def test_shift_preserves_ranking_and_rejects_mismatched_universe():
    evaluation = pd.DataFrame(
        {"listing_id": ["a", "b"], "action_id": ["fuel", "fuel"]}
    )
    risk = evaluation.copy()
    risk["predicted_pre_action_absolute_price_error"] = [5.0, 3.0]
    score, shift = combine_risk_and_posterror(evaluation, risk, np.array([8.0, 1.0]))
    assert shift == pytest.approx(4.0)
    assert score[SCORE_COLUMN].gt(0).all()
    assert score.loc[1, SCORE_COLUMN] > score.loc[0, SCORE_COLUMN]
    with pytest.raises(ValueError, match="universes differ"):
        combine_risk_and_posterror(evaluation, risk.iloc[:1], np.array([8.0, 1.0]))


def test_target_backfill_requires_identical_frozen_actions():
    original = pd.DataFrame({"listing_id": ["a"], "action_id": ["fuel"], "value_absolute_price_error": [3.0]})
    enriched = original.copy()
    enriched["risk_target_absolute_price_error_before"] = [10.0]
    out = add_before_error_from_matched_actions(original, enriched)
    assert out["risk_target_absolute_price_error_before"].tolist() == [10.0]
    enriched.loc[0, "value_absolute_price_error"] = 4.0
    with pytest.raises(ValueError, match="changed frozen column"):
        add_before_error_from_matched_actions(original, enriched)


def test_calibration_label_backfill_uses_listing_keys():
    actions = pd.DataFrame({"listing_id": ["a"], "action_id": ["fuel"], "before_prediction_log": [np.log1p(90.0)]})
    labels = pd.DataFrame({"listing_id": ["a"], "target_log": [np.log1p(100.0)]})
    out = add_before_error_from_labels(actions, labels)
    assert out["risk_target_absolute_price_error_before"].iloc[0] == pytest.approx(10.0)
