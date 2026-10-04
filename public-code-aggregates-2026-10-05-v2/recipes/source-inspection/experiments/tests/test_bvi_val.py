import numpy as np
import pandas as pd
import pytest

from bvi_val import (
    OneSidedActionValueCalibrator,
    evaluate_policy,
    policy_post_action_predictions,
    realized_action_value,
    select_heterogeneous_cost_actions,
    select_unit_cost_actions,
)


def test_realized_action_value_matches_declared_objective():
    target = np.log1p([100.0, 200.0])
    before = np.log1p([80.0, 240.0])
    after = np.log1p([95.0, 210.0])
    squared = realized_action_value(
        target_log=target,
        before_prediction_log=before,
        after_prediction_log=after,
        objective="squared_log_error",
    )
    expected_squared = np.square(target - before) - np.square(target - after)
    np.testing.assert_allclose(squared, expected_squared)
    absolute = realized_action_value(
        target_log=target,
        before_prediction_log=before,
        after_prediction_log=after,
        objective="absolute_price_error",
    )
    np.testing.assert_allclose(absolute, [15.0, 30.0])


def test_calibrator_uses_group_correction_and_pooled_fallback():
    predicted = np.asarray([3.0, 3.0, 3.0, 3.0, 4.0, 4.0])
    realized = np.asarray([1.0, 2.0, 2.5, 2.8, 3.0, 3.5])
    actions = np.asarray(["mileage"] * 4 + ["fuel"] * 2)
    calibrator = OneSidedActionValueCalibrator(alpha=0.25, minimum_group_size=3).fit(
        predicted, realized, actions
    )
    bounds = calibrator.lower_bound([5.0, 5.0], ["mileage", "fuel"])
    audit = calibrator.audit()
    assert "mileage" in audit["action_corrections"]
    assert "fuel" not in audit["action_corrections"]
    assert bounds[0] == pytest.approx(5.0 - audit["action_corrections"]["mileage"])
    assert bounds[1] == pytest.approx(5.0 - audit["pooled_correction"])


def test_unit_cost_selector_takes_one_positive_action_per_listing():
    actions = pd.DataFrame(
        {
            "listing_id": ["a", "a", "b", "c"],
            "action_id": ["fuel", "mileage", "fuel", "image"],
            "lower_value": [0.4, 0.7, -0.1, 0.6],
        }
    )
    selected = select_unit_cost_actions(actions, budget_fraction=2 / 3)
    assert selected[["listing_id", "action_id"]].values.tolist() == [
        ["a", "mileage"],
        ["c", "image"],
    ]
    assert not selected["listing_id"].duplicated().any()


def test_heterogeneous_selector_solves_multiple_choice_budget():
    actions = pd.DataFrame(
        {
            "listing_id": ["a", "a", "b", "c"],
            "action_id": ["cheap", "expensive", "field", "image"],
            "lower_value": [4.0, 8.0, 6.0, 6.5],
            "cost": [1.0, 3.0, 2.0, 2.0],
        }
    )
    selected = select_heterogeneous_cost_actions(actions, budget=4.0)
    assert set(map(tuple, selected[["listing_id", "action_id"]].to_numpy())) == {
        ("b", "field"),
        ("c", "image"),
    }
    assert selected["cost"].sum() <= 4.0


def test_policy_predictions_and_metrics_use_only_selected_actions():
    candidates = pd.DataFrame(
        {
            "listing_id": ["a", "a", "b"],
            "action_id": ["fuel", "mileage", "fuel"],
            "target_log": np.log1p([100.0, 100.0, 200.0]),
            "before_prediction_log": np.log1p([80.0, 80.0, 250.0]),
            "after_prediction_log": np.log1p([90.0, 100.0, 205.0]),
        }
    )
    selected = pd.DataFrame(
        {"listing_id": ["a"], "action_id": ["mileage"], "lower_value": [1.0]}
    )
    post = policy_post_action_predictions(candidates, selected)
    a = post.set_index("listing_id").loc["a"]
    b = post.set_index("listing_id").loc["b"]
    assert a["post_prediction_log"] == pytest.approx(np.log1p(100.0))
    assert b["post_prediction_log"] == pytest.approx(np.log1p(250.0))
    metrics = evaluate_policy(candidates, selected)
    assert metrics["action_count"] == 1
    assert metrics["post_rmsle"] < metrics["before_rmsle"]
    assert metrics["harmful_action_rate"] == 0.0


def test_invalid_duplicate_actions_are_rejected():
    actions = pd.DataFrame(
        {
            "listing_id": ["a", "a"],
            "action_id": ["fuel", "fuel"],
            "lower_value": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        select_unit_cost_actions(actions, budget_fraction=1.0)
