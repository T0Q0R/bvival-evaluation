import pandas as pd
import pytest

from select_bvival_fixed_field_development import select_fixed_field


def fixture_tables():
    actions = pd.DataFrame({
        "listing_id": ["a", "a", "b", "b", "c", "c"],
        "action_id": ["fuel", "mileage"] * 3,
        "prediction_is_oof": [True] * 6,
        "risk_target_absolute_price_error_before": [100.0] * 6,
        "value_absolute_price_error": [20.0, 1.0, 0.0, 30.0, 0.0, 1.0],
    })
    risks = pd.DataFrame({
        "listing_id": ["a", "a", "b", "b", "c", "c"],
        "action_id": ["fuel", "mileage"] * 3,
        "predicted_pre_action_absolute_price_error": [9.0, 9.0, 8.0, 8.0, 7.0, 7.0],
    })
    return actions, risks


def test_selects_budget_best_field_not_largest_global_mean():
    actions, risks = fixture_tables()
    # Budget 1/3 picks listing a. Fuel has highest gain there, whereas
    # mileage has the higher all-listing mean (32/3 versus fuel's 20/3).
    result = select_fixed_field(actions, risks, budget_fraction=1 / 3)
    assert result["selected_action"] == "fuel"
    assert result["capacity"] == 1
    assert result["listing_count"] == 3
    assert result["evaluation_labels_read"] is False


def test_rejects_non_oof_development_outcomes():
    actions, risks = fixture_tables()
    actions.loc[0, "prediction_is_oof"] = False
    with pytest.raises(ValueError, match="OOF"):
        select_fixed_field(actions, risks, budget_fraction=0.1)
