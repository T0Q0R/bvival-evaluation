import numpy as np
import pandas as pd
import pytest

from evaluate_bvival_policies import (
    absolute_price_error_by_listing,
    error_budget_auc,
    evaluate_score_policies,
    merge_additional_scores,
    paired_bootstrap_relative_mae,
    paired_bootstrap_relative_rmsle,
    validate_and_join,
)


def joined_table():
    return pd.DataFrame(
        {
            "listing_id": ["a", "b", "c", "d"],
            "action_id": ["field", "field", "field", "field"],
            "target_log": np.log1p([100.0, 200.0, 300.0, 400.0]),
            "before_prediction_log": np.log1p([80.0, 240.0, 330.0, 350.0]),
            "after_prediction_log": np.log1p([100.0, 205.0, 305.0, 390.0]),
            "score_good": [0.9, 0.8, 0.7, 0.6],
            "score_bad": [0.1, 0.2, 0.3, 0.4],
        }
    )


def test_validate_join_rejects_mismatched_action_universe():
    table = joined_table()
    outcomes = table.drop(columns=["score_good", "score_bad"])
    scores = table[["listing_id", "action_id", "score_good", "score_bad"]].iloc[:-1]
    with pytest.raises(ValueError, match="universes differ"):
        validate_and_join(outcomes, scores, ["score_good", "score_bad"])


def test_evaluator_builds_budget_curve_and_action_mix():
    curve, action_mix, selections = evaluate_score_policies(
        joined_table(),
        score_columns=["score_good", "score_bad"],
        budgets=[0.25, 0.50],
        objective="squared_log_error",
    )
    assert len(curve) == 4
    assert not action_mix.empty
    assert len(selections[("score_good", 0.25)]) == 1
    assert curve["positive_gain_capture"].between(0, 1).all()


def test_zero_score_policy_represents_no_acquisition_at_every_budget():
    table = joined_table()
    table["score_no_acquisition"] = 0.0
    curve, _, selections = evaluate_score_policies(
        table,
        score_columns=["score_no_acquisition"],
        budgets=[0.25, 0.50],
        objective="squared_log_error",
    )
    assert curve["action_count"].eq(0).all()
    assert curve["post_rmsle"].eq(curve["before_rmsle"]).all()
    assert selections[("score_no_acquisition", 0.25)].empty


def test_paired_bootstrap_returns_positive_gain_for_better_policy():
    candidate = pd.DataFrame(
        {"listing_id": ["a", "b", "c"], "squared_log_error": [1.0, 1.0, 1.0]}
    )
    reference = pd.DataFrame(
        {"listing_id": ["a", "b", "c"], "squared_log_error": [4.0, 4.0, 4.0]}
    )
    result = paired_bootstrap_relative_rmsle(
        candidate, reference, repetitions=100, seed=13
    )
    assert result["relative_rmsle_gain_percent"] == pytest.approx(50.0)
    assert result["ci_lower_percent"] == pytest.approx(50.0)


def test_error_budget_auc_rejects_duplicate_budget():
    curve = pd.DataFrame({"budget": [0.1, 0.1], "post_rmsle": [0.2, 0.1]})
    with pytest.raises(ValueError, match="one row"):
        error_budget_auc(curve)


def test_absolute_price_objective_uses_mae_for_paired_comparison():
    table = joined_table()
    selected = table.iloc[[0]][["listing_id", "action_id"]]
    candidate = absolute_price_error_by_listing(table, selected)
    reference = absolute_price_error_by_listing(table, table.iloc[0:0])
    result = paired_bootstrap_relative_mae(
        candidate, reference, repetitions=100, seed=13
    )
    assert result["relative_mae_gain_percent"] > 0
    assert "relative_rmsle_gain_percent" not in result
    curve = pd.DataFrame({"budget": [0.1, 0.2], "post_mae_price": [10.0, 8.0]})
    assert error_budget_auc(curve, "post_mae_price") == pytest.approx(9.0)


def test_additional_score_merge_requires_same_action_universe():
    base = pd.DataFrame({"listing_id": ["a", "b"], "action_id": ["fuel", "fuel"], "score_a": [1.0, 2.0]})
    additional = base[["listing_id", "action_id"]].copy()
    additional["score_b"] = [2.0, 1.0]
    out = merge_additional_scores(base, additional)
    assert out["score_b"].tolist() == [2.0, 1.0]
    with pytest.raises(ValueError, match="universes differ"):
        merge_additional_scores(base, additional.iloc[:1])
