import pandas as pd
import pytest

from assemble_bvival_policy_scores import assemble_policy_scores, merge_one_to_one


def fixtures():
    development = pd.DataFrame(
        {
            "listing_id": ["d1", "d1", "d2", "d2"],
            "action_id": ["fuel_type", "mileage_km"] * 2,
            "value_squared_log_error": [0.1, 0.4, 0.2, 0.6],
            "value_absolute_price_error": [100.0, 300.0, 150.0, 400.0],
        }
    )
    features = pd.DataFrame(
        {
            "listing_id": ["a", "a", "b", "b"],
            "action_id": ["fuel_type", "mileage_km"] * 2,
            "before_prediction_std": [0.2, 0.3, 0.4, 0.5],
        }
    )
    values = pd.DataFrame(
        {
            "listing_id": ["a", "a", "b", "b"],
            "action_id": ["fuel_type", "mileage_km"] * 2,
            "mean_value_squared_log_error": [0.2, 0.5, 0.1, 0.4],
            "lower_value_squared_log_error": [0.1, 0.3, -0.1, 0.2],
        }
    )
    risks = pd.DataFrame(
        {
            "listing_id": ["a", "a", "b", "b"],
            "action_id": ["fuel_type", "mileage_km"] * 2,
            "predicted_pre_action_squared_log_error": [0.7, 0.8, 0.3, 0.4],
        }
    )
    return development, features, values, risks


def test_assembler_emits_all_local_policy_scores_without_outcomes():
    development, features, values, risks = fixtures()
    scores, prior = assemble_policy_scores(
        development_actions=development,
        value_scores=values,
        risk_scores=risks,
        evaluation_features=features,
        objective="squared_log_error",
        random_seed=13,
    )
    assert prior["fuel_type"] == pytest.approx(0.15)
    assert prior["mileage_km"] == pytest.approx(0.5)
    assert {
        "score_no_acquisition",
        "score_random",
        "score_global_field_prior",
        "score_uncertainty_only",
        "score_disagreement_only",
        "score_mean_value",
        "score_conservative_lower_value",
    }.issubset(scores.columns)
    assert len(scores) == 4


def test_development_mean_fixed_field_uses_no_evaluation_outcomes():
    development, features, values, risks = fixtures()
    scores, _ = assemble_policy_scores(
        development_actions=development, value_scores=values,
        risk_scores=risks, evaluation_features=features,
        objective="squared_log_error", random_seed=13,
        fixed_action="development_mean",
    )
    fixed = scores.set_index(["listing_id", "action_id"])["score_fixed_field_risk"]
    assert fixed.loc[("a", "fuel_type")] == 0
    assert fixed.loc[("a", "mileage_km")] == 0.8
    assert fixed.loc[("b", "fuel_type")] == 0
    assert fixed.loc[("b", "mileage_km")] == 0.4


def test_assembler_refuses_evaluation_target():
    development, features, values, risks = fixtures()
    features["target_log"] = 1.0
    with pytest.raises(ValueError, match="outcomes"):
        assemble_policy_scores(
            development_actions=development,
            value_scores=values,
            risk_scores=risks,
            evaluation_features=features,
            objective="squared_log_error",
            random_seed=13,
        )


def test_assembler_refuses_eur_price_proxy():
    development, features, values, risks = fixtures()
    features["price_value_EUR"] = 20_000.0
    with pytest.raises(ValueError, match="outcomes"):
        assemble_policy_scores(
            development_actions=development, value_scores=values,
            risk_scores=risks, evaluation_features=features,
            objective="squared_log_error", random_seed=13,
        )


def test_mae_assembler_requires_mae_matched_risk_score():
    development, features, values, risks = fixtures()
    values = values.rename(
        columns={
            "mean_value_squared_log_error": "mean_value_absolute_price_error",
            "lower_value_squared_log_error": "lower_value_absolute_price_error",
        }
    )
    with pytest.raises(ValueError, match="predicted_pre_action_absolute_price_error"):
        assemble_policy_scores(
            development_actions=development,
            value_scores=values,
            risk_scores=risks,
            evaluation_features=features,
            objective="absolute_price_error",
            random_seed=13,
        )


def test_merge_requires_identical_pair_universe():
    _, features, values, _ = fixtures()
    with pytest.raises(ValueError, match="universe differs"):
        merge_one_to_one(features, values.iloc[:-1], "values")
