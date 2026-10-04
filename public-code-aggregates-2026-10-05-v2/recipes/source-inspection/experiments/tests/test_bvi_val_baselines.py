import numpy as np
import pandas as pd
import pytest

from bvi_val_baselines import (
    build_baseline_score_table,
    external_baseline_record,
    fit_global_action_prior,
    fixed_field_risk_scores,
    stable_random_score,
)


def candidate_table():
    return pd.DataFrame(
        {
            "listing_id": ["a", "a", "b", "b"],
            "action_id": ["fuel", "mileage", "fuel", "mileage"],
            "mean": [0.1, 0.3, -0.1, 0.2],
            "lower": [-0.1, 0.2, -0.2, 0.1],
            "width": [0.8, 0.8, 0.4, 0.4],
            "disagreement": [0.2, 0.2, 0.7, 0.7],
        }
    )


def test_stable_random_score_is_repeatable_and_pair_specific():
    assert stable_random_score("a", "fuel", 13) == stable_random_score("a", "fuel", 13)
    assert stable_random_score("a", "fuel", 13) != stable_random_score("a", "mileage", 13)


def test_global_prior_uses_development_targets_only():
    train = pd.DataFrame(
        {"action_id": ["fuel", "fuel", "mileage"], "value": [1.0, 3.0, 5.0]}
    )
    assert fit_global_action_prior(train, target_column="value") == {
        "fuel": 2.0,
        "mileage": 5.0,
    }


def test_baseline_table_routes_risk_to_best_prior_action():
    table = build_baseline_score_table(
        candidate_table(),
        action_prior={"fuel": 0.1, "mileage": 0.4},
        mean_value_column="mean",
        lower_value_column="lower",
        uncertainty_column="width",
        disagreement_column="disagreement",
        random_seed=13,
    )
    routed = table[table["score_uncertainty_only"] > -np.inf]
    assert routed["action_id"].tolist() == ["mileage", "mileage"]
    assert table["score_mean_value"].tolist() == [0.1, 0.3, -0.1, 0.2]
    assert table["score_no_acquisition"].eq(0.0).all()


def test_fixed_field_risk_never_routes_to_other_field():
    scores = fixed_field_risk_scores(
        candidate_table(), action_id="fuel", risk_column="width"
    )
    assert scores.tolist() == [0.8, 0.0, 0.4, 0.0]
    table = build_baseline_score_table(
        candidate_table(), action_prior={"fuel": 0.1, "mileage": 0.4},
        mean_value_column="mean", lower_value_column="lower",
        uncertainty_column="width", disagreement_column="disagreement",
        random_seed=13, fixed_action="fuel",
    )
    assert table["score_fixed_field_risk"].tolist() == [0.8, 0.0, 0.4, 0.0]


def test_completed_external_baseline_cannot_be_a_proxy():
    with pytest.raises(ValueError, match="exact"):
        external_baseline_record(
            name="ACO",
            source_url="https://proceedings.mlr.press/v235/valancius24a.html",
            implementation_status="completed",
            exact_official_implementation=False,
            reason="local approximation",
        )


def test_duplicate_candidate_actions_are_rejected():
    table = pd.concat([candidate_table(), candidate_table().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        build_baseline_score_table(
            table,
            action_prior={"fuel": 0.1},
            mean_value_column="mean",
            lower_value_column="lower",
            uncertainty_column="width",
            disagreement_column="disagreement",
            random_seed=13,
        )
