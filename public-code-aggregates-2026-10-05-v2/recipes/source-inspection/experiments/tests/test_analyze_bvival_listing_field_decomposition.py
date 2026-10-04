import numpy as np
import pandas as pd
import pytest

from analyze_bvival_listing_field_decomposition import (
    best_value_field,
    decompose,
    fixed_field_with_fallback,
)


def test_fixed_field_fallback_preserves_selected_listing_set():
    scores = pd.DataFrame(
        {
            "listing_id": ["a", "a", "b"],
            "action_id": ["fixed", "other", "other"],
            "score_mean_value": [1.0, 2.0, 3.0],
        }
    )
    listings = pd.Series(["a", "b"])
    alternative = best_value_field(scores, listings)
    selected, fallback_count = fixed_field_with_fallback(
        scores, listings, "fixed", alternative
    )
    assert fallback_count == 1
    assert dict(zip(selected.listing_id, selected.action_id)) == {"a": "fixed", "b": "other"}


def test_crossed_decomposition_separates_listing_and_field_effects():
    rows = []
    for listing, price, before, risk, fixed_gain, other_gain, value_score in (
        ("a", 100.0, 90.0, 10.0, 0.0, 0.0, 1.0),
        ("b", 200.0, 180.0, 5.0, 15.0, 20.0, 9.0),
        ("c", 300.0, 290.0, 1.0, 0.0, 0.0, 0.5),
    ):
        for action, gain, action_score in (
            ("fixed", fixed_gain, value_score),
            ("other", other_gain, value_score + (1.0 if listing == "b" else -1.0)),
        ):
            rows.append(
                {
                    "listing_id": listing,
                    "action_id": action,
                    "target_log": np.log1p(price),
                    "before_prediction_log": np.log1p(before),
                    "after_prediction_log": np.log1p(before + gain),
                    "score_uncertainty_only": risk if action == "fixed" else -np.inf,
                    "score_mean_value": action_score,
                }
            )
    table = pd.DataFrame(rows)
    scores = table[["listing_id", "action_id", "score_uncertainty_only", "score_mean_value"]]
    outcomes = table[
        ["listing_id", "action_id", "target_log", "before_prediction_log", "after_prediction_log"]
    ]
    result, selections = decompose(
        scores,
        outcomes,
        fixed_field="fixed",
        budget=0.3,
        bootstrap_repetitions=100,
        bootstrap_seed=13,
    )
    assert result["selected_count_per_cell"] == {cell: 1 for cell in result["selected_count_per_cell"]}
    assert result["listing_set_overlap"] == 0
    assert result["post_reveal_mae_by_cell"]["risk_listings_fixed_field"] > result["post_reveal_mae_by_cell"]["value_listings_fixed_field"]
    assert result["post_reveal_mae_by_cell"]["value_listings_fixed_field"] > result["post_reveal_mae_by_cell"]["value_listings_value_field"]
    assert len(selections) == 4


def test_decomposition_rejects_duplicate_score_pairs():
    scores = pd.DataFrame(
        {
            "listing_id": ["a", "a"],
            "action_id": ["fixed", "fixed"],
            "score_uncertainty_only": [1.0, 1.0],
            "score_mean_value": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        decompose(
            scores,
            pd.DataFrame(),
            fixed_field="fixed",
            budget=0.1,
            bootstrap_repetitions=10,
            bootstrap_seed=13,
        )
