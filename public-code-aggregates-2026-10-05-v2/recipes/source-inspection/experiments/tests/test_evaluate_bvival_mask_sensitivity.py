import numpy as np
import pandas as pd

from evaluate_bvival_mask_sensitivity import evaluate_mask


def test_sensitivity_selects_globally_before_subsetting():
    rows = []
    scores = []
    for index, listing in enumerate(("a", "b", "c", "d")):
        for action in ("fuel", "mileage"):
            rows.append({
                "listing_id": listing, "action_id": action,
                "target_log": np.log1p(100.0),
                "before_prediction_log": np.log1p(90.0),
                "after_prediction_log": np.log1p(95.0),
            })
            scores.append({
                "listing_id": listing, "action_id": action,
                "candidate": 10.0 - index if action == "fuel" else 0.0,
                "reference": 10.0 - index if action == "mileage" else 0.0,
            })
    mask = pd.DataFrame({
        "listing_id": ["a", "b", "c", "d"],
        "quasi_key_seen_in_non_test": [False, False, True, True],
    })
    result = evaluate_mask(
        pd.DataFrame(rows), pd.DataFrame(scores), mask,
        candidate_score="candidate", reference_score="reference",
        budget_fraction=0.5, repetitions=20, seed=13,
    )
    novel, seen = result
    assert novel["n"] == 2 and seen["n"] == 2
    assert novel["candidate_action_count"] == 2
    assert seen["candidate_action_count"] == 0
