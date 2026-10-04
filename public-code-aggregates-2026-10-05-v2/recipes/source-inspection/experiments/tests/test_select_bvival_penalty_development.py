import pandas as pd

from select_bvival_penalty_development import choose_penalty


def test_penalty_selection_uses_heldout_rmsle_and_harm_gate():
    curve = pd.DataFrame({
        "policy": [
            "score_mean_value", "score_penalty_0p25", "score_penalty_0p5",
            "score_penalty_1p0",
        ],
        "budget": [0.10] * 4,
        "post_rmsle": [0.20, 0.19, 0.18, 0.21],
        "action_count": [10] * 4,
        "harmful_action_rate": [0.30, 0.28, 0.35, 0.20],
    })
    selected = choose_penalty(curve)
    assert selected["selected_penalty"] == 0.25
    assert selected["safety_constraint_satisfied"]
