import pandas as pd

from fit_bvival_risk_baseline import crossfit_development_risk
from generate_bvival_prediction_pairs import stable_fold


def test_risk_crossfit_keeps_each_listing_out_of_its_training_fold(tmp_path):
    rows = []
    for index in range(12):
        listing = f"listing-{index}"
        for action in ("fuel_type", "mileage_km"):
            rows.append({
                "listing_id": listing,
                "action_id": action,
                "before_prediction_log": 9.0 + 0.01 * index,
                "risk_target_absolute_price_error_before": 100.0 + 20 * index,
            })
    actions = pd.DataFrame(rows)
    features = actions.drop(columns="risk_target_absolute_price_error_before")
    scores, audit = crossfit_development_risk(
        train_actions=actions,
        development_features=features,
        target_column="risk_target_absolute_price_error_before",
        score_column="predicted_pre_action_absolute_price_error",
        feature_columns=["action_id", "before_prediction_log"],
        categorical_columns=["action_id"],
        seeds=[13], folds=2, iterations=8, depth=2,
        learning_rate=0.05, l2_leaf_reg=10.0,
        model_dir=tmp_path,
    )
    assert len(scores) == len(actions)
    assert len(audit) == 2
    assert scores["predicted_pre_action_absolute_price_error"].ge(0).all()
    assert scores["risk_fold"].tolist() == [
        stable_fold(identifier, 2) for identifier in scores["listing_id"]
    ]
