import numpy as np
import pandas as pd
import pytest

from generate_bvival_prediction_pairs import (
    ACTION_FIELDS,
    augment_mask_aware_training,
    infer_feature_columns,
    predict_listing_actions,
    stable_fold,
)


class FakeModel:
    def __init__(self, offset):
        self.offset = offset

    def predict(self, frame):
        mileage = pd.to_numeric(frame["mileage_km"], errors="coerce").fillna(0.0)
        return 8.0 + mileage.to_numpy() / 100000.0 + self.offset


def fixture_frame():
    return pd.DataFrame(
        {
            "listing_id": ["a", "b", "c"],
            "source_row_number": [0, 1, 2],
            "price_value_JOD": [10000.0, 12000.0, 15000.0],
            "target_log": np.log1p([10000.0, 12000.0, 15000.0]),
            "make": ["x", "y", "z"],
            "year": [2020, 2021, 2022],
            "mileage_km": [10000.0, 20000.0, np.nan],
            "mileage_text": ["10,000 - 19,999", "20,000 - 29,999", "Unknown"],
            "transmission": ["Automatic", "Manual", "Automatic"],
            "fuel_type": ["Gasoline", "Electric", "Hybrid"],
            "engine_type": ["Gasoline", "Electric", "Hybrid"],
            "feat_body_condition": ["Excellent", "Good", "Excellent"],
        }
    )


def test_training_augmentation_balances_clean_and_masked_weight_per_listing():
    frame = fixture_frame()
    augmented, target, weight = augment_mask_aware_training(frame)
    assert len(augmented) == len(target) == len(weight)
    assert set(augmented["action_mask"]) == {
        "none",
        "all_hidden",
        *(f"reveal:{action}" for action in ACTION_FIELDS),
    }
    clean_weight = weight[augmented["action_mask"].eq("none")].sum()
    masked_weight = weight[~augmented["action_mask"].eq("none")].sum()
    assert clean_weight == 3.0
    assert masked_weight == pytest.approx(3.0)


def test_prediction_pairs_share_one_all_hidden_pre_action_state():
    frame = fixture_frame()
    feature_columns, categorical = infer_feature_columns(frame)
    assert "mileage_text" not in feature_columns
    assert "engine_type" not in feature_columns
    pairs, features = predict_listing_actions(
        [FakeModel(0.0), FakeModel(0.1)],
        frame,
        feature_columns=feature_columns,
        categorical_columns=categorical,
        fold_id=pd.Series([0, 1, 2], index=frame.index),
        prediction_is_oof=True,
    )
    assert features[list(ACTION_FIELDS)].isna().all().all()
    assert features["before_prediction_log"].notna().all()
    assert "mileage_text" not in features.columns
    assert "engine_type" not in features.columns
    assert features["missing_action_field_count"].eq(len(ACTION_FIELDS)).all()
    before_counts = pairs.groupby("listing_id")["before_prediction_log"].nunique()
    assert before_counts.eq(1).all()
    assert pairs["prediction_is_oof"].all()
    assert pairs[["listing_id", "action_id"]].duplicated().sum() == 0


def test_after_state_reveals_only_the_selected_action():
    frame = fixture_frame()
    feature_columns, categorical = infer_feature_columns(frame)
    pairs, _ = predict_listing_actions(
        [FakeModel(0.0)],
        frame,
        feature_columns=feature_columns,
        categorical_columns=categorical,
        fold_id=-1,
        prediction_is_oof=True,
    )
    listing_a = pairs[pairs["listing_id"].eq("a")].set_index("action_id")
    assert listing_a.loc["mileage_km", "after_prediction_log"] > listing_a.loc[
        "mileage_km", "before_prediction_log"
    ]
    assert listing_a.loc["fuel_type", "after_prediction_log"] == listing_a.loc[
        "fuel_type", "before_prediction_log"
    ]


def test_listing_without_observed_action_field_has_no_candidate_for_it():
    frame = fixture_frame()
    feature_columns, categorical = infer_feature_columns(frame)
    pairs, _ = predict_listing_actions(
        [FakeModel(0.0)],
        frame,
        feature_columns=feature_columns,
        categorical_columns=categorical,
        fold_id=-1,
        prediction_is_oof=True,
    )
    assert not ((pairs["listing_id"] == "c") & (pairs["action_id"] == "mileage_km")).any()


def test_stable_fold_is_deterministic():
    assert stable_fold("listing-a", 5) == stable_fold("listing-a", 5)


def test_non_jod_price_column_is_excluded_from_features():
    frame = fixture_frame().rename(columns={"price_value_JOD": "price_value_MAD"})
    columns, _ = infer_feature_columns(frame)
    assert "price_value_MAD" not in columns


def test_custom_service_history_action_preserves_hidden_information_state():
    actions = ("mileage_km", "transmission", "fuel_type", "service_history")
    frame = fixture_frame().rename(columns={"feat_body_condition": "service_history"})
    features, categorical = infer_feature_columns(frame, action_fields=actions)
    pairs, pre = predict_listing_actions(
        [FakeModel(0.0)], frame,
        feature_columns=features, categorical_columns=categorical,
        fold_id=-1, prediction_is_oof=True, action_fields=actions,
    )
    assert set(pairs["action_id"]) == set(actions)
    assert pre[list(actions)].isna().all().all()
    assert pre["missing_action_field_count"].eq(4).all()
    assert "feat_body_condition" not in pre.columns
