import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluate_bvival_subgroups import evaluate_groups


def test_prespecified_subgroups_compare_same_listings_and_flag_harm():
    config_path = Path(__file__).parents[1] / "configs/jucars_bvival_subgroups.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["primary_budget_fraction"] = 0.5
    config["minimum_group_size_for_safety_gate"] = 1
    config["material_relative_rmsle_deterioration_percent"] = 0.0
    joined = pd.DataFrame({
        "listing_id": ["a", "b"],
        "action_id": ["mileage_km", "mileage_km"],
        "target_log": np.log1p([5000.0, 15000.0]),
        "before_prediction_log": np.log1p([4000.0, 12000.0]),
        "after_prediction_log": np.log1p([5000.0, 10000.0]),
        "score_uncertainty_only": [2.0, 1.0],
        "score_conservative_lower_value": [-1.0, 2.0],
    })
    features = pd.DataFrame({
        "listing_id": ["a", "b"],
        "make": ["Hyundai", "Toyota"],
        "year": [2020, 2010],
        "mileage_km": [5499.0, np.nan],
    })
    report = evaluate_groups(joined, features, config)
    assert set(report["dimension"]) == {
        "maker_group", "model_year_group", "price_group",
        "mileage_availability_group",
    }
    hyundai = report.query("dimension == 'maker_group' and group == 'Hyundai'").iloc[0]
    assert hyundai["reference_action_count"] == 1
    assert hyundai["candidate_action_count"] == 0
    assert hyundai["material_deterioration"]


def test_mucars_subgroups_use_mad_mae_and_frozen_training_cutpoints():
    config_path = Path(__file__).parents[1] / "configs/mucars_bvival_v5_subgroups.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["primary_budget_fraction"] = 0.5
    config["minimum_group_size_for_safety_gate"] = 1
    joined = pd.DataFrame({
        "listing_id": ["a", "b"],
        "action_id": ["transmission", "transmission"],
        "target_log": np.log1p([50000.0, 160000.0]),
        "before_prediction_log": np.log1p([40000.0, 150000.0]),
        "after_prediction_log": np.log1p([50000.0, 140000.0]),
        "score_uncertainty_only": [2.0, 1.0],
        "score_mean_value": [-1.0, 2.0],
    })
    features = pd.DataFrame({
        "listing_id": ["a", "b"],
        "make": ["Renault", "Dacia"],
        "year": [2020, 2005],
        "mileage_km": [55000.0, np.nan],
    })
    report = evaluate_groups(joined, features, config)
    assert set(report["primary_loss_metric"]) == {"post_mae_price"}
    assert "at most 74000 MAD" in set(report["group"])
    renault = report.query("dimension == 'maker_group' and group == 'Renault'").iloc[0]
    assert renault["reference_post_mae_price"] == 0.0
    assert renault["candidate_post_mae_price"] == pytest.approx(10000.0)
    assert renault["material_deterioration"]
