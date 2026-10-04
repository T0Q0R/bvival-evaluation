import pandas as pd

from build_autoscout24_quasi_duplicate_mask import QUASI_KEY, build_mask


def test_mask_uses_features_only_and_marks_exact_cross_split_keys():
    spec = {
        "make": "A", "model": "B", "registration_year": 2020,
        "mileage_km": 50000, "power_hp": 150,
        "body_type": "Sedan", "country_code": "DE",
    }
    rows = [
        {"listing_id": "train", "split": "train", **spec},
        {"listing_id": "test-same", "split": "test", **spec},
        {"listing_id": "test-different", "split": "test", **{**spec, "mileage_km": 70000}},
    ]
    frame = pd.DataFrame(rows)
    assert set(QUASI_KEY).issubset(frame.columns)
    mask = build_mask(frame).set_index("listing_id")
    assert bool(mask.loc["test-same", "quasi_key_seen_in_non_test"])
    assert not bool(mask.loc["test-different", "quasi_key_seen_in_non_test"])
    assert "train" not in mask.index
