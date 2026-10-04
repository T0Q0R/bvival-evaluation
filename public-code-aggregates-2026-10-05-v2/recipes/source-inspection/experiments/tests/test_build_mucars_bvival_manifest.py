import numpy as np
import pandas as pd

from build_mucars_bvival_manifest import (
    build_manifest,
    equipment_count,
    mileage_band_numeric,
)


def test_mucars_field_parsers_use_bands_and_equipment_counts():
    assert mileage_band_numeric("200 000 - 249 999") == 224999.5
    assert mileage_band_numeric("Plus de 500 000") == 500000.0
    assert equipment_count("['Airbags', 'GPS']") == 2.0
    assert np.isnan(equipment_count("malformed"))


def test_manifest_filters_price_and_new_cars_then_deduplicates_nonprice():
    base = {
        "Brand": "Toyota", "Model": "Corolla", "Year": "2018",
        "Condition": "Good", "Mileage": "100 000 - 109 999",
        "Gearbox": "Manual", "Fiscal Power": "6 CV", "Fuel": "Diesel",
        "Equipment": "['Airbags']", "Number of Doors": 5,
        "Origin": "Local", "First Owner": "No", "Location": "Rabat",
        "Sector": "Center", "Price": 100000,
    }
    source = pd.DataFrame([
        base,
        {**base, "Price": 110000},
        {**base, "Model": "Polo", "Price": 1},
        {**base, "Model": "Civic", "Condition": "New"},
    ])
    manifest, audit = build_manifest(source)
    assert len(manifest) == 1
    assert audit["duplicate_nonprice_fingerprints_removed"] == 1
    assert manifest["price_value_MAD"].iloc[0] == 100000
    assert manifest["mileage_km"].iloc[0] == 104999.5
    assert audit["test_performance_opened"] is False
