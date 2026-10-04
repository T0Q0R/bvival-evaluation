import pandas as pd

from audit_autoscout24_candidate import is_plausible_vin
from build_autoscout24_bvival_manifest import build_manifest


def test_vin_screen_excludes_placeholders_and_accepts_structural_vin():
    assert is_plausible_vin("WAUZZZF30L1044529")
    assert not is_plausible_vin("00000000000000000")
    assert not is_plausible_vin("M10BM2VP000C092")


def test_manifest_uses_numeric_raw_mileage_and_collapses_valid_vin_duplicates():
    base = {
        "vin": "WAUZZZF30L1044529", "price": 20_000, "price_currency": "EUR",
        "is_used": True, "make": "Audi", "model": "A3",
        "registration_date": "2020-03-01", "vehicle_type": "Car",
        "body_type": "Sedan", "power_hp": 150, "nr_seats": 5, "nr_doors": 4,
        "country_code": "DE", "seller_is_dealer": True, "nr_prev_owners": 1,
        "mileage_km_raw": 50_000, "transmission": "Automatic",
        "fuel_category": "Gasoline", "has_full_service_history": True,
    }
    source = pd.DataFrame([
        {**base, "id": "ad-a"},
        {**base, "id": "ad-b", "price": 21_000, "country_code": "NL"},
        {**base, "id": "ad-c", "vin": "00000000000000000", "mileage_km_raw": 75_000},
        {**base, "id": "ad-future", "vin": None, "registration_date": "2026-01-01"},
    ])
    manifest, audit = build_manifest(source)
    assert len(manifest) == 2
    assert audit["duplicate_group_rows_removed"] == 1
    assert audit["pre_dedup_filter_count"] == 3
    assert sorted(manifest["mileage_km"].tolist()) == [50_000, 75_000]
    assert manifest["listing_id"].is_unique
    assert audit["test_performance_opened"] is False
