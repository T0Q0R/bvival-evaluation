import numpy as np
import pandas as pd

from build_jucars_bvival_manifest import ACTION_FIELDS, build_manifest, split_from_listing_id


def fixture_frame():
    rows = []
    for index in range(30):
        rows.append(
            {
                "price_value_JOD": 5000.0 + index,
                "make": f"make-{index % 3}",
                "model": f"model-{index}",
                "year": 2010 + index % 10,
                "mileage_km": 10000.0 + index,
                "condition": "Used",
                "transmission": "Automatic" if index % 2 else "Manual",
                "fuel_type": "Gasoline",
                "feat_body_condition": "Excellent",
                "city": "Amman",
            }
        )
    rows.append(dict(rows[0]))
    rows.append({**rows[1], "condition": "New"})
    rows.append({**rows[2], "price_value_JOD": np.nan})
    return pd.DataFrame(rows)


def test_manifest_deduplicates_non_target_fingerprints_and_splits_ids():
    manifest, audit = build_manifest(fixture_frame())
    assert manifest["listing_id"].is_unique
    assert audit["conservative_non_target_duplicate_rows_removed"] == 1
    assert audit["retained_used_listing_count"] == 30
    assert audit["cross_split_listing_id_overlap"] == 0
    assert set(manifest["split"]).issubset({"train", "validation", "calibration", "test"})
    assert audit["performance_metrics_computed"] is False


def test_price_does_not_determine_listing_fingerprint():
    frame = fixture_frame().iloc[:1].copy()
    alternate = frame.copy()
    alternate["price_value_JOD"] = 99999.0
    combined = pd.concat([frame, alternate], ignore_index=True)
    manifest, audit = build_manifest(combined)
    assert len(manifest) == 1
    assert audit["conservative_non_target_duplicate_rows_removed"] == 1


def test_action_fields_are_all_audited():
    _, audit = build_manifest(fixture_frame())
    assert set(audit["action_eligibility_counts"]) == set(ACTION_FIELDS)


def test_split_assignment_is_deterministic():
    identifier = "a" * 64
    assert split_from_listing_id(identifier) == split_from_listing_id(identifier)
