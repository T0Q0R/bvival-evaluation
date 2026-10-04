import pandas as pd
import pytest

from join_bvival_evaluation_outcomes import (
    build_evaluation_outcomes,
    file_sha256,
    verify_frozen_score_file,
    verify_score_universe,
)


def fixtures():
    pairs = pd.DataFrame(
        {
            "listing_id": ["a", "a", "b"],
            "action_id": ["fuel", "mileage", "fuel"],
            "before_prediction_log": [8.0, 8.0, 9.0],
            "after_prediction_log": [8.1, 8.2, 8.9],
        }
    )
    labels = pd.DataFrame({"listing_id": ["a", "b"], "target_log": [8.15, 8.95]})
    scores = pairs[["listing_id", "action_id"]].copy()
    scores["score"] = [0.1, 0.2, 0.3]
    return pairs, labels, scores


def test_outcome_join_requires_identical_frozen_score_universe():
    pairs, labels, scores = fixtures()
    with pytest.raises(ValueError, match="universes differ"):
        build_evaluation_outcomes(
            prediction_pairs=pairs,
            labels=labels,
            policy_scores=scores.iloc[:-1],
        )


def test_outcome_join_emits_locked_evaluator_schema():
    pairs, labels, scores = fixtures()
    output = build_evaluation_outcomes(
        prediction_pairs=pairs,
        labels=labels,
        policy_scores=scores,
    )
    assert output.columns.tolist() == [
        "listing_id",
        "action_id",
        "target_log",
        "before_prediction_log",
        "after_prediction_log",
    ]
    assert output["target_log"].tolist() == [8.15, 8.15, 8.95]


def test_outcome_join_rejects_missing_sealed_label():
    pairs, labels, scores = fixtures()
    with pytest.raises(ValueError, match="no sealed target"):
        build_evaluation_outcomes(
            prediction_pairs=pairs,
            labels=labels.iloc[:1],
            policy_scores=scores,
        )


def test_secondary_score_freeze_rejects_hash_and_universe_changes(tmp_path):
    _, _, scores = fixtures()
    path = tmp_path / "scores.csv"
    scores.to_csv(path, index=False)
    verified, actual = verify_frozen_score_file(path, file_sha256(path))
    assert actual == file_sha256(path)
    verify_score_universe(scores, verified)
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_frozen_score_file(path, "0" * 64)
    with pytest.raises(ValueError, match="universes differ"):
        verify_score_universe(scores, verified.iloc[:-1])
