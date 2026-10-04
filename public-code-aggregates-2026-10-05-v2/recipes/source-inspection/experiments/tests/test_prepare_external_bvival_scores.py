import json
from argparse import Namespace

import pandas as pd
import pytest

import prepare_external_bvival_scores as adapter
from prepare_external_bvival_scores import validate_provenance, validate_score_contract


def provenance():
    return {
        "source_name": "MUCars-2024",
        "method_name": "Example published-method adaptation",
        "paper_url": "https://example.org/paper",
        "implementation_url": "https://example.org/code",
        "implementation_commit": "abc123",
        "adaptation_description": "One-step regression adaptation; not exact published method.",
        "implementation_class": "independent_adaptation",
        "objective": "absolute_price_error",
        "analysis_status": "post_test_exploratory",
        "score_direction": "higher_is_better",
        "training_splits": ["train", "validation"],
        "evaluation_outcomes_used": False,
    }


def frames():
    universe = pd.DataFrame({"listing_id": ["a", "a", "b"], "action_id": ["fuel", "age", "fuel"]})
    scores = universe.copy()
    scores["score_external_afa"] = [0.3, 0.1, -0.2]
    return universe, scores


def test_accepts_complete_label_free_scores():
    universe, scores = frames()
    assert validate_score_contract(universe, scores).equals(scores)
    validate_provenance(provenance())


def test_rejects_missing_pair_and_duplicate():
    universe, scores = frames()
    with pytest.raises(ValueError, match="exactly"):
        validate_score_contract(universe, scores.iloc[:2])
    with pytest.raises(ValueError, match="duplicate"):
        validate_score_contract(universe, pd.concat([scores, scores.iloc[[0]]]))


def test_rejects_label_column_and_nonfinite_score():
    universe, scores = frames()
    with pytest.raises(ValueError, match="only"):
        validate_score_contract(universe, scores.assign(target_log=1.0))
    with pytest.raises(ValueError, match="finite"):
        validate_score_contract(universe, scores.assign(score_external_afa=[0.2, float("inf"), 0.1]))
    with pytest.raises(ValueError, match="only"):
        validate_score_contract(universe.assign(target_log=1.0), scores)


def test_cli_writes_audited_exploratory_scores(tmp_path, monkeypatch):
    universe, scores = frames()
    universe_path = tmp_path / "universe.csv"
    scores_path = tmp_path / "scores.csv"
    provenance_path = tmp_path / "provenance.json"
    output_dir = tmp_path / "output"
    universe.to_csv(universe_path, index=False)
    scores.to_csv(scores_path, index=False)
    provenance_path.write_text(json.dumps(provenance()), encoding="utf-8")
    monkeypatch.setattr(
        adapter,
        "parse_args",
        lambda: Namespace(
            candidate_universe=universe_path,
            external_scores=scores_path,
            provenance=provenance_path,
            output_dir=output_dir,
        ),
    )
    adapter.main()
    output_path = output_dir / "external_afa_scores_exploratory.csv"
    assert pd.read_csv(output_path).equals(scores)
    audit = json.loads((output_dir / "external_afa_audit.json").read_text(encoding="utf-8"))
    assert audit["output_scores_sha256"] == adapter.sha256(output_path)
    assert audit["candidate_listing_count"] == 2
    assert audit["adapter_status"] == "contract_checked_not_method_verified"


@pytest.mark.parametrize(
    "changed",
    [
        {"analysis_status": "frozen_prospective"},
        {"evaluation_outcomes_used": True},
        {"objective": "squared_log_error"},
        {"training_splits": ["train", "test"]},
        {"score_direction": "lower_is_better"},
    ],
)
def test_opened_source_contract_refuses_misleading_claims(changed):
    record = provenance()
    record.update(changed)
    with pytest.raises(ValueError):
        validate_provenance(record)
