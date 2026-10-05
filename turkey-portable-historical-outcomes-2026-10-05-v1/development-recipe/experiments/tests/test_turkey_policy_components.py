import copy
import json
from pathlib import Path

import numpy as np
import pytest

from turkey_execution_contract import ACTIONS, budget_count
from turkey_policy_components import allocate_policies, context_frame, fit_policy_heads, make_head, post_mae, score_head, validate_material


def material(n, prefix, plan):
    contexts = [{"marka": "brand", "seri": "series", "yil": str(2010 + i % 10), "kasa_tipi": "sedan",
                 "renk": "white", "kimden": "private", "boyali_sayisi": "", "degisen_sayisi": "",
                 "before_prediction_log": 7.0 + i / 100, "before_disagreement_log": .01} for i in range(n)]
    before = np.arange(n) + 100.0
    after = np.repeat(before[:, None], 3, axis=1) - np.array([10.0, -5.0, 20.0])
    return {"keys": [f"{prefix}_{i:03}" for i in range(n)], "contexts": contexts, "valid_mask": np.ones(n, dtype=bool),
            "target_origin": "train_nested_oof" if prefix == "train" else "validation_train_only_base",
            "targets": {"before_absolute_error": before, "after_absolute_error": after,
                        "signed_gain": before[:, None] - after, "common_training_scale": float(before.mean())}}


@pytest.fixture
def fixture():
    directory = Path(__file__).parents[1] / "configs"
    plan = json.loads((directory / "turkey_execution_plan_v1_2026-09-30.json").read_text())
    config = json.loads((directory / "turkey_price_free_cohort_2026-09-30.json").read_text())
    return material(80, "train", plan), material(20, "validation", plan), plan, config


class MeanHead:
    def fit(self, X, target):
        assert set(X).isdisjoint({"fiyat", "record_key", *ACTIONS})
        self.mean = float(np.mean(target))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean)


def test_head_schedule_and_train_only_rows(fixture):
    train, valid, plan, config = fixture
    seen = []
    def factory(plan, candidate, seed, categorical):
        seen.append((candidate, seed, categorical))
        return MeanHead()
    result = fit_policy_heads(train, valid, plan, config, factory=factory)
    assert len(seen) == len(result["fit_audit"]) == 21
    assert len(result["candidate_trials"]) == 12
    assert result["chosen_candidates"] == {k: "head01" for k in ("risk", "value", "posterror")}
    assert all(not a["validation_targets_in_fit"] and a["fit_record_count"] == 80 for a in result["fit_audit"])
    assert all(a["fit_rows"] == (80 if a["kind"] == "risk" else 240) for a in result["fit_audit"])
    assert result["fixed_field"] == result["global_field"] == ACTIONS[2]


@pytest.mark.parametrize("extra", ["fiyat", "kilometre", "vites_tipi", "yakit_tipi", "record_key", "after_prediction_log", "signed_gain"])
def test_policy_context_rejects_hidden_or_outcome_extras(fixture, extra):
    train, _, plan, config = fixture
    train["contexts"][0][extra] = "leaked"
    with pytest.raises(ValueError, match="allowlist"):
        context_frame(train["contexts"], plan, config)


def test_action_head_only_adds_known_action_id(fixture):
    train, _, plan, config = fixture
    with pytest.raises(ValueError, match="Unknown candidate"):
        context_frame([{**train["contexts"][0], "action_id": "engine"}], plan, config, action_head=True)


def test_negative_value_is_not_clipped(fixture):
    train, _, plan, config = fixture
    model = MeanHead()
    model.mean = -2.0
    assert (score_head(model, train["contexts"], plan, config, kind="value", scale=100) == -200).all()
    assert (score_head(model, train["contexts"], plan, config, kind="posterror", scale=100) == 0).all()


@pytest.mark.parametrize("mutation", ["wrong_origin", "overlap", "gain_identity", "scale", "mask_shape", "target_extra"])
def test_bad_material_fails_before_fit(fixture, mutation):
    train, valid, plan, config = fixture
    if mutation == "wrong_origin":
        train["target_origin"] = "in_sample"
    elif mutation == "overlap":
        valid["keys"][0] = train["keys"][0]
    elif mutation == "gain_identity":
        train["targets"]["signed_gain"][0, 0] = 999
    elif mutation == "scale":
        train["targets"]["common_training_scale"] = 1.0
    elif mutation == "mask_shape":
        train["valid_mask"] = train["valid_mask"][:-1]
    else:
        train["targets"]["price"] = 123
    with pytest.raises(ValueError):
        fit_policy_heads(train, valid, plan, config, factory=lambda *args: pytest.fail("Invalid input reached fitting"))


def test_allocations_exact_even_negative_scores_and_same_risk_cohort(fixture):
    _, valid, plan, _ = fixture
    n = len(valid["keys"])
    result = allocate_policies(valid["keys"], np.zeros(n), np.full((n, 3), -99), np.full((n, 3), 999),
                               np.repeat([[.1, .2, .7]], n, axis=0), plan, fixed_field=ACTIONS[1], global_field=ACTIONS[2])
    for budget, policies in result.items():
        assert not policies["no_acquisition"]
        assert all(len(a) == budget_count(n, float(budget)) for p, a in policies.items() if p != "no_acquisition")
        common = {k for k, _ in policies["risk_fixed_validation_best"]}
        assert all({k for k, _ in policies[p]} == common for p in ("risk_global_train_field", "risk_benefit_field", "risk_neural_field"))
        assert all(a == ACTIONS[0] for _, a in policies["benefit"])
        assert all(a == ACTIONS[2] for _, a in policies["risk_neural_field"])


def test_invalid_target_selected_does_not_reroute(fixture):
    _, valid, _, _ = fixture
    valid["valid_mask"][0] = False
    for field in ("before_absolute_error", "after_absolute_error", "signed_gain"):
        valid["targets"][field] = valid["targets"][field][1:]
    assert post_mae(valid, [(valid["keys"][0], ACTIONS[2])]) == pytest.approx(valid["targets"]["before_absolute_error"].mean())


def test_real_catboost_risk_and_action_heads_fit_and_reload(fixture, tmp_path):
    import joblib
    train, valid, plan, config = fixture
    for candidate in plan["heads"]["candidates"]:
        candidate.update(iterations=3, depth=2)
    result = fit_policy_heads(train, valid, plan, config, model_directory=tmp_path)
    assert len(list(tmp_path.glob("*.joblib"))) == 9
    for kind in ("risk", "value", "posterror"):
        loaded = joblib.load(tmp_path / f"{kind}_seed_42.joblib")
        actual = score_head(loaded, valid["contexts"], plan, config, kind=kind, scale=result["common_training_scale"])
        assert np.array_equal(actual, result["per_seed_scores"][kind]["42"])


def test_validation_target_perturbation_never_enters_fit_supervision(fixture):
    train, valid, plan, config = fixture
    seen = []
    class Spy(MeanHead):
        def fit(self, X, target):
            # Stable JSON nulls avoid NaN != NaN in a missing-field comparison.
            seen.append((X.to_json(orient="records"), np.asarray(target).tolist()))
            return super().fit(X, target)
    original = fit_policy_heads(train, valid, plan, config, factory=lambda *args: Spy())
    first = copy.deepcopy(seen)
    seen.clear()
    changed = copy.deepcopy(valid)
    for name in ("before_absolute_error", "after_absolute_error", "signed_gain"):
        changed["targets"][name] *= 100
    changed["targets"]["common_training_scale"] *= 100
    altered = fit_policy_heads(train, changed, plan, config, factory=lambda *args: Spy())
    assert seen == first
    assert original["common_training_scale"] == altered["common_training_scale"]


def test_direct_posterror_ranks_predicted_difference_not_error_or_probability(fixture):
    _, _, plan, _ = fixture
    keys = ["a", "b", "c", "d"]
    risk = np.array([100., 90., 80., 70.])
    posterror = np.array([[100., 100., 100.], [100., 100., 100.], [20., 10., 30.], [10., 20., 30.]])
    value = np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [500., 2., 3.]])
    probability = np.repeat([[0., 0., 1.]], 4, axis=0)
    policies = allocate_policies(keys, risk, value, posterror, probability, plan, fixed_field=ACTIONS[0], global_field=ACTIONS[1])["0.10"]
    assert policies["direct_posterror"] == [("c", ACTIONS[1])]
    assert policies["benefit"] == [("d", ACTIONS[0])]
    assert policies["risk_fixed_validation_best"] == [("a", ACTIONS[0])]
    assert policies["risk_neural_field"] == [("a", ACTIONS[2])]


def test_synthetic_reduction_keeps_source_plan_unchanged(fixture):
    from run_turkey_policy_synthetic_integration import smoke_plan
    _, _, plan, config = fixture
    original = copy.deepcopy(plan)
    reduced = smoke_plan(plan)
    assert plan == original and reduced["neural_comparator"]["maximum_epochs"] == 3
    assert plan["neural_comparator"]["maximum_epochs"] == 100
