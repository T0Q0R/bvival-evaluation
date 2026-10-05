import json
from pathlib import Path

import numpy as np
import pytest

from turkey_information_boundary import valuation_state
from turkey_valuation_components import (
    ValuationStateGuard, ensemble_prediction, fit_valuation_model,
    make_valuation_model, signed_action_targets,
)


@pytest.fixture
def configs():
    directory = Path(__file__).parents[1] / "configs"
    return (json.loads((directory / "turkey_execution_plan_v1_2026-09-30.json").read_text()),
            json.loads((directory / "turkey_price_free_cohort_2026-09-30.json").read_text()))


def initial(i=0):
    return {"marka": "synthetic_brand_" + str(i % 2), "seri": "synthetic_series", "yil": str(2010 + i),
            "kasa_tipi": "sedan", "renk": "white", "kimden": "private", "boyali_sayisi": "", "degisen_sayisi": ""}


def synthetic_states(cohort):
    states, prices = [], []
    for i in range(8):
        states.append(valuation_state(initial(i), cohort))
        prices.append(1000 + i * 100)
        for action, value in [("kilometre", str(10000 + i)), ("vites_tipi", "automatic"), ("yakit_tipi", "petrol")]:
            states.append(valuation_state(initial(i), cohort, revealed_action=action, revealed_value=value))
            prices.append(1000 + i * 100)
    return states, prices


@pytest.mark.parametrize("candidate", ["cb01", "et01", "hg01"])
def test_each_declared_family_fits_only_synthetic_states_and_predicts_finite(candidate, configs):
    plan, cohort = configs
    states, prices = synthetic_states(cohort)
    model = fit_valuation_model(make_valuation_model(plan, candidate, 42), states, prices)
    before = valuation_state(initial(0), cohort)
    unseen = valuation_state({**initial(0), "marka": "validation_only_brand"}, cohort)
    predictions = model.predict([before, unseen])
    assert predictions.shape == (2,) and np.isfinite(predictions).all()
    if candidate != "cb01":
        fitted = model.named_steps["train_fitted_preprocess"]
        encoder = fitted.named_transformers_["categorical"]
        assert all("validation_only_brand" not in values for values in encoder.categories_)
        matrix = model.named_steps["float32"].transform(fitted.transform(model.named_steps["state_guard"].transform(states)))
        assert matrix.dtype == np.float32 and np.isfinite(matrix).all()


@pytest.mark.parametrize("candidate", ["cb01", "et01", "hg01"])
def test_every_family_refuses_metadata_outcomes_and_full_reveal(candidate, configs):
    plan, cohort = configs
    states, prices = synthetic_states(cohort)
    model = make_valuation_model(plan, candidate, 42)
    bad = [{**states[0], "fiyat": "DO_NOT_PARSE"}]
    with pytest.raises(ValueError, match="exactly the allowed"):
        fit_valuation_model(model, bad, [1000])
    bad = [{**states[0], "kilometre": "20000", "hidden_kilometre": 0,
            "vites_tipi": "automatic", "hidden_vites_tipi": 0}]
    with pytest.raises(ValueError, match="At most one"):
        fit_valuation_model(model, bad, [1000])


def test_numeric_missing_indicators_preserve_all_empty_optional_fields(configs):
    plan, cohort = configs
    states, prices = synthetic_states(cohort)
    model = fit_valuation_model(make_valuation_model(plan, "et01", 42), states, prices)
    imputer = model.named_steps["train_fitted_preprocess"].named_transformers_["numeric"]
    for field in ("boyali_sayisi", "degisen_sayisi"):
        assert plan["valuation"]["numeric_fields"].index(field) in imputer.indicator_.features_
    assert imputer.keep_empty_features


def test_guard_rejects_actual_hidden_value_and_nonfinite_numeric(configs):
    plan, cohort = configs
    guard = ValuationStateGuard(plan["valuation"]["categorical_fields"], plan["valuation"]["numeric_fields"])
    state = valuation_state(initial(), cohort)
    with pytest.raises(ValueError, match="Hidden field"):
        guard.transform([{**state, "yakit_tipi": "diesel"}])
    with pytest.raises(ValueError, match="must be finite"):
        guard.transform([{**state, "yil": "NaN"}])


def test_factory_has_no_unplanned_seed_or_candidate(configs):
    plan, _ = configs
    with pytest.raises(ValueError, match="seed"):
        make_valuation_model(plan, "cb01", 999)
    with pytest.raises(ValueError, match="candidate"):
        make_valuation_model(plan, "new", 42)


def test_ensemble_averages_log_then_expm1_and_population_spread():
    result = ensemble_prediction([[1, -3], [2, -2], [3, -1]])
    assert result["prediction_log"].tolist() == [2, 0]
    assert result["prediction_price"].tolist() == [pytest.approx(np.expm1(2)), 0]
    assert result["disagreement_log"].tolist() == [pytest.approx(np.sqrt(2 / 3))] * 2
    assert result["prediction_price"][0] != pytest.approx(np.expm1([1, 2, 3]).mean())
    with pytest.raises(ValueError, match="Overflow"):
        ensemble_prediction([[1000], [1000], [1000]])


def test_signed_targets_keep_harmful_actions_and_share_one_scale():
    result = signed_action_targets([100, 200], np.log1p([90, 180]), np.log1p([[100, 70, 110], [200, 170, 160]]))
    assert result["signed_gain"] == pytest.approx(np.array([[10, -20, 0], [20, -10, -20]]))
    assert result["common_training_scale"] == pytest.approx(15)


def test_invalid_fit_targets_and_shape_do_not_reach_training(configs):
    plan, cohort = configs
    state = valuation_state(initial(), cohort)
    model = make_valuation_model(plan, "cb01", 42)
    with pytest.raises(ValueError, match="aligned"):
        fit_valuation_model(model, [state], [0])
    with pytest.raises(ValueError, match="complete three-action"):
        signed_action_targets([100], [1], [[1, 2]])
