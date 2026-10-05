import copy
import json
from pathlib import Path

import numpy as np
import pytest

from build_turkey_price_free_manifest import digest, fold
from train_turkey_nested_valuation import four_states, policy_training_material, run_nested_valuation, validate_dataset
from turkey_execution_contract import ACTIONS


def synthetic_dataset(config, n_train=160, n_validation=20):
    rows, initial, actions, prices, validity = [], {}, {}, {}, {}
    for i in range(n_train + n_validation):
        split = "train" if i < n_train else "validation"
        group = digest(["synthetic_train" if split == "train" else "synthetic_validation", i // 2])
        key = digest(["synthetic_record", i])
        rows.append({"record_key": key, "group_hash": group, "profile_hash": digest(["synthetic_profile", i]),
                     "split": split, "oof_fold": str(fold(group, config)) if split == "train" else ""})
        initial[key] = {"marka": "synthetic_brand_" + str(i % 2), "seri": "synthetic_series", "yil": str(2010 + i % 10),
                        "kasa_tipi": "sedan", "renk": "white", "kimden": "private",
                        "boyali_sayisi": "", "degisen_sayisi": ""}
        actions[key] = dict(zip(ACTIONS, [str(10000 + i * 100), "otomatik", "benzin"]))
        prices[key], validity[key] = 1000 + i * 10, True
    return {"rows": rows, "initial": initial, "actions": actions, "prices": prices, "validity": validity}


class SyntheticMeanModel:
    def __init__(self, bias):
        self.bias = bias

    def fit(self, states, y):
        assert all("record_key" not in state and "fiyat" not in state for state in states)
        assert all(sum(1 - state["hidden_" + action] for action in ACTIONS) <= 1 for state in states)
        self.log_mean = float(np.mean(y))
        return self

    def predict(self, states):
        return np.full(len(states), self.log_mean + self.bias)


def synthetic_factory(plan, candidate, seed):
    return SyntheticMeanModel((int(candidate[-2:]) - 1) * .001 + seed / 1e8)


@pytest.fixture
def material():
    directory = Path(__file__).parents[1] / "configs"
    plan = json.loads((directory / "turkey_execution_plan_v1_2026-09-30.json").read_text())
    config = json.loads((directory / "turkey_price_free_cohort_2026-09-30.json").read_text())
    return synthetic_dataset(config), plan, config


def test_orchestration_runs_exact_258_fits_and_complete_four_state_oof(material):
    data, plan, config = material
    result = run_nested_valuation(data, plan, config, model_factory=synthetic_factory)
    assert result["fit_count"] == 258 and len(result["candidate_trials"]) == 72
    assert len(result["oof"]) == 160 and len(result["validation"]) == 20
    assert not result["validation_targets_used_for_base_selection"]
    assert not result["policy_neural_or_test_stage_completed"]
    assert all(row["fit_and_excluded_overlap"] == 0 and row["fit_state_count"] == row["fit_record_count"] * 4 for row in result["fit_audit"])
    for prediction in result["oof"].values():
        assert len(prediction["after_prediction_log"]) == 3
        assert np.shape(prediction["per_seed_state_logs"]) == (3, 4)


def test_changing_outer_holdout_targets_cannot_change_its_selected_model_or_oof_predictions(material):
    data, plan, config = material
    original = run_nested_valuation(data, plan, config, model_factory=synthetic_factory)
    changed = copy.deepcopy(data)
    heldout = {row["record_key"] for row in data["rows"] if row["split"] == "train" and row["oof_fold"] == "0"}
    for key in heldout:
        changed["prices"][key] *= 100
    altered = run_nested_valuation(changed, plan, config, model_factory=synthetic_factory)
    assert original["chosen_candidates"]["outer_0"] == altered["chosen_candidates"]["outer_0"]
    assert {key: original["oof"][key] for key in heldout} == {key: altered["oof"][key] for key in heldout}
    assert [row for row in original["candidate_trials"] if row["stage"] == "outer_0"] == [row for row in altered["candidate_trials"] if row["stage"] == "outer_0"]


def test_validation_targets_do_not_select_or_fit_shared_valuation_model(material):
    data, plan, config = material
    original = run_nested_valuation(data, plan, config, model_factory=synthetic_factory)
    changed = copy.deepcopy(data)
    for row in changed["rows"]:
        if row["split"] == "validation":
            changed["prices"][row["record_key"]] *= 100
    altered = run_nested_valuation(changed, plan, config, model_factory=synthetic_factory)
    assert original["chosen_candidates"] == altered["chosen_candidates"]
    assert original["oof"] == altered["oof"] and original["validation"] == altered["validation"]


@pytest.mark.parametrize("mutation", ["test_label", "hidden_input", "wrong_fold", "cross_group", "missing_action"])
def test_bad_data_is_refused_before_any_model_fit(material, mutation):
    data, plan, config = copy.deepcopy(material)
    key = data["rows"][0]["record_key"]
    if mutation == "test_label":
        data["rows"][0]["split"] = "test"
    elif mutation == "hidden_input":
        data["initial"][key]["kilometre"] = "10000"
    elif mutation == "wrong_fold":
        data["rows"][0]["oof_fold"] = str((int(data["rows"][0]["oof_fold"]) + 1) % 5)
    elif mutation == "cross_group":
        data["rows"][-1]["group_hash"] = data["rows"][0]["group_hash"]
    else:
        del data["actions"][key]["yakit_tipi"]
    with pytest.raises(ValueError):
        run_nested_valuation(data, plan, config, model_factory=lambda *a: pytest.fail("Must reject before fitting"))


def test_four_state_features_never_contain_targets_and_training_context_cannot_have_acquisition_values(material):
    data, plan, config = material
    result = run_nested_valuation(data, plan, config, model_factory=synthetic_factory)
    material = policy_training_material(data, result, config)
    for context in material["pre_action_contexts"].values():
        assert set(context) == set(config["initial_visible_fields"] + ["before_prediction_log", "before_disagreement_log"])
        assert not set(ACTIONS) & set(context)
    keys = material["record_keys"][:2]
    states = four_states(data, keys, config)
    assert len(states) == 8 and all("price" not in state for state in states)
    assert material["signed_gain"].shape == (160, 3)


def test_initial_metadata_and_extra_price_keys_are_not_silently_dropped(material):
    data, plan, config = material
    changed = copy.deepcopy(data)
    changed["prices"]["not_a_development_record"] = 1234
    with pytest.raises(ValueError, match="Targets"):
        validate_dataset(changed, plan, config)
