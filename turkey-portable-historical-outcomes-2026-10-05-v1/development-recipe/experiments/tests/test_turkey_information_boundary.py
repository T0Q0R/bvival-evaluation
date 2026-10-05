import json
from pathlib import Path

import pytest

from turkey_information_boundary import initial_context, policy_context, valuation_state


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "configs/turkey_price_free_cohort_2026-09-30.json").read_text())


def context(config):
    return {f: "" for f in config["initial_visible_fields"]}


@pytest.mark.parametrize("field", ["kilometre", "vites_tipi", "yakit_tipi", "model", "motor_gucu",
                                   "motor_hacmi", "id", "record_key", "group_hash", "fiyat",
                                   "after_prediction_log", "value_absolute_price_error"])
def test_hidden_values_metadata_and_outcomes_cannot_enter_initial_context(field, config):
    with pytest.raises(ValueError, match="pre-action"):
        initial_context({**context(config), field: "FORBIDDEN"}, config)


def test_initial_state_hides_all_actions_and_single_reveal_changes_only_one(config):
    before = valuation_state(context(config), config)
    after = valuation_state(context(config), config, revealed_action="kilometre", revealed_value="123000")
    assert all(before[f] == "" and before["hidden_" + f] == 1 for f in config["action_fields"])
    assert after["kilometre"] == "123000" and after["hidden_kilometre"] == 0
    assert all(after[f] == before[f] for f in config["initial_visible_fields"])
    assert all(after[f] == "" and after["hidden_" + f] == 1 for f in ["vites_tipi", "yakit_tipi"])


def test_full_action_dictionary_is_not_accepted_as_single_reveal(config):
    with pytest.raises(ValueError, match="scalar"):
        valuation_state(context(config), config, revealed_action="kilometre",
                        revealed_value={"kilometre": "123000", "yakit_tipi": "dizel"})
    with pytest.raises(ValueError, match="explicitly selected"):
        valuation_state(context(config), config, revealed_value="123000")


def test_policy_context_uses_common_before_summaries_not_revealed_values(config):
    values = context(config)
    first = policy_context(values, config, before_prediction_log=10, before_disagreement_log=0.1, action_id="kilometre")
    second = policy_context(values, config, before_prediction_log=10, before_disagreement_log=0.1, action_id="yakit_tipi")
    assert {k: v for k, v in first.items() if k != "action_id"} == {k: v for k, v in second.items() if k != "action_id"}
    assert not set(config["action_fields"]) & set(first)
    with pytest.raises(ValueError, match="pre-action"):
        policy_context({**values, "yakit_tipi": "dizel"}, config, before_prediction_log=10, before_disagreement_log=0)


@pytest.mark.parametrize("prediction,spread", [(float("nan"), 0), (10, float("inf")), (10, -1)])
def test_invalid_before_summaries_fail_closed(prediction, spread, config):
    with pytest.raises(ValueError, match="finite"):
        policy_context(context(config), config, before_prediction_log=prediction, before_disagreement_log=spread)
