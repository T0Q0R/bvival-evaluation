import copy
import json
from pathlib import Path

import numpy as np
import pytest

from build_turkey_price_free_manifest import digest, fold
from turkey_execution_contract import (
    ACTIONS, budget_count, classify_evidence, holm_adjust, nested_training_roles,
    paired_cluster_effect, price_is_valid, random_actions, select_actions,
    select_candidate, select_on_common_cohort, target_validity_inventory, validate_plan,
)


@pytest.fixture
def configs():
    directory = Path(__file__).parents[1] / "configs"
    return (json.loads((directory / "turkey_execution_plan_v1_2026-09-30.json").read_text()),
            json.loads((directory / "turkey_price_free_cohort_2026-09-30.json").read_text()))


def test_real_design_config_obeys_role_and_inference_contract(configs):
    validate_plan(*configs)


@pytest.mark.parametrize("path,value", [
    (("action_fields",), list(reversed(ACTIONS))),
    (("capacity_rule",), "reroute_after_price_filter"),
    (("valuation", "inner_folds"), 1),
    (("valuation", "family_selection"), "test_mae"),
    (("valuation", "numeric_fields"), ["fiyat"]),
    (("heads", "features"), ["kilometre"]),
    (("heads", "value_scores_keep_negative"), False),
    (("neural_comparator", "official_method_reproduction_completed"), True),
    (("neural_comparator", "required_before_test"), False),
    (("statistics", "multiplicity"), "none"),
    (("price_validity", "maximum_invalid_fraction_per_partition"), .2),
    (("release_gates", "currently_price_release_allowed"), True),
    (("release_gates", "test_rerouting_after_price_validity"), True),
])
def test_protocol_mutations_are_rejected(configs, path, value):
    plan, cohort = copy.deepcopy(configs)
    location = plan
    for key in path[:-1]:
        location = location[key]
    location[path[-1]] = value
    with pytest.raises(ValueError):
        validate_plan(plan, cohort)


def test_unbalanced_family_grid_is_rejected(configs):
    plan, cohort = copy.deepcopy(configs)
    plan["valuation"]["candidates"].pop()
    with pytest.raises(ValueError, match="four distinct"):
        validate_plan(plan, cohort)


@pytest.mark.parametrize("n,b,expected", [(7802, .05, 391), (7802, .1, 781), (7802, .2, 1561),
                                         (10, .1, 1), (1, .1, 1), (0, .1, 0)])
def test_capacity_uses_exact_decimal_ceiling(n, b, expected):
    assert budget_count(n, b) == expected


@pytest.mark.parametrize("value,valid", [
    ("1234.5", True), (" 1e3 ", True), (12, True), (0, False), ("-1", False),
    ("NaN", False), (float("inf"), False), ("1,200", False), ("₺1200", False),
    (True, False), (None, False), ("1e999", False), ("1e-999", False),
])
def test_price_validity_is_explicit_not_a_locale_guess(value, valid):
    assert price_is_valid(value) is valid


def test_validity_counts_do_not_change_feature_capacity_and_use_exact_one_percent_boundary():
    rows = [{"record_key": str(i), "split": "train", "oof_fold": str(i % 5)} for i in range(500)]
    valid = {r["record_key"]: True for r in rows}
    valid["0"] = False
    result = target_validity_inventory(rows, valid, released_splits=["train"])
    assert result["partitions"]["train"]["feature_eligible_records"] == 500
    assert result["partitions"]["train"]["price_valid_records"] == 499
    assert budget_count(result["partitions"]["train"]["feature_eligible_records"], .1) == 50
    valid["5"] = False
    with pytest.raises(ValueError, match="Training fold"):
        target_validity_inventory(rows, valid, released_splits=["train"])


def test_validity_inventory_cannot_silently_drop_keys_or_accept_price_values():
    rows = [{"record_key": "a", "split": "test", "oof_fold": ""}]
    for invalid in [{}, {"a": 12345}, {"a": True, "unreleased": True}]:
        with pytest.raises(ValueError, match="universe"):
            target_validity_inventory(rows, invalid, released_splits=["test"])
    with pytest.raises(ValueError, match="exceeds 1%"):
        target_validity_inventory(rows, {"a": False}, released_splits=["test"])


def test_negative_scores_still_use_identical_full_capacity_and_action_ties():
    selected = select_actions(["b", "a", "c"], ACTIONS, [[-3, -3, -4], [-3, -3, -4], [-9, -8, -7]], .5)
    assert selected == [("a", ACTIONS[0]), ("b", ACTIONS[0])]
    assert select_actions(["x"], ACTIONS, [[0, 0, 0]], .1) == [("x", ACTIONS[0])]


def test_action_allocation_is_order_invariant_and_fixed_field_uses_its_score():
    keys, scores = ["c", "a", "b"], [[10, 2, 1], [4, 8, 0], [1, 7, 3]]
    assert select_actions(keys, ACTIONS, scores, .5) == [("c", ACTIONS[0]), ("a", ACTIONS[1])]
    assert select_actions(keys[::-1], ACTIONS, scores[::-1], .5) == select_actions(keys, ACTIONS, scores, .5)
    assert select_actions(keys, ACTIONS, scores, .5, fixed_action=ACTIONS[1]) == [("a", ACTIONS[1]), ("b", ACTIONS[1])]


@pytest.mark.parametrize("keys,scores", [(["a", "a"], [[0, 0, 0], [0, 0, 0]]),
                                        (["a"], [[0, float("nan"), 0]]), (["a"], [[1, 2]])])
def test_bad_score_universe_is_rejected(keys, scores):
    with pytest.raises(ValueError):
        select_actions(keys, ACTIONS, scores, .1)


def test_random_allocation_is_seeded_order_independent_and_has_one_action_each():
    keys = [str(i) for i in range(100)]
    selected = random_actions(keys, ACTIONS, .1, namespace="synthetic", seed=42)
    assert selected == random_actions(keys[::-1], ACTIONS, .1, namespace="synthetic", seed=42)
    assert len(selected) == len({key for key, _ in selected}) == 10
    assert all(action in ACTIONS for _, action in selected)


def test_crossed_selector_cannot_change_common_cohort():
    assert select_on_common_cohort(["a", "b", "c"], ACTIONS, [[1, 2, 3], [9, 0, 0], [0, 8, 0]], ["a", "c"]) == [("a", ACTIONS[2]), ("c", ACTIONS[1])]
    with pytest.raises(ValueError, match="Common cohort"):
        select_on_common_cohort(["a"], ACTIONS, [[0, 0, 0]], ["unknown"])


def test_nested_roles_remove_entire_outer_groups_and_keep_augmented_rows_together(configs):
    _, cohort = configs
    rows = []
    for i in range(40):
        group = digest(["synthetic group", i // 2])
        rows.append({"record_key": str(i), "group_hash": group, "split": "train", "oof_fold": str(fold(group, cohort))})
    for outer in range(5):
        roles = nested_training_roles(rows, cohort, outer_fold=outer, namespace="synthetic nested")
        per_group = {}
        for row in roles:
            pair = row["role"], row["inner_fold"]
            assert per_group.setdefault(row["group_hash"], pair) == pair
            assert (row["inner_fold"] == "") == (row["role"] == "outer_holdout")
        heldout_keys = {row["record_key"] for row in rows if int(row["oof_fold"]) == outer}
        assert heldout_keys == {row["record_key"] for row in roles if row["role"] == "outer_holdout"}


def test_nested_roles_reject_validation_and_saved_fold_tampering(configs):
    _, cohort = configs
    group = digest(["synthetic"])
    row = {"record_key": "a", "group_hash": group, "split": "validation", "oof_fold": str(fold(group, cohort))}
    with pytest.raises(ValueError, match="train rows"):
        nested_training_roles([row], cohort, outer_fold=0, namespace="s")
    row.update(split="train", oof_fold=str((fold(group, cohort) + 1) % 5))
    with pytest.raises(ValueError, match="frozen group"):
        nested_training_roles([row], cohort, outer_fold=0, namespace="s")


def test_model_selection_ties_are_predefined_and_nonfinite_values_rejected():
    assert select_candidate({"z": 1.0, "a": 1.0}) == "a"
    with pytest.raises(ValueError):
        select_candidate({"a": float("nan")})


def test_cluster_bootstrap_preserves_listing_not_group_weighted_estimand():
    # One large group of three listings: listing-weighted MAE is 8, not 6.
    result = paired_cluster_effect([10, 10, 10, 2], [9, 9, 9, 1], ["big", "big", "big", "small"], repetitions=1000)
    assert result["reference_mae"] == 8
    assert result["candidate_mae"] == 7
    assert result["relative_effect_percent"] == 12.5
    assert result["listing_count"] == 4 and result["attribute_group_count"] == 2
    assert result["relative_ci95_percent"] == [10.0, 50.0]
    assert result["centered_bootstrap_p_two_sided"] == pytest.approx(1 / 1001)


def test_cluster_bootstrap_null_and_negative_effect_are_not_hidden():
    equal = paired_cluster_effect([2, 3, 4], [2, 3, 4], ["a", "b", "c"], repetitions=1000)
    assert equal["relative_effect_percent"] == 0
    assert equal["relative_ci95_percent"] == [0, 0]
    assert equal["centered_bootstrap_p_two_sided"] == 1
    worse = paired_cluster_effect([2, 3, 4], [3, 4, 5], ["a", "b", "c"], repetitions=1000)
    assert worse["relative_effect_percent"] < 0 and worse["relative_ci95_percent"][1] < 0


def test_cluster_bootstrap_is_paired_and_order_invariant():
    a = paired_cluster_effect([3, 5, 7, 9], [2, 5, 6, 7], ["x", "x", "y", "z"], repetitions=1000)
    b = paired_cluster_effect([9, 7, 5, 3], [7, 6, 5, 2], ["z", "y", "x", "x"], repetitions=1000)
    assert a == b


@pytest.mark.parametrize("reference,candidate,groups", [([1], [1], ["a"]),
                                                       ([0, 0], [1, 1], ["a", "b"]),
                                                       ([1, 2], [1], ["a", "b"]),
                                                       ([1, 2], [1, -1], ["a", "b"])])
def test_unusable_bootstrap_inputs_fail_closed(reference, candidate, groups):
    with pytest.raises(ValueError):
        paired_cluster_effect(reference, candidate, groups, repetitions=1000)


def test_holm_adjustment_returns_original_test_order():
    assert holm_adjust([.03, .01]) == [.03, .02]
    assert holm_adjust([.8, .6]) == [1, 1]


def test_small_effect_and_lower_bound_are_separate_claims():
    seeds = {13: .2, 42: .3, 2026: .4}
    small = classify_evidence(.28, .04, .01, seeds)
    assert small["directional_support"] and not small["practical_screen_support"]
    point_only = classify_evidence(2.32, 1.75, .01, seeds)
    assert point_only["practical_screen_support"] and not point_only["ci_lower_above_practical_reference"]
    assert not classify_evidence(2.32, 1.75, .01, {**seeds, 42: -.1})["directional_support"]
    assert not classify_evidence(2.32, 1.75, .06, seeds)["directional_support"]
    assert not point_only["registered_confirmatory_claim"]
