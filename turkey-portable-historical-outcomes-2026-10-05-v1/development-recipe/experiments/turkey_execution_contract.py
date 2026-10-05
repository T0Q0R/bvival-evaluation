"""Executable design components, not a label reader or training runner.

Allocation consumes pre-action scores only. Statistics consume already-frozen
policy losses, never reroute/refit. Attribute clusters are not verified entities.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, ROUND_CEILING

import numpy as np

from build_turkey_price_free_manifest import digest, fold, validate_config


ACTIONS = ["kilometre", "vites_tipi", "yakit_tipi"]
POLICIES = ["no_acquisition", "random", "risk_fixed_validation_best", "risk_global_train_field",
            "benefit", "direct_posterror", "risk_benefit_field", "risk_neural_field"]
PLAIN_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def validate_plan(plan: dict, cohort_config: dict) -> None:
    validate_config(cohort_config)
    if (plan["version"] != "turkey-bvival-execution-plan-v1"
            or plan["action_fields"] != ACTIONS or cohort_config["action_fields"] != ACTIONS
            or plan["seeds"] != [13, 42, 2026] or plan["selection_seed"] != 42
            or plan["budgets"] != [0.05, 0.10, 0.20] or plan["primary_budget"] != 0.10
            or plan["policies"] != POLICIES
            or plan["capacity_rule"] != "ceil_fraction_of_feature_eligible_partition_no_outcome_rerouting"
            or plan["allocation_rule"] != "exact_capacity_one_field_per_listing_even_if_scores_negative"
            or plan["tie_rule"] != "descending_score_then_record_key_then_declared_action_order"):
        raise ValueError("Changed action, seed, policy, or capacity contract")
    valuation, heads, stats, gates = (plan[k] for k in ("valuation", "heads", "statistics", "release_gates"))
    valuation_fields = valuation["categorical_fields"] + valuation["numeric_fields"]
    expected_state_fields = set(cohort_config["initial_visible_fields"] + ACTIONS + ["hidden_" + f for f in ACTIONS])
    if (set(valuation_fields) != expected_state_fields or len(valuation_fields) != len(expected_state_fields)
            or set(valuation["categorical_fields"]) != {"marka", "seri", "kasa_tipi", "renk", "kimden", "vites_tipi", "yakit_tipi"}):
        raise ValueError("Valuation fields must match exactly the one-reveal input allowlist")
    if (valuation["outer_folds"] != 5 or valuation["inner_folds"] != 3
            or valuation["training_states"] != ["all_hidden", *ACTIONS]
            or valuation["family_selection"] != "train_only_nested_group_cv_uniform_state_price_mae"
            or valuation["final_selection"] != "five_existing_train_group_folds_uniform_state_price_mae"
            or valuation["preprocessing"]["fit_scope"] != "current_fit_groups_only_no_validation_calibration_or_test_fit"):
        raise ValueError("Valuation nesting or preprocessing scope changed")
    candidates = valuation["candidates"]
    if (Counter(c["family"] for c in candidates)
            != {"catboost": 4, "extra_trees": 4, "hist_gradient_boosting": 4}
            or len({c["id"] for c in candidates}) != 12):
        raise ValueError("Valuation families need four distinct candidates each")
    param_keys = {
        "catboost": {"iterations", "depth", "learning_rate", "l2_leaf_reg"},
        "extra_trees": {"n_estimators", "min_samples_leaf", "max_features", "max_depth"},
        "hist_gradient_boosting": {"max_iter", "max_leaf_nodes", "learning_rate", "l2_regularization"},
    }
    for candidate in candidates:
        params = candidate["params"]
        if set(params) != param_keys[candidate["family"]] or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                for v in params.values()):
            raise ValueError("Invalid or undeclared valuation parameter")
    if valuation["fixed_options"]["hist_gradient_boosting"]["early_stopping"] is not False:
        raise ValueError("Do not add uncontrolled internal early stopping")
    expected_features = cohort_config["initial_visible_fields"] + ["before_prediction_log", "before_disagreement_log"]
    if (heads["features"] != expected_features or heads["action_heads_add"] != "action_id"
            or len(heads["candidates"]) != 4 or len({c["id"] for c in heads["candidates"]}) != 4
            or heads["selection_seed"] != 42 or heads["value_scores_keep_negative"] is not True
            or heads["final_fit"] != "train_oof_targets_only_three_seeds_no_train_validation_refit"):
        raise ValueError("Policy input, grid or fit contract changed")
    for candidate in heads["candidates"]:
        if set(candidate) != {"id", "iterations", "depth", "learning_rate", "l2_leaf_reg"} or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                for key, v in candidate.items() if key != "id"):
            raise ValueError("Invalid or undeclared policy-head parameter")
    neural = plan["neural_comparator"]
    if (neural["identity"] != "independent_one_step_expected_loss_adaptation_not_official_GDFS"
            or neural["official_method_reproduction_completed"] is not False
            or neural["required_before_test"] is not True
            or neural["learning_rates"] != [0.0003, 0.001, 0.003]
            or neural["selection_seed"] != 42):
        raise ValueError("Neural identity/budget or required comparator changed")
    expected_family = [
        {"id": "H1", "candidate": "benefit", "reference": "risk_fixed_validation_best", "budget": 0.10},
        {"id": "H2", "candidate": "benefit", "reference": "direct_posterror", "budget": 0.10},
    ]
    if (stats["primary_family"] != expected_family or stats["repetitions"] != 10000
            or stats["seed"] != 2026 or stats["practical_reference_percent"] != 2.0
            or stats["bootstrap_unit"] != "coarse_attribute_group_paired_all_listings_in_group"
            or stats["estimand_weighting"] != "listing_weighted_not_group_weighted"
            or stats["p_value"] != "two_sided_centered_cluster_bootstrap_absolute_mae_difference_plus_one"
            or stats["multiplicity"] != "Holm_two_primary_tests_alpha_0.05"):
        raise ValueError("Primary family or statistical decision changed")
    validity = plan["price_validity"]
    if (validity["rule"] != "finite_positive_plain_numeric_no_quantile_or_residual_filter"
            or validity["maximum_invalid_fraction_per_partition"] != 0.01
            or validity["minimum_valid_fraction_per_train_oof_fold"] != 0.99
            or validity["invalid_targets"] != "omit_from_loss_only_keep_feature_cohort_and_frozen_allocation"):
        raise ValueError("Target validity cannot redefine the cohort/capacity")
    if (gates["currently_price_release_allowed"] is not False
            or gates["development"] != ["train", "validation"]
            or gates["test_openings"] != 1 or gates["refit_on_validation_or_calibration"] is not False
            or gates["test_rerouting_after_price_validity"] is not False
            or gates["runner_integration_and_synthetic_tests_required"] is not True):
        raise ValueError("This design-only freeze does not authorize label access")


def budget_count(n: int, fraction: float) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("Capacity denominator must be a nonnegative listing count")
    b = Decimal(str(fraction))
    if not b.is_finite() or not 0 < b <= 1:
        raise ValueError("Budget fraction must be in (0, 1]")
    return int((n * b).to_integral_value(rounding=ROUND_CEILING))


def price_is_valid(value) -> bool:
    """Validity of a supplied scalar; does not access or parse any source cell."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return False
    token = str(value).strip()
    if not PLAIN_NUMBER.fullmatch(token):
        return False
    try:
        number = Decimal(token)
        converted = float(number)
    except (InvalidOperation, ValueError, OverflowError):
        return False
    return number.is_finite() and number > 0 and math.isfinite(converted) and converted > 0


def target_validity_inventory(split_rows, validity, *, released_splits):
    """Count explicit validity booleans after a separately authorized release.

    No price values/label files accepted or opened. Never changes allocations.
    Failure must stop downstream execution, not trigger a new cohort selection.
    """
    if (not released_splits or len(set(released_splits)) != len(released_splits)
            or not set(released_splits).issubset({"train", "validation", "calibration", "test"})):
        raise ValueError("Known distinct released splits required")
    rows = [r for r in split_rows if r["split"] in released_splits]
    keys = [r["record_key"] for r in rows]
    if (len(set(keys)) != len(keys) or set(validity) != set(keys)
            or any(type(value) is not bool for value in validity.values())):
        raise ValueError("Exact released record universe and validity booleans required")
    result = {}
    for split in released_splits:
        group = [r for r in rows if r["split"] == split]
        valid = sum(validity[r["record_key"]] for r in group)
        if not group or (len(group) - valid) * 100 > len(group):
            raise ValueError("Invalid target fraction exceeds 1% or a released split is empty; stop")
        result[split] = {"feature_eligible_records": len(group), "price_valid_records": valid,
                         "invalid_records": len(group) - valid}
    if "train" in released_splits:
        for outer in range(5):
            group = [r for r in rows if r["split"] == "train" and int(r["oof_fold"]) == outer]
            valid = sum(validity[r["record_key"]] for r in group)
            if not group or (len(group) - valid) * 100 > len(group):
                raise ValueError("Training fold validity below 99% or fold empty; stop")
    return {"partitions": result, "allocation_denominators_unchanged": True}


def _score_matrix(record_keys, actions, scores):
    if (any(not isinstance(k, str) or not k for k in record_keys)
            or len(set(record_keys)) != len(record_keys) or list(actions) != ACTIONS):
        raise ValueError("Unique nonempty keys and declared action order required")
    matrix = np.asarray(scores, dtype=float)
    if matrix.shape != (len(record_keys), len(actions)) or not np.isfinite(matrix).all():
        raise ValueError("Complete finite pre-action score matrix required")
    return matrix


def select_actions(record_keys, actions, scores, fraction, *, fixed_action=None):
    """Exact capacity, including negative scores. Does not consume outcome values."""
    matrix = _score_matrix(record_keys, actions, scores)
    if fixed_action is not None and fixed_action not in actions:
        raise ValueError("Unknown fixed field")
    choices = (np.full(len(record_keys), actions.index(fixed_action), dtype=int)
               if fixed_action is not None else np.argmax(matrix, axis=1))
    ranking = sorted(range(len(record_keys)), key=lambda i: (-matrix[i, choices[i]], record_keys[i], int(choices[i])))
    selected = ranking[:budget_count(len(record_keys), fraction)]
    return [(record_keys[i], actions[choices[i]]) for i in selected]


def select_on_common_cohort(record_keys, actions, field_scores, selected_keys):
    matrix = _score_matrix(record_keys, actions, field_scores)
    if len(set(selected_keys)) != len(selected_keys) or not set(selected_keys).issubset(record_keys):
        raise ValueError("Common cohort must be unique known record keys")
    positions = {key: i for i, key in enumerate(record_keys)}
    return [(key, actions[int(np.argmax(matrix[positions[key]]))]) for key in selected_keys]


def random_actions(record_keys, actions, fraction, *, namespace: str, seed: int):
    # Matrix check enforces the same three-action universe and valid keys.
    _score_matrix(record_keys, actions, np.zeros((len(record_keys), len(actions))))
    ordered = sorted(record_keys, key=lambda key: (digest([namespace, "listing", seed, key]), key))
    return [(key, actions[int(digest([namespace, "field", seed, key])[:16], 16) % len(actions)])
            for key in ordered[:budget_count(len(record_keys), fraction)]]


def nested_training_roles(rows, cohort_config, *, outer_fold: int, namespace: str):
    """Group-only assignments; excludes outer groups before inner selection."""
    if outer_fold not in range(5):
        raise ValueError("Unknown outer fold")
    validate_config(cohort_config)
    groups = defaultdict(set)
    keys = set()
    for row in rows:
        if row["split"] != "train" or row["record_key"] in keys:
            raise ValueError("Nested roles accept unique train rows only")
        keys.add(row["record_key"])
        saved = int(row["oof_fold"])
        if saved != fold(row["group_hash"], cohort_config):
            raise ValueError("Train fold changed from frozen group hash")
        groups[row["group_hash"]].add(saved)
    if any(len(folds) != 1 for folds in groups.values()):
        raise ValueError("Group crosses outer fold")
    result = []
    for row in sorted(rows, key=lambda r: r["record_key"]):
        heldout = int(row["oof_fold"]) == outer_fold
        inner = "" if heldout else int(digest([namespace, outer_fold, row["group_hash"]])[:8], 16) % 3
        result.append({"record_key": row["record_key"], "group_hash": row["group_hash"],
                       "role": "outer_holdout" if heldout else "outer_fit", "inner_fold": inner})
    return result


def select_candidate(validation_losses: dict[str, float]) -> str:
    if not validation_losses or any(not math.isfinite(v) or v < 0 for v in validation_losses.values()):
        raise ValueError("Candidate losses must be finite and nonnegative")
    return min(validation_losses, key=lambda name: (validation_losses[name], name))


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Finite p values in [0,1] required")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty(len(p))
    out[order] = adjusted
    return out.tolist()


def paired_cluster_effect(reference_errors, candidate_errors, group_keys, *, repetitions=10000, seed=2026):
    """Listing-weighted paired cluster bootstrap with frozen losses/allocations.

    p is an approximate centered-bootstrap test of zero absolute MAE difference.
    The percentile relative CI is not a simultaneous or retraining interval.
    """
    reference, candidate = (np.asarray(x, dtype=float) for x in (reference_errors, candidate_errors))
    if (reference.ndim != 1 or reference.shape != candidate.shape or not len(reference)
            or len(group_keys) != len(reference) or not np.isfinite(reference).all()
            or not np.isfinite(candidate).all() or (reference < 0).any() or (candidate < 0).any()
            or any(not isinstance(g, str) or not g for g in group_keys)):
        raise ValueError("Aligned finite nonnegative listing losses and group keys required")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 100:
        raise ValueError("At least 100 bootstrap replicates required")
    # Stable group order and grouped sums make resampling row-order invariant.
    labels = sorted(set(group_keys))
    if len(labels) < 2 or reference.mean() <= 0:
        raise ValueError("At least two groups and positive reference MAE required")
    positions = {key: i for i, key in enumerate(labels)}
    indices = np.array([positions[key] for key in group_keys])
    counts = np.bincount(indices, minlength=len(labels))
    rsum = np.bincount(indices, weights=reference, minlength=len(labels))
    csum = np.bincount(indices, weights=candidate, minlength=len(labels))
    rng = np.random.default_rng(seed)
    effects, differences = [], []
    for start in range(0, repetitions, 128):
        draws = rng.integers(len(labels), size=(min(128, repetitions - start), len(labels)))
        denominators = counts[draws].sum(axis=1)
        r = rsum[draws].sum(axis=1) / denominators
        c = csum[draws].sum(axis=1) / denominators
        if (r <= 0).any():
            raise ValueError("Undefined relative effect in a zero-reference replicate; do not discard draws")
        effects.extend((100 * (r - c) / r).tolist())
        differences.extend((r - c).tolist())
    difference = float(reference.mean() - candidate.mean())
    effect = float(100 * difference / reference.mean())
    p = (1 + np.count_nonzero(np.abs(np.asarray(differences) - difference) >= abs(difference))) / (repetitions + 1)
    return {"listing_count": len(reference), "attribute_group_count": len(labels),
            "reference_mae": float(reference.mean()), "candidate_mae": float(candidate.mean()),
            "absolute_mae_difference": difference, "relative_effect_percent": effect,
            "relative_ci95_percent": np.quantile(effects, [.025, .975]).tolist(),
            "centered_bootstrap_p_two_sided": float(p), "repetitions": repetitions, "seed": seed,
            "allocation_and_models_held_fixed": True, "verified_entity_independence": False}


def classify_evidence(effect_percent, ci_lower, holm_p, seed_effects, *, practical_reference=2.0):
    if set(seed_effects) != {13, 42, 2026} or any(not math.isfinite(v) for v in seed_effects.values()):
        raise ValueError("Exactly the three prespecified finite seed effects required")
    if not all(math.isfinite(v) for v in (effect_percent, ci_lower, holm_p)) or not 0 <= holm_p <= 1:
        raise ValueError("Invalid effect, lower interval or adjusted p")
    directional = effect_percent > 0 and ci_lower > 0 and holm_p < .05 and all(v > 0 for v in seed_effects.values())
    practical = directional and effect_percent >= practical_reference
    return {"directional_support": directional, "practical_screen_support": practical,
            "ci_lower_above_practical_reference": practical and ci_lower >= practical_reference,
            "registered_confirmatory_claim": False}
