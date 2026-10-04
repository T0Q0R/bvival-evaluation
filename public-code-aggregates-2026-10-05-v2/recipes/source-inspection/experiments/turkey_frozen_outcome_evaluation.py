"""Pure evaluation of already-frozen policies on explicitly supplied outcomes.

No workbook, price-file I/O, model fitting, field selection or release authority.
All policy seeds use the SAME three-model ensemble valuation, as frozen in plan.
"""
from __future__ import annotations

import numpy as np

from turkey_execution_contract import (ACTIONS, POLICIES, budget_count, classify_evidence,
                                       holm_adjust, paired_cluster_effect, price_is_valid,
                                       target_validity_inventory, validate_plan)
from turkey_valuation_components import signed_action_targets


def prepare_outcomes(keys, groups, labels, predictions, *, split):
    """Invalid targets omit losses only; exact complete feature universe required."""
    if (split not in {"calibration", "test"} or not isinstance(keys, list) or not keys
            or any(not isinstance(k, str) or not k for k in keys) or keys != sorted(set(keys))
            or set(groups) != set(keys) or any(not isinstance(v, str) or not v for v in groups.values())
            or set(labels) != set(keys) or set(predictions) != set(keys)):
        raise ValueError("Exact ordered held-out feature/outcome/group/prediction universe required")
    validity, prices = {}, {}
    for key in keys:
        label = labels[key]
        if set(label) != {"price", "price_valid"} or type(label["price_valid"]) is not bool:
            raise ValueError("Explicit price/boolean validity only")
        valid = label["price_valid"]
        if (valid and not price_is_valid(label["price"])) or (not valid and label["price"] != ""):
            raise ValueError("Label value and validity disagree")
        if set(predictions[key]) != {"before_prediction_log", "after_prediction_log"}:
            raise ValueError("Shared valuation before/three-after logs only, no policy outcomes")
        before = predictions[key]["before_prediction_log"]
        after = np.asarray(predictions[key]["after_prediction_log"], dtype=float)
        if (isinstance(before, bool) or not isinstance(before, (int, float, np.number))
                or not np.isfinite(before) or after.shape != (3,) or not np.isfinite(after).all()):
            raise ValueError("Finite aligned before/three-after predictions required for ALL feature records")
        validity[key] = valid
        if valid:
            prices[key] = float(label["price"])
    rows = [{"record_key": k, "split": split} for k in keys]
    inventory = target_validity_inventory(rows, validity, released_splits=[split])
    valid_keys = [k for k in keys if validity[k]]
    targets = signed_action_targets([prices[k] for k in valid_keys],
                                    [predictions[k]["before_prediction_log"] for k in valid_keys],
                                    [predictions[k]["after_prediction_log"] for k in valid_keys])
    return {"keys": keys, "valid_keys": valid_keys, "groups": [groups[k] for k in valid_keys],
            "prices": np.asarray([prices[k] for k in valid_keys]),
            "before_prediction_log": np.maximum([predictions[k]["before_prediction_log"] for k in valid_keys], 0),
            "after_prediction_log": np.maximum([predictions[k]["after_prediction_log"] for k in valid_keys], 0),
            "targets": targets, "target_validity_inventory": inventory, "split": split}


def fixed_policy_losses(material, allocation, *, expected_count):
    """One-field replacement on frozen selected keys; invalid rows NEVER backfilled."""
    if (not isinstance(allocation, list) or len(allocation) != expected_count
            or any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in allocation)
            or len({key for key, _ in allocation}) != len(allocation)
            or any(key not in material["keys"] or action not in ACTIONS for key, action in allocation)):
        raise ValueError("Exact frozen capacity and unique known listing/action pairs required")
    positions = {key: i for i, key in enumerate(material["valid_keys"])}
    targets = material["targets"]
    losses = np.asarray(targets["before_absolute_error"]).copy()
    log_predictions = material["before_prediction_log"].copy()
    gains = []
    for key, action in allocation:
        if key in positions:
            i, a = positions[key], ACTIONS.index(action)
            losses[i] = targets["after_absolute_error"][i, a]
            log_predictions[i] = material["after_prediction_log"][i, a]
            gains.append(targets["signed_gain"][i, a])
    return losses, {"post_mae": float(losses.mean()), "selected_feature_records": len(allocation),
                    "rmsle_descriptive_only": float(np.sqrt(np.mean((log_predictions - np.log1p(material["prices"])) ** 2))),
                    "selected_price_valid_records": len(gains), "invalid_selected_not_backfilled": len(allocation) - len(gains),
                    "positive_realized_gain_count": int(np.sum(np.asarray(gains) > 0)),
                    "negative_realized_gain_count": int(np.sum(np.asarray(gains) < 0)),
                    "zero_realized_gain_count": int(np.sum(np.asarray(gains) == 0))}


def relative_effect(reference, candidate):
    if not np.isfinite(reference) or not np.isfinite(candidate) or min(reference, candidate) < 0:
        raise ValueError("Finite nonnegative losses required")
    return None if reference == 0 else 100 * (reference - candidate) / reference


def evaluate_frozen_outcomes(keys, groups, labels, predictions, allocations, plan, config, *, split, evidence_status):
    """Calibration descriptive only; test has exactly two frozen primary analyses.

    No third-party preregistration, verified-entity or business-effect guarantee.
    Synthetic computations must never become source results or hypothesis claims.
    Caller still must verify freeze/release bindings BEFORE supplying any labels.
    """
    if evidence_status not in {"synthetic_only", "authorized_heldout_only"}:
        raise ValueError("Explicit synthetic/authorized-heldout evidence role required")
    validate_plan(plan, config)
    material = prepare_outcomes(keys, groups, labels, predictions, split=split)
    if set(allocations) != {"ensemble", "per_seed"} or set(allocations["per_seed"]) != {str(s) for s in plan["seeds"]}:
        raise ValueError("All frozen ensemble/seed allocations required")
    budgets = {f"{b:.2f}" for b in plan["budgets"]}
    vectors, summaries = {}, {}
    for name, decisions in [("ensemble", allocations["ensemble"]), *allocations["per_seed"].items()]:
        if set(decisions) != budgets:
            raise ValueError("All frozen budgets required, no selective reporting")
        vectors[name], summaries[name] = {}, {}
        for budget, policies in decisions.items():
            if set(policies) != set(POLICIES):
                raise ValueError("All eight frozen policies required, including negative/null results")
            vectors[name][budget], summaries[name][budget] = {}, {}
            for policy, action in policies.items():
                count = 0 if policy == "no_acquisition" else budget_count(len(keys), float(budget))
                loss, summary = fixed_policy_losses(material, action, expected_count=count)
                vectors[name][budget][policy], summaries[name][budget][policy] = loss, summary
            for policy, summary in summaries[name][budget].items():
                summary["relative_effect_vs_risk_fixed_percent"] = relative_effect(
                    summaries[name][budget]["risk_fixed_validation_best"]["post_mae"], summary["post_mae"])
                summary["relative_effect_vs_direct_posterror_percent"] = relative_effect(
                    summaries[name][budget]["direct_posterror"]["post_mae"], summary["post_mae"])
    result = {"evidence_status": evidence_status, "split": split, "feature_eligible_records": len(keys),
              "price_valid_records": len(material["valid_keys"]), "target_validity_inventory": material["target_validity_inventory"],
              "summaries": summaries, "shared_ensemble_valuation_for_all_policy_seeds": True,
              "seed_runs_not_independent_market_samples": True, "source_currency_not_independently_verified": True,
              "real_request_or_transaction_or_profit_evidence": False, "refits": 0, "reselection": False,
              "calibration_diagnostic_only": split == "calibration", "registered_confirmatory_claim": False,
              "caller_must_verify_freeze_and_release_bindings": True, "primary_comparisons": []}
    if split == "calibration":
        result["confidence_intervals_or_pvalues_computed"] = False
        return result
    comparisons = []
    for primary in plan["statistics"]["primary_family"]:
        budget = f"{primary['budget']:.2f}"
        reference, candidate = primary["reference"], primary["candidate"]
        effect = paired_cluster_effect(vectors["ensemble"][budget][reference], vectors["ensemble"][budget][candidate],
                                       material["groups"], repetitions=plan["statistics"]["repetitions"], seed=plan["statistics"]["seed"])
        seed_effects = {s: relative_effect(summaries[str(s)][budget][reference]["post_mae"],
                                          summaries[str(s)][budget][candidate]["post_mae"]) for s in plan["seeds"]}
        if any(v is None for v in seed_effects.values()):
            raise ValueError("Zero-reference seed effect undefined; do not drop or rerun seeds")
        comparisons.append({"id": primary["id"], "reference": reference, "candidate": candidate, "budget": budget,
                            **effect, "per_seed_point_effect_percent": seed_effects})
    adjusted = holm_adjust([c["centered_bootstrap_p_two_sided"] for c in comparisons])
    for comparison, p in zip(comparisons, adjusted):
        comparison["holm_adjusted_p_two_sided"] = p
        comparison["decision_rule_readout"] = classify_evidence(comparison["relative_effect_percent"],
            comparison["relative_ci95_percent"][0], p, comparison["per_seed_point_effect_percent"],
            practical_reference=plan["statistics"]["practical_reference_percent"])
        comparison["synthetic_rule_check_not_source_hypothesis_support"] = evidence_status == "synthetic_only"
    result.update(primary_comparisons=comparisons, confidence_intervals_or_pvalues_computed=True,
                  confidence_intervals_conditional_on_frozen_models_and_allocations=True,
                  confidence_intervals_not_retraining_or_independent_market_sampling=True,
                  only_two_primary_tests_no_secondary_significance=True)
    return result
