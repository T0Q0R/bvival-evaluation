"""Train-OOF-only heads and outcome-free allocations; no workbook/label I/O.

Validation chooses settings, so validation losses are not replication evidence.
Only initial context and before summaries enter model inputs. Supervision and
realized after-state losses are kept separate from scoring/allocation APIs.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from build_turkey_price_free_manifest import digest
from train_turkey_nested_valuation import validate_dataset
from turkey_execution_contract import ACTIONS, POLICIES, budget_count, random_actions, select_actions, select_candidate, select_on_common_cohort
from turkey_information_boundary import policy_context
from turkey_valuation_components import signed_action_targets


def context_frame(contexts, plan, config, *, action_head=False):
    """Reject target, locator and hidden-value columns before making a matrix."""
    features = plan["heads"]["features"] + (["action_id"] if action_head else [])
    rows = []
    if not contexts:
        raise ValueError("Nonempty pre-action contexts required")
    for context in contexts:
        if set(context) != set(features):
            raise ValueError("Exact policy context allowlist required")
        initial = {f: context[f] for f in config["initial_visible_fields"]}
        if any(not isinstance(v, str) for v in initial.values()):
            raise ValueError("Initial values must be normalized strings")
        row = policy_context(initial, config, before_prediction_log=context["before_prediction_log"],
                             before_disagreement_log=context["before_disagreement_log"],
                             action_id=context.get("action_id") if action_head else None)
        rows.append(row)
    frame = pd.DataFrame(rows)[features]
    categorical = [f for f in plan["valuation"]["categorical_fields"] if f in features]
    if action_head:
        categorical.append("action_id")
    for field in categorical:
        frame[field] = frame[field].replace("", "__MISSING__")
    for field in set(features) - set(categorical):
        values = []
        for value in frame[field]:
            if value == "":
                values.append(np.nan)
            else:
                number = float(value)
                if not np.isfinite(number):
                    raise ValueError("Nonfinite nonempty policy numeric value")
                values.append(number)
        frame[field] = values
    return frame, categorical


def make_material(data, valuation, config, split):
    """Separate supervision from all-record contexts, preserving invalid rows."""
    if split not in {"train", "validation"}:
        raise ValueError("Only development partitions accepted")
    keys = sorted(row["record_key"] for row in data["rows"] if row["split"] == split)
    records = valuation["oof" if split == "train" else "validation"]
    if set(records) != set(keys):
        raise ValueError("Prediction universe differs from frozen partition")
    contexts = [policy_context(data["initial"][key], config,
                               before_prediction_log=records[key]["before_prediction_log"],
                               before_disagreement_log=records[key]["before_disagreement_log"]) for key in keys]
    mask = np.array([data["validity"][key] for key in keys], dtype=bool)
    valid_keys = [key for key, valid in zip(keys, mask) if valid]
    targets = signed_action_targets([data["prices"][key] for key in valid_keys],
                                   [records[key]["before_prediction_log"] for key in valid_keys],
                                   [records[key]["after_prediction_log"] for key in valid_keys])
    return {"keys": keys, "contexts": contexts, "valid_mask": mask, "targets": targets,
            "target_origin": "train_nested_oof" if split == "train" else "validation_train_only_base"}


def validate_material(material, plan, config, *, train):
    if set(material) != {"keys", "contexts", "valid_mask", "targets", "target_origin"}:
        raise ValueError("Unexpected material roles")
    keys, mask, targets = material["keys"], np.asarray(material["valid_mask"]), material["targets"]
    if (not keys or len(set(keys)) != len(keys) or any(not isinstance(k, str) or not k for k in keys)
            or mask.dtype != np.bool_ or mask.shape != (len(keys),) or not mask.any()
            or len(material["contexts"]) != len(keys)
            or material["target_origin"] != ("train_nested_oof" if train else "validation_train_only_base")):
        raise ValueError("Invalid development supervision role or universe")
    context_frame(material["contexts"], plan, config)
    if set(targets) != {"before_absolute_error", "after_absolute_error", "signed_gain", "common_training_scale"}:
        raise ValueError("Unexpected supervision fields")
    before, after, gain = [np.asarray(targets[f], dtype=float) for f in
                           ("before_absolute_error", "after_absolute_error", "signed_gain")]
    if (before.shape != (int(mask.sum()),) or after.shape != (int(mask.sum()), 3) or gain.shape != after.shape
            or any(not np.isfinite(x).all() for x in (before, after, gain))
            or (before < 0).any() or (after < 0).any()
            or not np.allclose(gain, before[:, None] - after, rtol=1e-10, atol=1e-8)
            or not np.isclose(targets["common_training_scale"], max(float(before.mean()), 1.0))):
        raise ValueError("Misaligned/nonfinite supervision or signed-gain identity")


def post_mae(material, allocation):
    """Evaluate an already selected allocation, never reallocate invalid targets."""
    keys, mask, targets = material["keys"], np.asarray(material["valid_mask"]), material["targets"]
    if (len({key for key, _ in allocation}) != len(allocation)
            or any(key not in keys or action not in ACTIONS for key, action in allocation)):
        raise ValueError("Allocation must have unique known listing-action pairs")
    positions = {key: i for i, key in enumerate(key for key, valid in zip(keys, mask) if valid)}
    losses = np.asarray(targets["before_absolute_error"]).copy()
    for key, action in allocation:
        if key in positions:
            losses[positions[key]] = targets["after_absolute_error"][positions[key], ACTIONS.index(action)]
    return float(losses.mean())


def make_head(plan, candidate_id, seed, categorical):
    if seed not in plan["seeds"]:
        raise ValueError("Unplanned policy seed")
    candidate = next(c for c in plan["heads"]["candidates"] if c["id"] == candidate_id)
    params = {k: v for k, v in candidate.items() if k != "id"}
    return CatBoostRegressor(**params, random_seed=seed, cat_features=categorical, loss_function="RMSE",
                             thread_count=4, verbose=False, allow_writing_files=False)


def score_head(model, contexts, plan, config, *, kind, scale):
    """No labels, acquisition values or after predictions in this scoring API."""
    if kind not in {"risk", "value", "posterror"} or not np.isfinite(scale) or scale < 1:
        raise ValueError("Unknown head or invalid common training scale")
    action_head = kind != "risk"
    expanded = ([{**context, "action_id": action} for context in contexts for action in ACTIONS]
                if action_head else contexts)
    frame, _ = context_frame(expanded, plan, config, action_head=action_head)
    prediction = np.asarray(model.predict(frame), dtype=float) * scale
    if prediction.shape != (len(contexts) * (3 if action_head else 1),) or not np.isfinite(prediction).all():
        raise ValueError("Finite aligned policy prediction required")
    if kind != "value":
        prediction = np.maximum(prediction, 0)
    return prediction.reshape(len(contexts), 3) if action_head else prediction


def allocate_policies(keys, risk, value, posterror, neural_probability, plan, *, fixed_field, global_field, seed=42):
    """Outcome-free, exact-capacity allocation for all eight frozen policies."""
    n = len(keys)
    risk, value, posterror, probability = [np.asarray(a, dtype=float) for a in
                                         (risk, value, posterror, neural_probability)]
    if (risk.shape != (n,) or any(a.shape != (n, 3) for a in (value, posterror, probability))
            or any(not np.isfinite(a).all() for a in (risk, value, posterror, probability))
            or (risk < 0).any() or (posterror < 0).any() or (probability < 0).any()
            or not np.allclose(probability.sum(axis=1), 1, rtol=1e-6, atol=1e-6)
            or fixed_field not in ACTIONS or global_field not in ACTIONS or seed not in plan["seeds"]):
        raise ValueError("Complete guarded policy scores/fields/seeds required")
    risk_matrix = np.repeat(risk[:, None], 3, axis=1)
    result = {}
    for fraction in plan["budgets"]:
        common = select_actions(keys, ACTIONS, risk_matrix, fraction, fixed_action=fixed_field)
        common_keys = [key for key, _ in common]
        policies = {
            "no_acquisition": [],
            "random": random_actions(keys, ACTIONS, fraction, namespace=plan["random_namespace"], seed=seed),
            "risk_fixed_validation_best": common,
            "risk_global_train_field": [(key, global_field) for key in common_keys],
            "benefit": select_actions(keys, ACTIONS, value, fraction),
            "direct_posterror": select_actions(keys, ACTIONS, risk[:, None] - posterror, fraction),
            "risk_benefit_field": select_on_common_cohort(keys, ACTIONS, value, common_keys),
            "risk_neural_field": select_on_common_cohort(keys, ACTIONS, probability, common_keys),
        }
        if set(policies) != set(POLICIES) or any(len(v) != budget_count(n, fraction) for k, v in policies.items() if k != "no_acquisition"):
            raise AssertionError("Incomplete policies or budget mismatch")
        result[f"{fraction:.2f}"] = policies
    return result


def fit_policy_heads(train, validation, plan, config, *, model_directory=None, factory=make_head):
    validate_material(train, plan, config, train=True)
    validate_material(validation, plan, config, train=False)
    if set(train["keys"]) & set(validation["keys"]):
        raise ValueError("Head fit and selection records overlap")
    scale = train["targets"]["common_training_scale"]
    train_contexts = [c for c, valid in zip(train["contexts"], train["valid_mask"]) if valid]
    chosen, scores, per_seed, fits, trials, models = {}, {}, {}, [], [], {}

    def fit(kind, candidate, seed, stage):
        action = kind != "risk"
        contexts = ([{**c, "action_id": a} for c in train_contexts for a in ACTIONS] if action else train_contexts)
        frame, categorical = context_frame(contexts, plan, config, action_head=action)
        target_name = {"risk": "before_absolute_error", "value": "signed_gain", "posterror": "after_absolute_error"}[kind]
        targets = np.asarray(train["targets"][target_name]).reshape(-1) / scale
        started = time.perf_counter()
        model = factory(plan, candidate, seed, categorical).fit(frame, targets)
        fits.append({"kind": kind, "candidate": candidate, "seed": seed, "stage": stage,
                     "fit_rows": len(frame), "fit_record_count": len(train_contexts),
                     "fit_record_keys_sha256": digest([k for k, valid in zip(train["keys"], train["valid_mask"]) if valid]),
                     "excluded_record_keys_sha256": digest(sorted(validation["keys"])),
                     "validation_targets_in_fit": False, "fit_seconds": time.perf_counter() - started})
        return model

    for kind in ("risk", "value", "posterror"):
        losses = {}
        for candidate in sorted(c["id"] for c in plan["heads"]["candidates"]):
            model = fit(kind, candidate, 42, "validation_selection")
            prediction = score_head(model, validation["contexts"], plan, config, kind=kind, scale=scale)
            if kind == "risk":
                loss = float(np.sqrt(np.mean((prediction[validation["valid_mask"]] - validation["targets"]["before_absolute_error"]) ** 2)))
            else:
                ranking = prediction if kind == "value" else scores["risk"][:, None] - prediction
                allocation = select_actions(validation["keys"], ACTIONS, ranking, plan["primary_budget"])
                loss = post_mae(validation, allocation)
            losses[candidate] = loss
            trials.append({"kind": kind, "candidate": candidate, "validation_selection_loss": loss,
                           "metric": "before_error_rmse" if kind == "risk" else "full_partition_post_mae_at_10_percent"})
        chosen[kind] = select_candidate(losses)
        seeds = {}
        for seed in plan["seeds"]:
            model = fit(kind, chosen[kind], seed, "final_train_oof_fit")
            seeds[str(seed)] = score_head(model, validation["contexts"], plan, config, kind=kind, scale=scale)
            models[(kind, seed)] = model
            if model_directory is not None:
                joblib.dump(model, Path(model_directory) / f"{kind}_seed_{seed}.joblib")
        per_seed[kind] = seeds
        scores[kind] = np.mean(list(seeds.values()), axis=0)
    cohort = select_actions(validation["keys"], ACTIONS, np.repeat(scores["risk"][:, None], 3, axis=1), plan["primary_budget"], fixed_action=ACTIONS[0])
    field_losses = {a: post_mae(validation, [(k, a) for k, _ in cohort]) for a in ACTIONS}
    fixed = min(ACTIONS, key=lambda a: (field_losses[a], ACTIONS.index(a)))
    global_field = ACTIONS[int(np.argmax(np.asarray(train["targets"]["signed_gain"]).mean(axis=0)))]
    return {"chosen_candidates": chosen, "scores": scores, "per_seed_scores": per_seed, "models": models,
            "common_training_scale": scale, "fixed_field": fixed, "global_field": global_field,
            "fixed_field_validation_losses": field_losses, "fit_audit": fits, "candidate_trials": trials,
            "validation_is_selection_not_replication": True, "official_GDFS_reproduction": False}


def prepare_policy_material(data, valuation, plan, config):
    validate_dataset(data, plan, config)
    if valuation.get("validation_targets_used_for_base_selection") is not False:
        raise ValueError("Train-only base selection audit required")
    return make_material(data, valuation, config, "train"), make_material(data, valuation, config, "validation")
