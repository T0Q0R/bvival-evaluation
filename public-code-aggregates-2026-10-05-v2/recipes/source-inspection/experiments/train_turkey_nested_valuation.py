"""Train/validation-only nested valuation orchestration for the frozen plan.

This is the valuation stage, not the unfinished policy/neural/test pipeline.
The production CLI requires a full-development readiness receipt before loading
labels. Synthetic callers exercise orchestration without any source access.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from build_turkey_price_free_manifest import digest, fold, write_csv
from turkey_development_io import bound_plan, load_development_dataset, require_development_receipt, sha256
from turkey_execution_contract import ACTIONS, nested_training_roles, price_is_valid, select_candidate, target_validity_inventory, validate_plan
from turkey_information_boundary import initial_context, policy_context, valuation_state
from turkey_valuation_components import ensemble_prediction, fit_valuation_model, make_valuation_model, signed_action_targets


def validate_dataset(data, plan, config):
    validate_plan(plan, config)
    if set(data) != {"rows", "initial", "actions", "prices", "validity"}:
        raise ValueError("Unexpected dataset role/payload")
    rows = data["rows"]
    keys = [row["record_key"] for row in rows]
    if (not keys or len(keys) != len(set(keys)) or any(not k for k in keys)
            or {row["split"] for row in rows} != {"train", "validation"}
            or any(set(row) != {"record_key", "group_hash", "profile_hash", "split", "oof_fold"} for row in rows)
            or set(data["initial"]) != set(keys) or set(data["actions"]) != set(keys)):
        raise ValueError("Exact train/validation record universe required, no calibration/test")
    target_validity_inventory(rows, data["validity"], released_splits=["train", "validation"])
    valid_keys = {key for key, valid in data["validity"].items() if valid}
    if set(data["prices"]) != valid_keys or any(not price_is_valid(p) for p in data["prices"].values()):
        raise ValueError("Targets must match validity flags exactly")
    groups = defaultdict(set)
    for row in rows:
        key = row["record_key"]
        initial_context(data["initial"][key], config)
        if set(data["actions"][key]) != set(ACTIONS) or any(not isinstance(v, str) or not v for v in data["actions"][key].values()):
            raise ValueError("Complete recorded action universe required")
        groups[row["group_hash"]].add(row["split"])
        if row["split"] == "train":
            if int(row["oof_fold"]) != fold(row["group_hash"], config):
                raise ValueError("Training fold differs from frozen group rule")
        elif row["oof_fold"]:
            raise ValueError("Validation cannot have a training OOF assignment")
    if any(len(parts) != 1 for parts in groups.values()):
        raise ValueError("Attribute group crosses train/validation")


def four_states(data, record_keys, config):
    states = []
    for key in record_keys:
        visible = data["initial"][key]
        states.append(valuation_state(visible, config))
        for action in ACTIONS:
            states.append(valuation_state(visible, config, revealed_action=action, revealed_value=data["actions"][key][action]))
    return states


def _fit(data, keys, forbidden_keys, plan, config, candidate, seed, stage, fits, factory):
    fit_keys = sorted(key for key in keys if data["validity"][key])
    if not fit_keys or set(fit_keys) & set(forbidden_keys):
        raise ValueError("Empty fit or excluded record in fitting scope")
    states = four_states(data, fit_keys, config)
    prices = [data["prices"][key] for key in fit_keys for _ in range(4)]
    started = time.perf_counter()
    model = fit_valuation_model(factory(plan, candidate, seed), states, prices)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    fits.append({"stage": stage, "candidate": candidate, "seed": seed,
                 "fit_record_count": len(fit_keys), "fit_state_count": len(states),
                 "fit_record_keys_sha256": digest(fit_keys), "excluded_record_keys_sha256": digest(sorted(forbidden_keys)),
                 "fit_and_excluded_overlap": 0, "fit_seconds": time.perf_counter() - started,
                 "process_peak_rss_bytes_cumulative": int(peak if sys.platform == "darwin" else peak * 1024)})
    return model


def _predict(model, data, keys, config):
    predictions = np.asarray(model.predict(four_states(data, keys, config)), dtype=float)
    if predictions.shape != (len(keys) * 4,) or not np.isfinite(predictions).all():
        raise ValueError("Four finite aligned state predictions required")
    return predictions.reshape(len(keys), 4)


def _loss_sum(data, keys, logs):
    valid_indices = [i for i, key in enumerate(keys) if data["validity"][key]]
    if not valid_indices:
        raise ValueError("No valid targets in CV evaluation fold")
    actual = np.asarray([data["prices"][keys[i]] for i in valid_indices])[:, None]
    with np.errstate(over="raise", invalid="raise"):
        try:
            losses = np.abs(actual - np.expm1(np.maximum(logs[valid_indices], 0)))
        except FloatingPointError:
            raise ValueError("Overflow in candidate evaluation; no silent clipping") from None
    return float(losses.sum()), int(losses.size)


def choose_on_train_folds(data, rows, assignments, plan, config, *, stage, excluded_keys, fits, trials, factory):
    # Selection partitions must only cover the permitted train subset.
    keys = {row["record_key"] for row in rows}
    if (set(assignments) != keys or keys & set(excluded_keys)
            or any(row["split"] != "train" for row in rows)):
        raise ValueError("Model selection requires train-only folds excluding outer holdout")
    per_group = defaultdict(set)
    for row in rows:
        per_group[row["group_hash"]].add(assignments[row["record_key"]])
    if any(len(folds) != 1 for folds in per_group.values()):
        raise ValueError("Attribute group crosses selection fold")
    losses = {}
    for candidate in sorted(c["id"] for c in plan["valuation"]["candidates"]):
        total, count = 0.0, 0
        for selected_fold in sorted(set(assignments.values())):
            heldout = sorted(key for key in keys if assignments[key] == selected_fold)
            fitting = sorted(keys - set(heldout))
            model = _fit(data, fitting, set(excluded_keys) | set(heldout), plan, config,
                         candidate, plan["selection_seed"], f"{stage}/selection_fold_{selected_fold}", fits, factory)
            loss, n = _loss_sum(data, heldout, _predict(model, data, heldout, config))
            total += loss
            count += n
        losses[candidate] = total / count
        trials.append({"stage": stage, "candidate": candidate, "uniform_four_state_price_mae": losses[candidate],
                       "scored_state_count": count, "selection_record_keys_sha256": digest(sorted(keys)),
                       "excluded_record_keys_sha256": digest(sorted(excluded_keys))})
    return select_candidate(losses)


def _prediction_records(models, data, keys, config):
    per_seed = np.asarray([_predict(model, data, keys, config) for model in models])
    summary = ensemble_prediction(per_seed.reshape(3, -1))
    logs = summary["prediction_log"].reshape(len(keys), 4)
    spread = summary["disagreement_log"].reshape(len(keys), 4)
    return {key: {"before_prediction_log": float(logs[i, 0]), "before_disagreement_log": float(spread[i, 0]),
                  "after_prediction_log": logs[i, 1:].tolist(), "per_seed_state_logs": per_seed[:, i].tolist()}
            for i, key in enumerate(keys)}


def run_nested_valuation(data, plan, config, *, model_factory=make_valuation_model, model_directory=None):
    validate_dataset(data, plan, config)
    train = sorted([row for row in data["rows"] if row["split"] == "train"], key=lambda r: r["record_key"])
    validation_keys = sorted(row["record_key"] for row in data["rows"] if row["split"] == "validation")
    fits, trials, oof, chosen = [], [], {}, {}
    for outer in range(5):
        roles = nested_training_roles(train, config, outer_fold=outer, namespace=plan["valuation"]["inner_namespace"])
        heldout = sorted(row["record_key"] for row in roles if row["role"] == "outer_holdout")
        inner_assignments = {row["record_key"]: row["inner_fold"] for row in roles if row["role"] == "outer_fit"}
        fitting_rows = [row for row in train if row["record_key"] in inner_assignments]
        if set(inner_assignments.values()) != {0, 1, 2} or not heldout:
            raise ValueError("Every outer fold requires all three inner folds and a holdout")
        stage = f"outer_{outer}"
        print(f"{stage}: selecting 12 candidates on three train-only inner folds", flush=True)
        chosen[stage] = choose_on_train_folds(data, fitting_rows, inner_assignments, plan, config,
                                             stage=stage, excluded_keys=set(heldout), fits=fits, trials=trials, factory=model_factory)
        models = []
        for seed in plan["seeds"]:
            model = _fit(data, inner_assignments, heldout, plan, config, chosen[stage], seed,
                         stage + "/oof_fit", fits, model_factory)
            models.append(model)
            if model_directory is not None:
                joblib.dump(model, model_directory / f"{stage}_seed_{seed}.joblib")
        oof.update(_prediction_records(models, data, heldout, config))
    if set(oof) != {row["record_key"] for row in train}:
        raise AssertionError("OOF record universe is incomplete or duplicated")
    final_assignments = {row["record_key"]: int(row["oof_fold"]) for row in train}
    chosen["final"] = choose_on_train_folds(data, train, final_assignments, plan, config,
                                           stage="final_train_cv", excluded_keys=set(validation_keys),
                                           fits=fits, trials=trials, factory=model_factory)
    final_models = []
    for seed in plan["seeds"]:
        model = _fit(data, final_assignments, validation_keys, plan, config, chosen["final"], seed,
                     "final_train_fit", fits, model_factory)
        final_models.append(model)
        if model_directory is not None:
            joblib.dump(model, model_directory / f"final_seed_{seed}.joblib")
    validation = _prediction_records(final_models, data, validation_keys, config)
    return {"oof": oof, "validation": validation, "chosen_candidates": chosen, "fit_audit": fits,
            "candidate_trials": trials, "fit_count": len(fits), "validation_targets_used_for_base_selection": False,
            "source_scope": ["train", "validation"], "policy_neural_or_test_stage_completed": False}


def policy_training_material(data, valuation, config):
    """Keep pre-action features and explicit allowed training targets separate."""
    records = valuation["oof"]
    keys = sorted(key for key in records if data["validity"][key])
    prices = [data["prices"][key] for key in keys]
    targets = signed_action_targets(prices, [records[key]["before_prediction_log"] for key in keys],
                                    [records[key]["after_prediction_log"] for key in keys])
    contexts = {key: policy_context(data["initial"][key], config,
                                    before_prediction_log=records[key]["before_prediction_log"],
                                    before_disagreement_log=records[key]["before_disagreement_log"]) for key in keys}
    return {"record_keys": keys, "pre_action_contexts": contexts,
            "before_absolute_error": targets["before_absolute_error"],
            "after_absolute_error": targets["after_absolute_error"], "signed_gain": targets["signed_gain"],
            "common_training_scale": targets["common_training_scale"]}


def save_valuation_run(result, output, *, plan_audit, label_directory, readiness_path):
    for split, predictions in [("train_oof", result["oof"]), ("validation", result["validation"])]:
        columns = ["record_key", "before_prediction_log", "before_disagreement_log", *["after_log_" + action for action in ACTIONS]]
        rows = [{"record_key": key, "before_prediction_log": row["before_prediction_log"],
                 "before_disagreement_log": row["before_disagreement_log"],
                 **dict(zip(columns[3:], row["after_prediction_log"]))} for key, row in sorted(predictions.items())]
        write_csv(output / f"{split}_valuation_predictions.csv", columns, rows)
        with (output / f"{split}_per_seed_predictions.json").open("x") as stream:
            json.dump({key: row["per_seed_state_logs"] for key, row in sorted(predictions.items())}, stream)
    audit = {key: result[key] for key in ("chosen_candidates", "fit_audit", "candidate_trials", "fit_count",
                                          "validation_targets_used_for_base_selection", "source_scope", "policy_neural_or_test_stage_completed")}
    audit.update(stage="TRAIN_VALIDATION_VALUATION_STAGE_ONLY", plan_sha256=plan_audit["plan_sha256"],
                 cohort_audit_sha256=plan_audit["cohort_audit_sha256"],
                 development_label_release_audit_sha256=sha256(label_directory / "development_label_release_audit.json"),
                 readiness_receipt_sha256=sha256(readiness_path), test_prices_read=False, policy_test_scores_frozen=False,
                 output_sha256={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()})
    with (output / "nested_valuation_audit.json").open("x") as stream:
        json.dump(audit, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-freeze", type=Path, required=True)
    parser.add_argument("--development-labels", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("New empty output directory required")
    plan, config, cohort, audit = bound_plan(args.plan_freeze, project_root=root)
    receipt = json.loads(args.readiness.read_text())
    require_development_receipt(receipt, audit, project_root=root)
    data = load_development_dataset(cohort, args.development_labels, config)
    if json.loads((args.development_labels / "development_label_release_audit.json").read_text())["plan_sha256"] != audit["plan_sha256"]:
        raise ValueError("Label release bound to a different plan")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.chmod(0o700)
    models = args.output_dir / "models"
    models.mkdir(mode=0o700)
    for name in ("train_turkey_nested_valuation.py", "turkey_development_io.py", "turkey_price_cells.py"):
        with (args.output_dir / (name + ".snapshot")).open("xb") as stream:
            stream.write((root / "experiments" / name).read_bytes())
    try:
        result = run_nested_valuation(data, plan, config, model_directory=models)
        save_valuation_run(result, args.output_dir, plan_audit=audit, label_directory=args.development_labels, readiness_path=args.readiness)
    except Exception as error:
        with (args.output_dir / "failed_valuation_attempt.json").open("x") as stream:
            json.dump({"stage": "FAILED_TRAIN_VALIDATION_VALUATION_ATTEMPT", "exception_class": type(error).__name__,
                       "test_prices_read": False, "source_snapshots_preserved": True}, stream)
        raise
    finally:
        for path in args.output_dir.rglob("*"):
            if path.is_file():
                path.chmod(0o400)
    print(json.dumps({"stage": "TRAIN_VALIDATION_VALUATION_STAGE_ONLY", "fit_count": result["fit_count"],
                      "test_prices_read": False, "policy_neural_or_test_stage_completed": False}))


if __name__ == "__main__":
    main()
