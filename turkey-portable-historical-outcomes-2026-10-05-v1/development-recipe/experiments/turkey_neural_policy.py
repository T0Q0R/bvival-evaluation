"""Independent one-step expected-loss adaptation, NOT official GDFS.

Enumerated signed nested-OOF gains supervise a fixed-predictor field selector.
No acquisition true values, after predictions, identifiers or target columns
enter the network. The common risk cohort is selected outside this module.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from turkey_execution_contract import ACTIONS, budget_count
from turkey_policy_components import context_frame, post_mae, validate_material


def make_preprocessor(plan, config):
    features = plan["heads"]["features"]
    categorical = [f for f in plan["valuation"]["categorical_fields"] if f in features]
    numeric = [f for f in features if f not in categorical]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                               ("scale", StandardScaler())]), numeric),
        ("categorical", OneHotEncoder(max_categories=64, handle_unknown="infrequent_if_exist",
                                      sparse_output=False, dtype=np.float32), categorical),
    ], remainder="drop", sparse_threshold=0)


def make_network(width, settings):
    hidden = settings["hidden_width"]
    return nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(settings["dropout"]),
                         nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 3))


def negative_expected_gain(logits, gains, scale):
    if logits.shape != gains.shape or logits.ndim != 2 or logits.shape[1] != 3 or not np.isfinite(scale) or scale < 1:
        raise ValueError("Aligned three-action logits/gains and train-only scale required")
    return (torch.softmax(logits, dim=1) * (-gains / scale)).sum(dim=1).mean()


def probabilities(network, matrix):
    network.eval()
    with torch.no_grad():
        result = torch.softmax(network(torch.from_numpy(matrix)), dim=1).numpy()
    if result.shape != (len(matrix), 3) or not np.isfinite(result).all():
        raise ValueError("Finite neural probabilities required")
    return result


def fit_network(train_x, gains, valid_x, settings, *, seed, learning_rate, epochs,
                scale, validation=None, cohort=None):
    """Fixed-epoch final fits receive no validation supervision/selection hook."""
    if (train_x.ndim != 2 or valid_x.ndim != 2 or train_x.shape[1] != valid_x.shape[1]
            or np.shape(gains) != (len(train_x), 3) or not len(train_x)
            or any(not np.isfinite(x).all() for x in (train_x, valid_x, gains))
            or epochs < 1 or (validation is None) != (cohort is None)):
        raise ValueError("Invalid neural fitting roles or matrices")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    network = make_network(train_x.shape[1], settings)
    opt = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=settings["weight_decay"])
    x, targets = torch.from_numpy(train_x), torch.from_numpy(np.asarray(gains, dtype=np.float32))
    best_state, best_epoch, best_loss, stale, history = None, None, np.inf, 0, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        network.train()
        accumulated = 0.0
        order = rng.permutation(len(x))
        for start in range(0, len(x), settings["batch_size"]):
            index = order[start:start + settings["batch_size"]]
            opt.zero_grad()
            loss = negative_expected_gain(network(x[index]), targets[index], scale)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite neural training loss")
            loss.backward()
            opt.step()
            accumulated += float(loss.detach()) * len(index)
        row = {"epoch": epoch, "train_scaled_negative_expected_gain": accumulated / len(x)}
        if validation is not None:
            probability = probabilities(network, valid_x)
            choices = np.argmax(probability, axis=1)
            positions = {key: i for i, key in enumerate(validation["keys"])}
            allocation = [(key, ACTIONS[int(choices[positions[key]])]) for key in cohort]
            current = post_mae(validation, allocation)
            row["validation_common_risk_post_mae"] = current
            if current < best_loss:
                best_state, best_epoch, best_loss, stale = copy.deepcopy(network.state_dict()), epoch, current, 0
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= settings["patience"]:
            break
    if validation is not None:
        network.load_state_dict(best_state)
    else:
        best_epoch = epochs
    return network, probabilities(network, valid_x), {
        "seed": seed, "learning_rate": learning_rate, "epochs_run": len(history), "selected_epoch": best_epoch,
        "validation_selection_post_mae": float(best_loss) if validation is not None else None,
        "validation_labels_used": validation is not None, "history": history,
        "fit_seconds": time.perf_counter() - started,
    }


def run_neural_policy(train, validation, plan, config, cohort, *, output_directory=None):
    validate_material(train, plan, config, train=True)
    validate_material(validation, plan, config, train=False)
    if (set(train["keys"]) & set(validation["keys"])
            or len(cohort) != budget_count(len(validation["keys"]), plan["primary_budget"])
            or len(set(cohort)) != len(cohort) or not set(cohort).issubset(validation["keys"])):
        raise ValueError("Separate development partitions and exact common risk cohort required")
    settings = plan["neural_comparator"]
    if (settings["identity"] != "independent_one_step_expected_loss_adaptation_not_official_GDFS"
            or settings["official_method_reproduction_completed"] is not False
            or settings["learning_rates"] != [0.0003, 0.001, 0.003]):
        raise ValueError("Neural comparator identity or declared grid changed")
    train_frame, _ = context_frame(train["contexts"], plan, config)
    valid_frame, _ = context_frame(validation["contexts"], plan, config)
    preprocess = make_preprocessor(plan, config)
    train_x = np.asarray(preprocess.fit_transform(train_frame.loc[np.asarray(train["valid_mask"])]), dtype=np.float32)
    valid_x = np.asarray(preprocess.transform(valid_frame), dtype=np.float32)
    if not np.isfinite(train_x).all() or not np.isfinite(valid_x).all():
        raise ValueError("Nonfinite train-fitted neural matrix")
    scale = train["targets"]["common_training_scale"]
    trials, finals, per_seed = [], [], {}
    torch.set_num_threads(4)
    for rate in settings["learning_rates"]:
        network, _, trial = fit_network(train_x, train["targets"]["signed_gain"], valid_x, settings,
                                        seed=42, learning_rate=rate, epochs=settings["maximum_epochs"], scale=scale,
                                        validation=validation, cohort=cohort)
        trials.append(trial)
        if output_directory is not None:
            torch.save(network.state_dict(), output_directory / f"candidate_lr_{rate}.pt")
    selected = min(trials, key=lambda t: (t["validation_selection_post_mae"], t["learning_rate"], t["selected_epoch"]))
    for seed in plan["seeds"]:
        network, probability, audit = fit_network(train_x, train["targets"]["signed_gain"], valid_x, settings,
                                                  seed=seed, learning_rate=selected["learning_rate"],
                                                  epochs=selected["selected_epoch"], scale=scale)
        per_seed[str(seed)] = probability.tolist()
        finals.append(audit)
        if output_directory is not None:
            torch.save(network.state_dict(), output_directory / f"final_seed_{seed}.pt")
    if output_directory is not None:
        joblib.dump(preprocess, output_directory / "train_fitted_preprocessing.joblib")
    return {"stage": "DEVELOPMENT_NEURAL_FIELD_SELECTOR_COMPLETED_NOT_OFFICIAL_GDFS",
            "identity": settings["identity"], "upstream_commit": settings["upstream_commit"],
            "official_method_reproduction_completed": False,
            "probability": np.mean(list(per_seed.values()), axis=0).tolist(), "per_seed_probability": per_seed,
            "selected_learning_rate": selected["learning_rate"], "selected_epoch": selected["selected_epoch"],
            "candidate_trials": trials, "final_fit_audit": finals, "network_fit_count": len(trials) + len(finals),
            "network_input_width": train_x.shape[1], "common_training_scale": scale,
            "risk_cohort_keys": cohort, "validation_is_selection_not_replication": True,
            "calibration_or_test_labels_used": False, "raw_acquisition_values_in_network": False}


def restore_material(value):
    result = copy.deepcopy(value)
    if not isinstance(result.get("valid_mask"), list) or any(type(v) is not bool for v in result["valid_mask"]):
        raise ValueError("Bridge validity flags must be JSON booleans, not truthy strings")
    result["valid_mask"] = np.asarray(result["valid_mask"], dtype=bool)
    for name in ("before_absolute_error", "after_absolute_error", "signed_gain"):
        result["targets"][name] = np.asarray(result["targets"][name], dtype=float)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(args.task.read_text())
    if set(task) != {"stage", "scope", "train", "validation", "plan", "config", "cohort", "expected_environment"}:
        raise ValueError("Exact neural bridge roles required")
    if task["stage"] not in {"synthetic_only", "authorized_development_only"} or task["scope"] != ["train", "validation"]:
        raise ValueError("No calibration/test neural inputs accepted")
    env = {"python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in task["expected_environment"]["packages"]}}
    if env != task["expected_environment"]:
        raise ValueError("Neural environment differs from frozen versions")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("New empty neural output required")
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output_dir.chmod(0o700)
    try:
        result = run_neural_policy(restore_material(task["train"]), restore_material(task["validation"]),
                                   task["plan"], task["config"], task["cohort"], output_directory=args.output_dir)
        result["runtime_environment"] = env
        with (args.output_dir / "neural_audit.json").open("x") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps({"stage": result["stage"], "network_fit_count": result["network_fit_count"]}))
    except Exception as error:
        with (args.output_dir / "failed_neural_attempt.json").open("x") as stream:
            json.dump({"stage": "FAILED_NEURAL_ATTEMPT", "exception_class": type(error).__name__}, stream)
        raise
    finally:
        for path in args.output_dir.rglob("*"):
            if path.is_file():
                path.chmod(0o400)


if __name__ == "__main__":
    main()
