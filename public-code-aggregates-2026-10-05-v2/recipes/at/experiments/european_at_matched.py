"""Matched P1--P4 components. No source, label-file, or evaluation I/O here."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from build_european_at_cohort import ACTIONS
from bvival_neighbor_available import AvailableNeighborGain

CONTEXT_FIELDS = ["brand", "family", "year", "before_prediction_log"]
CAT_CONTEXT = ["brand", "family"]
STATE_COLUMNS = ["brand", "family", "year", *ACTIONS, *["hidden_" + a for a in ACTIONS]]


def states(records):
    output = []
    for r in records:
        if set(r["context"]) != {"brand", "family", "year"} or set(r["actions"]) != set(ACTIONS):
            raise ValueError("Exact initial context and action allowlists required")
        for reveal in (None, *ACTIONS):
            s = dict(r["context"])
            for a in ACTIONS:
                hidden = int(a != reveal)
                s["hidden_" + a] = hidden
                s[a] = ("__HIDDEN__" if a in {"fuel", "transmission"} else np.nan) if hidden else r["actions"][a]
            output.append(s)
    return pd.DataFrame(output, columns=STATE_COLUMNS)


def contexts(records, log_predictions):
    logs = np.asarray(log_predictions, float)
    if logs.shape != (len(records), 5) or not np.isfinite(logs).all():
        raise ValueError("Five aligned finite state predictions required")
    return [{**r["context"], "before_prediction_log": float(logs[i, 0])}
            for i, r in enumerate(records)]


def context_frame(items, *, actions=False):
    expected = set(CONTEXT_FIELDS) | ({"action_id"} if actions else set())
    if not items or any(set(x) != expected for x in items):
        raise ValueError("Only exact pre-action context fields are permitted")
    frame = pd.DataFrame(items, columns=CONTEXT_FIELDS + (["action_id"] if actions else []))
    if not np.isfinite(frame[["year", "before_prediction_log"]].to_numpy(float)).all():
        raise ValueError("Nonfinite context")
    return frame


def gain_frame(items):
    return context_frame([{**c, "action_id": a} for c in items for a in ACTIONS], actions=True)


def new_model(options, seed, categorical):
    allowed = {"iterations", "depth", "learning_rate", "l2_leaf_reg", "loss_function", "thread_count"}
    params = {k: v for k, v in options.items() if k in allowed}
    return CatBoostRegressor(**params, random_seed=seed, cat_features=categorical,
                             allow_writing_files=False, verbose=False)


def fit_base(records, prices, config):
    y = np.asarray(prices, float)
    if y.shape != (len(records),) or not len(y) or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Aligned positive training prices required")
    matrix, target = states(records), np.repeat(np.log1p(y), 5)
    models = []
    for seed in config["model_seeds"]:
        model = new_model(config["base_predictor"], seed, ["brand", "family", "fuel", "transmission"])
        model.fit(matrix, target)
        models.append(model)
    return models


def predict_base(models, records):
    matrix = states(records)
    logs = np.asarray([m.predict(matrix) for m in models], float).mean(axis=0).reshape(len(records), 5)
    logs = np.maximum(logs, 0)
    if not np.isfinite(logs).all():
        raise ValueError("Nonfinite base prediction")
    return logs


def errors(prices, logs):
    y, z = np.asarray(prices, float), np.asarray(logs, float)
    if z.shape != (len(y), 5) or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Aligned valid observed targets required")
    with np.errstate(over="raise", invalid="raise"):
        result = np.abs(y[:, None] - np.expm1(z))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite error")
    return result


def fit_heads(items, losses, config):
    loss = np.asarray(losses, float)
    if loss.shape != (len(items), 5) or not np.isfinite(loss).all() or (loss < 0).any():
        raise ValueError("Five aligned OOF loss targets required")
    scale = max(float(loss[:, 0].mean()), 1.0)
    gains = loss[:, 0, None] - loss[:, 1:]
    gx, rx = gain_frame(items), context_frame(items)
    gain_models, risk_models = [], []
    for seed in config["model_seeds"]:
        g = new_model(config["gain_head"], seed, [*CAT_CONTEXT, "action_id"])
        g.fit(gx, (gains / scale).reshape(-1))
        gain_models.append(g)
        r = new_model(config["risk_head"], seed, CAT_CONTEXT)
        r.fit(rx, loss[:, 0] / scale)
        risk_models.append(r)
    return gain_models, risk_models, gains, scale


def predict_heads(gain_models, risk_models, items):
    gx, rx = gain_frame(items), context_frame(items)
    gain = np.asarray([m.predict(gx) for m in gain_models]).mean(axis=0).reshape(len(items), 4)
    risk = np.asarray([m.predict(rx) for m in risk_models]).mean(axis=0)
    if not np.isfinite(gain).all() or not np.isfinite(risk).all():
        raise ValueError("Nonfinite policy score")
    return gain, risk


def fit_neighbor(keys, items, gains, config):
    opts = config["neighbor"]
    model = AvailableNeighborGain(CAT_CONTEXT, ["year", "before_prediction_log"],
                                  max_categories=opts["max_categories"], working_memory=opts["working_memory_mib"])
    return model.fit_gains(keys, items, gains)


def allocation(keys, scores, actions, fraction):
    n = len(keys)
    s, a = np.asarray(scores, float), np.asarray(actions, int)
    if (not n or len(set(keys)) != n or s.shape != (n,) or a.shape != (n,) or
            not np.isfinite(s).all() or (a < 0).any() or (a >= 4).any() or not 0 < fraction <= 1):
        raise ValueError("Aligned unique keys and finite action scores required")
    budget = math.ceil(fraction * n)
    order = sorted(range(n), key=lambda i: (-s[i], keys[i]))[:budget]
    result = np.full(n, -1, dtype=int)
    result[order] = a[order]
    return result


def policies(keys, gain, risk, neighbor, fixed_field, fraction):
    g, r, ng = np.asarray(gain, float), np.asarray(risk, float), np.asarray(neighbor, float)
    n = len(keys)
    if g.shape != (n, 4) or ng.shape != (n, 4) or r.shape != (n,) or fixed_field not in ACTIONS:
        raise ValueError("Aligned common action universe and selected field required")
    fixed = ACTIONS.index(fixed_field)
    best, near = np.argmax(g, axis=1), np.argmax(ng, axis=1)
    return {
        "P1": allocation(keys, r, np.full(n, fixed), fraction),
        "P2": allocation(keys, g[:, fixed], np.full(n, fixed), fraction),
        "P3": allocation(keys, g[np.arange(n), best], best, fraction),
        "P4": allocation(keys, ng[np.arange(n), near], near, fraction),
    }


def selected_errors(losses, action_vector):
    a = np.asarray(action_vector, int)
    loss = np.asarray(losses, float)
    if loss.shape != (len(a), 5) or (a < -1).any() or (a >= 4).any():
        raise ValueError("Aligned five-state loss and immutable actions required")
    return loss[np.arange(len(a)), a + 1]


def observed_mae(logs, prices, action_vector):
    valid = np.array([p is not None for p in prices])
    if not valid.any():
        raise ValueError("No valid observed targets")
    loss = errors([p for p in prices if p is not None], np.asarray(logs)[valid])
    return float(selected_errors(loss, np.asarray(action_vector)[valid]).mean())


def bootstrap_pair(candidate, reference, *, seed=2026, replicates=10000):
    c, r = np.asarray(candidate, float), np.asarray(reference, float)
    if (not len(c) or c.shape != r.shape or not np.isfinite(c).all() or not np.isfinite(r).all()
            or (c < 0).any() or (r < 0).any() or r.mean() <= 0
            or not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1):
        raise ValueError("Aligned positive-reference paired errors required")
    rng, draws = np.random.default_rng(seed), []
    for offset in range(0, replicates, 100):
        ix = rng.integers(0, len(c), (min(100, replicates - offset), len(c)))
        cm, rm = c[ix].mean(axis=1), r[ix].mean(axis=1)
        if (rm <= 0).any():
            raise ValueError("Percentage improvement undefined for zero-error bootstrap reference")
        draws.extend((100 * (rm - cm) / rm).tolist())
    return {"relative_mae_improvement_percent": float(100 * (r.mean() - c.mean()) / r.mean()),
            "nominal_97_5_percent_interval": np.quantile(draws, [.0125, .9875]).tolist(),
            "replicates": replicates, "seed": seed, "paired_unique_proxy_representatives": len(c),
            "conditional_fixed_policies_not_retrained_or_reallocated": True}


def summarize(logs, prices, choices, config):
    valid = np.array([p is not None for p in prices])
    if not valid.any() or set(choices) != {"P1", "P2", "P3", "P4"}:
        raise ValueError("Valid targets and all four unchanged policies required")
    budget = math.ceil(config['budget_fraction'] * len(prices))
    if any(len(a) != len(prices) or sum(i >= 0 for i in a) != budget for a in choices.values()):
        raise ValueError("Every policy must retain the same full-candidate capacity")
    loss = errors([p for p in prices if p is not None], np.asarray(logs)[valid])
    per = {p: selected_errors(loss, np.asarray(a)[valid]) for p, a in choices.items()}
    summaries = {}
    for p, a in choices.items():
        a = np.asarray(a)
        selected_valid = a[valid] >= 0
        delta = loss[selected_valid, 0] - per[p][selected_valid]
        summaries[p] = {
            "mae_EUR_valid_target_cohort": float(per[p].mean()),
            "actions_all_nonprice_candidates": int((a >= 0).sum()),
            "actions_on_valid_target_records": int(selected_valid.sum()),
            "selected_field_counts_all_candidates": dict(Counter(ACTIONS[i] for i in a if i >= 0)),
            "valid_actions_reducing_absolute_error": int((delta > 0).sum()),
            "valid_actions_increasing_absolute_error": int((delta < 0).sum()),
            "valid_actions_unchanged_absolute_error": int((delta == 0).sum()),
            "net_absolute_error_reduction_EUR": float(delta.sum()),
        }
    comparisons = {f"P3_vs_{q}": bootstrap_pair(per['P3'], per[q], seed=config['bootstrap']['seed'],
                                               replicates=config['bootstrap']['replicates']) for q in ('P2', 'P4')}
    return {"nonprice_candidates_N": len(prices), "valid_observed_target_n": int(valid.sum()),
            "invalid_observed_targets": int((~valid).sum()), "before_mae_EUR": float(loss[:, 0].mean()),
            "policies": summaries, "primary_comparisons": comparisons}
