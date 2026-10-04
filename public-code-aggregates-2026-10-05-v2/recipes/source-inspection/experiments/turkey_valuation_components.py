"""Guarded shared-predictor components; no source I/O or full experiment runner.

Fitted only on caller-supplied states/targets. Group nesting and correct training
scope remain the responsibility of a separately audited integration runner.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from turkey_execution_contract import ACTIONS, price_is_valid


def _float32(matrix):
    return np.asarray(matrix, dtype=np.float32)


class ValuationStateGuard(TransformerMixin, BaseEstimator):
    """Reject extras and malformed one-reveal states before any learned transform."""

    def __init__(self, categorical_fields, numeric_fields, missing_token="__MISSING__"):
        self.categorical_fields = categorical_fields
        self.numeric_fields = numeric_fields
        self.missing_token = missing_token

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        expected = self.categorical_fields + self.numeric_fields
        if set(frame.columns) != set(expected) or len(frame.columns) != len(expected):
            raise ValueError("Valuation input must contain exactly the allowed state fields, no metadata/targets")
        for row in frame.to_dict("records"):
            revealed = 0
            for field in ACTIONS:
                hidden = row["hidden_" + field]
                if not isinstance(hidden, (int, np.integer)) or isinstance(hidden, bool) or hidden not in (0, 1):
                    raise ValueError("Hidden masks must be binary integers")
                value = row[field]
                if not isinstance(value, str) or (hidden == 1 and value != "") or (hidden == 0 and not value):
                    raise ValueError("Hidden field values must be empty; a reveal requires one scalar string")
                revealed += 1 - hidden
            if revealed > 1:
                raise ValueError("At most one field may be revealed")
        for field in self.categorical_fields:
            if any(not isinstance(value, str) for value in frame[field]):
                raise ValueError("Categorical values must be source-normalized strings")
            frame[field] = frame[field].replace("", self.missing_token)
        for field in self.numeric_fields:
            values = []
            for value in frame[field]:
                if value == "":
                    values.append(np.nan)
                else:
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        raise ValueError("Nonempty numeric state value is not numeric") from None
                    if not math.isfinite(number):
                        raise ValueError("Nonempty numeric state value must be finite")
                    values.append(number)
            frame[field] = values
        return frame[expected]


def make_valuation_model(plan, candidate_id, seed):
    if seed not in plan["seeds"]:
        raise ValueError("Unplanned model seed")
    matches = [c for c in plan["valuation"]["candidates"] if c["id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError("Exactly one declared valuation candidate required")
    candidate = matches[0]
    valuation, family = plan["valuation"], candidate["family"]
    categorical, numeric = valuation["categorical_fields"], valuation["numeric_fields"]
    prep = valuation["preprocessing"]
    guard = ValuationStateGuard(categorical, numeric, prep["categorical_missing_token"])
    params = {**copy.deepcopy(candidate["params"]), **copy.deepcopy(valuation["fixed_options"][family])}
    if family == "catboost":
        model = CatBoostRegressor(**params, random_seed=seed, cat_features=categorical)
        return Pipeline([("state_guard", guard), ("model", model)])
    transform = ColumnTransformer([
        ("numeric", SimpleImputer(**prep["numeric_imputer"]), numeric),
        ("categorical", OneHotEncoder(max_categories=prep["onehot_max_categories"],
                                      handle_unknown=prep["onehot_unknown"], sparse_output=False,
                                      dtype=np.float32), categorical),
    ], remainder="drop", sparse_threshold=0)
    if family == "extra_trees":
        model = ExtraTreesRegressor(**params, random_state=seed)
    elif family == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(**params, random_state=seed)
    else:
        raise ValueError("Unplanned valuation family")
    return Pipeline([("state_guard", guard), ("train_fitted_preprocess", transform),
                     ("float32", FunctionTransformer(_float32)), ("model", model)])


def fit_valuation_model(model, states, prices):
    if len(states) != len(prices) or not len(states) or not all(price_is_valid(p) for p in prices):
        raise ValueError("Every fitting state requires an aligned finite positive numeric price")
    target = np.log1p(np.asarray(prices, dtype=float))
    return model.fit(states, target)


def ensemble_prediction(log_predictions):
    predictions = np.asarray(log_predictions, dtype=float)
    if predictions.ndim != 2 or predictions.shape[0] != 3 or not np.isfinite(predictions).all():
        raise ValueError("Three aligned finite seed prediction vectors required")
    logs = np.maximum(predictions.mean(axis=0), 0.0)
    with np.errstate(over="raise", invalid="raise"):
        try:
            prices = np.expm1(logs)
        except FloatingPointError:
            raise ValueError("Overflow in price prediction; do not clip silently") from None
    return {"prediction_log": logs, "prediction_price": prices,
            "disagreement_log": predictions.std(axis=0, ddof=0)}


def signed_action_targets(prices, before_logs, after_logs):
    if not all(price_is_valid(p) for p in prices):
        raise ValueError("Invalid explicit training target")
    targets = np.asarray(prices, dtype=float)
    before, after = np.asarray(before_logs, dtype=float), np.asarray(after_logs, dtype=float)
    if targets.ndim != 1 or before.shape != targets.shape or after.shape != (len(targets), 3):
        raise ValueError("Aligned before and complete three-action predictions required")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Predictions must be finite")
    with np.errstate(over="raise", invalid="raise"):
        try:
            before_errors = np.abs(targets - np.expm1(np.maximum(before, 0)))
            after_errors = np.abs(targets[:, None] - np.expm1(np.maximum(after, 0)))
        except FloatingPointError:
            raise ValueError("Overflow in action targets") from None
    return {"before_absolute_error": before_errors, "after_absolute_error": after_errors,
            "signed_gain": before_errors[:, None] - after_errors,
            "common_training_scale": max(float(before_errors.mean()), 1.0)}
