"""Core policy primitives for prospective BVI-Val experiments.

BVI-Val treats missing-information handling as a one-step allocation problem.
This module deliberately contains no DVM-CAR paths and performs no test-set I/O.
It can therefore be unit-tested before a prospective evaluation source is frozen.

The primary policy target is reduction in squared log error because minimizing
its mean is exactly aligned with RMSLE. A separate absolute-price objective is
provided for management analyses; the two objectives must not be blended after
evaluation outcomes are inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack


SUPPORTED_OBJECTIVES = ("squared_log_error", "absolute_price_error")


def _as_finite_vector(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _check_same_length(**vectors: np.ndarray) -> None:
    lengths = {name: len(values) for name, values in vectors.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Vectors have inconsistent lengths: {lengths}")


def realized_action_value(
    *,
    target_log: Iterable[float],
    before_prediction_log: Iterable[float],
    after_prediction_log: Iterable[float],
    objective: str = "squared_log_error",
) -> np.ndarray:
    """Return per-listing loss reduction after acquiring one action.

    Positive values mean that the action improved the prediction. For the
    absolute-price objective, log predictions are transformed with ``expm1`` and
    clipped at zero before error calculation.
    """

    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(
            f"Unsupported objective {objective!r}; expected one of {SUPPORTED_OBJECTIVES}"
        )
    target = _as_finite_vector(target_log, "target_log")
    before = _as_finite_vector(before_prediction_log, "before_prediction_log")
    after = _as_finite_vector(after_prediction_log, "after_prediction_log")
    _check_same_length(target=target, before=before, after=after)

    if objective == "squared_log_error":
        return np.square(target - before) - np.square(target - after)

    target_price = np.maximum(np.expm1(target), 0.0)
    before_price = np.maximum(np.expm1(before), 0.0)
    after_price = np.maximum(np.expm1(after), 0.0)
    return np.abs(target_price - before_price) - np.abs(target_price - after_price)


def _finite_sample_upper_quantile(scores: np.ndarray, alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    if len(scores) == 0:
        raise ValueError("at least one calibration score is required")
    rank = min(int(np.ceil((len(scores) + 1) * (1.0 - alpha))), len(scores))
    return float(np.partition(scores, rank - 1)[rank - 1])


@dataclass
class OneSidedActionValueCalibrator:
    """Marginal lower calibration for predicted action values.

    The nonconformity score is ``predicted_value - realized_value``. Subtracting
    its finite-sample upper quantile from a future prediction yields a marginal
    one-sided lower bound under exchangeability. Per-action corrections are
    used only when their calibration groups meet ``minimum_group_size``;
    otherwise the pooled correction is used.
    """

    alpha: float = 0.10
    minimum_group_size: int = 100
    nonnegative_correction: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        if self.minimum_group_size < 1:
            raise ValueError("minimum_group_size must be positive")
        self.pooled_correction_: float | None = None
        self.action_corrections_: dict[str, float] = {}
        self.action_counts_: dict[str, int] = {}

    def fit(
        self,
        predicted_value: Iterable[float],
        realized_value: Iterable[float],
        action_id: Iterable[str] | None = None,
    ) -> "OneSidedActionValueCalibrator":
        predicted = _as_finite_vector(predicted_value, "predicted_value")
        realized = _as_finite_vector(realized_value, "realized_value")
        _check_same_length(predicted=predicted, realized=realized)
        scores = predicted - realized
        pooled = _finite_sample_upper_quantile(scores, self.alpha)
        self.pooled_correction_ = max(pooled, 0.0) if self.nonnegative_correction else pooled
        self.action_corrections_ = {}
        self.action_counts_ = {}

        if action_id is None:
            return self
        actions = np.asarray(list(action_id), dtype=object)
        if actions.ndim != 1 or len(actions) != len(predicted):
            raise ValueError("action_id must be one-dimensional and match prediction length")
        for action in sorted({str(value) for value in actions}):
            mask = np.asarray([str(value) == action for value in actions], dtype=bool)
            count = int(mask.sum())
            self.action_counts_[action] = count
            if count >= self.minimum_group_size:
                correction = _finite_sample_upper_quantile(scores[mask], self.alpha)
                self.action_corrections_[action] = (
                    max(correction, 0.0) if self.nonnegative_correction else correction
                )
        return self

    def lower_bound(
        self,
        predicted_value: Iterable[float],
        action_id: Iterable[str] | None = None,
    ) -> np.ndarray:
        predicted = _as_finite_vector(predicted_value, "predicted_value")
        if self.pooled_correction_ is None:
            raise RuntimeError("fit must be called before lower_bound")
        if action_id is None:
            return predicted - self.pooled_correction_
        actions = np.asarray(list(action_id), dtype=object)
        if actions.ndim != 1 or len(actions) != len(predicted):
            raise ValueError("action_id must be one-dimensional and match prediction length")
        corrections = np.asarray(
            [
                self.action_corrections_.get(str(action), self.pooled_correction_)
                for action in actions
            ],
            dtype=float,
        )
        return predicted - corrections

    def audit(self) -> Mapping[str, object]:
        if self.pooled_correction_ is None:
            raise RuntimeError("fit must be called before audit")
        return {
            "alpha": self.alpha,
            "minimum_group_size": self.minimum_group_size,
            "nonnegative_correction": self.nonnegative_correction,
            "pooled_correction": self.pooled_correction_,
            "action_corrections": dict(self.action_corrections_),
            "action_counts": dict(self.action_counts_),
        }


REQUIRED_ACTION_COLUMNS = ("listing_id", "action_id", "lower_value")


def _validate_action_table(actions: pd.DataFrame, require_cost: bool = False) -> pd.DataFrame:
    required = set(REQUIRED_ACTION_COLUMNS)
    if require_cost:
        required.add("cost")
    missing = sorted(required.difference(actions.columns))
    if missing:
        raise ValueError(f"Action table is missing columns: {missing}")
    if actions.empty:
        return actions.copy()
    working = actions.copy().reset_index(drop=False).rename(columns={"index": "source_row"})
    if working[["listing_id", "action_id"]].isna().any().any():
        raise ValueError("listing_id and action_id must not be missing")
    if working.duplicated(["listing_id", "action_id"]).any():
        raise ValueError("Each listing-action pair must be unique")
    working["lower_value"] = pd.to_numeric(working["lower_value"], errors="raise")
    if not np.isfinite(working["lower_value"]).all():
        raise ValueError("lower_value contains non-finite values")
    if require_cost:
        working["cost"] = pd.to_numeric(working["cost"], errors="raise")
        if not np.isfinite(working["cost"]).all() or (working["cost"] <= 0).any():
            raise ValueError("cost must contain finite positive values")
    return working


def select_unit_cost_actions(
    actions: pd.DataFrame,
    *,
    budget_fraction: float,
    lambda_cost: float = 0.0,
) -> pd.DataFrame:
    """Select at most one action per listing under a unit-cost budget."""

    if not 0.0 <= budget_fraction <= 1.0:
        raise ValueError("budget_fraction must lie in [0, 1]")
    if lambda_cost < 0:
        raise ValueError("lambda_cost must be nonnegative")
    working = _validate_action_table(actions)
    if working.empty or budget_fraction == 0.0:
        return working.iloc[0:0].copy()

    working["net_lower_value"] = working["lower_value"] - lambda_cost
    working = working[working["net_lower_value"] > 0].copy()
    if working.empty:
        return working
    working = working.sort_values(
        ["listing_id", "net_lower_value", "action_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop_duplicates("listing_id", keep="first")
    eligible_listings = int(actions["listing_id"].nunique())
    capacity = min(int(np.ceil(eligible_listings * budget_fraction)), len(working))
    selected = working.sort_values(
        ["net_lower_value", "listing_id", "action_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(capacity)
    return selected.reset_index(drop=True)


def select_heterogeneous_cost_actions(
    actions: pd.DataFrame,
    *,
    budget: float,
    lambda_cost: float = 0.0,
    time_limit_seconds: float | None = None,
) -> pd.DataFrame:
    """Solve the one-action-per-listing budget problem as a binary MILP.

    The objective maximizes ``lower_value - lambda_cost * cost``. Candidates
    with nonpositive net lower value are discarded before optimization.
    """

    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if lambda_cost < 0:
        raise ValueError("lambda_cost must be nonnegative")
    working = _validate_action_table(actions, require_cost=True)
    if working.empty or budget == 0:
        return working.iloc[0:0].copy()
    working["net_lower_value"] = working["lower_value"] - lambda_cost * working["cost"]
    working = working[working["net_lower_value"] > 0].reset_index(drop=True)
    if working.empty:
        return working

    listing_codes, unique_listings = pd.factorize(working["listing_id"], sort=True)
    row_index = listing_codes
    col_index = np.arange(len(working))
    listing_matrix = coo_matrix(
        (np.ones(len(working)), (row_index, col_index)),
        shape=(len(unique_listings), len(working)),
    ).tocsr()
    cost_matrix = coo_matrix(
        (working["cost"].to_numpy(dtype=float), (np.zeros(len(working)), col_index)),
        shape=(1, len(working)),
    ).tocsr()
    constraint_matrix = vstack([listing_matrix, cost_matrix], format="csr")
    lower = np.full(len(unique_listings) + 1, -np.inf)
    upper = np.concatenate([np.ones(len(unique_listings)), np.asarray([budget])])
    options: dict[str, float | bool] = {"disp": False}
    if time_limit_seconds is not None:
        if time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        options["time_limit"] = float(time_limit_seconds)
    result = milp(
        c=-working["net_lower_value"].to_numpy(dtype=float),
        integrality=np.ones(len(working), dtype=int),
        bounds=Bounds(np.zeros(len(working)), np.ones(len(working))),
        constraints=LinearConstraint(constraint_matrix, lower, upper),
        options=options,
    )
    if result.x is None or not result.success:
        raise RuntimeError(f"Budget allocator failed: status={result.status}, {result.message}")
    selected = working[np.asarray(result.x) > 0.5].copy()
    if selected["listing_id"].duplicated().any():
        raise AssertionError("MILP selected more than one action for a listing")
    if float(selected["cost"].sum()) > budget + 1e-8:
        raise AssertionError("MILP solution exceeds the budget")
    return selected.sort_values(["listing_id", "action_id"], kind="mergesort").reset_index(
        drop=True
    )


def policy_post_action_predictions(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Construct one post-action prediction per listing for policy evaluation."""

    required = {
        "listing_id",
        "action_id",
        "target_log",
        "before_prediction_log",
        "after_prediction_log",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Candidate table is missing evaluation columns: {missing}")
    if candidates.duplicated(["listing_id", "action_id"]).any():
        raise ValueError("Candidate listing-action pairs must be unique")
    consistency = candidates.groupby("listing_id").agg(
        target_values=("target_log", "nunique"),
        before_values=("before_prediction_log", "nunique"),
    )
    if (consistency[["target_values", "before_values"]] > 1).any().any():
        raise ValueError("Target and before-action prediction must be constant within listing")

    base = candidates.groupby("listing_id", as_index=False).first()[
        ["listing_id", "target_log", "before_prediction_log"]
    ]
    base["post_prediction_log"] = base["before_prediction_log"]
    base["selected_action_id"] = pd.NA
    if selected.empty:
        return base
    chosen = selected[["listing_id", "action_id"]].merge(
        candidates[["listing_id", "action_id", "after_prediction_log"]],
        on=["listing_id", "action_id"],
        how="left",
        validate="one_to_one",
    )
    if chosen["after_prediction_log"].isna().any():
        raise ValueError("Selected action is absent from the candidate table")
    if chosen["listing_id"].duplicated().any():
        raise ValueError("Selected table contains multiple actions for one listing")
    indexed = base.set_index("listing_id")
    chosen_index = chosen.set_index("listing_id")
    indexed.loc[chosen_index.index, "post_prediction_log"] = chosen_index[
        "after_prediction_log"
    ]
    indexed.loc[chosen_index.index, "selected_action_id"] = chosen_index["action_id"]
    return indexed.reset_index()


def evaluate_policy(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    objective: str = "squared_log_error",
) -> Mapping[str, float | int]:
    """Evaluate a selected one-step policy on listing-level predictions."""

    post = policy_post_action_predictions(candidates, selected)
    target = _as_finite_vector(post["target_log"], "target_log")
    before = _as_finite_vector(post["before_prediction_log"], "before_prediction_log")
    after = _as_finite_vector(post["post_prediction_log"], "post_prediction_log")
    before_rmsle = float(np.sqrt(np.mean(np.square(target - before))))
    after_rmsle = float(np.sqrt(np.mean(np.square(target - after))))
    target_price = np.maximum(np.expm1(target), 0.0)
    before_price = np.maximum(np.expm1(before), 0.0)
    after_price = np.maximum(np.expm1(after), 0.0)
    before_mae = float(np.mean(np.abs(target_price - before_price)))
    after_mae = float(np.mean(np.abs(target_price - after_price)))

    chosen = post[post["selected_action_id"].notna()]
    if chosen.empty:
        harmful_rate = float("nan")
    else:
        chosen_value = realized_action_value(
            target_log=chosen["target_log"],
            before_prediction_log=chosen["before_prediction_log"],
            after_prediction_log=chosen["post_prediction_log"],
            objective=objective,
        )
        harmful_rate = float(np.mean(chosen_value < 0))
    return {
        "listing_count": int(len(post)),
        "action_count": int(len(chosen)),
        "action_rate": float(len(chosen) / len(post)) if len(post) else float("nan"),
        "before_rmsle": before_rmsle,
        "post_rmsle": after_rmsle,
        "relative_rmsle_improvement_percent": (
            100.0 * (before_rmsle - after_rmsle) / before_rmsle
            if before_rmsle > 0
            else float("nan")
        ),
        "before_mae_price": before_mae,
        "post_mae_price": after_mae,
        "harmful_action_rate": harmful_rate,
    }
