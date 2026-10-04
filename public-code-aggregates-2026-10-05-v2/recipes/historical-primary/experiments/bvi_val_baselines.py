"""Budget-matched baseline policies for BVI-Val.

These functions implement transparent local baselines. They intentionally do
not impersonate published active-feature-acquisition methods: ACO or another
external method must be run from a verified official implementation or marked
unavailable with the reason recorded.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


def stable_random_score(listing_id: object, action_id: object, seed: int) -> float:
    payload = f"{seed}|{listing_id}|{action_id}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64 - 1)


def fit_global_action_prior(
    train_actions: pd.DataFrame,
    *,
    target_column: str,
) -> dict[str, float]:
    required = {"action_id", target_column}
    missing = sorted(required.difference(train_actions.columns))
    if missing:
        raise ValueError(f"Training action table is missing columns: {missing}")
    values = pd.to_numeric(train_actions[target_column], errors="raise")
    if not np.isfinite(values).all():
        raise ValueError(f"{target_column} contains non-finite values")
    working = train_actions[["action_id"]].copy()
    working["target"] = values
    return {
        str(key): float(value)
        for key, value in working.groupby("action_id", sort=True)["target"].mean().items()
    }


def _require_candidate_columns(candidates: pd.DataFrame, columns: Iterable[str]) -> None:
    required = {"listing_id", "action_id", *columns}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Candidate table is missing columns: {missing}")
    if candidates.duplicated(["listing_id", "action_id"]).any():
        raise ValueError("Candidate listing-action pairs must be unique")


def _risk_routing_scores(
    candidates: pd.DataFrame,
    *,
    risk_column: str,
    action_prior: Mapping[str, float],
) -> np.ndarray:
    """Rank listings by risk and choose each listing's best global-prior action."""

    _require_candidate_columns(candidates, [risk_column])
    working = candidates[["listing_id", "action_id", risk_column]].copy()
    working["risk"] = pd.to_numeric(working[risk_column], errors="raise")
    if not np.isfinite(working["risk"]).all():
        raise ValueError(f"{risk_column} contains non-finite values")
    working["prior"] = working["action_id"].astype(str).map(action_prior)
    fallback = min(action_prior.values()) if action_prior else 0.0
    working["prior"] = working["prior"].fillna(fallback)
    best_index = (
        working.sort_values(
            ["listing_id", "prior", "action_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("listing_id", keep="first")
        .index
    )
    score = np.full(len(working), -np.inf)
    score[best_index.to_numpy()] = working.loc[best_index, "risk"].to_numpy(dtype=float)
    return score


def fixed_field_risk_scores(
    candidates: pd.DataFrame, *, action_id: str, risk_column: str
) -> np.ndarray:
    """Route the global listing-risk budget to one development-chosen field.

    Zero scores for other fields keep the budget denominator equal to all
    candidate listings; selection requires a strictly positive score.
    """

    _require_candidate_columns(candidates, [risk_column])
    if not action_id or action_id not in set(candidates["action_id"].astype(str)):
        raise ValueError("Fixed action must occur among candidate actions")
    risk = pd.to_numeric(candidates[risk_column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(risk).all():
        raise ValueError(f"{risk_column} contains non-finite values")
    chosen = candidates["action_id"].astype(str).eq(action_id).to_numpy()
    return np.where(chosen, np.maximum(risk, 0.0), 0.0)


def build_baseline_score_table(
    candidates: pd.DataFrame,
    *,
    action_prior: Mapping[str, float],
    mean_value_column: str,
    lower_value_column: str,
    uncertainty_column: str,
    disagreement_column: str,
    random_seed: int,
    fixed_action: str | None = None,
) -> pd.DataFrame:
    """Return one comparable score column per transparent local policy."""

    _require_candidate_columns(
        candidates,
        [mean_value_column, lower_value_column, uncertainty_column, disagreement_column],
    )
    output = candidates[["listing_id", "action_id"]].copy()
    output["score_no_acquisition"] = 0.0
    output["score_random"] = [
        stable_random_score(listing, action, random_seed)
        for listing, action in zip(output["listing_id"], output["action_id"])
    ]
    fallback = min(action_prior.values()) if action_prior else 0.0
    output["score_global_field_prior"] = (
        candidates["action_id"].astype(str).map(action_prior).fillna(fallback).to_numpy(dtype=float)
    )
    output["score_uncertainty_only"] = _risk_routing_scores(
        candidates, risk_column=uncertainty_column, action_prior=action_prior
    )
    output["score_disagreement_only"] = _risk_routing_scores(
        candidates, risk_column=disagreement_column, action_prior=action_prior
    )
    if fixed_action is not None:
        output["score_fixed_field_risk"] = fixed_field_risk_scores(
            candidates, action_id=fixed_action, risk_column=uncertainty_column
        )
    output["score_mean_value"] = pd.to_numeric(
        candidates[mean_value_column], errors="raise"
    ).to_numpy(dtype=float)
    output["score_conservative_lower_value"] = pd.to_numeric(
        candidates[lower_value_column], errors="raise"
    ).to_numpy(dtype=float)
    numeric = output.drop(columns=["listing_id", "action_id"])
    allowed_nonfinite = {"score_uncertainty_only", "score_disagreement_only"}
    for column in numeric:
        values = numeric[column].to_numpy(dtype=float)
        if column in allowed_nonfinite:
            if np.isnan(values).any() or np.isposinf(values).any():
                raise ValueError(f"{column} contains invalid values")
        elif not np.isfinite(values).all():
            raise ValueError(f"{column} contains non-finite values")
    return output


def external_baseline_record(
    *,
    name: str,
    source_url: str,
    implementation_status: str,
    exact_official_implementation: bool,
    reason: str,
) -> dict[str, object]:
    """Create an auditable status record for published external baselines."""

    if not name or not source_url or not implementation_status or not reason:
        raise ValueError("External baseline records require all descriptive fields")
    if implementation_status == "completed" and not exact_official_implementation:
        raise ValueError("A completed external baseline must use a verified exact implementation")
    return {
        "name": name,
        "source_url": source_url,
        "implementation_status": implementation_status,
        "exact_official_implementation": exact_official_implementation,
        "reason": reason,
    }
