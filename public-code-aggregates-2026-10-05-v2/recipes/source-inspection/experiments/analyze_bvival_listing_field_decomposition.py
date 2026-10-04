"""Post-test exploratory decomposition of listing and field selection.

This analysis crosses the *frozen* listing sets selected by risk and predicted
action value with the *frozen* field rules (development-chosen fixed field and
highest predicted action-value field). It does not train, retune, or introduce
a new held-out test. Since the motivation followed test opening, all results
are explicitly exploratory and intervals are descriptive, not confirmatory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bvi_val import select_unit_cost_actions
from evaluate_bvival_policies import (
    absolute_price_error_by_listing,
    file_sha256,
    paired_bootstrap_relative_mae,
    validate_and_join,
)


RISK_SCORE = "score_uncertainty_only"
VALUE_SCORE = "score_mean_value"
CELL_NAMES = (
    "risk_listings_fixed_field",
    "risk_listings_value_field",
    "value_listings_fixed_field",
    "value_listings_value_field",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-policy-scores", type=Path, required=True)
    parser.add_argument("--evaluation-outcomes", type=Path, required=True)
    parser.add_argument("--fixed-field", required=True)
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def select_frozen_policy(scores: pd.DataFrame, score_column: str, budget: float) -> pd.DataFrame:
    required = {"listing_id", "action_id", score_column}
    if missing := sorted(required.difference(scores.columns)):
        raise ValueError(f"Frozen score table is missing columns: {missing}")
    finite = scores[np.isfinite(pd.to_numeric(scores[score_column], errors="raise"))]
    actions = finite[["listing_id", "action_id", score_column]].rename(
        columns={score_column: "lower_value"}
    )
    selected = select_unit_cost_actions(actions, budget_fraction=budget)
    return selected[["listing_id", "action_id"]].copy()


def best_value_field(scores: pd.DataFrame, listings: pd.Series) -> pd.DataFrame:
    candidates = scores[scores["listing_id"].isin(listings)].copy()
    if candidates.empty:
        return candidates[["listing_id", "action_id"]]
    values = pd.to_numeric(candidates[VALUE_SCORE], errors="raise")
    if not np.isfinite(values).all():
        raise ValueError("Action-value scores must be finite for selected listings")
    candidates["value_score"] = values
    result = (
        candidates.sort_values(
            ["listing_id", "value_score", "action_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("listing_id", keep="first")
        [["listing_id", "action_id"]]
    )
    if result["listing_id"].nunique() != listings.nunique():
        raise ValueError("At least one selected listing lacks an eligible action")
    return result.reset_index(drop=True)


def fixed_field_with_fallback(
    scores: pd.DataFrame,
    listings: pd.Series,
    fixed_field: str,
    fallback: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Keep the listing set and fall back to its frozen alternative if absent."""

    if not fixed_field:
        raise ValueError("fixed_field must be specified from development evidence")
    fixed = scores[
        scores["listing_id"].isin(listings) & scores["action_id"].eq(fixed_field)
    ][["listing_id", "action_id"]]
    if fixed.duplicated("listing_id").any():
        raise ValueError("A listing has duplicate fixed-field candidate actions")
    missing = listings[~listings.isin(fixed["listing_id"])]
    fallback_rows = fallback[fallback["listing_id"].isin(missing)]
    result = pd.concat([fixed, fallback_rows], ignore_index=True)
    if result["listing_id"].nunique() != listings.nunique() or len(result) != len(listings):
        raise ValueError("Fixed-field fallback failed to preserve the listing set")
    return result.sort_values("listing_id", kind="mergesort").reset_index(drop=True), len(missing)


def decompose(
    scores: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    fixed_field: str,
    budget: float,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    if scores.duplicated(["listing_id", "action_id"]).any():
        raise ValueError("Frozen scores contain duplicate listing-action pairs")
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive")
    # Decisions below use frozen scores only. Outcomes are joined afterwards.
    risk_original = select_frozen_policy(scores, RISK_SCORE, budget)
    value_original = select_frozen_policy(scores, VALUE_SCORE, budget)
    risk_listings = risk_original["listing_id"]
    value_listings = value_original["listing_id"]
    risk_value_field = best_value_field(scores, risk_listings)
    value_value_field = best_value_field(scores, value_listings)
    risk_fixed, risk_fallback = fixed_field_with_fallback(
        scores, risk_listings, fixed_field, risk_original
    )
    value_fixed, value_fallback = fixed_field_with_fallback(
        scores, value_listings, fixed_field, value_value_field
    )
    cells = dict(
        zip(
            CELL_NAMES,
            (risk_fixed, risk_value_field, value_fixed, value_value_field),
            strict=True,
        )
    )
    joined = validate_and_join(outcomes, scores, [RISK_SCORE, VALUE_SCORE])
    errors = {
        name: absolute_price_error_by_listing(joined, actions)
        for name, actions in cells.items()
    }
    mae = {
        name: float(frame["absolute_price_error"].mean())
        for name, frame in errors.items()
    }
    contrasts = {}
    for contrast, candidate, reference in (
        ("listing_effect_at_fixed_field", "value_listings_fixed_field", "risk_listings_fixed_field"),
        ("field_effect_on_value_listings", "value_listings_value_field", "value_listings_fixed_field"),
        ("field_effect_on_risk_listings", "risk_listings_value_field", "risk_listings_fixed_field"),
        ("total_value_vs_risk_fixed", "value_listings_value_field", "risk_listings_fixed_field"),
    ):
        contrasts[contrast] = paired_bootstrap_relative_mae(
            errors[candidate],
            errors[reference],
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        )
        contrasts[contrast]["candidate_cell"] = candidate
        contrasts[contrast]["reference_cell"] = reference
    selection_rows = [
        actions.assign(cell=name) for name, actions in cells.items()
    ]
    selection_table = pd.concat(selection_rows, ignore_index=True)
    result = {
        "analysis_status": "post-test exploratory; no prospective inference",
        "independent_resampling_unit": "listing",
        "budget": budget,
        "fixed_field": fixed_field,
        "test_listing_count": int(joined["listing_id"].nunique()),
        "selected_count_per_cell": {name: len(actions) for name, actions in cells.items()},
        "listing_set_overlap": int(len(set(risk_listings) & set(value_listings))),
        "risk_original_action_agreement_with_fixed": float(
            risk_original.merge(risk_fixed, on="listing_id", suffixes=("_original", "_fixed"))
            .eval("action_id_original == action_id_fixed")
            .mean()
        ),
        "value_original_action_agreement_with_value_field": float(
            value_original.merge(value_value_field, on="listing_id", suffixes=("_original", "_value"))
            .eval("action_id_original == action_id_value")
            .mean()
        ),
        "fixed_field_unavailable_fallback_count": {
            "risk_listings": risk_fallback,
            "value_listings": value_fallback,
        },
        "post_reveal_mae_by_cell": mae,
        "contrasts": contrasts,
        "uncertainty_boundary": (
            "Percentile intervals resample listings while keeping fitted models, scores, "
            "selected listing sets, and actions fixed. The four contrasts were devised "
            "after test opening; intervals are descriptive and not multiplicity adjusted."
        ),
    }
    return result, selection_table


def main() -> None:
    args = parse_args()
    scores = pd.read_csv(args.frozen_policy_scores)
    outcomes = pd.read_csv(args.evaluation_outcomes)
    result, selection_table = decompose(
        scores,
        outcomes,
        fixed_field=args.fixed_field,
        budget=args.budget,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["frozen_policy_scores_sha256"] = file_sha256(args.frozen_policy_scores)
    result["evaluation_outcomes_sha256"] = file_sha256(args.evaluation_outcomes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections_path = args.output_dir / "exploratory_crossed_selections.csv"
    result_path = args.output_dir / "exploratory_decomposition.json"
    selection_table.to_csv(selections_path, index=False)
    result["exploratory_crossed_selections_sha256"] = file_sha256(selections_path)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
