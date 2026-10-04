"""Descriptive predeclared feature-mask sensitivity on frozen global policies.

Actions are selected once on the full test universe at the original global
budget, then per-listing errors are restricted by a pretest feature-only mask.
No subgroup reranking, model fitting, or policy tuning is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bvi_val import select_unit_cost_actions
from evaluate_bvival_policies import (
    absolute_price_error_by_listing,
    paired_bootstrap_relative_mae,
    validate_and_join,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def global_selection(joined: pd.DataFrame, score_column: str, budget: float) -> pd.DataFrame:
    finite = joined[np.isfinite(joined[score_column].to_numpy(dtype=float))]
    actions = finite[["listing_id", "action_id", score_column]].rename(
        columns={score_column: "lower_value"}
    )
    return select_unit_cost_actions(actions, budget_fraction=budget)


def evaluate_mask(
    outcomes: pd.DataFrame,
    scores: pd.DataFrame,
    mask: pd.DataFrame,
    *,
    candidate_score: str,
    reference_score: str,
    budget_fraction: float,
    repetitions: int,
    seed: int,
) -> list[dict]:
    joined = validate_and_join(outcomes, scores, [candidate_score, reference_score])
    if mask["listing_id"].duplicated().any():
        raise ValueError("Mask listing IDs must be unique")
    if set(mask["listing_id"]) != set(joined["listing_id"]):
        raise ValueError("Mask must cover the full test listing universe")
    if "quasi_key_seen_in_non_test" not in mask.columns:
        raise ValueError("Mask lacks quasi_key_seen_in_non_test")
    candidate_selected = global_selection(joined, candidate_score, budget_fraction)
    reference_selected = global_selection(joined, reference_score, budget_fraction)
    candidate_errors = absolute_price_error_by_listing(joined, candidate_selected)
    reference_errors = absolute_price_error_by_listing(joined, reference_selected)
    groups = mask[["listing_id", "quasi_key_seen_in_non_test"]].copy()
    groups["quasi_key_seen_in_non_test"] = groups["quasi_key_seen_in_non_test"].astype(str).str.lower()
    if not groups["quasi_key_seen_in_non_test"].isin(["true", "false"]).all():
        raise ValueError("Mask values must be Boolean")
    result = []
    for seen in (False, True):
        selected_ids = set(groups.loc[
            groups["quasi_key_seen_in_non_test"].eq(str(seen).lower()), "listing_id"
        ])
        candidate_group = candidate_errors.loc[candidate_errors["listing_id"].isin(selected_ids)]
        reference_group = reference_errors.loc[reference_errors["listing_id"].isin(selected_ids)]
        if candidate_group.empty:
            continue
        comparison = paired_bootstrap_relative_mae(
            candidate_group, reference_group, repetitions=repetitions, seed=seed
        )
        comparison.update({
            "group": "novel_exact_spec_key" if not seen else "key_seen_in_non_test",
            "candidate_action_count": int(candidate_selected["listing_id"].isin(selected_ids).sum()),
            "reference_action_count": int(reference_selected["listing_id"].isin(selected_ids).sum()),
            "candidate_post_mae": float(candidate_group["absolute_price_error"].mean()),
            "reference_post_mae": float(reference_group["absolute_price_error"].mean()),
        })
        result.append(comparison)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-outcomes", type=Path, required=True)
    parser.add_argument("--frozen-policy-scores", type=Path, required=True)
    parser.add_argument("--pretest-mask", type=Path, required=True)
    parser.add_argument("--candidate-score", default="score_mean_value")
    parser.add_argument("--reference-score", default="score_uncertainty_only")
    parser.add_argument("--budget-fraction", type=float, default=0.1)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = evaluate_mask(
        pd.read_csv(args.evaluation_outcomes),
        pd.read_csv(args.frozen_policy_scores),
        pd.read_csv(args.pretest_mask),
        candidate_score=args.candidate_score,
        reference_score=args.reference_score,
        budget_fraction=args.budget_fraction,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "quasi_duplicate_sensitivity.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    audit = {
        "protocol": "posttest-computation-of-pretest-feature-mask-descriptive-sensitivity",
        "global_policy_selection_before_masking": True,
        "candidate_score": args.candidate_score,
        "reference_score": args.reference_score,
        "budget_fraction": args.budget_fraction,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "input_sha256": {
            "outcomes": file_sha256(args.evaluation_outcomes),
            "scores": file_sha256(args.frozen_policy_scores),
            "pretest_mask": file_sha256(args.pretest_mask),
        },
        "output_sha256": file_sha256(output),
        "interpretation": "descriptive exact-spec sensitivity; no verified duplicate identity",
    }
    (args.output_dir / "quasi_duplicate_sensitivity_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
