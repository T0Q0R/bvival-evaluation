"""Exploratory group diagnostics for frozen global AutoScout24 policies.

No subgroup hypothesis test is performed. Price tertiles come exclusively
from the training partition. Policy actions are selected globally first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_bvival_mask_sensitivity import global_selection
from evaluate_bvival_policies import absolute_price_error_by_listing, validate_and_join


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_groups(
    outcomes: pd.DataFrame,
    scores: pd.DataFrame,
    features: pd.DataFrame,
    train: pd.DataFrame,
    *,
    budget_fraction: float = 0.1,
) -> tuple[pd.DataFrame, list[float]]:
    candidate = "score_mean_value"
    reference = "score_uncertainty_only"
    joined = validate_and_join(outcomes, scores, [candidate, reference])
    if features["listing_id"].duplicated().any():
        raise ValueError("Test feature listing IDs must be unique")
    if not {"country_code", "registration_year", "listing_id"}.issubset(features):
        raise ValueError("Test features lack country or registration year")
    train_prices = pd.to_numeric(train["price_value_EUR"], errors="raise")
    low, high = [float(value) for value in train_prices.quantile([1 / 3, 2 / 3])]
    selections = {
        policy: global_selection(joined, policy, budget_fraction)
        for policy in (candidate, reference)
    }
    errors = {
        policy: absolute_price_error_by_listing(joined, selections[policy])
        for policy in (candidate, reference)
    }
    listing_targets = joined[["listing_id", "target_log"]].drop_duplicates("listing_id")
    info = features[["listing_id", "country_code", "registration_year"]].merge(
        listing_targets, on="listing_id", how="inner", validate="one_to_one"
    )
    if len(info) != len(features) or len(info) != joined["listing_id"].nunique():
        raise ValueError("Feature/outcome listing universes differ")
    price = np.maximum(np.expm1(info["target_log"].to_numpy(dtype=float)), 0.0)
    year = pd.to_numeric(info["registration_year"], errors="raise")
    info["country"] = info["country_code"].fillna("Unknown")
    info["registration_group"] = np.select(
        [year <= 2018, year <= 2022],
        ["2018 or earlier", "2019 to 2022"], default="2023 or later",
    )
    info["price_group"] = np.select(
        [price <= low, price <= high],
        ["training lower tertile", "training middle tertile"],
        default="training upper tertile",
    )
    for policy, label in ((candidate, "candidate"), (reference, "reference")):
        info = info.merge(
            errors[policy].rename(columns={"absolute_price_error": f"{label}_error"}),
            on="listing_id", how="left", validate="one_to_one",
        )
        info[f"{label}_selected"] = info["listing_id"].isin(
            selections[policy]["listing_id"]
        )
    rows = []
    for dimension in ("country", "registration_group", "price_group"):
        for group, part in info.groupby(dimension, dropna=False, sort=True):
            candidate_mae = float(part["candidate_error"].mean())
            reference_mae = float(part["reference_error"].mean())
            rows.append({
                "dimension": dimension,
                "group": str(group),
                "listing_count": int(len(part)),
                "candidate_action_count": int(part["candidate_selected"].sum()),
                "reference_action_count": int(part["reference_selected"].sum()),
                "candidate_post_mae_EUR": candidate_mae,
                "reference_post_mae_EUR": reference_mae,
                "relative_mae_gain_percent": (
                    100.0 * (reference_mae - candidate_mae) / reference_mae
                    if reference_mae > 0 else float("nan")
                ),
                "status": "exploratory_descriptive_no_multiplicity_corrected_inference",
            })
    return pd.DataFrame(rows), [low, high]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-outcomes", type=Path, required=True)
    parser.add_argument("--frozen-policy-scores", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, cutpoints = evaluate_groups(
        pd.read_csv(args.evaluation_outcomes),
        pd.read_csv(args.frozen_policy_scores),
        pd.read_csv(args.test_features),
        pd.read_csv(args.train),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "exploratory_group_mae.csv"
    report.to_csv(output, index=False)
    audit = {
        "protocol": "posttest-exploratory-group-diagnostic",
        "global_policy_selection_before_grouping": True,
        "train_price_tertile_cutpoints_EUR": cutpoints,
        "input_sha256": {
            "outcomes": file_sha256(args.evaluation_outcomes),
            "scores": file_sha256(args.frozen_policy_scores),
            "test_features": file_sha256(args.test_features),
            "train": file_sha256(args.train),
        },
        "output_sha256": file_sha256(output),
        "nonclaim": "Descriptive only; no subgroup superiority, interactions, or causality",
    }
    (args.output_dir / "exploratory_group_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
