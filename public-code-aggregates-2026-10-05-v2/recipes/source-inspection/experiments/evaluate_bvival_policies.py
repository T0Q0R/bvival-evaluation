"""Locked evaluator for frozen BVI-Val policy scores.

This is the only BVI-Val program intended to join evaluation outcomes to
already-frozen policy scores. It performs no model fitting or policy tuning.
The reference policy must be declared on the command line before evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from bvi_val import (
    evaluate_policy,
    policy_post_action_predictions,
    realized_action_value,
    select_unit_cost_actions,
)


OUTCOME_COLUMNS = (
    "listing_id",
    "action_id",
    "target_log",
    "before_prediction_log",
    "after_prediction_log",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-outcomes", type=Path, required=True)
    parser.add_argument("--frozen-policy-scores", type=Path, required=True)
    parser.add_argument("--additional-policy-scores", type=Path)
    parser.add_argument("--analysis-status", choices=("frozen_prospective", "post_test_exploratory"), default="frozen_prospective")
    parser.add_argument("--score-columns", required=True)
    parser.add_argument("--reference-policy", required=True)
    parser.add_argument("--objective", choices=("squared_log_error", "absolute_price_error"), required=True)
    parser.add_argument("--budgets", default="0.01,0.05,0.10,0.20,0.30")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze-id", required=True)
    parser.add_argument(
        "--include-oracle-diagnostic",
        action="store_true",
        help=(
            "Add a label-informed realized-gain upper bound inside the evaluator. "
            "This is never a deployable policy or frozen-score comparator."
        ),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("Budget list must be nonempty and unique")
    if any(item <= 0 or item > 1 for item in values):
        raise ValueError("Each budget must lie in (0, 1]")
    return sorted(values)


def validate_and_join(
    outcomes: pd.DataFrame,
    scores: pd.DataFrame,
    score_columns: Iterable[str],
) -> pd.DataFrame:
    missing_outcomes = sorted(set(OUTCOME_COLUMNS).difference(outcomes.columns))
    if missing_outcomes:
        raise ValueError(f"Evaluation outcomes are missing columns: {missing_outcomes}")
    score_columns = list(score_columns)
    missing_scores = sorted({"listing_id", "action_id", *score_columns}.difference(scores.columns))
    if missing_scores:
        raise ValueError(f"Frozen policy scores are missing columns: {missing_scores}")
    for name, frame in [("outcomes", outcomes), ("scores", scores)]:
        if frame.duplicated(["listing_id", "action_id"]).any():
            raise ValueError(f"{name} contains duplicate listing-action rows")
    joined = outcomes[list(OUTCOME_COLUMNS)].merge(
        scores[["listing_id", "action_id", *score_columns]],
        on=["listing_id", "action_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        counts = joined["_merge"].value_counts().to_dict()
        raise ValueError(f"Outcome and score listing-action universes differ: {counts}")
    joined = joined.drop(columns="_merge")
    for column in (*OUTCOME_COLUMNS[2:], *score_columns):
        joined[column] = pd.to_numeric(joined[column], errors="raise")
        values = joined[column].to_numpy(dtype=float)
        if np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError(f"{column} contains invalid values")
    return joined


def merge_additional_scores(base: pd.DataFrame, additional: pd.DataFrame) -> pd.DataFrame:
    keys = ["listing_id", "action_id"]
    for name, frame in [("base", base), ("additional", additional)]:
        if missing := sorted(set(keys).difference(frame.columns)):
            raise ValueError(f"{name} score table lacks keys: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} score table contains duplicate listing-action pairs")
    overlap = sorted((set(base.columns) & set(additional.columns)) - set(keys))
    if overlap:
        raise ValueError(f"Score tables have overlapping non-key columns: {overlap}")
    joined = base.merge(additional, on=keys, how="outer", validate="one_to_one", indicator=True)
    if len(joined) != len(base) or not joined["_merge"].eq("both").all():
        raise ValueError("Base and additional score universes differ")
    return joined.drop(columns="_merge")


def positive_gain_capture(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    objective: str,
) -> float:
    gain = realized_action_value(
        target_log=candidates["target_log"],
        before_prediction_log=candidates["before_prediction_log"],
        after_prediction_log=candidates["after_prediction_log"],
        objective=objective,
    )
    working = candidates[["listing_id", "action_id"]].copy()
    working["positive_gain"] = np.maximum(gain, 0.0)
    denominator = float(working["positive_gain"].sum())
    if denominator == 0:
        return float("nan")
    chosen = selected[["listing_id", "action_id"]].merge(
        working, on=["listing_id", "action_id"], how="left", validate="one_to_one"
    )
    return float(chosen["positive_gain"].sum() / denominator)


def squared_error_by_listing(candidates: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    post = policy_post_action_predictions(candidates, selected)
    post["squared_log_error"] = np.square(
        post["target_log"].to_numpy(dtype=float)
        - post["post_prediction_log"].to_numpy(dtype=float)
    )
    return post[["listing_id", "squared_log_error"]]


def absolute_price_error_by_listing(
    candidates: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    post = policy_post_action_predictions(candidates, selected)
    target = np.maximum(np.expm1(post["target_log"].to_numpy(dtype=float)), 0.0)
    prediction = np.maximum(
        np.expm1(post["post_prediction_log"].to_numpy(dtype=float)), 0.0
    )
    post["absolute_price_error"] = np.abs(target - prediction)
    return post[["listing_id", "absolute_price_error"]]


def paired_bootstrap_relative_rmsle(
    candidate_errors: pd.DataFrame,
    reference_errors: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    paired = candidate_errors.merge(
        reference_errors,
        on="listing_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_reference"),
    )
    if len(paired) != len(candidate_errors) or len(paired) != len(reference_errors):
        raise ValueError("Candidate and reference policies must cover identical listings")
    candidate = paired["squared_log_error_candidate"].to_numpy(dtype=float)
    reference = paired["squared_log_error_reference"].to_numpy(dtype=float)
    candidate_rmsle = float(np.sqrt(candidate.mean()))
    reference_rmsle = float(np.sqrt(reference.mean()))
    point = 100.0 * (reference_rmsle - candidate_rmsle) / reference_rmsle
    rng = np.random.default_rng(seed)
    boot = np.empty(repetitions, dtype=float)
    n = len(paired)
    for index in range(repetitions):
        sample = rng.integers(0, n, size=n)
        candidate_sample = float(np.sqrt(candidate[sample].mean()))
        reference_sample = float(np.sqrt(reference[sample].mean()))
        boot[index] = 100.0 * (reference_sample - candidate_sample) / reference_sample
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return {
        "n": n,
        "relative_rmsle_gain_percent": point,
        "ci_lower_percent": float(lower),
        "ci_upper_percent": float(upper),
    }


def paired_bootstrap_relative_mae(
    candidate_errors: pd.DataFrame,
    reference_errors: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    paired = candidate_errors.merge(
        reference_errors,
        on="listing_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_reference"),
    )
    if len(paired) != len(candidate_errors) or len(paired) != len(reference_errors):
        raise ValueError("Candidate and reference policies must cover identical listings")
    candidate = paired["absolute_price_error_candidate"].to_numpy(dtype=float)
    reference = paired["absolute_price_error_reference"].to_numpy(dtype=float)
    reference_mae = float(reference.mean())
    if reference_mae <= 0:
        raise ValueError("Reference MAE must be positive")
    point = 100.0 * (reference_mae - float(candidate.mean())) / reference_mae
    rng = np.random.default_rng(seed)
    boot = np.empty(repetitions, dtype=float)
    n = len(paired)
    for index in range(repetitions):
        sample = rng.integers(0, n, size=n)
        reference_sample = float(reference[sample].mean())
        boot[index] = (
            100.0 * (reference_sample - float(candidate[sample].mean())) / reference_sample
            if reference_sample > 0 else np.nan
        )
    lower, upper = np.nanquantile(boot, [0.025, 0.975])
    return {
        "n": n,
        "relative_mae_gain_percent": point,
        "ci_lower_percent": float(lower),
        "ci_upper_percent": float(upper),
    }


def error_budget_auc(curve: pd.DataFrame, metric_column: str = "post_rmsle") -> float:
    ordered = curve.sort_values("budget")
    if ordered["budget"].duplicated().any():
        raise ValueError("A policy curve must have one row per budget")
    budgets = ordered["budget"].to_numpy(dtype=float)
    error = ordered[metric_column].to_numpy(dtype=float)
    if len(budgets) < 2:
        return float("nan")
    return float(np.trapezoid(error, budgets) / (budgets[-1] - budgets[0]))


def evaluate_score_policies(
    joined: pd.DataFrame,
    *,
    score_columns: Iterable[str],
    budgets: Iterable[float],
    objective: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, float], pd.DataFrame]]:
    rows = []
    action_mix_rows = []
    selections: dict[tuple[str, float], pd.DataFrame] = {}
    for score_column in score_columns:
        finite = joined[np.isfinite(joined[score_column].to_numpy(dtype=float))].copy()
        for budget in budgets:
            action_table = finite[["listing_id", "action_id", score_column]].rename(
                columns={score_column: "lower_value"}
            )
            selected = select_unit_cost_actions(action_table, budget_fraction=budget)
            selections[(score_column, float(budget))] = selected
            metrics = dict(evaluate_policy(joined, selected, objective=objective))
            metrics.update(
                {
                    "policy": score_column,
                    "budget": float(budget),
                    "positive_gain_capture": positive_gain_capture(
                        joined, selected, objective=objective
                    ),
                }
            )
            rows.append(metrics)
            counts = selected["action_id"].value_counts()
            for action, count in counts.items():
                action_mix_rows.append(
                    {
                        "policy": score_column,
                        "budget": float(budget),
                        "action_id": action,
                        "count": int(count),
                        "share_of_actions": float(count / len(selected)),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(action_mix_rows), selections


def main() -> None:
    args = parse_args()
    score_columns = [item.strip() for item in args.score_columns.split(",") if item.strip()]
    if len(score_columns) != len(set(score_columns)) or not score_columns:
        raise ValueError("Score columns must be nonempty and unique")
    if args.reference_policy not in score_columns:
        raise ValueError("reference-policy must be one of the frozen score columns")
    budgets = parse_float_list(args.budgets)
    outcomes = pd.read_csv(args.evaluation_outcomes)
    scores = pd.read_csv(args.frozen_policy_scores)
    if args.additional_policy_scores:
        if args.analysis_status != "post_test_exploratory":
            raise ValueError("Additional scores require post_test_exploratory status")
        scores = merge_additional_scores(scores, pd.read_csv(args.additional_policy_scores))
    joined = validate_and_join(outcomes, scores, score_columns)
    oracle_policy_name = "diagnostic_oracle_realized_gain"
    if args.include_oracle_diagnostic:
        joined[oracle_policy_name] = realized_action_value(
            target_log=joined["target_log"],
            before_prediction_log=joined["before_prediction_log"],
            after_prediction_log=joined["after_prediction_log"],
            objective=args.objective,
        )
        score_columns = [*score_columns, oracle_policy_name]
    curve, action_mix, selections = evaluate_score_policies(
        joined,
        score_columns=score_columns,
        budgets=budgets,
        objective=args.objective,
    )

    auc_metric_column = (
        "post_rmsle" if args.objective == "squared_log_error" else "post_mae_price"
    )
    summaries = []
    for policy, group in curve.groupby("policy", sort=True):
        summaries.append(
            {
                "policy": policy,
                "error_budget_auc": error_budget_auc(group, auc_metric_column),
                "auc_metric": auc_metric_column,
                "mean_harmful_action_rate": float(group["harmful_action_rate"].mean()),
            }
        )
    comparisons = []
    for budget in budgets:
        if args.objective == "squared_log_error":
            reference_errors = squared_error_by_listing(
                joined, selections[(args.reference_policy, float(budget))]
            )
        else:
            reference_errors = absolute_price_error_by_listing(
                joined, selections[(args.reference_policy, float(budget))]
            )
        for policy in score_columns:
            if policy == args.reference_policy:
                continue
            if args.objective == "squared_log_error":
                candidate_errors = squared_error_by_listing(
                    joined, selections[(policy, float(budget))]
                )
                comparison = paired_bootstrap_relative_rmsle(
                    candidate_errors, reference_errors,
                    repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed,
                )
            else:
                candidate_errors = absolute_price_error_by_listing(
                    joined, selections[(policy, float(budget))]
                )
                comparison = paired_bootstrap_relative_mae(
                    candidate_errors, reference_errors,
                    repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed,
                )
            comparison.update(
                {"policy": policy, "reference_policy": args.reference_policy, "budget": budget}
            )
            comparisons.append(comparison)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.output_dir / "bvival_error_budget_curve.csv"
    action_mix_path = args.output_dir / "bvival_action_mix.csv"
    summary_path = args.output_dir / "bvival_policy_summary.csv"
    comparison_path = args.output_dir / "bvival_reference_comparisons.csv"
    curve.to_csv(curve_path, index=False)
    action_mix.to_csv(action_mix_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.DataFrame(comparisons).to_csv(comparison_path, index=False)
    audit = {
        "protocol": "v4-bvival-prospective" if args.analysis_status == "frozen_prospective" else "post-test exploratory comparator",
        "analysis_status": args.analysis_status,
        "freeze_id": args.freeze_id,
        "objective": args.objective,
        "auc_metric": auc_metric_column,
        "reference_policy": args.reference_policy,
        "score_columns": score_columns,
        "oracle_diagnostic_included": args.include_oracle_diagnostic,
        "oracle_diagnostic_name": (
            oracle_policy_name if args.include_oracle_diagnostic else None
        ),
        "budgets": budgets,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "evaluation_outcomes_sha256": file_sha256(args.evaluation_outcomes),
        "frozen_policy_scores_sha256": file_sha256(args.frozen_policy_scores),
        "additional_policy_scores_sha256": (
            file_sha256(args.additional_policy_scores) if args.additional_policy_scores else None
        ),
        "outputs": {
            path.name: file_sha256(path)
            for path in [curve_path, action_mix_path, summary_path, comparison_path]
        },
        "nonclaim": (
            "The evaluator reports predictive decision evidence and does not establish "
            "causal profit, welfare, or selection-conditional coverage."
        ),
    }
    (args.output_dir / "bvival_evaluation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
