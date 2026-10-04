"""Assemble the frozen, label-free policy score table for BVI-Val.

All inputs are development artifacts or evaluation *features*.  The program
refuses evaluation outcomes and writes a hash-audited table consumed by the
locked evaluator after the external test labels are opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from bvi_val_baselines import build_baseline_score_table, fit_global_action_prior


FORBIDDEN_EVALUATION_COLUMNS = {
    "target_log",
    "price_value_JOD",
    "after_prediction_log",
    "value_squared_log_error",
    "value_absolute_price_error",
    "risk_target_squared_log_error_before",
    "risk_target_absolute_log_error_before",
    "risk_target_absolute_price_error_before",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-actions", type=Path, required=True)
    parser.add_argument("--value-scores", type=Path, required=True)
    parser.add_argument("--risk-scores", type=Path, required=True)
    parser.add_argument("--evaluation-features", type=Path, required=True)
    parser.add_argument(
        "--objective",
        choices=("squared_log_error", "absolute_price_error"),
        default="squared_log_error",
    )
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument(
        "--fixed-action",
        default=None,
        help=(
            "Explicit development-chosen action, or 'development_mean' to choose "
            "the largest development mean realized gain (lexical tie-break)."
        ),
    )
    parser.add_argument(
        "--fixed-action-file", type=Path,
        help="JSON from development-only fixed-field selection; mutually exclusive with --fixed-action",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_one_to_one(base: pd.DataFrame, other: pd.DataFrame, name: str) -> pd.DataFrame:
    keys = ["listing_id", "action_id"]
    for frame_name, frame in [("base", base), (name, other)]:
        missing = sorted(set(keys).difference(frame.columns))
        if missing:
            raise ValueError(f"{frame_name} is missing identifiers: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{frame_name} contains duplicate listing-action pairs")
    overlap = sorted(set(base.columns).intersection(other.columns).difference(keys))
    if overlap:
        raise ValueError(f"{name} repeats non-key columns already present: {overlap}")
    merged = base.merge(other, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError(
            f"{name} listing-action universe differs: "
            + str(merged["_merge"].value_counts().to_dict())
        )
    return merged.drop(columns="_merge")


def assemble_policy_scores(
    *,
    development_actions: pd.DataFrame,
    value_scores: pd.DataFrame,
    risk_scores: pd.DataFrame,
    evaluation_features: pd.DataFrame,
    objective: str,
    random_seed: int,
    fixed_action: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    leaked = sorted(
        column for column in evaluation_features.columns
        if column in FORBIDDEN_EVALUATION_COLUMNS or column.startswith("price_value_")
    )
    if leaked:
        raise ValueError(
            "Evaluation features contain outcomes and are refused: " + ", ".join(leaked)
        )
    target = f"value_{objective}"
    action_prior = fit_global_action_prior(development_actions, target_column=target)
    if fixed_action == "development_mean" and not action_prior:
        raise ValueError("Cannot choose fixed field from an empty development action table")
    resolved_fixed_action = (
        sorted(action_prior, key=lambda action: (-action_prior[action], action))[0]
        if fixed_action == "development_mean" else fixed_action
    )
    if resolved_fixed_action is not None and resolved_fixed_action not in action_prior:
        raise ValueError("Fixed action is absent from development action table")
    candidates = evaluation_features.copy()
    candidates = merge_one_to_one(candidates, value_scores, "value_scores")
    candidates = merge_one_to_one(candidates, risk_scores, "risk_scores")
    required = {
        f"mean_value_{objective}",
        f"lower_value_{objective}",
        f"predicted_pre_action_{objective}",
        "before_prediction_std",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Merged candidates are missing score inputs: {missing}")
    scores = build_baseline_score_table(
        candidates,
        action_prior=action_prior,
        mean_value_column=f"mean_value_{objective}",
        lower_value_column=f"lower_value_{objective}",
        uncertainty_column=f"predicted_pre_action_{objective}",
        disagreement_column="before_prediction_std",
        random_seed=random_seed,
        fixed_action=resolved_fixed_action,
    )
    return scores, action_prior


def main() -> None:
    args = parse_args()
    if args.fixed_action is not None and args.fixed_action_file is not None:
        raise ValueError("Use only one of --fixed-action and --fixed-action-file")
    fixed_action = args.fixed_action
    fixed_action_selection = None
    if args.fixed_action_file is not None:
        fixed_action_selection = json.loads(args.fixed_action_file.read_text(encoding="utf-8"))
        if fixed_action_selection.get("evidence_status") != "development_selection_only":
            raise ValueError("Fixed-field file must be a development-only selection")
        fixed_action = fixed_action_selection.get("selected_action")
        if not isinstance(fixed_action, str) or not fixed_action:
            raise ValueError("Fixed-field file lacks selected_action")
    inputs = {
        "development_actions": args.development_actions,
        "value_scores": args.value_scores,
        "risk_scores": args.risk_scores,
        "evaluation_features": args.evaluation_features,
    }
    frames = {name: pd.read_csv(path) for name, path in inputs.items()}
    scores, action_prior = assemble_policy_scores(
        development_actions=frames["development_actions"],
        value_scores=frames["value_scores"],
        risk_scores=frames["risk_scores"],
        evaluation_features=frames["evaluation_features"],
        objective=args.objective,
        random_seed=args.random_seed,
        fixed_action=fixed_action,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_fixed_action = (
        sorted(action_prior, key=lambda action: (-action_prior[action], action))[0]
        if fixed_action == "development_mean" else fixed_action
    )
    score_path = args.output_dir / f"frozen_policy_scores_{args.objective}.csv"
    scores.to_csv(score_path, index=False)
    audit = {
        "protocol": "v4-bvival-prospective",
        "objective": args.objective,
        "evaluation_outcomes_read": False,
        "random_seed": args.random_seed,
        "fixed_action_argument": fixed_action,
        "fixed_action_file": (
            {"path": str(args.fixed_action_file), "sha256": file_sha256(args.fixed_action_file),
             "budget_fraction": fixed_action_selection["budget_fraction"]}
            if args.fixed_action_file is not None else None
        ),
        "resolved_fixed_action": resolved_fixed_action,
        "fixed_action_selection_rule": (
            "maximum development mean realized action value with lexical tie-break"
            if fixed_action == "development_mean" else
            fixed_action_selection["selection_rule"]
            if fixed_action_selection is not None else "explicit development choice"
            if fixed_action is not None else None
        ),
        "development_action_prior": action_prior,
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in inputs.items()
        },
        "output": {"path": str(score_path), "sha256": file_sha256(score_path)},
        "score_columns": [
            column for column in scores.columns if column.startswith("score_")
        ],
    }
    (args.output_dir / f"policy_score_assembly_audit_{args.objective}.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
