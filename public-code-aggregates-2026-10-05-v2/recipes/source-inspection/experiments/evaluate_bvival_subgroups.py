"""Prespecified descriptive group safety gate for a locked BVI-Val result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bvi_val import policy_post_action_predictions, select_unit_cost_actions
from evaluate_bvival_policies import validate_and_join


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-outcomes", type=Path, required=True)
    parser.add_argument("--frozen-policy-scores", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--subgroup-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze-id", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_groups(
    post: pd.DataFrame, test_features: pd.DataFrame, config: dict
) -> pd.DataFrame:
    if "target_log" in test_features or any(
        column.startswith("price_value_") for column in test_features.columns
    ):
        raise ValueError("Test feature input must not contain target or price")
    required = {"listing_id", "make", "year", "mileage_km"}
    missing = sorted(required.difference(test_features.columns))
    if missing or test_features["listing_id"].duplicated().any():
        raise ValueError(f"Test features have missing columns or duplicate IDs: {missing}")
    working = post.merge(
        test_features[["listing_id", "make", "year", "mileage_km"]],
        on="listing_id", how="left", validate="one_to_one", indicator=True,
    )
    if not working["_merge"].eq("both").all():
        raise ValueError("Policy listings and test-feature listings differ")
    working = working.drop(columns="_merge")
    makers = set(config["maker_groups"])
    working["maker_group"] = working["make"].where(
        working["make"].isin(makers), config["maker_other_group"]
    )
    year = pd.to_numeric(working["year"], errors="raise")
    year_groups = config["model_year_groups"]
    if len(year_groups) < 2 or year_groups[-1]["minimum"] is not None:
        raise ValueError("model_year_groups must end with an open lower group")
    working["model_year_group"] = np.select(
        [year >= group["minimum"] for group in year_groups[:-1]],
        [group["label"] for group in year_groups[:-1]],
        default=year_groups[-1]["label"],
    )
    price = np.maximum(np.expm1(working["target_log"].to_numpy(dtype=float)), 0.0)
    low, high = config.get("price_cutpoints", config.get("price_cutpoints_JOD"))
    currency = config.get("price_currency", "JOD")
    working["price_group"] = np.select(
        [price <= low, price <= high],
        [f"at most {low:g} {currency}", f"{low:g} to {high:g} {currency}"],
        default=f"above {high:g} {currency}",
    )
    working["mileage_availability_group"] = np.where(
        working["mileage_km"].notna(),
        "numeric mileage available", "numeric mileage missing",
    )
    return working


def group_metrics(frame: pd.DataFrame, objective: str) -> dict[str, float | int]:
    target = frame["target_log"].to_numpy(dtype=float)
    before = frame["before_prediction_log"].to_numpy(dtype=float)
    after = frame["post_prediction_log"].to_numpy(dtype=float)
    chosen = frame["selected_action_id"].notna().to_numpy()
    before_error = np.abs(np.expm1(target) - np.expm1(before))
    after_error = np.abs(np.expm1(target) - np.expm1(after))
    if objective == "absolute_price_error":
        value = before_error - after_error
    elif objective == "squared_log_error":
        value = np.square(target - before) - np.square(target - after)
    else:
        raise ValueError(f"Unknown subgroup objective: {objective}")
    return {
        "listing_count": int(len(frame)),
        "action_count": int(chosen.sum()),
        "post_rmsle": float(np.sqrt(np.square(target - after).mean())),
        "post_mae_price": float(after_error.mean()),
        "harmful_action_rate": (
            float(np.mean(value[chosen] < 0)) if chosen.any() else float("nan")
        ),
    }


def evaluate_groups(
    joined: pd.DataFrame, test_features: pd.DataFrame, config: dict
) -> pd.DataFrame:
    budget = float(config["primary_budget_fraction"])
    objective = config.get("objective", "squared_log_error")
    metric = "post_mae_price" if objective == "absolute_price_error" else "post_rmsle"
    policies = [config["reference_policy"], config["candidate_policy"]]
    posts = {}
    for policy in policies:
        finite = joined[np.isfinite(joined[policy].to_numpy(dtype=float))]
        selected = select_unit_cost_actions(
            finite[["listing_id", "action_id", policy]].rename(
                columns={policy: "lower_value"}
            ),
            budget_fraction=budget,
        )
        posts[policy] = assign_groups(
            policy_post_action_predictions(joined, selected), test_features, config
        )
    rows = []
    for dimension in (
        "maker_group", "model_year_group", "price_group",
        "mileage_availability_group",
    ):
        for group in sorted(posts[policies[0]][dimension].dropna().unique()):
            reference = group_metrics(posts[policies[0]].loc[
                posts[policies[0]][dimension].eq(group)
            ], objective)
            candidate = group_metrics(posts[policies[1]].loc[
                posts[policies[1]][dimension].eq(group)
            ], objective)
            if reference[metric] == 0:
                change = 0.0 if candidate[metric] == 0 else float("inf")
            else:
                change = 100.0 * (
                    candidate[metric] - reference[metric]
                ) / reference[metric]
            eligible = reference["listing_count"] >= int(
                config["minimum_group_size_for_safety_gate"]
            )
            rows.append({
                "dimension": dimension,
                "group": group,
                "listing_count": reference["listing_count"],
                "reference_post_rmsle": reference["post_rmsle"],
                "candidate_post_rmsle": candidate["post_rmsle"],
                "reference_post_mae_price": reference["post_mae_price"],
                "candidate_post_mae_price": candidate["post_mae_price"],
                "relative_primary_loss_change_percent": change,
                "primary_loss_metric": metric,
                "reference_action_count": reference["action_count"],
                "candidate_action_count": candidate["action_count"],
                "reference_harmful_action_rate": reference["harmful_action_rate"],
                "candidate_harmful_action_rate": candidate["harmful_action_rate"],
                "safety_gate_eligible": bool(eligible),
                "material_deterioration": bool(
                    eligible and change > float(
                        config.get(
                            "material_relative_loss_deterioration_percent",
                            config.get("material_relative_rmsle_deterioration_percent"),
                        )
                    )
                ),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = json.loads(args.subgroup_config.read_text(encoding="utf-8"))
    policies = [config["reference_policy"], config["candidate_policy"]]
    outcomes = pd.read_csv(args.evaluation_outcomes)
    scores = pd.read_csv(args.frozen_policy_scores)
    test_features = pd.read_csv(args.test_features)
    joined = validate_and_join(outcomes, scores, policies)
    report = evaluate_groups(joined, test_features, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "bvival_prespecified_subgroups.csv"
    report.to_csv(report_path, index=False)
    audit = {
        "freeze_id": args.freeze_id,
        "protocol": config["protocol"],
        "descriptive_safety_gate_passed": bool(~report["material_deterioration"].any()),
        "input_hashes": {
            "evaluation_outcomes": file_sha256(args.evaluation_outcomes),
            "frozen_policy_scores": file_sha256(args.frozen_policy_scores),
            "test_features": file_sha256(args.test_features),
            "subgroup_config": file_sha256(args.subgroup_config),
        },
        "report_sha256": file_sha256(report_path),
        "nonclaim": "Descriptive group gate; no subgroup inference or conditional coverage guarantee",
    }
    (args.output_dir / "bvival_subgroup_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
