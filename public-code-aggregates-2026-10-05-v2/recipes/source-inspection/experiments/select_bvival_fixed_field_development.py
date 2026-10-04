"""Choose a budget-matched fixed verification field using development only.

The risk scores rank listings. For each candidate field, the selector evaluates
the same global 10% budget on OOF action-value outcomes, then freezes one field
for an external evaluation. No evaluation/test input is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VALUE_COLUMN = "value_absolute_price_error"
BEFORE_ERROR_COLUMN = "risk_target_absolute_price_error_before"
RISK_COLUMN = "predicted_pre_action_absolute_price_error"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_fixed_field(
    actions: pd.DataFrame, risk_scores: pd.DataFrame, *, budget_fraction: float
) -> dict:
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must lie in (0, 1]")
    keys = ["listing_id", "action_id"]
    needed_actions = {*keys, VALUE_COLUMN, BEFORE_ERROR_COLUMN, "prediction_is_oof"}
    needed_risk = {*keys, RISK_COLUMN}
    if not needed_actions.issubset(actions.columns):
        raise ValueError(f"Development actions lack {sorted(needed_actions - set(actions))}")
    if not needed_risk.issubset(risk_scores.columns):
        raise ValueError(f"Development risk scores lack {sorted(needed_risk - set(risk_scores))}")
    if actions.duplicated(keys).any() or risk_scores.duplicated(keys).any():
        raise ValueError("Listing-action pairs must be unique")
    oof = actions["prediction_is_oof"].astype(str).str.lower()
    if not oof.eq("true").all():
        raise ValueError("All development action outcomes must use OOF base predictions")
    joined = actions[[*keys, VALUE_COLUMN, BEFORE_ERROR_COLUMN]].merge(
        risk_scores[[*keys, RISK_COLUMN]], on=keys, how="outer",
        validate="one_to_one", indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Development action/risk candidate universes differ")
    joined = joined.drop(columns="_merge")
    for column in (VALUE_COLUMN, BEFORE_ERROR_COLUMN, RISK_COLUMN):
        joined[column] = pd.to_numeric(joined[column], errors="raise")
        if not np.isfinite(joined[column]).all():
            raise ValueError(f"{column} contains non-finite values")
    grouped_before = joined.groupby("listing_id")[BEFORE_ERROR_COLUMN]
    if (grouped_before.max() - grouped_before.min() > 1e-7).any():
        raise ValueError("One listing has inconsistent pre-action error across fields")

    listing_count = int(joined["listing_id"].nunique())
    capacity = int(np.ceil(listing_count * budget_fraction))
    before_mae = float(grouped_before.first().mean())
    rows = []
    for action_id, field_rows in joined.groupby("action_id", sort=True):
        ranked = field_rows.loc[field_rows[RISK_COLUMN] > 0].sort_values(
            [RISK_COLUMN, "listing_id"], ascending=[False, True], kind="stable"
        )
        chosen = ranked.head(capacity)
        gain_per_listing = float(chosen[VALUE_COLUMN].sum() / listing_count)
        rows.append({
            "action_id": str(action_id),
            "available_listing_count": int(len(field_rows)),
            "selected_action_count": int(len(chosen)),
            "development_post_mae": before_mae - gain_per_listing,
            "development_gain_per_listing": gain_per_listing,
        })
    if not rows:
        raise ValueError("No candidate fields")
    rows.sort(key=lambda row: (row["development_post_mae"], row["action_id"]))
    return {
        "selected_action": rows[0]["action_id"],
        "selection_rule": "minimum development post-MAE at the fixed global budget; lexical tie-break",
        "risk_score_provenance": "development-fitted risk scores; field choice only, not performance inference",
        "budget_fraction": budget_fraction,
        "listing_count": listing_count,
        "capacity": capacity,
        "before_mae": before_mae,
        "candidate_fields": rows,
        "evidence_status": "development_selection_only",
        "evaluation_labels_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-actions", type=Path, required=True)
    parser.add_argument("--development-risk-scores", type=Path, required=True)
    parser.add_argument("--budget-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = select_fixed_field(
        pd.read_csv(args.development_actions),
        pd.read_csv(args.development_risk_scores),
        budget_fraction=args.budget_fraction,
    )
    selection["input_sha256"] = {
        "development_actions": file_sha256(args.development_actions),
        "development_risk_scores": file_sha256(args.development_risk_scores),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "selected_action": selection["selected_action"],
        "candidate_fields": selection["candidate_fields"],
    }, indent=2))


if __name__ == "__main__":
    main()
