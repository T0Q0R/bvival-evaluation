"""Build a leakage-audited BVI-Val listing-action training table.

The input prediction-pair table must have been generated out of fold. This
builder does not fit the valuation model; it validates provenance, constructs
objective-aligned action values, and emits only explicitly allowlisted
pre-acquisition features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from bvi_val import realized_action_value


PAIR_COLUMNS = (
    "listing_id",
    "action_id",
    "fold_id",
    "prediction_is_oof",
    "before_prediction_log",
    "after_prediction_log",
)
LABEL_COLUMNS = ("listing_id", "target_log")
FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "label",
    "price",
    "after_prediction",
    "revealed",
    "realized",
    "gain",
    "loss_reduction",
    "error_before",
    "risk_target",
)

RISK_TARGET_COLUMNS = (
    "risk_target_squared_log_error_before",
    "risk_target_absolute_log_error_before",
    "risk_target_absolute_price_error_before",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-pairs", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pre-action-features", type=Path, required=True)
    parser.add_argument(
        "--feature-columns",
        required=True,
        help="Comma-separated pre-acquisition feature allowlist.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evidence-status",
        choices=(
            "development_only",
            "supportive_reused_source",
            "opened_posthoc",
            "prospective_confirmatory",
        ),
        required=True,
    )
    parser.add_argument(
        "--freeze-id",
        help="Required for prospective_confirmatory evidence and assigned before label access.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing columns: {missing}")


def _strict_boolean(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    normalized = values.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        bad = sorted(values[normalized.isna()].astype(str).unique().tolist())
        raise ValueError(f"prediction_is_oof contains invalid values: {bad}")
    return normalized.astype(bool)


def validate_feature_allowlist(feature_columns: Iterable[str]) -> list[str]:
    columns = [column.strip() for column in feature_columns if column.strip()]
    if not columns:
        raise ValueError("At least one pre-acquisition feature must be allowlisted")
    if len(columns) != len(set(columns)):
        raise ValueError("Feature allowlist contains duplicates")
    forbidden = [
        column
        for column in columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"Feature allowlist contains forbidden outcome fields: {forbidden}")
    return columns


def build_action_dataset(
    *,
    prediction_pairs: pd.DataFrame,
    labels: pd.DataFrame,
    pre_action_features: pd.DataFrame,
    feature_columns: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate inputs and construct a long listing-action value table."""

    features = validate_feature_allowlist(feature_columns)
    _require_columns(prediction_pairs, PAIR_COLUMNS, "prediction_pairs")
    _require_columns(labels, LABEL_COLUMNS, "labels")
    pair_feature_columns = {"before_prediction_log"}
    pre_feature_columns = [column for column in features if column not in pair_feature_columns]
    _require_columns(
        pre_action_features,
        ["listing_id", "action_id", *pre_feature_columns],
        "pre_action_features",
    )
    if prediction_pairs.duplicated(["listing_id", "action_id"]).any():
        raise ValueError("prediction_pairs contains duplicate listing-action rows")
    if labels["listing_id"].duplicated().any():
        raise ValueError("labels contains duplicate listing IDs")
    if pre_action_features.duplicated(["listing_id", "action_id"]).any():
        raise ValueError("pre_action_features contains duplicate listing-action rows")

    pairs = prediction_pairs[list(PAIR_COLUMNS)].copy()
    pairs["prediction_is_oof"] = _strict_boolean(pairs["prediction_is_oof"])
    if not pairs["prediction_is_oof"].all():
        count = int((~pairs["prediction_is_oof"]).sum())
        raise ValueError(f"All prediction pairs must be out of fold; found {count} violations")
    for column in ("before_prediction_log", "after_prediction_log"):
        pairs[column] = pd.to_numeric(pairs[column], errors="raise")
        if not np.isfinite(pairs[column]).all():
            raise ValueError(f"{column} contains non-finite values")

    label_subset = labels[list(LABEL_COLUMNS)].copy()
    label_subset["target_log"] = pd.to_numeric(label_subset["target_log"], errors="raise")
    if not np.isfinite(label_subset["target_log"]).all():
        raise ValueError("target_log contains non-finite values")
    working = pairs.merge(label_subset, on="listing_id", how="left", validate="many_to_one")
    if working["target_log"].isna().any():
        raise ValueError("Some prediction pairs have no matching label")
    working = working.merge(
        pre_action_features[["listing_id", "action_id", *pre_feature_columns]],
        on=["listing_id", "action_id"],
        how="left",
        validate="one_to_one",
    )
    missing_feature_rows = int(working[features].isna().all(axis=1).sum())
    if missing_feature_rows:
        raise ValueError(
            f"{missing_feature_rows} listing-action rows have no matching pre-action feature row"
        )

    working["value_squared_log_error"] = realized_action_value(
        target_log=working["target_log"],
        before_prediction_log=working["before_prediction_log"],
        after_prediction_log=working["after_prediction_log"],
        objective="squared_log_error",
    )
    working["value_absolute_price_error"] = realized_action_value(
        target_log=working["target_log"],
        before_prediction_log=working["before_prediction_log"],
        after_prediction_log=working["after_prediction_log"],
        objective="absolute_price_error",
    )
    residual_before = (
        working["target_log"].to_numpy(dtype=float)
        - working["before_prediction_log"].to_numpy(dtype=float)
    )
    working["risk_target_squared_log_error_before"] = np.square(residual_before)
    working["risk_target_absolute_log_error_before"] = np.abs(residual_before)
    working["risk_target_absolute_price_error_before"] = np.abs(
        np.maximum(np.expm1(working["target_log"].to_numpy(dtype=float)), 0.0)
        - np.maximum(np.expm1(working["before_prediction_log"].to_numpy(dtype=float)), 0.0)
    )

    output_columns = list(dict.fromkeys([
        "listing_id",
        "action_id",
        "fold_id",
        "prediction_is_oof",
        "before_prediction_log",
        *features,
        "value_squared_log_error",
        "value_absolute_price_error",
        *RISK_TARGET_COLUMNS,
    ]))
    output = working[output_columns].copy()
    audit = {
        "listing_count": int(output["listing_id"].nunique()),
        "listing_action_count": int(len(output)),
        "action_counts": {
            str(key): int(value)
            for key, value in output["action_id"].value_counts().sort_index().items()
        },
        "fold_counts": {
            str(key): int(value)
            for key, value in output["fold_id"].value_counts().sort_index().items()
        },
        "all_predictions_oof": bool(output["prediction_is_oof"].all()),
        "feature_allowlist": features,
        "forbidden_columns_excluded": ["target_log", "after_prediction_log"],
        "positive_value_rate_squared_log_error": float(
            (output["value_squared_log_error"] > 0).mean()
        ),
        "positive_value_rate_absolute_price_error": float(
            (output["value_absolute_price_error"] > 0).mean()
        ),
    }
    return output, audit


def main() -> None:
    args = parse_args()
    if args.evidence_status == "prospective_confirmatory" and not args.freeze_id:
        raise ValueError("--freeze-id is required for prospective_confirmatory evidence")
    feature_columns = [value.strip() for value in args.feature_columns.split(",")]
    pairs = pd.read_csv(args.prediction_pairs)
    labels = pd.read_csv(args.labels)
    features = pd.read_csv(args.pre_action_features)
    output, audit = build_action_dataset(
        prediction_pairs=pairs,
        labels=labels,
        pre_action_features=features,
        feature_columns=feature_columns,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "bvival_action_dataset.csv"
    output.to_csv(output_path, index=False)
    audit.update(
        {
            "evidence_status": args.evidence_status,
            "freeze_id": args.freeze_id,
            "inputs": {
                "prediction_pairs": {
                    "path": str(args.prediction_pairs),
                    "sha256": file_sha256(args.prediction_pairs),
                },
                "labels": {"path": str(args.labels), "sha256": file_sha256(args.labels)},
                "pre_action_features": {
                    "path": str(args.pre_action_features),
                    "sha256": file_sha256(args.pre_action_features),
                },
            },
            "output_sha256": file_sha256(output_path),
            "nonclaim": (
                "Dataset construction alone does not establish policy superiority or "
                "selection-conditional coverage."
            ),
        }
    )
    (args.output_dir / "bvival_action_dataset_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
