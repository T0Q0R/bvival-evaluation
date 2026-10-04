"""Validate an external AFA score table before exploratory BVI-Val evaluation.

This adapter does not implement, fit, or certify a published method. It only
checks the label-free score contract for the three *already opened* sources.
Any resulting comparison is post-test exploratory, even when model fitting
uses only development data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ("listing_id", "action_id")
SCORE = "score_external_afa"
OPENED_SOURCES = {"MUCars-2024", "JUCars-2024", "AutoScout24-2025"}
REQUIRED_PROVENANCE = {
    "source_name",
    "method_name",
    "paper_url",
    "implementation_url",
    "implementation_commit",
    "adaptation_description",
    "implementation_class",
    "objective",
    "analysis_status",
    "score_direction",
    "training_splits",
    "evaluation_outcomes_used",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_provenance(record: dict) -> None:
    missing = sorted(REQUIRED_PROVENANCE.difference(record))
    if missing:
        raise ValueError(f"External baseline provenance lacks: {missing}")
    if record["source_name"] not in OPENED_SOURCES:
        raise ValueError("This adapter only covers the three already opened sources")
    if record["analysis_status"] != "post_test_exploratory":
        raise ValueError("Opened-source external comparisons must be post-test exploratory")
    if record["evaluation_outcomes_used"] is not False:
        raise ValueError("External scores must be generated without evaluation outcomes")
    if record["objective"] != "absolute_price_error":
        raise ValueError("Use the same absolute-price-error objective as the manuscript")
    if record["score_direction"] != "higher_is_better":
        raise ValueError("External scores must rank larger expected benefit first")
    if record["implementation_class"] not in {"official_adapted", "independent_adaptation"}:
        raise ValueError("Declare official_adapted or independent_adaptation")
    if record["implementation_class"] == "independent_adaptation" and not record["adaptation_description"]:
        raise ValueError("Independent adaptations require a nonempty description")
    if record["training_splits"] not in (["train"], ["train", "validation"]):
        raise ValueError("External fitting may use only train and optional validation")
    for key in ("method_name", "paper_url", "implementation_url", "implementation_commit"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"{key} must be nonempty")


def validate_score_contract(
    candidate_universe: pd.DataFrame,
    external_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Require exact pair coverage and no labels or auxiliary columns."""

    if set(external_scores.columns) != {*KEYS, SCORE}:
        raise ValueError("External score file must contain only listing_id, action_id, score_external_afa")
    if set(candidate_universe.columns) != set(KEYS):
        raise ValueError("Candidate universe must contain only listing_id and action_id")
    for name, frame in (("candidate", candidate_universe), ("external", external_scores)):
        if frame[list(KEYS)].isna().any().any():
            raise ValueError(f"{name} has null listing-action keys")
        if frame.duplicated(list(KEYS)).any():
            raise ValueError(f"{name} has duplicate listing-action pairs")
    universe = candidate_universe[list(KEYS)].copy()
    universe["_candidate_order"] = np.arange(len(universe))
    merged = universe.merge(external_scores, on=list(KEYS), how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("External scores must cover exactly the candidate listing-action universe")
    values = pd.to_numeric(merged[SCORE], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("External scores must all be finite")
    if len(merged) == 0:
        raise ValueError("Candidate universe is empty")
    merged = merged.sort_values("_candidate_order", kind="mergesort")
    result = merged[list(KEYS)].reset_index(drop=True)
    result[SCORE] = pd.to_numeric(merged[SCORE], errors="raise").to_numpy(dtype=float)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-universe", type=Path, required=True)
    parser.add_argument("--external-scores", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    validate_provenance(provenance)
    scores = validate_score_contract(
        pd.read_csv(args.candidate_universe), pd.read_csv(args.external_scores)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "external_afa_scores_exploratory.csv"
    scores.to_csv(output, index=False)
    audit = {
        **provenance,
        "adapter_status": "contract_checked_not_method_verified",
        "candidate_universe_sha256": sha256(args.candidate_universe),
        "input_scores_sha256": sha256(args.external_scores),
        "provenance_sha256": sha256(args.provenance),
        "output_scores_sha256": sha256(output),
        "candidate_action_count": len(scores),
        "candidate_listing_count": int(scores["listing_id"].nunique()),
    }
    (args.output_dir / "external_afa_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
