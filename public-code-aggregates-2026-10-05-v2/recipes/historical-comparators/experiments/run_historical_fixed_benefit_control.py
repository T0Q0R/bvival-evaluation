"""Necessary retrospective comparator on unchanged historical score matrices.

No fitting, new source, new confirmatory evidence, or external release. Prepare
all label-free selections before this run reads previously used outcomes. This
local ordering cannot undo historical test use. Outputs are local, append-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bvi_val import select_unit_cost_actions
from evaluate_bvival_policies import absolute_price_error_by_listing, validate_and_join

KEYS = ["listing_id", "action_id"]
JOINT = "score_mean_value"
FIXED = "score_fixed_field_mean_value"
RISK = "score_uncertainty_only"
NONE = "score_no_acquisition"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def bound_path(root: Path, relative: str, expected: str | None = None) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Project-relative paths required")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing/nonregular input: {relative}")
    if expected is not None and digest(path) != expected:
        raise ValueError(f"Input hash differs: {relative}")
    return path


def fixed_field(priors: dict[str, float]) -> str:
    if not priors or not all(np.isfinite(value) for value in priors.values()):
        raise ValueError("Finite nonempty historical development priors required")
    return min(priors, key=lambda field: (-priors[field], field))


def prepare(scores: pd.DataFrame, *, field: str, budget: float) -> tuple:
    """Only keys and scores; non-fixed zero rows preserve the whole denominator."""
    required = {*KEYS, JOINT, RISK, NONE}
    if set(scores.columns) != required:
        raise ValueError("Prepare accepts only the specified score columns, never outcomes")
    if scores[KEYS].isna().any().any() or scores.duplicated(KEYS).any():
        raise ValueError("Unique nonmissing listing-action keys required")
    if not 0 < budget <= 1:
        raise ValueError("Budget must be in (0,1]")
    if not np.isfinite(scores[JOINT]).all() or not scores[NONE].eq(0).all():
        raise ValueError("Joint scores must be finite and no-acquisition scores zero")
    if np.isnan(scores[RISK]).any() or np.isposinf(scores[RISK]).any():
        raise ValueError("Historical risk scores permit only finite values or negative infinity")
    if field not in set(scores.action_id):
        raise ValueError("Fixed field absent from the action universe")
    out = scores.copy()
    out[FIXED] = np.where(out.action_id.eq(field), out[JOINT], 0.0)
    n = out.listing_id.nunique()
    expected_capacity = math.ceil(budget * n)
    selected = {}
    for policy in [FIXED, JOINT, RISK, NONE]:
        # The original evaluator excludes nonfinite risk rows before selection.
        finite = out.loc[np.isfinite(out[policy]), KEYS + [policy]]
        selected[policy] = select_unit_cost_actions(
            finite.rename(columns={policy: "lower_value"}), budget_fraction=budget
        )
    if any(len(selected[p]) != expected_capacity for p in [FIXED, JOINT, RISK]):
        raise ValueError("Capacity mismatch: do not force negative scores or shrink N")
    fixed_ids = set(out.loc[out.action_id.eq(field), "listing_id"])
    joint_ids = set(selected[JOINT].listing_id)
    overlap = selected[FIXED][KEYS].merge(selected[JOINT][KEYS], on=KEYS)
    receipt = {
        "n_listings": int(n), "n_action_rows": len(out), "capacity": expected_capacity,
        "selected_count": {p: len(v) for p, v in selected.items()},
        "fixed_field": field, "fixed_field_eligible_listings": len(fixed_ids),
        "joint_selected_listings_without_fixed_field": len(joint_ids - fixed_ids),
        "listing_selection_overlap": len(set(selected[FIXED].listing_id) & joint_ids),
        "listing_action_selection_overlap": len(overlap),
        "joint_action_counts": {str(k): int(v) for k, v in selected[JOINT].action_id.value_counts().items()},
        "outcomes_read_in_prepare": False, "independent_confirmation": False,
    }
    return out, selected, receipt


def paired_intervals(candidate: pd.DataFrame, reference: pd.DataFrame, *,
                     repetitions: int, seed: int, family_size: int) -> dict:
    if repetitions < 1 or family_size < 1:
        raise ValueError("Positive repetition and family counts required")
    paired = candidate.merge(reference, on="listing_id", validate="one_to_one",
                             suffixes=("_candidate", "_reference"))
    if len(paired) != len(candidate) or len(paired) != len(reference) or not len(paired):
        raise ValueError("Identical nonempty listing cohorts required")
    c = paired.absolute_price_error_candidate.to_numpy(float)
    r = paired.absolute_price_error_reference.to_numpy(float)
    if not np.isfinite(c).all() or not np.isfinite(r).all() or np.any(c < 0) or np.any(r < 0) or r.mean() <= 0:
        raise ValueError("Finite nonnegative errors and positive reference MAE required")
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions)
    for i in range(repetitions):
        ix = rng.integers(len(c), size=len(c))
        denom = r[ix].mean()
        samples[i] = 100 * (denom - c[ix].mean()) / denom if denom > 0 else np.nan
    if not np.isfinite(samples).all():
        raise ValueError("Zero-reference bootstrap resample: relative interval undefined; stop rather than silently omit")
    tail = .05 / (2 * family_size)
    return {
        "relative_mae_gain_percent": float(100 * (r.mean() - c.mean()) / r.mean()),
        "ci95_percent": np.quantile(samples, [.025, .975]).tolist(),
        "family_adjusted_interval_percent": np.quantile(samples, [tail, 1-tail]).tolist(),
        "family_adjusted_nominal_confidence": 1 - .05 / family_size,
        "bootstrap_unit": "listing_not_verified_vehicle", "n": len(c),
        "repetitions": repetitions, "seed": seed,
        "conditioning": "fixed models and selected actions; excludes training/selection/market uncertainty",
        "historical_adaptive_analysis_multiplicity_controlled": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["analysis_status"] != "post_test_exploratory" or config["independent_confirmation"] is not False:
        raise ValueError("Historical analysis cannot be declared confirmatory")
    if str(Path(sys.executable).absolute()) != config["runtime_python"]:
        raise ValueError("Use the configured venv interpreter without resolving its symlink")
    if config["fixed_field_rule"] != "maximum_recorded_development_mean_gain_lexical_tie":
        raise ValueError("Unexpected fixed-field rule")
    if len(config["sources"]) != config["comparison_family_size"] or len({s["name"] for s in config["sources"]}) != len(config["sources"]):
        raise ValueError("Comparison family must cover every source once")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prepared = []
    # Complete every source's selection before reading any old outcomes in this run.
    for source in config["sources"]:
        directory = args.output_dir / source["name"]
        directory.mkdir()
        audit_path = bound_path(args.project_root, source["assembly_audit"])
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit["objective"] != "absolute_price_error" or audit["output"]["sha256"] != source["scores_sha256"]:
            raise ValueError("Historical assembly binding mismatch")
        field = fixed_field(audit["development_action_prior"])
        if field != source["fixed_field"]:
            raise ValueError("Fixed field differs from recorded development choice")
        score_path = bound_path(args.project_root, source["scores"], source["scores_sha256"])
        scores = pd.read_csv(score_path, usecols=KEYS+[JOINT, RISK, NONE], dtype={k: str for k in KEYS})
        augmented, selections, receipt = prepare(scores, field=field, budget=config["budget_fraction"])
        if receipt["n_listings"] != source["expected_n"]:
            raise ValueError("Historical cohort count differs")
        paths = {}
        for policy, selection in selections.items():
            path = directory / (policy + "_selection.csv")
            selection[KEYS].to_csv(path, index=False)
            paths[path.name] = digest(path)
        receipt.update({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": digest(args.config), "runner_sha256": digest(Path(__file__)),
            "assembly_audit_sha256": digest(audit_path), "original_scores_sha256": digest(score_path),
            "selection_hashes": paths, "source": source["name"],
        })
        write_json(directory / "label_free_selection_receipt.json", receipt)
        prepared.append((source, directory, augmented, selections, receipt))
    results = []
    for source, directory, scores, selections, receipt in prepared:
        path = bound_path(args.project_root, source["outcomes"], source["outcomes_sha256"])
        old_curve_path = bound_path(args.project_root, source["historical_curve"], source["historical_curve_sha256"])
        outcomes = pd.read_csv(path, dtype={k: str for k in KEYS})
        joined = validate_and_join(outcomes, scores, [FIXED, JOINT, RISK, NONE])
        errors = {p: absolute_price_error_by_listing(joined, s) for p, s in selections.items()}
        mae = {p: float(e.absolute_price_error.mean()) for p, e in errors.items()}
        old_curve = pd.read_csv(old_curve_path)
        for policy in [JOINT, RISK, NONE]:
            row = old_curve.loc[old_curve.policy.eq(policy) & np.isclose(old_curve.budget, config["budget_fraction"])]
            if len(row) != 1 or not np.isclose(mae[policy], float(row.iloc[0].post_mae_price), atol=1e-8, rtol=1e-12):
                raise ValueError("Historical MAE reconstruction mismatch")
        result = {
            "source": source["name"], "analysis_status": config["analysis_status"],
            "independent_confirmation": False, "price_unit": source["price_unit"],
            "budget_fraction": config["budget_fraction"], "selection": receipt,
            "mae": mae, "historical_MAE_reconstruction_passed": True,
            "outcomes_sha256": digest(path), "historical_curve_sha256": digest(old_curve_path),
            "contrast": "joint_mean_benefit_vs_development_prior_fixed_field_mean_benefit",
            "joint_minus_fixed_MAE_reduction": mae[FIXED] - mae[JOINT],
            "inference": paired_intervals(errors[JOINT], errors[FIXED],
                repetitions=config["bootstrap_repetitions"], seed=config["bootstrap_seed"],
                family_size=config["comparison_family_size"]),
        }
        write_json(directory / "aggregate_result.json", result)
        results.append(result)
        print(json.dumps({"source": source["name"], "mae": mae, "inference": result["inference"]}, ensure_ascii=False), flush=True)
    write_json(args.output_dir / "aggregate_results.json", {
        "analysis_status": "post_test_exploratory", "independent_confirmation": False,
        "no_model_fit": True, "no_new_source": True, "submission_ready": False,
        "config_sha256": digest(args.config), "runner_sha256": digest(Path(__file__)),
        "all_selections_written_before_outcome_read_in_this_run_only": True,
        "source_results": results,
    })


if __name__ == "__main__":
    main()
