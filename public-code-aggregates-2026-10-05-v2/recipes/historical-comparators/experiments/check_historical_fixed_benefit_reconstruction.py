"""Independent stdlib re-accounting of historical cached outcomes and actions.

Not a new source or independent retraining. No row values are printed. Does
not call the policy evaluator used to generate the aggregate results.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, config_path, output):
    config = json.loads(config_path.read_text())
    aggregate = json.loads((output / "aggregate_results.json").read_text())
    if (aggregate["config_sha256"] != sha(config_path)
            or aggregate["independent_confirmation"] is not False
            or aggregate["runner_sha256"] != sha(root / "experiments/run_historical_fixed_benefit_control.py")):
        raise ValueError("Configuration/code/status differs from the completed run")
    results = []
    for source, result in zip(config["sources"], aggregate["source_results"], strict=True):
        if source["name"] != result["source"] or result["analysis_status"] != "post_test_exploratory":
            raise ValueError("Source order/status differs")
        path = root / source["outcomes"]
        if sha(path) != source["outcomes_sha256"] or result["outcomes_sha256"] != sha(path):
            raise ValueError("Outcomes binding differs")
        if sha(root / source["scores"]) != source["scores_sha256"]:
            raise ValueError("Historical scores binding differs")
        listing_states, actions = {}, {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["listing_id"], row["action_id"])
                if key in actions:
                    raise ValueError("Duplicate outcome action")
                target, before, after = (max(math.expm1(float(row[col])), 0.) for col in
                                        ("target_log", "before_prediction_log", "after_prediction_log"))
                state = (target, before)
                previous = listing_states.setdefault(key[0], state)
                if not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9) for a, b in zip(previous, state)):
                    raise ValueError("Target/before state differs among a listing's actions")
                actions[key] = after
        n = len(listing_states)
        if n != source["expected_n"]:
            raise ValueError("Whole cohort size differs")
        means, selected_sets = {}, {}
        receipt = result["selection"]
        for policy, expected_mae in result["mae"].items():
            selection_path = output / source["name"] / (policy + "_selection.csv")
            if sha(selection_path) != receipt["selection_hashes"][selection_path.name]:
                raise ValueError("Selected actions changed")
            selected = {}
            with selection_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    listing, action = row["listing_id"], row["action_id"]
                    if listing in selected or (listing, action) not in actions:
                        raise ValueError("Invalid/repeated selected listing")
                    selected[listing] = action
            selected_sets[policy] = set(selected.items())
            if len(selected) != receipt["selected_count"][policy]:
                raise ValueError("Selected capacity differs")
            errors = [abs(target - (actions[(listing, selected[listing])] if listing in selected else before))
                      for listing, (target, before) in listing_states.items()]
            means[policy] = math.fsum(errors) / n
            if not math.isclose(means[policy], expected_mae, rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError("Stdlib MAE differs from reported MAE")
        fixed = "score_fixed_field_mean_value"
        joint = "score_mean_value"
        effect = 100 * (means[fixed] - means[joint]) / means[fixed]
        if not math.isclose(effect, result["inference"]["relative_mae_gain_percent"], abs_tol=1e-10):
            raise ValueError("Effect orientation/arithmetic differs")
        if len(selected_sets[fixed] & selected_sets[joint]) != receipt["listing_action_selection_overlap"]:
            raise ValueError("Action overlap differs")
        results.append({"source": source["name"], "n": n,
                        "four_policy_whole_cohort_MAEs_reconstructed": True,
                        "target_before_consistency_checked": True,
                        "effect_recomputed_percent": effect})
    return {"status": "STDLIB_HISTORICAL_ACCOUNTING_VERIFIED_NOT_INDEPENDENT_CONFIRMATION",
            "source_results": results, "config_sha256": sha(config_path),
            "aggregate_sha256": sha(output / "aggregate_results.json"),
            "checker_sha256": sha(Path(__file__)), "new_source_or_retraining": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.project_root, args.config, args.output_dir)
    with (args.output_dir / "stdlib_reconstruction_receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2))
