"""Recover AutoScout24 final/strict-selection contracts without empirical replay.

Default reads metadata, code, and notes only. Explicit model/header opt-ins hash
and load 18 bound local CatBoost models and parse four target headers only.
No listing rows, predictions, targets or source prices are decoded; no fits or
outcome computations. This is not a data-publication permission decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mucars_reconstruction import PARAMETER_KEYS, safe_path, recorded_graph_checks
from check_remaining_source_metadata import AUTO, digest, RIGHTS_NOTE
from plan_mucars_replay import static_import_closure

AUDITS = {**AUTO, **{k: "experiments/outputs/autoscout24_bvival_v2_" + v for k, v in {
    "selection_pairs": "full_selection_pairs/field_selection_pair_audit.json",
    "selection_train": "full_selection_train_actions/bvival_action_dataset_audit.json",
    "selection_validation": "full_selection_validation_actions/bvival_action_dataset_audit.json",
    "selection_risk": "full_selection_risk/risk_baseline_fit_audit.json",
    "decision": "full_fixed_field/selection.json",
}.items()}}
CONFIG = "experiments/configs/autoscout24_bvival_v2_full.json"
FREEZE = "experiments/autoscout24_bvival_v2_full_final_pretest_freeze.json"
SEEDS = [13, 42, 2026]
ENTRYPOINTS = ["experiments/" + p + ".py" for p in (
    "build_autoscout24_bvival_manifest", "generate_bvival_prediction_pairs",
    "generate_bvival_field_selection_pairs", "build_bvival_action_dataset",
    "fit_bvival_value_policy", "fit_bvival_risk_baseline", "select_bvival_fixed_field_development",
    "assemble_bvival_policy_scores", "join_bvival_evaluation_outcomes", "evaluate_bvival_policies")]


def selection_checks(a, config, hashes):
    checks = []
    p, t, v, r, d, s = (a[k] for k in ("selection_pairs", "selection_train", "selection_validation", "selection_risk", "decision", "assembly"))

    def eq(name, left, right):
        if left is None or right is None or type(left) is not type(right) or left != right:
            raise ValueError("AutoScout selection mismatch: " + name)
        checks.append(name)

    def link(name, left, right):
        if any(not isinstance(h, str) or len(h) != 64 or any(ch not in "0123456789abcdef" for ch in h) for h in (left, right)):
            raise ValueError("Malformed recorded hash: " + name)
        eq(name, left, right)

    for split in ("train", "validation"):
        link("cohort_to_selection_" + split, a["cohort"]["output_hashes"]["splits/" + split + ".csv"], p["inputs"][split + "_sha256"])
        eq("selection_" + split + "_count", p[split + "_listing_count"], a["cohort"]["split_counts"][split])
    for phase, target in (("train_oof", t), ("validation_heldout", v)):
        for producer, consumer in (("pairs", "prediction_pairs"), ("features", "pre_action_features"), ("labels", "labels")):
            link(phase + "_" + producer + "_to_targets", p["outputs"][phase][producer], target["inputs"][consumer]["sha256"])
        eq(phase + "_out_of_sample", target["all_predictions_oof"], True)
        key = "train" if phase == "train_oof" else "validation"
        eq(phase + "_pair_count", target["listing_action_count"], p[key + "_pair_count"])
        eq(phase + "_listing_count", target["listing_count"], p[key + "_listing_count"])
    link("train_targets_to_selection_risk", t["output_sha256"], r["inputs"]["train_actions_sha256"])
    link("validation_features_to_selection_risk", p["outputs"]["validation_heldout"]["features"], r["inputs"]["evaluation_features_sha256"])
    link("validation_targets_to_field_decision", v["output_sha256"], d["input_sha256"]["development_actions"])
    link("selection_risk_to_field_decision", r["evaluation_risk_scores_sha256"], d["input_sha256"]["development_risk_scores"])
    link("field_decision_to_final_assembly", hashes[AUDITS["decision"]], s["fixed_action_file"]["sha256"])
    eq("fixed_field_matches_decision", d["selected_action"], s["resolved_fixed_action"])
    eq("selection_no_train_validation_overlap", p["train_validation_id_overlap"], 0)
    eq("validation_labels_not_base_fit", p["validation_labels_used_in_base_fit"], False)
    eq("selection_test_not_read", p["sealed_test_input_read"], False)
    eq("field_decision_not_test_estimate", d["evaluation_labels_read"], False)
    eq("field_decision_scope", d["evidence_status"], "development_selection_only")
    eq("validation_selection_denominator", d["listing_count"], p["validation_listing_count"])
    eq("fixed_budget", d["budget_fraction"], config["primary_budget_fraction"])
    eq("fixed_capacity", d["capacity"], math.ceil(d["listing_count"] * d["budget_fraction"]))
    eq("risk_selection_objective", r["objective"], "absolute_price_error")
    eq("head_schema", r["feature_columns"], a["risk"]["feature_columns"])
    eq("base_schema", p["feature_columns"], a["pairs"]["feature_columns"])
    eq("base_settings", p["model_parameters"], a["pairs"]["model_parameters"])
    for role, obj in (("final_base", a["pairs"]), ("selection_base", p), ("value", a["value"]), ("risk", a["risk"]), ("selection_risk", r)):
        eq(role + "_seeds", obj["seeds"], SEEDS)
    eq("declared_action_set", config["action_fields"], p["action_fields"])
    eq("evaluation_single_budget", a["evaluation"]["budgets"], [config["primary_budget_fraction"]])
    rows = d["candidate_fields"]
    if len(rows) != len(p["action_fields"]) or {x["action_id"] for x in rows} != set(p["action_fields"]):
        raise ValueError("All four candidate-field records required")
    if any(not math.isfinite(x["development_post_mae"]) for x in rows):
        raise ValueError("Nonfinite recorded selection criterion")
    eq("minimum_validation_post_mae_with_lexical_tie", d["selected_action"],
       min(rows, key=lambda x: (x["development_post_mae"], x["action_id"]))["action_id"])
    return checks


def freeze_check(root, a):
    path = safe_path(root, FREEZE)
    f = json.loads(path.read_text())
    checked, deferred = [], []
    for relative, recorded in f["files_sha256"].items():
        if Path(relative).suffix == ".csv":
            deferred.append({"path": relative, "recorded_sha256": recorded, "row_bytes_checked": False})
        elif Path(relative).suffix in {".py", ".json", ".md"}:
            actual = digest(safe_path(root, relative))
            checked.append({"path": relative, "frozen_sha256": recorded, "current_sha256": actual, "matches": actual == recorded})
        else:
            raise ValueError("Unclassified freeze artifact")
    recorded = f["files_sha256"]
    bindings = {"experiments/outputs/autoscout24_bvival_v2_full_scores/frozen_policy_scores_absolute_price_error.csv": a["assembly"]["output"]["sha256"],
                "experiments/data/processed/autoscout24_bvival_v2/sealed_labels/test_labels.csv": a["cohort"]["output_hashes"]["sealed_labels/test_labels.csv"],
                "experiments/outputs/autoscout24_bvival_v2_full_pairs/evaluation_prediction_pairs.csv": a["pairs"]["outputs"]["evaluation"]["pairs"]}
    if any(recorded.get(p) != h for p, h in bindings.items()):
        raise ValueError("Freeze does not bind recorded primary graph")
    return {"freeze_sha256": digest(path), "code_and_metadata_comparisons": checked,
            "recorded_csv_bindings_not_opened": deferred, "primary_recorded_row_bindings_match": True,
            "historical_pretest_flags_preserved_not_current_unopened_status": True,
            "full_upstream_source_identity_certified": False}


def model_specs(a):
    specs = []
    groups = [("final_base", a["pairs"]["models"], a["pairs"]),
              ("selection_base", a["selection_pairs"]["train_only_base_models"], a["selection_pairs"]),
              ("final_risk", a["risk"]["models"], a["risk"]),
              ("selection_risk", a["selection_risk"]["models"], a["selection_risk"])]
    groups += [(objective, value["models"], a["value"]) for objective, value in a["value"]["objectives"].items()]
    if {x[0] for x in groups} != {"final_base", "selection_base", "final_risk", "selection_risk", "absolute_price_error", "squared_log_error"}:
        raise ValueError("Six historical model roles required")
    for role, models, obj in groups:
        if len(models) != 3 or sorted(m["seed"] for m in models) != SEEDS:
            raise ValueError("Three original models per role required")
        for m in models:
            s = m["seed"]
            prefix = "experiments/outputs/autoscout24_bvival_v2_"
            expected = (prefix + f"full_pairs/base_models/mask_aware_catboost_seed_{s}.cbm" if role == "final_base" else
                        prefix + f"full_selection_pairs/train_only_base_seed_{s}.cbm" if role == "selection_base" else
                        prefix + f"full_risk/models/seed_{s}.cbm" if role == "final_risk" else
                        prefix + f"full_selection_risk/models/seed_{s}.cbm" if role == "selection_risk" else
                        prefix + f"full_value/models/{role}/seed_{s}.cbm")
            if m.get("model_path", m.get("path", expected)) != expected:
                raise ValueError("Saved-model role/path mismatch")
            extra = ["action_mask"] if role.endswith("base") else []
            specs.append({"role": role, "seed": s, "path": expected, "sha256": m["sha256"],
                          "feature_names": obj["feature_columns"] + extra,
                          "categorical_columns": obj["categorical_columns"] + extra})
    return specs


def inspect_models(root, a):
    from catboost import CatBoostRegressor
    from importlib.metadata import version
    records = []
    for spec in model_specs(a):
        path = safe_path(root, spec["path"])
        if digest(path) != spec["sha256"]:
            raise ValueError("Original model hash changed")
        m = CatBoostRegressor()
        m.load_model(str(path))
        cat_indices = [spec["feature_names"].index(c) for c in spec["categorical_columns"]]
        if m.feature_names_ != spec["feature_names"] or m.get_cat_feature_indices() != cat_indices:
            raise ValueError("Saved-model feature/categorical schema changed")
        p = m.get_all_params()
        is_base = spec["role"].endswith("base")
        if (m.tree_count_ != 400 or p["iterations"] != 400 or p["depth"] != (8 if is_base else 6)
                or p["random_seed"] != spec["seed"] or p["loss_function"] != "RMSE" or p["task_type"] != "CPU"
                or p["l2_leaf_reg"] != 10 or not math.isclose(p["learning_rate"], .08 if is_base else .05, rel_tol=1e-6)):
            raise ValueError("Saved model differs from declared 400-tree contract")
        internal = json.loads(m.get_metadata()["params"])
        records.append({**spec, "categorical_feature_indices": cat_indices, "tree_count": m.tree_count_,
                        "parameters": {k: p[k] for k in PARAMETER_KEYS},
                        "serialized_thread_count": internal.get("system_options", {}).get("thread_count")})
    return {"records": records, "reader_catboost_version": version("catboost"),
            "training_environment_certified": False, "fitting_called": False, "prediction_called": False}


def inspect_headers(root, a):
    records = []
    for role in ("development", "calibration", "selection_train", "selection_validation"):
        relative = str(Path(AUDITS[role]).parent / "bvival_action_dataset.csv")
        path = safe_path(root, relative)
        if digest(path) != a[role]["output_sha256"]:
            raise ValueError("Target-table bytes differ from audited hash")
        with path.open(newline="", encoding="utf-8") as stream:
            header = next(csv.reader([stream.readline()]))
        if len(header) != len(set(header)) or not set(a[role]["feature_allowlist"]).issubset(header):
            raise ValueError("Malformed target schema")
        records.append({"role": role, "path": relative, "sha256": a[role]["output_sha256"], "header": header})
    return {"records": records, "all_four_headers_equal": all(x["header"] == records[0]["header"] for x in records),
            "opaque_row_bytes_hashed": True, "data_rows_parsed": False,
            "current_generator_schema_and_generated_hashes_verified": False}


def recover(root, models=False, headers=False):
    root = Path(root).resolve(strict=True)
    a, hashes = {}, {}
    for role, relative in AUDITS.items():
        path = safe_path(root, relative)
        a[role], hashes[relative] = json.loads(path.read_text()), digest(path)
    config = json.loads(safe_path(root, CONFIG).read_text())
    graph = recorded_graph_checks({k: a[k] for k in AUTO})
    selection = selection_checks(a, config, hashes)
    return {"date": "2026-10-01", "stage": "AUTOSCOUT24_FINAL_AND_STRICT_SELECTION_CONTRACT_RECOVERED_NOT_REPLAYED",
            "audit_file_sha256": hashes, "config_sha256": digest(safe_path(root, CONFIG)),
            "final_graph_checks": graph, "strict_selection_graph_checks": selection,
            "historical_test_labels_already_opened": True, "historical_freeze_comparison": freeze_check(root, a),
            "current_static_import_closure": static_import_closure(root, ENTRYPOINTS),
            "source_path": a["cohort"]["source_path"], "recorded_source_sha256_not_raw_bytes_checked": a["cohort"]["source_sha256"],
            "split_counts": a["cohort"]["split_counts"], "action_fields": a["pairs"]["action_fields"],
            "final_base_fit_scope": "train_plus_validation_after_fixed_field_selected_on_separate_train_only_branch",
            "selection_branch_counts": {k: a["selection_pairs"][k] for k in ("train_listing_count", "validation_listing_count", "train_pair_count", "validation_pair_count")},
            "validation_fixed_field": a["decision"]["selected_action"], "validation_capacity": a["decision"]["capacity"],
            "validation_field_selection_is_not_test_performance": True, "recorded_config": config,
            "saved_model_parameter_inspection": inspect_models(root, a) if models else None,
            "target_header_inspection": inspect_headers(root, a) if headers else None,
            "rights_correction_note": RIGHTS_NOTE, "rights_correction_note_sha256": digest(safe_path(root, RIGHTS_NOTE)),
            "underlying_data_or_derivative_publication_rights_adjudicated": False,
            "raw_source_records_parsed": False, "target_or_score_rows_parsed": False,
            "study_model_fitted_or_prediction_called": False, "outcome_evaluation_called": False,
            "empirical_replay_completed": False, "independent_confirmation": False,
            "public_release_created": False, "submission_ready": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inspect-models", action="store_true")
    p.add_argument("--inspect-target-headers", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = recover(args.root, args.inspect_models, args.inspect_target_headers)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"stage": report["stage"], "report_sha256": digest(args.output),
                          "final_checks": len(report["final_graph_checks"]), "selection_checks": len(report["strict_selection_graph_checks"])}))
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
