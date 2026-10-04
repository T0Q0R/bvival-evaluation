"""Reconstruct unchanged core comparisons from newly source-rebuilt artifacts.

No original cached row files or valuation refitting. The source recipes must
already have completed in private trees. The historical tests remain opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_source_bundle import safe_file
from historical_reconstruction_contract import AUDITS, sha256
from check_historical_reconstruction_package import verify as verify_primary
from historical_comparator_package import (
    verify, PRIMARY_MANIFEST_SHA, FIXED_TEMPLATE, RUNTIME_TOKEN,
    NEIGHBOR_CONFIG, NEIGHBOR_ROW_HASHES,
)
from run_mucars_outcome_replay import check_runtime
from run_mucars_scoring_replay import LAUNCHER
from run_historical_reconstruction_package import run_phase, write_new

SOURCES = ("mucars", "jucars", "autoscout24")


def verify_parent(root, source, profile):
    root = Path(root).resolve(strict=True)
    receipt_path = safe_file(root, "copied-tree-reconstruction-receipt.json")
    receipt = json.loads(receipt_path.read_text())
    if (receipt["stage"] != "COPIED_TREE_PRIMARY_SOURCE_TO_RESULTS_VERIFIED"
            or receipt["source"] != source or receipt["package_manifest_sha256"] != PRIMARY_MANIFEST_SHA
            or receipt["all_four_primary_tables_byte_identical"] is not True
            or receipt["new_independent_confirmation"] is not False
            or receipt["primary_aggregate_sha256"] != profile["reference_aggregate_hashes"]):
        raise ValueError("Completed source-rebuilt parent required")
    scoring = root / profile["scoring_root"]
    outcome_relative = "experiments/replays/" + source + "-reconstructed-primary-outcomes-copied-tree-v1"
    outcome = root / outcome_relative
    freeze_path = safe_file(scoring, "scoring-freeze.json")
    result_path = safe_file(outcome, "outcome-replay-receipt.json")
    if sha256(freeze_path) != receipt["scoring_freeze_sha256"] or sha256(result_path) != receipt["outcome_receipt_sha256"]:
        raise ValueError("Parent freeze/outcome receipt drift")
    freeze, result = json.loads(freeze_path.read_text()), json.loads(result_path.read_text())
    if (result["historical_tests_already_opened"] is not True
            or result["all_primary_table_byte_hashes_match"] is not True
            or result["scoring_freeze_sha256"] != sha256(freeze_path)):
        raise ValueError("Parent outcome status/hash contract differs")
    for directory, hashes in ((scoring, freeze["artifact_sha256"]), (outcome, result["artifact_sha256"])):
        for relative, expected in hashes.items():
            if sha256(safe_file(directory, relative)) != expected:
                raise ValueError("Reconstructed parent artifact drift")
    if sha256(safe_file(scoring, "bound-plan.json")) != receipt["plan_sha256"]:
        raise ValueError("Parent plan drift")
    return {"root": root, "scoring": scoring, "outcome": outcome,
            "freeze": freeze, "receipt_sha256": sha256(receipt_path)}


def copied_inputs(primary, parents, neighbor_config):
    """Explicit aliases for freshly generated ROWS and original REFERENCE metadata.

    A legacy JUCars audit describes the matching generated projection. It is
    identified as historical reference metadata, not a new producer receipt.
    """
    records = []
    for spec in neighbor_config["sources"]:
        source = spec["name"].lower()
        parent = parents[source]
        pair_stage = "final-pairs" if source == "autoscout24" else "pairs"
        target_stage = {"mucars": "development-actions", "jucars": "development-legacy",
                        "autoscout24": "development-targets"}[source]
        for role, folder, filename in (
            ("pairs", spec["pairs"], "prediction_pair_generation_audit.json"),
            ("value", spec["value"], "value_policy_fit_audit.json"),
            ("assembly", spec["scores"], "policy_score_assembly_audit_absolute_price_error.json"),
            ("join", spec["outcomes"], "evaluation_outcome_join_audit.json"),
            ("evaluation", spec["historical_evaluation"], "bvival_evaluation_audit.json"),
            ("development_value" if source == "jucars" else "development",
             spec["development_actions"], "bvival_action_dataset_audit.json"),
        ):
            origin = safe_file(primary, AUDITS[source][role])
            records.append({"target": "experiments/outputs/" + folder + "/" + filename,
                            "origin": origin, "sha256": sha256(origin), "source": source,
                            "kind": "historical_reference_metadata_not_new_producer"})
        rows = [(parent["scoring"], pair_stage + "/" + filename, spec["pairs"] + "/" + filename)
                for filename in ("development_pre_action_features.csv", "evaluation_pre_action_features.csv",
                                 "development_prediction_pairs.csv", "evaluation_prediction_pairs.csv", "development_labels.csv")]
        rows += [(parent["scoring"], target_stage + "/bvival_action_dataset.csv",
                  spec["development_actions"] + "/bvival_action_dataset.csv"),
                 (parent["scoring"], "scores/frozen_policy_scores_absolute_price_error.csv",
                  spec["scores"] + "/frozen_policy_scores_absolute_price_error.csv"),
                 (parent["outcome"], "evaluation-inputs/evaluation_outcomes.csv",
                  spec["outcomes"] + "/evaluation_outcomes.csv"),
                 (parent["outcome"], "evaluation-mae/bvival_error_budget_curve.csv",
                  spec["historical_evaluation"] + "/bvival_error_budget_curve.csv")]
        for directory, relative, target in rows:
            origin = safe_file(directory, relative)
            records.append({"target": "experiments/outputs/" + target, "origin": origin,
                            "sha256": sha256(origin), "source": source,
                            "kind": "new_source_reconstruction_generated_rows"})
    if len({r["target"] for r in records}) != len(records):
        raise ValueError("Duplicate input alias")
    return records


def fixed_content(result):
    value = json.loads(json.dumps(result))
    # These three execution provenance fields necessarily differ. ALL remaining
    # selection counts/hashes, four MAEs and interval/conditioning fields stay.
    for key in ("created_at_utc", "config_sha256", "runner_sha256"):
        value["selection"].pop(key)
    return value


def neighbor_content(result):
    value = dict(result)
    value.pop("timing_seconds")
    return value


def compare_outputs(package, workspace):
    fixed, neighbors = {}, {}
    for source in SOURCES:
        old = json.loads(safe_file(package, "references/fixed/" + source + ".json").read_text())
        new = json.loads(safe_file(workspace, "fixed-results/" + source + "/aggregate_result.json").read_text())
        if fixed_content(old) != fixed_content(new):
            raise ValueError("Fixed comparator scientific result/selection differs: " + source)
        for filename, expected in old["selection"]["selection_hashes"].items():
            if sha256(safe_file(workspace, "fixed-results/" + source + "/" + filename)) != expected:
                raise ValueError("Fixed comparator exact selection differs")
        fixed[source] = {"four_MAEs_intervals_and_conditioning_exact": True,
                         "four_selection_files_byte_identical": True}
    for name, hashes in NEIGHBOR_ROW_HASHES.items():
        old = json.loads(safe_file(package, "references/neighbor/" + name + ".json").read_text())
        new = json.loads(safe_file(workspace, "neighbor-results/" + name + "/results.json").read_text())
        if neighbor_content(old) != neighbor_content(new):
            raise ValueError("Neighbor scientific result differs: " + name)
        for filename, expected in hashes.items():
            if sha256(safe_file(workspace, "neighbor-results/" + name + "/" + filename)) != expected:
                raise ValueError("Neighbor exact score/allocation bytes differ: " + name)
        neighbors[name] = {"nine_policy_results_and_all_boundaries_exact": True,
                           "scores_and_allocations_byte_identical": True}
    return {"fixed_benefit": fixed, "neighbor": neighbors,
            "numeric_tolerance": 0, "all_scientific_outputs_exact": True,
            "ignored_fields": {"fixed": ["selection.created_at_utc", "selection.config_sha256", "selection.runner_sha256"],
                               "neighbor": ["timing_seconds"]}}


def execute(package, primary, copies, workspace, python, opt_in=False):
    package, primary = Path(package).resolve(strict=True), Path(primary).resolve(strict=True)
    workspace, python = Path(workspace).absolute(), Path(python).absolute()
    package_check, primary_check = verify(package), verify_primary(primary)
    if sha256(safe_file(primary, "manifest.json")) != PRIMARY_MANIFEST_SHA:
        raise ValueError("Original primary recipe package required")
    if (workspace.exists() or workspace.is_symlink() or any(p.is_symlink() for p in workspace.parents)
            or workspace.is_relative_to(package) or workspace.is_relative_to(primary)):
        raise ValueError("New private workspace outside packages required")
    profiles = json.loads(safe_file(primary, "manifest.json").read_text())["sources"]
    parents = {s: verify_parent(copies[s], s, profiles[s]) for s in SOURCES}
    if any(workspace.is_relative_to(p["root"]) for p in parents.values()):
        raise ValueError("Do not write inside a reconstructed parent")
    for parent in parents.values():
        check_runtime(parent["scoring"], parent["freeze"], python)
    cfg = json.loads(safe_file(package, NEIGHBOR_CONFIG).read_text())
    inputs = copied_inputs(primary, parents, cfg)
    if not opt_in:
        return {"stage": "COMPARATOR_INPUT_PREFLIGHT_NO_EXECUTION",
                "package_check": package_check, "primary_package_check": primary_check,
                "generated_input_aliases": len(inputs), "parent_artifacts_hash_checked": True,
                "row_values_parsed": False, "workspace_created": False}
    workspace.mkdir(parents=True, mode=0o700, exist_ok=False)
    phase, completed = "copy", []
    started = datetime.now(timezone.utc).isoformat()
    try:
        manifest = json.loads(safe_file(package, "manifest.json").read_text())
        for record in manifest["files"]:
            target = workspace / record["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_file(package, record["file"]), target)
            if sha256(target) != record["sha256"]:
                raise ValueError("Copied companion code/config/reference drift")
        aliases = []
        for item in inputs:
            target = workspace / item["target"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with item["origin"].open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            if sha256(target) != item["sha256"]:
                raise ValueError("Copied newly generated input differs")
            aliases.append({**{k: v for k, v in item.items() if k != "origin"},
                            "parent_relative_origin": str(item["origin"].relative_to(
                                primary if item["kind"].startswith("historical_reference") else parents[item["source"]]["root"]))})
        write_new(workspace / "input-alias-provenance.json", aliases)
        fixed_cfg = json.loads(safe_file(package, FIXED_TEMPLATE).read_text())
        if fixed_cfg["runtime_python"] != RUNTIME_TOKEN:
            raise ValueError("Exact runtime placeholder required")
        fixed_cfg["runtime_python"] = str(python)
        write_new(workspace / "fixed-config.json", fixed_cfg)
        logs = workspace / "execution-logs"
        logs.mkdir()
        commands = {
            "fixed-benefit": ("experiments/run_historical_fixed_benefit_control.py",
                ["--project-root", str(workspace), "--config", str(workspace / "fixed-config.json"),
                 "--output-dir", str(workspace / "fixed-results")]),
            "fixed-stdlib-accounting": ("experiments/check_historical_fixed_benefit_reconstruction.py",
                ["--project-root", str(workspace), "--config", str(workspace / "fixed-config.json"),
                 "--output-dir", str(workspace / "fixed-results")]),
            "neighbor": ("experiments/run_bvival_neighbor_historical.py",
                ["--config", str(workspace / NEIGHBOR_CONFIG), "--output", str(workspace / "neighbor-results")]),
        }
        for phase, (script, arguments) in commands.items():
            command = [str(python), "-I", "-B", "-c", LAUNCHER, str(safe_file(workspace, script)), *arguments]
            completed.append({"phase": phase, **run_phase(command, workspace, logs / (phase + ".txt"))})
        phase = "concordance"
        concordance = compare_outputs(package, workspace)
        for item in inputs:
            if sha256(item["origin"]) != item["sha256"] or sha256(safe_file(workspace, item["target"])) != item["sha256"]:
                raise ValueError("Reference/generated inputs changed during execution")
        for record in manifest["files"]:
            if sha256(safe_file(workspace, record["file"])) != record["sha256"]:
                raise ValueError("Companion code/config/reference changed during execution")
        artifacts = {str(p.relative_to(workspace)): sha256(p) for p in sorted(workspace.rglob("*")) if p.is_file()}
        receipt = {"stage": "SOURCE_REBUILT_HISTORICAL_CORE_COMPARATORS_EXACTLY_RECONSTRUCTED",
                   "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                   "companion_manifest_sha256": sha256(safe_file(package, "manifest.json")),
                   "primary_package_manifest_sha256": PRIMARY_MANIFEST_SHA,
                   "parent_copy_receipt_sha256": {s: p["receipt_sha256"] for s, p in parents.items()},
                   "completed_phases": completed, "concordance": concordance,
                   "artifact_sha256": artifacts, "new_valuation_fits_called": False,
                   "fixed_rule_capacity_neighbor_scales_or_bootstrap_changed": False,
                   "old_cached_ROW_inputs_used": False, "historical_reference_audits_used": True,
                   "historical_tests_already_opened": True, "new_independent_confirmation": False,
                   "fresh_dependency_installation": False, "full_study_including_every_SI_analysis_reconstructed": False,
                   "publicly_archived_or_software_license_granted": False, "submission_ready": False}
        write_new(workspace / "comparator-reconstruction-receipt.json", receipt)
        return {k: v for k, v in receipt.items() if k != "artifact_sha256"}
    except BaseException as error:
        write_new(workspace / "failed-comparator-reconstruction.json",
                  {"phase": phase, "exception_class": type(error).__name__, "reason": str(error),
                   "completed_phases": completed, "attempt_retained": True,
                   "no_automatic_retry_tuning_or_tolerance_widening": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package-root", "primary-package", "workspace", "python", "mucars-copy", "jucars-copy", "autoscout24-copy"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    copies = {s: getattr(args, s + "_copy") for s in SOURCES}
    print(json.dumps(execute(args.package_root, args.primary_package, copies, args.workspace, args.python, args.execute), indent=2))
