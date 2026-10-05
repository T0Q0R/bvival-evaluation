"""Verified cached phase targets -> unchanged saved valuation/frozen policies.

No XLSX I/O, price release, fitting, field selection or model calibration. The
accounting erratum is REQUIRED; legacy raw count is never silently propagated.
Calibration descriptive only; test uses the original two-comparison contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from run_turkey_development import file_hashes, jsonable, save_json, verify_files
from train_turkey_nested_valuation import _prediction_records
from turkey_development_io import sha256
from turkey_execution_contract import ACTIONS, POLICIES
from turkey_frozen_outcome_evaluation import evaluate_frozen_outcomes
from turkey_price_access_accounting_v2 import corrected_accounting
from turkey_source_access_binding import (RAW_ROWS, bind_original_source_coordinates,
                                          canonical_source_ledger, source_prerequisite_preflight)
from turkey_source_price_release import (load_released_labels, observed_environment, selected_coordinates,
                                         verify_release_package, verify_source_readiness)
from verify_turkey_heldout_score_freeze import verify_score_freeze
from verify_turkey_price_free_bundle import table


SNAPSHOTS = ["evaluate_turkey_released_phase.py", "tests/test_evaluate_turkey_released_phase.py",
             "turkey_price_access_accounting_v2.py", "tests/test_turkey_price_access_accounting_v2.py",
             "turkey_frozen_outcome_evaluation.py", "verify_turkey_heldout_score_freeze.py",
             "train_turkey_nested_valuation.py"]


def verify_accounting_erratum(directory, release, binding, release_audit, *, project_root):
    """No target decoder/workbook; bind the separate correction to old release."""
    path = directory / "phase_accounting_erratum.json"
    if not path.is_file():
        raise ValueError("Verified phase accounting erratum required before cached prices")
    audit = json.loads(path.read_text())
    if (audit.get("stage") != "APPEND_ONLY_PHASE_ACCOUNTING_ERRATUM_NO_TARGET_REREAD"
            or audit.get("phase") != binding["phase"] or audit.get("evidence_status") != binding["evidence_status"]
            or audit.get("input_release_audit_sha256") != sha256(release / "once_phase_release_audit.json")
            or audit.get("input_phase_binding_sha256") != sha256(release / "phase_binding.json")
            or Path(audit.get("input_release_directory", "")).absolute() != release.absolute()
            or any(audit.get(f) is not False for f in ("source_workbook_opened_here", "cached_target_values_decoded_here",
                                                      "original_package_modified", "models_or_actions_changed", "phase_reopened"))):
        raise ValueError("Accounting erratum role/phase/release binding differs")
    verify_files(directory, audit["artifact_sha256"], ignored=[path.name])
    for name in ("turkey_price_access_accounting_v2.py", "tests/test_turkey_price_access_accounting_v2.py"):
        if (directory / (Path(name).name + ".snapshot")).read_bytes() != (project_root / "experiments" / name).read_bytes():
            raise ValueError("Accounting correction runtime/test snapshot differs")
    reader_hash = sha256(release / "turkey_price_cells.py.snapshot")
    corrected = corrected_accounting(release_audit["cell_access"], binding["expected_data_rows"],
        set(selected_coordinates(binding)), legacy_reader_sha256=reader_hash, successful_parse=release_audit["release_succeeded"])
    if (audit.get("legacy_reader_sha256") != reader_hash or audit.get("original_cell_access") != release_audit["cell_access"]
            or audit.get("corrected_cell_access") != corrected):
        raise ValueError("Accounting correction differs from independently reconstructed counters")
    return {"audit_sha256": sha256(path), "corrected_cell_access": corrected}


def validate_phase_alignment(package, coordinates, binding, readiness_audit_hash, *, phase):
    if phase not in {"calibration", "test"} or package["audit"]["evidence_status"] != "authorized_development_only":
        raise ValueError("Real complete frozen phase required, no synthetic substitution")
    config = package["config"]
    expected = {"stage": "BOUND_ONCE_PHASE_REQUIRES_VERIFIED_ENTRY", "evidence_status": "authorized_heldout_only",
        "phase": phase, "source_sha256": coordinates["source_version"]["sha256"],
        "plan_sha256": coordinates["plan_sha256"], "score_freeze_audit_sha256": package["audit_sha256"],
        "coordinate_manifest_sha256": coordinates["coordinate_manifest_sha256"],
        "source_readiness_audit_sha256": readiness_audit_hash, "manifest": coordinates["manifest"],
        "expected_headers": config["expected_headers"], "expected_data_rows": RAW_ROWS,
        "frozen_partition_keys": sorted(r["record_key"] for r in coordinates["manifest"] if r["split"] == phase)}
    selected_coordinates(binding)
    if (binding != expected or binding["frozen_partition_keys"] != package["scores"][phase]["before"]["keys"]
            or binding["frozen_partition_keys"] != sorted(package["partitions"][phase]["groups"])
            or package["audit"]["original_plan_sha256"] != coordinates["plan_sha256"]
            or package["audit"]["cohort_audit_sha256"] != coordinates["cohort_audit_sha256"]):
        raise ValueError("Released targets/frozen scores/original full phase do not align")


def verify_inputs(freeze, development, plan, release, erratum, readiness, *, phase, project_root):
    # No cached label/model/acquisition decode until all provenance checks pass.
    if not (release / "once_phase_release_audit.json").is_file():
        raise ValueError("Completed cached phase release required; never open workbook here")
    package = verify_score_freeze(freeze, development, project_root=project_root, plan_directory=plan)
    prerequisites = source_prerequisite_preflight(freeze, development, plan, project_root=project_root)
    verify_source_readiness(readiness, prerequisites, project_root=project_root)
    coordinates = bind_original_source_coordinates(plan, project_root=project_root)
    binding = json.loads((release / "phase_binding.json").read_text())
    validate_phase_alignment(package, coordinates, binding, sha256(readiness / "source_release_readiness_audit.json"), phase=phase)
    audit = verify_release_package(release, binding, canonical_source_ledger())
    accounting = verify_accounting_erratum(erratum, release, binding, audit, project_root=project_root)
    dev_audit = json.loads((development / "development_execution_audit.json").read_text())
    if observed_environment(dev_audit["runtime_environment"]) != dev_audit["runtime_environment"]:
        raise ValueError("Saved valuation runtime differs before model/target access")
    return package, coordinates, binding, accounting


def phase_candidate_values(cohort, config, full_keys, phase_keys):
    # This decodes the complete previously price-free acquisition CSV, not just
    # selected actions. Only the requested phase is used in outcome predictions;
    # no candidate values are supplied to policy scoring or new field selection.
    rows = table(cohort / "acquisition_fields.csv", ["record_key", *config["action_fields"]])
    if ({r["record_key"] for r in rows} != set(full_keys) or not set(phase_keys).issubset(full_keys)
            or set(config["action_fields"]) != set(ACTIONS)):
        raise ValueError("Full bound candidate table and phase key universe required")
    values = {r["record_key"]: {f: r[f] for f in ACTIONS} for r in rows}
    return {k: values[k] for k in phase_keys}


def verify_prediction_replay(first, second, keys, frozen_before, seeds):
    if set(first) != set(keys) or set(second) != set(keys) or frozen_before["keys"] != keys:
        raise ValueError("Complete aligned phase prediction replay required")
    for key, index in zip(keys, range(len(keys))):
        a, b = first[key], second[key]
        expected = {"before_prediction_log", "before_disagreement_log", "after_prediction_log", "per_seed_state_logs"}
        if set(a) != expected or set(b) != expected:
            raise ValueError("Exact valuation prediction roles required, no policy scores")
        for field, shape in (("before_prediction_log", ()), ("before_disagreement_log", ()),
                             ("after_prediction_log", (3,)), ("per_seed_state_logs", (3, 4))):
            aa, bb = np.asarray(a[field], dtype=float), np.asarray(b[field], dtype=float)
            if (aa.shape != shape or bb.shape != shape or not np.isfinite(aa).all() or not np.isfinite(bb).all()
                    or not np.allclose(aa, bb, rtol=1e-12, atol=1e-12)):
                raise ValueError("Saved valuation replay changed or is nonfinite")
        raw = np.asarray(a["per_seed_state_logs"], dtype=float)
        if (not np.allclose([raw[i, 0] for i in range(3)],
                           [frozen_before["per_seed_before_raw_log"][str(s)][index] for s in seeds], rtol=1e-12, atol=1e-12)
                or not np.isclose(a["before_prediction_log"], frozen_before["before_prediction_log"][index], rtol=1e-12, atol=1e-12)
                or not np.isclose(a["before_disagreement_log"], frozen_before["before_disagreement_log"][index], rtol=1e-12, atol=1e-12)
                or not np.allclose([a["before_prediction_log"], *a["after_prediction_log"]],
                                   np.maximum(raw.mean(axis=0), 0), rtol=1e-12, atol=1e-12)
                or not np.isclose(a["before_disagreement_log"], raw[:, 0].std(ddof=0), rtol=1e-12, atol=1e-12)):
            raise ValueError("Valuation before/raw seed/ensemble differs from frozen action contexts")


def render_note(result, accounting):
    lines = ["# 原冻结策略的来源分区评估", "", f"分区：{result['split']}。有效目标：{result['price_valid_records']}。",
             "校准分区仅为描述性诊断，不作主假设检验，不用于模型/策略选择。" if result["calibration_diagnostic_only"]
             else "测试主比较按原双主比较协议；次指标只作描述，不根据本报告重选方案。",
             f"访问日志原始行数使用已核对的更正值：{accounting['corrected_cell_access']['source_data_rows_scanned']}。",
             "金额单位尚未独立核实；模拟字段揭示不是真实核验、成交价、成本节约或利润。", "",
             "| 预算 | 策略 | 全有效分区 MAE | 相对 risk-fixed (%) | 相对 direct-posterror (%) | 选中记录 | 负揭示收益数 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for budget in sorted(result["summaries"]["ensemble"]):
        for policy in POLICIES:
            row = result["summaries"]["ensemble"][budget][policy]
            effects = ["—" if row[f] is None else f"{row[f]:.6f}" for f in
                       ("relative_effect_vs_risk_fixed_percent", "relative_effect_vs_direct_posterror_percent")]
            lines.append(f"| {budget} | {policy} | {row['post_mae']:.6f} | {effects[0]} | {effects[1]} | "
                         f"{row['selected_feature_records']} | {row['negative_realized_gain_count']} |")
    lines += ["", "完整三预算、八策略、ensemble/三个种子保存在 JSON；种子不是独立市场样本。",
              "无效目标仅从损失中剔除，不改变特征队列/容量或补选。没有重新拟合或选择。", ""]
    return "\n".join(lines)


def evaluate_source_phase(freeze, development, plan, release, erratum, readiness, output, *, phase, project_root):
    if phase not in {"calibration", "test"} or output.exists():
        raise ValueError("Known phase and new nonexistent evaluation output required")
    package, coordinates, binding, accounting = verify_inputs(freeze, development, plan, release, erratum, readiness,
                                                             phase=phase, project_root=project_root)
    output.mkdir(parents=True, mode=0o700)
    step, cached_price_access_started = "created", False
    try:
        for name in SNAPSHOTS:
            with (output / (Path(name).name + ".snapshot")).open("xb") as stream:
                stream.write((project_root / "experiments" / name).read_bytes())
        keys = package["scores"][phase]["before"]["keys"]
        material = package["partitions"][phase]
        initial = {r["record_key"]: r["initial"] for r in material["records"]}
        step = "candidate_values_after_action_freeze"
        candidates = phase_candidate_values(project_root / package["plan"]["cohort_directory"], package["config"],
            [r["record_key"] for r in coordinates["manifest"]], keys)
        data = {"initial": initial, "actions": candidates}
        predictions = []
        for step in ("saved_valuation_first_pass", "saved_valuation_reload_pass"):
            models = [joblib.load(development / "valuation_models" / f"final_seed_{s}.joblib") for s in package["plan"]["seeds"]]
            predictions.append(_prediction_records(models, data, keys, package["config"]))
        verify_prediction_replay(*predictions, keys, package["scores"][phase]["before"], package["plan"]["seeds"])
        save_json(output / "phase_saved_valuation_predictions.json", predictions[0])
        step, cached_price_access_started = "verified_cached_targets", True
        labels = load_released_labels(release, binding, canonical_source_ledger())
        pure_predictions = {k: {f: predictions[0][k][f] for f in ("before_prediction_log", "after_prediction_log")} for k in keys}
        step = "fixed_policy_outcome_evaluation"
        result = evaluate_frozen_outcomes(keys, material["groups"], labels, pure_predictions,
            package["scores"][phase]["allocations"], package["plan"], package["config"],
            split=phase, evidence_status="authorized_heldout_only")
        save_json(output / "phase_fixed_evaluation.json", result)
        with (output / "phase_fixed_evaluation.md").open("x") as stream:
            stream.write(render_note(result, accounting))
        step = "input_integrity_recheck"
        final = verify_inputs(freeze, development, plan, release, erratum, readiness, phase=phase, project_root=project_root)
        if final[1:] != (coordinates, binding, accounting) or final[0]["audit_sha256"] != package["audit_sha256"]:
            raise ValueError("Original phase provenance changed during evaluation")
        audit = {"stage": "SOURCE_RELEASED_PHASE_FIXED_EVALUATION_COMPLETE", "phase": phase,
                 "evidence_status": "authorized_heldout_only", "feature_records": len(keys),
                 "price_valid_records": result["price_valid_records"], "primary_comparisons": len(result["primary_comparisons"]),
                 "confidence_intervals_or_pvalues_computed": result["confidence_intervals_or_pvalues_computed"],
                 "score_freeze_audit_sha256": package["audit_sha256"],
                 "release_audit_sha256": sha256(release / "once_phase_release_audit.json"),
                 "accounting_erratum": accounting, "model_prediction_replay_passed": True,
                 "predictions_match_frozen_before_contexts": True, "models_refitted": 0, "policies_reselected": False,
                 "raw_workbook_or_new_price_release_here": False, "cached_price_values_decoded_here": True,
                 "test_cached_prices_decoded_here": phase == "test", "simulated_all_three_field_outcomes": True,
                 "full_acquisition_csv_decoded_only_requested_phase_used": True,
                 "candidate_values_supplied_to_policy_scoring": False, "calibration_used_for_selection": False,
                 "registered_confirmatory_claim": False, "real_request_transaction_or_profit_evidence": False,
                 "artifact_sha256": file_hashes(output)}
        save_json(output / "released_phase_evaluation_audit.json", audit)
        verify_files(output, audit["artifact_sha256"], ignored=["released_phase_evaluation_audit.json"])
        return audit
    except Exception as error:
        save_json(output / "failed_released_phase_evaluation.json", {"step": step, "exception_class": type(error).__name__,
            "phase": phase, "cached_price_access_started": cached_price_access_started,
            "raw_workbook_or_new_price_release_here": False, "models_refitted": 0, "policies_reselected": False})
        raise
    finally:
        for path in output.iterdir():
            if path.is_file():
                path.chmod(0o400)


def main():
    parser = argparse.ArgumentParser(description="Cached released phase fixed evaluation; no workbook/price release")
    for flag in ("score-freeze", "development-run", "plan-freeze", "phase-release", "accounting-erratum", "readiness", "output-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--phase", choices=["calibration", "test"], required=True)
    args = parser.parse_args()
    audit = evaluate_source_phase(args.score_freeze, args.development_run, args.plan_freeze, args.phase_release,
        args.accounting_erratum, args.readiness, args.output_dir, phase=args.phase, project_root=Path(__file__).absolute().parents[1])
    print(json.dumps({k: audit[k] for k in ("stage", "phase", "feature_records", "price_valid_records", "primary_comparisons",
                                           "raw_workbook_or_new_price_release_here")}))


if __name__ == "__main__":
    main()
