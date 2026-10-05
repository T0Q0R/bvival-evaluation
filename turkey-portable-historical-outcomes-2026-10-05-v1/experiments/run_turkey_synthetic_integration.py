"""Real-model, reduced-compute synthetic integration only; no source prices.

No readiness receipt is issued. Pending policy/neural integration keeps real
development label access closed. This is not a scientific experiment result.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
import zipfile
from io import BytesIO
from pathlib import Path

from build_turkey_price_free_manifest import digest, fold
from train_turkey_nested_valuation import policy_training_material, run_nested_valuation
from turkey_development_io import bound_plan, sha256
from turkey_execution_contract import ACTIONS
from turkey_price_cells import read_selected_prices


def synthetic_dataset(config):
    rows, initial, actions, prices, validity = [], {}, {}, {}, {}
    for i in range(100):
        split = "train" if i < 80 else "validation"
        group = digest(["synthetic_smoke_group", split, i])
        key = digest(["synthetic_smoke_record", i])
        rows.append({"record_key": key, "group_hash": group, "profile_hash": digest(["synthetic_profile", i]),
                     "split": split, "oof_fold": str(fold(group, config)) if split == "train" else ""})
        initial[key] = {"marka": "synthetic_brand_" + str(i % 2), "seri": "synthetic_series",
                        "yil": str(2010 + i % 10), "kasa_tipi": "sedan", "renk": "white", "kimden": "private",
                        "boyali_sayisi": "", "degisen_sayisi": ""}
        actions[key] = dict(zip(ACTIONS, [str(10000 + i * 100), "otomatik", "benzin"]))
        prices[key], validity[key] = 1000 + (i % 10) * 100, True
    return {"rows": rows, "initial": initial, "actions": actions, "prices": prices, "validity": validity}


def reduced_synthetic_plan(original):
    plan = copy.deepcopy(original)
    plan["evidence_status"] = "synthetic_smoke_only_not_source_development"
    for candidate in plan["valuation"]["candidates"]:
        params = candidate["params"]
        if candidate["family"] == "catboost":
            params.update(iterations=5, depth=3)
        elif candidate["family"] == "extra_trees":
            params.update(n_estimators=8, max_depth=3)
        else:
            params.update(max_iter=5, max_leaf_nodes=5)
    return plan


def synthetic_price_check():
    data = BytesIO()
    header = {"A": "marka", "B": "seri", "C": "model", "D": "fiyat"}
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("xl/workbook.xml", '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Synthetic" sheetId="1" r:id="rId1"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        columns = ''.join(f'<c r="{column}1" t="inlineStr"><is><t>{name}</t></is></c>' for column, name in header.items())
        archive.writestr("xl/worksheets/sheet1.xml", '<worksheet><sheetData><row r="1">' + columns + '</row>'
                         '<row r="2"><c r="D2"><v>1000</v></c></row><row r="3"><c r="D3"><v>2000</v></c></row>'
                         '<row r="4"><c r="D4"><v>UNREAD_SYNTHETIC_TEST_PRICE &invalid;</v></c></row></sheetData></worksheet>')
    data.seek(0)
    with zipfile.ZipFile(data) as archive:
        labels, access = read_selected_prices(archive, header, {1, 2}, expected_data_rows=3)
    if not all(row["price_valid"] for row in labels.values()) or access["unselected_price_values_decoded"] != 0:
        raise AssertionError("Synthetic price boundary failed")
    return access


def run_synthetic_check(plan_freeze, output, *, project_root):
    if output.exists() and any(output.iterdir()):
        raise ValueError("New empty synthetic output directory required")
    original, config, _, plan_audit = bound_plan(plan_freeze, project_root=project_root)
    plan, data = reduced_synthetic_plan(original), synthetic_dataset(config)
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o700)
    model_directory = output / "models"
    model_directory.mkdir(mode=0o700)
    sources = ["turkey_price_cells.py", "turkey_development_io.py", "train_turkey_nested_valuation.py",
               "run_turkey_synthetic_integration.py", "tests/test_turkey_price_cells.py",
               "tests/test_turkey_development_io.py", "tests/test_turkey_nested_valuation.py",
               "tests/test_turkey_synthetic_integration.py"]
    for name in sources:
        with (output / (Path(name).name + ".snapshot")).open("xb") as stream:
            stream.write((project_root / "experiments" / name).read_bytes())
    with (output / "synthetic_smoke_config.json").open("x") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
    with (output / "cohort_config_snapshot.json").open("x") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
    started = time.perf_counter()
    try:
        price_access = synthetic_price_check()
        result = run_nested_valuation(data, plan, config, model_directory=model_directory)
        targets = policy_training_material(data, result, config)
        if result["fit_count"] != 258 or len(result["candidate_trials"]) != 72:
            raise AssertionError("Incomplete synthetic grid/fold schedule")
        expected_context = set(config["initial_visible_fields"] + ["before_prediction_log", "before_disagreement_log"])
        if any(set(c) != expected_context for c in targets["pre_action_contexts"].values()):
            raise AssertionError("Hidden values entered policy context")
        audit = {"stage": "SYNTHETIC_PRICE_AND_NESTED_VALUATION_INTEGRATION_PASSED_NOT_SOURCE_READY",
                 "source_price_values_parsed": 0, "source_models_trained": False, "source_test_scores_computed": False,
                 "synthetic_real_estimator_fits": result["fit_count"], "synthetic_train_records": len(result["oof"]),
                 "synthetic_validation_records": len(result["validation"]), "synthetic_candidate_trials": len(result["candidate_trials"]),
                 "synthetic_price_access": price_access, "hyperparameters_reduced_for_synthetic_smoke": True,
                 "policy_context_contains_only_initial_and_before_summaries": True,
                 "nested_fit_and_excluded_record_overlap": 0, "validation_targets_used_for_base_selection": False,
                 "training_oof_targets_have_signed_gain": True, "elapsed_seconds": time.perf_counter() - started,
                 "plan_sha256": plan_audit["plan_sha256"], "cohort_audit_sha256": plan_audit["cohort_audit_sha256"],
                 "runtime_environment": plan_audit["runtime_environment"],
                 "policy_and_neural_integration_passed": False, "development_label_release_allowed": False,
                 "readiness_receipt_issued": False, "official_GDFS_reproduction_completed": False,
                 "source_scope_this_run": "synthetic_data_only_no_source_workbook_opened",
                 "fit_audit": result["fit_audit"], "candidate_trials": result["candidate_trials"],
                 "artifact_sha256": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}}
        with (output / "synthetic_integration_audit.json").open("x") as stream:
            json.dump(audit, stream, indent=2)
            stream.write("\n")
    except Exception as error:
        with (output / "failed_synthetic_attempt.json").open("x") as stream:
            json.dump({"exception_class": type(error).__name__, "source_price_values_parsed": 0,
                       "source_models_trained": False, "source_test_scores_computed": False}, stream)
        raise
    finally:
        for path in output.rglob("*"):
            if path.is_file():
                path.chmod(0o400)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_synthetic_check(args.plan_freeze, args.output_dir, project_root=Path(__file__).resolve().parents[1])
    print(json.dumps({key: result[key] for key in ("stage", "synthetic_real_estimator_fits", "elapsed_seconds",
                                                  "source_price_values_parsed", "development_label_release_allowed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
