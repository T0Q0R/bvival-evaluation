"""Recheck the original field selector on bound validation caches, no fits.

This reopens already-used validation action/risk rows only, not source/test
records. It is deterministic cached selection reconstruction, not a new
validation experiment, new strategy choice or complete model reconstruction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_autoscout24_reconstruction import AUDITS, digest
from check_mucars_reconstruction import safe_path
from run_mucars_scoring_replay import child_environment
from run_turkey_cached_outcome_replay import save_new, output_path, verify_hashes

REPORT = "release-bvival/autoscout24-reconstruction-2026-10-01.json"
REPORT_HASH = "3956332b97d5be334e1f97f3e83e3f5b0f9b1bf7f8df106374e421557b7404fd"
SELECTOR = "experiments/select_bvival_fixed_field_development.py"
WORKER = '''
import json,sys,runpy
from pathlib import Path
import pandas as pd
source,actions,risk,output=map(Path,sys.argv[1:])
allowed={actions.resolve(),risk.resolve()}
read_csv=pd.read_csv
def bounded_read(path,*a,**k):
    if not isinstance(path,(str,Path)) or Path(path).resolve() not in allowed:
        raise RuntimeError("ONLY_BOUND_VALIDATION_CACHES_ALLOWED")
    return read_csv(path,*a,**k)
pd.read_csv=bounded_read
sys.argv=[str(source),"--development-actions",str(actions),"--development-risk-scores",str(risk),
          "--budget-fraction","0.1","--output",str(output)]
runpy.run_path(str(source),run_name="__main__")
'''


def bindings(root):
    report_path = safe_path(root, REPORT)
    if digest(report_path) != REPORT_HASH:
        raise ValueError("Bound recovery report changed")
    report = json.loads(report_path.read_text())
    matches = [r for r in report["historical_freeze_comparison"]["code_and_metadata_comparisons"] if r["path"] == SELECTOR]
    if len(matches) != 1 or matches[0]["matches"] is not True:
        raise ValueError("Selector must match its historical freeze")
    protected = dict(report["audit_file_sha256"])
    protected.update({REPORT: REPORT_HASH, SELECTOR: matches[0]["frozen_sha256"]})
    paths = [str(Path(AUDITS["selection_validation"]).parent / "bvival_action_dataset.csv"),
             str(Path(AUDITS["selection_risk"]).parent / "evaluation_risk_scores.csv")]
    d = json.loads(safe_path(root, AUDITS["decision"]).read_text())
    if d["budget_fraction"] != .1 or d["evidence_status"] != "development_selection_only" or d["evaluation_labels_read"] is not False:
        raise ValueError("Original validation-only 10% decision required")
    protected.update({paths[0]: d["input_sha256"]["development_actions"], paths[1]: d["input_sha256"]["development_risk_scores"]})
    verify_hashes(root, protected)
    return report, protected, paths


def execute(root, python, relative):
    root = Path(root).resolve(strict=True)
    output = output_path(root, relative)
    report, protected, paths = bindings(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    try:
        save_new(output / "bound-inputs.json", {"stage": "CACHED_VALIDATION_SELECTION_REPLAY_NOT_NEW_MODEL_SELECTION",
                 "protected_file_sha256": protected, "worker_source": WORKER,
                 "runner_sha256": digest(Path(__file__)), "python_executable_sha256": digest(python),
                 "historical_test_already_opened": True, "new_fitting_or_inference": False})
        child = subprocess.run([str(python), "-I", "-B", "-c", WORKER, str(root / SELECTOR),
                                *[str(root / p) for p in paths], str(output / "selection.json")],
                               capture_output=True, text=True, env=child_environment(python), timeout=60)
        for name, data in (("worker.stdout.txt", child.stdout), ("worker.stderr.txt", child.stderr)):
            with (output / name).open("x") as stream:
                stream.write(data)
        if child.returncode:
            raise RuntimeError("Cached selector failed; retained worker.stderr.txt")
        expected = protected[AUDITS["decision"]]
        if digest(output / "selection.json") != expected:
            raise ValueError("New selection bytes differ from frozen decision; no replacement or retuning")
        decision = json.loads((output / "selection.json").read_text())
        verify_hashes(root, protected)
        receipt = {"date": "2026-10-01", "stage": "AUTOSCOUT24_CACHED_VALIDATION_FIELD_SELECTION_REPRODUCED",
                   "selection_sha256": expected, "selection_bytes_match_frozen_decision": True,
                   "selected_action": decision["selected_action"], "budget_fraction": decision["budget_fraction"],
                   "validation_listing_count": decision["listing_count"], "validation_capacity": decision["capacity"],
                   "all_four_candidate_fields_retained": decision["candidate_fields"],
                   "protected_file_hashes_checked_before_and_after": len(protected),
                   "existing_validation_action_and_risk_rows_parsed": True, "new_fit_or_prediction_called": False,
                   "raw_source_or_test_label_records_read": False, "historical_test_already_opened": True,
                   "fixed_field_for_test_changed": False, "independent_confirmation": False,
                   "full_training_scoring_or_outcome_pipeline_reconstructed": False,
                   "underlying_data_publication_permissions_adjudicated": False, "public_release_created": False,
                   "submission_ready": False}
        save_new(output / "cached-selection-receipt.json", receipt)
        return receipt
    except Exception as error:
        save_new(output / "failed-cached-selection.json", {"exception_class": type(error).__name__, "message": str(error),
                 "historical_files_not_replaced": True})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--output", default="experiments/replays/autoscout24-cached-field-selection-v1")
    p.add_argument("--execute-cached-selection", action="store_true", required=True)
    args = p.parse_args()
    r = execute(args.root, args.python, args.output)
    print(json.dumps({"stage": r["stage"], "selected_action": r["selected_action"],
                      "capacity": r["validation_capacity"], "exact_bytes": r["selection_bytes_match_frozen_decision"]}))


if __name__ == "__main__":
    main()
