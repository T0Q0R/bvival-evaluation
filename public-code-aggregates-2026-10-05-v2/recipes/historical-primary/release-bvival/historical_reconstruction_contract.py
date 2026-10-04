"""Exact metadata-only bindings for three historical primary reconstructions.

Imports release/planning helpers only, never scientific training modules.
Original receipts are creation-time records, not current execution statuses.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from check_source_bundle import safe_file
from check_mucars_reconstruction import AUDITS as MU_AUDITS
from check_jucars_reconstruction import AUDITS as JU_AUDITS
from check_autoscout24_reconstruction import AUDITS as AUTO_AUDITS, CONFIG

VERSION = "historical-primary-source-reconstruction-2026-10-04-v1"
LOCK = "environment/resolved-main-macos-arm64.lock"
LOCK_SOURCE = "release-bvival/output/hash-lock-check-2026-10-01-X1Ohm9/resolved-main-macos-arm64.lock"
LOCK_SHA = "206739a71c7554641dabf2c41d1a8ec184af9a7cf0481b233f1cb343abfced0a"
PLAN_SHAS = {
    "mucars": "bfe5325edbc22b3b5d5cde44dab90d8436fea0e7a5ea6d8b168c0ed5bf01257f",
    "jucars": "201c8fbd418034aa545c0ee5aa70dbd1f476cfd359dbf4c2c9db36eb8eaa81b3",
    "autoscout24": "aaaa6753a119c1e081a70b71d7eee04b5ce17a0f60d9ea1a2907e635262dd683",
}
RECOVERY_SHAS = {
    "mucars": "bf7c03e2abe88e34f87735a7439ec166da7f195f09a320993043d2b65874ca95",
    "jucars": "a066cc5ac617ccdfe14b239de00da1754e7ddb7fa453c5fdb948e9b995e9f236",
    "autoscout24": "3956332b97d5be334e1f97f3e83e3f5b0f9b1bf7f8df106374e421557b7404fd",
}
AUDITS = {"mucars": MU_AUDITS, "jucars": JU_AUDITS, "autoscout24": AUTO_AUDITS}
UNITS = {"mucars": "MAD", "jucars": "JOD", "autoscout24": "EUR"}
DOIS = {"mucars": "10.17632/vjrbcb2rrt.2", "jucars": "10.17632/ddcz486x5t.2",
        "autoscout24": "10.5281/zenodo.17643343"}
TABLE_NAMES = {"bvival_error_budget_curve.csv", "bvival_action_mix.csv",
               "bvival_policy_summary.csv", "bvival_reference_comparisons.csv"}
ENTRYPOINTS = ["release-bvival/run_historical_reconstruction_package.py",
               "release-bvival/check_historical_reconstruction_package.py"] + [
    "release-bvival/run_" + name + "_" + phase + "_replay.py"
    for name in PLAN_SHAS for phase in ("scoring", "outcome")]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plan_path(name):
    if name not in PLAN_SHAS:
        raise ValueError("Unknown historical source")
    return "release-bvival/" + name + "-reconstructed-primary-plan-2026-10-01.json"


def recovery_path(name):
    plan_path(name)
    return "release-bvival/" + name + "-reconstruction-2026-10-01.json"


def closure_entrypoints(bound):
    # Subprocess argv is not a Python import. Include every pinned study module
    # explicitly, not only the release wrappers' AST import dependencies.
    return sorted(set(ENTRYPOINTS) | set(bound["study"]))


def bindings(root):
    """Read only pinned plans/receipts; require their exact allowlisted audits."""
    root = Path(root).resolve(strict=True)
    metadata, study, sources = {}, {}, {}
    for name in PLAN_SHAS:
        pp, rp = plan_path(name), recovery_path(name)
        for relative, expected in ((pp, PLAN_SHAS[name]), (rp, RECOVERY_SHAS[name])):
            if sha256(safe_file(root, relative)) != expected:
                raise ValueError("Fixed historical metadata drift: " + relative)
            metadata[relative] = expected
        plan = json.loads(safe_file(root, pp).read_text())
        recovery = json.loads(safe_file(root, rp).read_text())
        if name != "autoscout24" and recovery["source_doi"] != DOIS[name]:
            raise ValueError("Dataset citation differs from recovered source")
        if set(recovery["audit_file_sha256"]) != set(AUDITS[name].values()):
            raise ValueError("Historical audit allowlist differs")
        metadata.update(recovery["audit_file_sha256"])
        closure = plan.get("static_study_source_closure", plan.get("static_source_closure"))
        for relative, expected in closure["local_source_sha256"].items():
            if relative in study and study[relative] != expected:
                raise ValueError("Source closures disagree")
            study[relative] = expected
        rules = plan.get("engineering_concordance_rules", plan.get("outcome_concordance"))
        if set(rules["reference_aggregate_hashes"]) != TABLE_NAMES:
            raise ValueError("Four aggregate reference tables required")
        directory = rules.get("original_aggregate_reference", rules.get("reference_directory"))
        for filename, expected in rules["reference_aggregate_hashes"].items():
            metadata[directory + "/" + filename] = expected
        if name == "mucars":
            inputs = {"source": {"path": plan["stages"][0]["argv_template"][3],
                                 "sha256": plan["source_sha256_to_verify_before_execution"]}}
        elif name == "jucars":
            inputs = plan["source_bindings_to_verify_before_execution"]
            study[plan["schema_adapter_source"]["path"]] = plan["schema_adapter_source"]["sha256"]
        else:
            inputs = {"source": plan["source_binding"]}
            metadata[CONFIG] = recovery["config_sha256"]
            metadata[recovery["rights_correction_note"]] = recovery["rights_correction_note_sha256"]
        sources[name] = {"inputs": inputs, "source_doi": DOIS[name], "price_unit": UNITS[name],
                         "plan": pp, "plan_sha256": PLAN_SHAS[name],
                         "scoring_root": plan["planned_output_root"],
                         "reference_directory": directory,
                         "reference_aggregate_hashes": rules["reference_aggregate_hashes"],
                         "historical_tests_already_opened": True}
    for relative, expected in {**metadata, **study}.items():
        if sha256(safe_file(root, relative)) != expected:
            raise ValueError("Bound metadata/source/reference drift: " + relative)
    return {"metadata": metadata, "study": study, "sources": sources}
