"""Metadata-only contract for a separate Turkey development reconstruction.

This is not the original once-release authority or a replacement for its ledger.
No held-out outcomes are admitted by this package. Historical tests are open.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

VERSION = "turkey-portable-development-2026-10-05-v2"
SCOPE = "source_rebuilt_development_and_price_free_scoring_not_outcomes"
PLAN = "experiments/outputs/turkey_execution_plan_freeze_v1_2026-09-30"
COHORT = "experiments/data/processed/turkey_price_free_cohort_v1_2026-09-30"
READINESS = "portable-execution/readiness"
DEVELOPMENT = "portable-execution/source-development"
SCORES = "portable-execution/heldout-price-free-scores"
SOURCE_SHA = "99aa8018807a72082704047a72d93aca202c4e30126c62f1dec8e01d1ad8f77b"
PLAN_AUDIT_SHA = "4ce5e47b6cbdabecd911394409a6328d88a597b3d3090642ade554fc86d12ab4"
COHORT_AUDIT_SHA = "e7a3668bae0507aaa0511f1cb27c4d25056e16bc793ba5f1f0897526f6bb448d"
COHORT_REFERENCE = "reference-metadata/cohort_freeze_audit.json"
FEATURE_CONFIG = "experiments/configs/turkey_price_free_cohort_2026-09-30.json"
FEATURE_PROTOCOL = "reference-metadata/prelabel_protocol_snapshot.md"
MAIN_LOCK = "environment/resolved-main-macos-arm64.lock"
NEURAL_LOCK = "environment/resolved-neural-macos-arm64.lock"
MAIN_LOCK_SOURCE = "release-bvival/output/hash-lock-check-2026-10-01-X1Ohm9/resolved-main-macos-arm64.lock"
NEURAL_LOCK_SOURCE = "release-bvival/output/neural-clean-install-2026-10-04-v1/resolved-neural-macos-arm64.lock"
FALSE_CAPABILITIES = (
    "source_rows_or_models_included", "original_release_ledger_included",
    "heldout_price_access_allowed", "new_independent_confirmation",
    "full_study_empirical_pipeline_portable", "public_download_verified_here",
)
STUDY_ENTRYPOINTS = (
    "experiments/build_turkey_price_free_manifest.py",
    "experiments/freeze_turkey_development_readiness.py",
    "experiments/run_turkey_development.py",
    "experiments/freeze_turkey_heldout_scores.py",
    "experiments/verify_turkey_heldout_score_freeze.py",
    # Path-launched subprocesses are not captured by AST imports alone.
    "experiments/turkey_neural_policy.py",
    "experiments/turkey_neural_replay.py",
    "experiments/turkey_price_free_neural_scoring.py",
    "experiments/run_turkey_heldout_freeze_check.py",
)
TEST_NAMES = (
    "test_turkey_feature_only.py", "test_build_turkey_price_free_manifest.py",
    "test_verify_turkey_price_free_bundle.py", "test_turkey_execution_contract.py",
    "test_turkey_information_boundary.py", "test_turkey_valuation_components.py",
    "test_turkey_price_cells.py", "test_turkey_development_io.py",
    "test_turkey_nested_valuation.py", "test_turkey_synthetic_integration.py",
    "test_turkey_policy_components.py", "test_turkey_neural_policy.py",
    "test_turkey_development_controller.py", "test_freeze_turkey_execution_plan.py",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def closure_roots():
    return sorted(set(STUDY_ENTRYPOINTS) | {
        "release-bvival/run_turkey_portable_development.py",
        "release-bvival/check_turkey_portable_development.py",
        *{"experiments/tests/" + name for name in TEST_NAMES},
    })
