"""Separate historical scoring/outcome companion, never once-release authority."""
from __future__ import annotations

VERSION = "turkey-portable-historical-outcomes-2026-10-05-v1"
SCOPE = "source_rebuilt_historical_scoring_and_outcomes_not_confirmation"
DEVELOPMENT_PACKAGE = "development-recipe"
DEVELOPMENT_MANIFEST_SHA = "997c01e2cbb5bb9c3f95bff84fbf1cbc409988f8a59203900ddf90d28c829972"
MISSING_SNAPSHOT = "experiments/tests/test_freeze_turkey_heldout_scores.py"
REFERENCE_FILES = {
    "calibration": (
        "experiments/outputs/turkey_calibration_fixed_evaluation_2026-09-30_v1/phase_fixed_evaluation.json",
        "80ba6dff70b49a51cf84049ab30f6d56ab911d7919de27ab6215cd9b579e096f",
    ),
    "test": (
        "experiments/outputs/turkey_test_fixed_evaluation_2026-10-01_v1/phase_fixed_evaluation.json",
        "0ac1a05cdb4ace95e86d18115825527689def300d435c0e97fc9b75338385e5e",
    ),
}
PRIOR_RELEASES = {
    "calibration": (
        "experiments/outputs/turkey_calibration_release_2026-09-30_v1/once_phase_release_audit.json",
        "26abd361406650b7388c5dfb173448da5e8a51d94b0115339db2e271af69fd97",
    ),
    "test": (
        "experiments/outputs/turkey_test_release_2026-10-01_v1/once_phase_release_audit.json",
        "d28f8216e6caed4d462e662ce70f891d6fbfe69c44b212aeba22e1a891f5e41e",
    ),
}
SCIENTIFIC_SOURCE_SHA = {
    "experiments/evaluate_turkey_released_phase.py": "9dca9182508b57ba2d9799a21c1426cdeaa7cc015af3fb242ac8a71914f45d1c",
    "experiments/turkey_frozen_outcome_evaluation.py": "274845ecf28b4a13df035a27cdd54a9f0858c20beb54386976adf1a5d4359153",
    "experiments/turkey_price_access_accounting_v2.py": "c0ff13976c49a1fb79cc46b0aa8dbb90ac20b12d142a9bae26cccda29a9fa6c4",
}
FALSE_CAPABILITIES = (
    "source_rows_or_models_included", "original_release_ledger_included",
    "original_once_release_authority_reissued", "new_independent_confirmation",
    "full_study_empirical_pipeline_verified_here", "public_download_verified_here",
)


def closure_roots():
    return ["release-bvival/run_turkey_portable_outcomes.py",
            "release-bvival/check_turkey_portable_outcomes.py", MISSING_SNAPSHOT,
            "release-bvival/tests/test_turkey_portable_outcomes.py"]
