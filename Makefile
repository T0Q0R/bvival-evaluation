# Integrity checking needs only Python 3.12+, not the research dependencies.
# For synthetic tests, select an already provisioned interpreter by absolute path.
BVI_PYTHON ?= python3
BVI_RELEASE := public-code-aggregates-2026-10-05-v2
BVI_TURKEY := turkey-portable-historical-outcomes-2026-10-05-v1

.PHONY: check test test-turkey

check:
	"$(BVI_PYTHON)" -I -B -S "$(BVI_RELEASE)/CHECK_RELEASE.py"
	"$(BVI_PYTHON)" -I -B -S "$(BVI_TURKEY)/release-bvival/check_turkey_portable_outcomes.py" --package-root "$(BVI_TURKEY)"

test:
	cd "$(BVI_RELEASE)/recipes/source-inspection" && "$(BVI_PYTHON)" -B -m pytest -q -p no:cacheprovider experiments/tests release-bvival/tests

test-turkey:
	cd "$(BVI_TURKEY)" && "$(BVI_PYTHON)" -B -m pytest -q -p no:cacheprovider release-bvival/tests/test_turkey_portable_outcomes.py
