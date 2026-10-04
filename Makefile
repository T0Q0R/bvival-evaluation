# Integrity checking needs only Python 3.12+, not the research dependencies.
# For synthetic tests, select an already provisioned interpreter by absolute path.
BVI_PYTHON ?= python3
BVI_RELEASE := public-code-aggregates-2026-10-05-v2

.PHONY: check test

check:
	"$(BVI_PYTHON)" -I -B -S "$(BVI_RELEASE)/CHECK_RELEASE.py"

test:
	cd "$(BVI_RELEASE)/recipes/source-inspection" && "$(BVI_PYTHON)" -B -m pytest -q -p no:cacheprovider experiments/tests release-bvival/tests
