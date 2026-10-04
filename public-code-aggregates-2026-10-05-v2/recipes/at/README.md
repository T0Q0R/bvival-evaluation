# AT 2023 fixed comparison: source-directed reconstruction

This local package supplies the actual fixed schema/cohort/fitting/scoring/
evaluation code, both synthetic guard suites, the original pre-target protocol,
configuration and plan receipt. It contains **no source rows, labels, derived
cohort, predictions, trained models or original outcome cache**. Unlike the
historical inspection plans, this task can construct its inputs from the one
specified source workbook; no private historical model is required.
Do not infer a verified external installation or reconstruction from that recipe.
Code licensing and public reviewer access remain pending author decisions.

The AT evaluation has now been opened. Re-running it is reconstruction, **not
new confirmation**. The supplied protocol and original receipt retain their
creation-time statements; a replay's new timestamps do not make it preregistered.
Do not change fixed choices to make the result positive or remove hash guards.

## Source and runtime

Obtain `AT.xlsx` yourself from [Mendeley Data v1](https://doi.org/10.17632/kz6hh7832p.1),
not another country/version or a CSV export. Required SHA256:

`b0eeff0ad4f91c4dda2190cfb31d4831ff50003da6fe9c325ddf47ac556d54c7`

Read the AT dictionary and [methods article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11255507/).
The source record states CC BY 4.0; this is not this project's adjudication of
underlying marketplace rights. Actual 93,517 workbook observations do not
match the article's 75,850 AT count. The task uses 18,761 earliest eligible
title/year proxies; they are not verified vehicle identities. New/used
commercial vehicles and the AutoScout platform family are included.

The original runtime was CPython 3.12.14, NumPy 2.3.5, pandas 2.2.3,
scikit-learn 1.9.1 and CatBoost 1.2.10. The supplied unchanged wheel-hash lock
also pins transitive main-runtime dependencies and pytest; it targets macOS
arm64 only, not Windows/Linux or the historical Turkey neural environment.
Use a compatible interpreter/environment. An optional **new** environment:

```sh
AT_RUN_ROOT="$(mktemp -d -t bvival-at-reconstruction)"
python3.12 -m venv "$AT_RUN_ROOT/venv"
AT_PYTHON="$AT_RUN_ROOT/venv/bin/python"
"$AT_PYTHON" -m pip install --require-hashes -r environment/resolved-main-macos-arm64.lock
```

This installation command may download wheels. It is not performed merely by
checking this package; do not mark installation verified without running it.
If using an existing compatible interpreter, set `AT_PYTHON` to its absolute
path and create a fresh `AT_RUN_ROOT` outside this package for private outputs.
The package root must not receive generated files, so its manifest stays valid.

## Safe checks first (no real targets)

From the delivered package root:

```sh
"$AT_PYTHON" -I -B -S release-bvival/check_source_bundle.py --package-root .
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments:release-bvival \
  "$AT_PYTHON" -B -m pytest -q -p no:cacheprovider experiments/tests
```

These are synthetic fixtures, including a tiny CatBoost smoke fit, not a
reproduction of any real-data MAE. The AST import check is not a complete
runtime/file-dependency certification. No manuscript QA or original score
vector is smuggled into these synthetic tests.

## Fixed empirical phases (explicitly reads targets in designated phases)

Set `AT_SOURCE` to the absolute path of your lawfully obtained workbook.
Every output directory below must be absent/empty. Commands are executed
from the delivered package root, while outputs stay under `AT_RUN_ROOT`.

```sh
"$AT_PYTHON" -B experiments/audit_european_at_schema.py \
  --source "$AT_SOURCE" --receipt "$AT_RUN_ROOT/header-receipt.json"
"$AT_PYTHON" -B experiments/audit_european_at_features.py \
  --source "$AT_SOURCE" --receipt "$AT_RUN_ROOT/feature-receipt.json"
"$AT_PYTHON" -B experiments/build_european_at_cohort.py \
  --source "$AT_SOURCE" --output-dir "$AT_RUN_ROOT/cohort"

"$AT_PYTHON" -B experiments/run_european_at_matched.py --stage freeze \
  --config experiments/configs/european_at_matched_comparison_2026-10-04_v1.json \
  --cohort "$AT_RUN_ROOT/cohort/private_price_free_cohort.json" \
  --output-dir "$AT_RUN_ROOT/freeze"

"$AT_PYTHON" -B experiments/run_european_at_matched.py --stage development \
  --source "$AT_SOURCE" --freeze-dir "$AT_RUN_ROOT/freeze" \
  --cohort "$AT_RUN_ROOT/cohort/private_price_free_cohort.json" \
  --output-dir "$AT_RUN_ROOT/development"

"$AT_PYTHON" -B experiments/run_european_at_matched.py --stage scores \
  --freeze-dir "$AT_RUN_ROOT/freeze" --development-dir "$AT_RUN_ROOT/development" \
  --cohort "$AT_RUN_ROOT/cohort/private_price_free_cohort.json" \
  --output-dir "$AT_RUN_ROOT/scores"

"$AT_PYTHON" -B experiments/run_european_at_matched.py --stage evaluate \
  --source "$AT_SOURCE" --freeze-dir "$AT_RUN_ROOT/freeze" \
  --score-dir "$AT_RUN_ROOT/scores" \
  --cohort "$AT_RUN_ROOT/cohort/private_price_free_cohort.json" \
  --output-dir "$AT_RUN_ROOT/evaluation"
```

Schema/features/cohort do not decode source price values. Development opens
only February–July development and August validation targets (12,495 here).
Scores accept **no workbook argument**. Evaluation first binds predictions,
scores and exact actions, then decodes only November–December targets
(3,251 here). September–October targets stay unread. First-access receipts
are audit records, not evidence of independent label custody.

Expected selected field: `power_kw`; neighbor k: 20; four policies use 326
actions. Whole-cohort MAEs in EUR: no acquisition 7804.1724, P1 7465.9535,
P2/P3 7364.4627, P4 7318.0484. P2/P3 full actions coincide, not population
equivalence. P3 versus P4 −0.6342% with nominal 97.5% interval
[−1.9674, 0.6218]. All eight validation choices are returned. Different
receipt timestamps/hashes on replay are expected; compare outcome values,
fixed configuration and accounting, not bit-identical whole audit bundles.

The targets are advertised prices and the request is synthetic revelation
of recorded specifications. There are no transaction prices, request logs,
actual costs or profit outcomes. Conditional paired bootstrap does not include
retraining, proxy-dependence or research-direction selection uncertainty.
Keep reconstructed rows/models/private scores local; permission to publish
aggregates and code is a separate author/institutional decision.
