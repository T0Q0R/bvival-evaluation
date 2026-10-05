# Turkey standalone development and price-free scoring recipe

Version: `turkey-portable-development-2026-10-05-v2`.
This separate reconstruction contract supplies **development and price-free
held-out scoring only**, not outcome evaluation or an original release ledger.
It never requires the author's original project tree, study models, real row
caches, released held-out targets or once-release ledger. Do not represent this
bounded recipe as full-study empirical portability or a new blind test.

## Source and two runtime roles

Obtain the December workbook from its original author repository, commit
`df15151fc655a54a43e47ea6917c400eaa7f8670`:
<https://github.com/masoudmaleki/used-car-temporal-ml/tree/df15151fc655a54a43e47ea6917c400eaa7f8670>.
Required filename `Aralık_2025.xlsx`; SHA256
`99aa8018807a72082704047a72d93aca202c4e30126c62f1dec8e01d1ad8f77b`;
4,155,327 bytes. Source-specific terms apply. This package does not redistribute
the workbook or grant rights over marketplace records.

The main and neural interpreters must separately match the frozen runtime
versions in `execution_plan_freeze_audit.json`. The two hash locks are for
CPython 3.12.14/macOS arm64, not Linux/Windows installers. Installing them is
not proof that empirical outputs were reproduced. Pass each interpreter's
absolute virtual-environment path, without resolving its symlink to a shared
base interpreter.

## Safe integrity check

```sh
python3 -I -B -S release-bvival/check_turkey_portable_development.py --package-root .
```

This uses only the standard library and does not read source records or import
scientific modules. It checks an exact inventory, hashes, source import closure
and explicit subprocess roots. Put all generated files outside the package.

## Reconstruction stages

Provide a new private workspace, source workbook and both runtime paths to:

```sh
python3 -I -B release-bvival/run_turkey_portable_development.py \
  --package-root /absolute/path/to/this/package \
  --workspace /absolute/path/to/new-private-workspace \
  --source /absolute/path/to/Aralık_2025.xlsx \
  --main-python /absolute/path/to/main-venv/bin/python \
  --neural-python /absolute/path/to/neural-venv/bin/python
```

Without `--prepare`, this verifies inputs and runtimes only, creating no output
and parsing no workbook cells. Add `--prepare` to copy the code/metadata package
and rebuild all non-price features. All ten produced feature/config/code files
must exactly match the historical metadata. The producer audit is retained as
the actual new-run receipt; the original cohort audit is explicitly reference
metadata used to preserve the historical plan's byte binding, not a fabricated
new creation date or execution receipt.

After preparation, use the bound workspace, with **no replacement inputs**:

```sh
python3 -I -B release-bvival/run_turkey_portable_development.py \
  --workspace /absolute/path/to/new-private-workspace --readiness
python3 -I -B release-bvival/run_turkey_portable_development.py \
  --workspace /absolute/path/to/new-private-workspace --fit
python3 -I -B release-bvival/run_turkey_portable_development.py \
  --workspace /absolute/path/to/new-private-workspace --score
```

Readiness runs all **included Turkey development synthetic tests**, then the
original full synthetic controller/reload checks. It is not a claim that every
historical whole-project test is included. Synthetic hyperparameters are
reduced; the real `--fit` retains the original 258 valuation, 21 head and six
neural fits, with unchanged grids, grouping, seeds, validation selection and
signed objectives. Real fitting takes considerably longer than the smoke run.

The `--score` stage requires completed development, binds both entire held-out
feature partitions, and replays scores and allocations without reading
calibration/test prices. No `--outcomes`, price-release, bypass, ledger-reset,
held-out re-selection or tolerance-widening option exists. Failed attempts are
retained; no automatic retry/resume or overwriting is allowed.

## Evidence boundary and privacy

The package contains code, configs, source/version hashes, reference metadata
and two environment locks, not source rows, source models, saved synthetic
models or labels. Private outputs include row-level features, development
targets, models and scores; **do not upload those outputs**. The source bytes
physically contain prices: selective parsing is local discipline, not a
cryptographic enclave or independent label custodian. Historical evaluation
has already been opened. Any execution here is reconstruction, not confirmation.

The original software licence is scoped by `LICENSING_SCOPE.md`. No journal
acceptance, complete empirical replay, platform-data permission or public
download is inferred from an integrity/readiness check. The original author's
protected ledger and historical reconstruction attempts remain unchanged.
