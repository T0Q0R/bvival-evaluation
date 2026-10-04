# Historical primary reconstruction — local preparation, not public release

This package reconstructs the primary MUCars, JUCars and AutoScout24 comparisons
from separately obtained source files. It includes executable code, original
parameter-recovery records, 33 audit JSONs and 12 aggregate reference CSVs.
It does NOT contain source rows, model binaries, prediction pairs, policy
scores, action targets or evaluation labels. Do not publish private replay
workspaces: they contain all of those generated row-level artifacts.

## Scope and source access

| Source | Original record | Required inputs | Reported price units |
|---|---|---|---|
| MUCars v2 | <https://doi.org/10.17632/vjrbcb2rrt.2> | `cars_dataframe.csv` | MAD |
| JUCars v2 | <https://doi.org/10.17632/ddcz486x5t.2> | `cars_jordan.csv`, `data_dictionary.csv` | JOD |
| AutoScout24 snapshot | <https://doi.org/10.5281/zenodo.17643343> | `autoscout24_dataset_20251108.csv` | EUR |

The manifest maps each input's exact SHA256 to its source, fixed plan and four
aggregate outputs. A differently serialized/downloaded source fails before
fitting; do not normalize or substitute it silently. Obtain inputs from the
original records subject to their actual terms. This package grants NO data
rights or software license. AutoScout24's depositor states research/education/
analysis use and attributed publication are allowed; this does not independently
clear platform rights or row-level redistribution. Its earlier copied audit
note is preserved as a creation-time record, not a final current rights decision.

## Check without decoding source records

```sh
python -I -B release-bvival/check_historical_reconstruction_package.py --package-root .
```

This checks exact code/import bindings, audit receipts, aggregate references and
capability boundaries. It does not reconstruct empirical results.

## Dedicated runtime

The wheel-hash lock is `environment/resolved-main-macos-arm64.lock`, for
macOS arm64 and CPython 3.12.14. It does not claim Windows/Linux portability.
Create an isolated environment and retain the actual pip installation report:

```sh
python3.12 -m venv /absolute/private-runtime/venv
/absolute/private-runtime/venv/bin/python -I -m pip --isolated install \
  --require-hashes -r environment/resolved-main-macos-arm64.lock \
  --report /absolute/private-runtime/install-report.json
```

The runners verify all pinned distributions and wheel hashes, isolated imports,
`pip check` and a native synthetic fit before fitting study models. The specific
local copied-tree validation reuses a previously installed dedicated runtime;
it is NOT a new installation or an independent-machine replication.

## Source-to-results command

From this package, use one NEW private workspace per source:

```sh
/absolute/private-runtime/venv/bin/python -I -B \
  release-bvival/run_historical_reconstruction_package.py \
  --package-root . --workspace /absolute/private-mucars-replay \
  --source mucars --raw-source /absolute/downloads/cars_dataframe.csv \
  --python /absolute/private-runtime/venv/bin/python \
  --install-report /absolute/private-runtime/install-report.json --execute
```

For JUCars, change `--source` to `jucars`, use its raw CSV and add
`--dictionary /absolute/downloads/data_dictionary.csv`. For AutoScout24 use
`--source autoscout24` and its snapshot CSV. Never reuse a workspace. Omit
`--execute` for byte-level package/input preflight only: no workspace is created,
no source records decoded, no empirical fitting or evaluation started.

The controller copies only allowlisted payloads plus your separately supplied
inputs. It executes the fixed scoring stages, hashes a NEW score checkpoint,
then reconstructs joined outcomes and 10,000-replicate listing-bootstrap tables.
AutoScout24 uses an explicit new-receipt mode: the original default remains
pinned to the old receipt, while this mode binds the new receipt's SHA256 and
retains every artifact, plan, score and zero-tolerance table check.

Outputs are in the private workspace:

- `experiments/replays/<source>-reconstructed-primary-v1/`: rebuilt cohort,
  OOF predictions, action targets, models, scores and `scoring-freeze.json`.
- `experiments/replays/<source>-reconstructed-primary-outcomes-copied-tree-v1/`:
  joined outcomes, four `evaluation-mae/` tables and concordance receipts.
- `copied-tree-reconstruction-receipt.json`: actual execution time, hashes,
  exact primary-table byte agreement and explicit non-claims.

The original author-chosen numerical tolerances are not widened. The controller
requires all FOUR aggregate table byte hashes to match as well. On failure the
attempt is retained, not overwritten, automatically retried or tuned.

## Evidence boundaries

All three historical tests were already opened. Old receipt dates/stage flags
are preserved; the new controller receipt supplies the actual execution date.
This is reconstruction of already known outcomes, NOT new confirmation,
external preregistration, prediction validation on untouched data, or journal
acceptance evidence. JUCars preserves the declared legacy target projection and
primary-only join; it does not claim to replay its omitted secondary-score guard.

This package does not reconstruct Turkey, AT, GDFS, all stronger retrospective
comparators, all subgroup tables or all supplementary analyses. AT has a separate
source-directed recipe. Do not rename this three-source primary package as a
full-study reproduction package. No public reviewer link exists and author
software-license/release authorization remains outstanding.

## 中文核对

这是三源主结果的本地源到结果重建包，不是新的确认性实验。原始文件需
从正式来源自行取得；复制目录会产生逐行数据，不能直接上传。完成本机
复制目录重建，不等于独立机器重装、全部补充实验复现或已经达到可投稿状态。
作者仍需确认软件许可、发布权限与真实审稿访问；本包不替作者作这些决定。
