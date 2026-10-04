# BVI-Val: consolidated local source inspection package

This is a current-workspace **source inspection and synthetic-test package**,
not a public deposit or independent replication. No software licence has been
selected. No marketplace records, labels, individual predictions, fitted
models, archive credentials, wheel binaries or manuscript PDFs are included.
Do not distribute externally before author rights/licensing review.

## What is consolidated

The earlier three-source experiment entrypoints and synthetic tests are joined
with the later MUCars, JUCars and AutoScout24 reconstruction/check/scoring/
outcome runners, and the Turkey saved-model/cached-outcome replay runner.
Local AST import dependencies, including parseable embedded Python workers,
are copied without executing them. The manifest pins the actual copied bytes.
Subprocess paths, all runtime branches and external input availability are
**not** certified by this static import closure.

`inspection-plans/` contains three unchanged reconstructed primary plans from
1 October, not new protocols. Their argv templates expose source-specific
hyperparameters, feature/categorical lists, seeds, folds, selection order,
budget and bootstrap settings. Output paths refer to the author's historical
reconstruction; audit receipts, unresolved runtime bindings and required
input artifacts are absent. Do not substitute test-selected choices or remove
those prerequisites to treat the plans as turnkey empirical commands.

## Safe local checks

Run from this package root, not the author's original project:

```sh
python -I -B -S release-bvival/check_source_bundle.py --package-root .
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments:release-bvival \
  python -m pytest -q -p no:cacheprovider experiments/tests release-bvival/tests
```

The second command uses synthetic fixtures. It is not a rerun on source data.
Use CPython 3.12 with the listed dependencies. The supplied wheel-hash lock
targets macOS arm64 only. Do not infer Windows/Linux compatibility from it.
Do not invoke empirical runners as a quick-start: they require original
hash-bound audit graphs and input artifacts that are deliberately absent here.
No guard or historical receipt should be rewritten to make a run succeed.

## Four-source capability contract

| Source | Code covered here | Inputs absent from this package | Interpretation |
|---|---|---|---|
| MUCars | Source manifest, pairs/targets, heads, scores, join/evaluation; reconstructed scoring/outcome runners | Original public source, recovered audit/model-parameter receipts, source/split bindings, runtime/install receipts and generated artifacts | Historical primary pipeline code can be inspected; this package alone is not an executable end-to-end empirical recipe |
| JUCars | Same stages plus legacy-target projection | Source and dictionary, separate historical value/risk target bindings, reconstruction and runtime receipts | Preserve inherited value-model provenance and primary-only scope; do not call it a uniform historical refit |
| AutoScout24 | Same stages plus train-only field-selection branch, cached selection checker | Lawfully acquired v2 source, separate selection/final-fit input bindings, reconstruction/runtime receipts | Field selection and final fitting are separate; repository metadata do not decide marketplace reuse rights |
| Turkey | Cached-outcome replay and its static evaluator/helper imports | Original saved models, score freeze, released test cache, canonical ledger, source-coordinate and accounting receipts, original bound runtimes | Not fresh nested fitting, neural-head fitting, policy rescoring or source-price release |

Helpers imported by the Turkey evaluator include development-related modules;
their presence is **not** a complete neural/development reconstruction bundle.
All four historical test sets have already been opened. Replaying any of them
cannot answer the teacher's request for a new independent strong-baseline test.

## Acquisition, rights and source units

Obtain source files through their original records, under the applicable terms:

- MUCars-2024 v2: https://doi.org/10.17632/vjrbcb2rrt.2
- JUCars-2024 v2: https://doi.org/10.17632/ddcz486x5t.2
- AutoScout24 2025: https://doi.org/10.5281/zenodo.17643343
- Turkey source repository pinned at commit
  `df15151fc655a54a43e47ea6917c400eaa7f8670`:
  https://github.com/masoudmaleki/used-car-temporal-ml/tree/df15151fc655a54a43e47ea6917c400eaa7f8670

These are reused-source identifiers, **not identifiers for this code package**.
The existing record routes were preserved; their live accessibility and current
terms were not reverified in this local packaging step. AutoScout24 record-level
MIT metadata are not a finding that platform records/derivatives may be reused
or redistributed. The authors must resolve that question before external
release. Source acquisition is not automated or performed here.

Turkey prices remain in **source price units**: author-described TL does not
establish an independently verified currency/scale. The task is simulated
revelation of already recorded fields and advertised-price error, not verified
transactions, field-request costs, labour savings or realised business profit.

## Environment and provenance

`environment/resolved-main-macos-arm64.lock` is an unchanged copy of the
previously recorded resolved wheel lock, including its creation-time caveats.
Its exact pins/hashes concern a local main/synthetic runtime, not the original
historical training environment or Turkey's distinct neural runtime. No
pretrained weights are needed for the covered synthetic tests/cached evaluator;
none are supplied. New installation on another machine was not attempted.

Three historical configurations are supplied for inspection. Subgroup configs
are diagnostic, not new fitting contracts. Historical strings such as sealed
test status describe their creation time, not current unseen status. The
package preserves code bytes; it does not claim those bytes match every
historical pretest freeze. Missing original artifact bindings remain explicit.

The companion aggregate package is
`four-source-review-prep-2026-10-02-v4` (separate, not bundled here). It checks
manuscript-to-aggregate numbers but does not replace original rows or fitting
code. Neither local package is a real reviewer-accessible public archive.

## Availability wording (current local status, not a submission-ready claim)

**Data Availability.** The study reuses four third-party sources identified
above. Aggregated result tables have been assembled in a local review package.
No raw or row-level derivative data are redistributed in either local package.
Third-party reuse and release permissions, especially AutoScout24, remain
unresolved; no public archive or reviewer-accessible repository is established.

**Code Availability.** A local source snapshot with static dependency checks,
synthetic tests and a platform-specific dependency lock has been assembled.
Full empirical reconstruction requires additional source data and original
hash-bound artifacts not included in that snapshot. Project licensing,
external access and persistent archiving remain pending author decisions.

## 中文核对

本包补的是老师此前无法核查的代码入口与依赖，不是新的性能证据。
本地复制包上的合成测试通过，也不等于独立机器完整重训通过。
公开仓库/审稿链接、软件许可证、AutoScout24 权利裁定仍需作者决定；
未解决之前，不把本包写成已公开或投稿就绪。不得将既有测试集重用
包装为“联合选择优于强简单基线”的新确认性实验。
