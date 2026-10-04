# BVI-Val: ranking and field choice in budgeted price prediction

Research code and aggregate evidence for **Risk Ranking, Benefit Ranking, and
Field Choice in Budgeted Vehicle Advertised-Price Prediction**.

## Access status and inspection

**Public release; unauthenticated ZIP download verified on 5 October 2026.**
The exact approved archive, original-software licence and licensing-scope
documents are committed in this repository. The archive was downloaded without
login, extracted into a new directory, and checked against all 298 file hashes
and the four frozen recipe checks. This verifies access and payload integrity,
not empirical retraining or independent human reproduction.

[Pinned archive](https://github.com/T0Q0R/bvival-evaluation/blob/3ca9aad12a2b6132b6808f2e33e2fe8254a1d702/BVI-Val_code-aggregates_2026-10-05_v2.zip).
The [directly browsable release tree](https://github.com/T0Q0R/bvival-evaluation/tree/341bc93e677681748dd537c0737756586558b41d/public-code-aggregates-2026-10-05-v2)
was also downloaded without login and verified: **298 files, zero missing,
unexpected or changed files**. All four subpackage checks passed on the actual
extraction. This separate check preserves the archive's original bytes.

Release filename: `BVI-Val_code-aggregates_2026-10-05_v2.zip`.
Version: `2026-10-05-v2`; 298 files inside the extracted package, including
297 payload files and its manifest. ZIP SHA256:

```text
81bb9acd9c1a2a3f6a35203c2538dbc90e2382ef6a6e3e1a217ab51cd99484a1
```

Extract the pinned ZIP and enter
`public-code-aggregates-2026-10-05-v2/`, then run:

```sh
python3 -I -B -S CHECK_RELEASE.py
```

Python 3.12+ is sufficient for this integrity check; no source data, installed
study packages or network access is required. It verifies hashes and four
frozen recipe subpackages. **It does not perform empirical retraining.**
Keep generated outputs outside the extracted package. Alternatively, after
cloning this repository, run `make check` at the outer repository root.

For the synthetic tests, provision a recipe-compatible research interpreter
first and run `make test BVI_PYTHON=/absolute/path/to/python`. The source-inspection
tests passed on the publicly downloaded package in an existing author-host
environment (360 tests, 31 subtests); this is not a fresh installation or
independent empirical replication. The integrity-only GitHub workflow uses
the standard library and does not run these dependency-requiring tests or train
models. Its execution status is visible under Actions, not inferred from the
presence of the workflow file.

## What is inside

| Path inside the released directory or ZIP | Content |
| --- | --- |
| `AGGREGATE_INDEX.md` and `aggregates/` | Source-specific results, negative findings, diagnostic summaries and interpretation index |
| `recipes/at/` | Austrian commercial-vehicle source-to-results recipe and synthetic guard tests |
| `recipes/historical-primary/` | Source-directed MUCars, JUCars and AutoScout24 primary reconstruction |
| `recipes/historical-comparators/` | Fixed-field benefit and neighbor controls using privately rebuilt primary artifacts |
| `recipes/source-inspection/` | Frozen implementation-inspection snapshot, including Turkey fitting/scoring helpers; not a complete Turkey retraining recipe |
| `SOURCE_UNITS.md` | Currency and price-unit evidence boundaries |
| `release-manifest.json` | Exact per-file hashes and provenance |

Read each recipe's README before obtaining the specified source version from
its depositor. No raw marketplace rows, row-level predictions/actions/labels,
trained models, private execution trees, full manuscript or author declaration
forms are distributed. The archive has original research software, not bundled
dependency wheels. Its hash-pinned environment targets macOS arm64/CPython
3.12.14; it is not a cross-platform installation claim.

## Research scope

The task simulates revealing recorded specifications to reduce advertised-price
prediction error under a fixed action capacity. Historical gains over risk
routing vary by source. Stronger fixed-field benefit controls explain much of
the apparent field-choice increment. In the locally prospectively fixed AT
follow-up, joint benefit and fixed-power benefit policies select **identical
actions** on 3,251 evaluation records (326 actions each); their MAE is EUR
7,364.4627. The neighbor policy has MAE EUR 7,318.0484; fitted-head superiority
is not established. The archive retains the historical positive findings,
negative results, failed pilot and distinct Turkey H2 comparison.

All released evaluations have already been opened: replay is reconstruction,
not new confirmation. AT shares the AutoScout platform family and includes
new/used commercial vehicles. Turkey/GDFS aggregate diagnostics and
implementation-inspection entry points are provided, but this version does not
contain a standalone full Turkey/GDFS training recipe. Two additional AT
diagnostics use frozen scores or non-price source information only; they are
post-result characterization, not new confirmatory performance evidence.
The study makes no transaction-price, real verification-cost, profit or universal
policy-superiority claim. Local reconstruction is not independent third-party
empirical reproduction.

## Licensing and version history

[MIT](LICENSE) applies to the project-original software within the
[specified scope](LICENSING_SCOPE.md), not source data or dependencies. See
[third-party notices](THIRD_PARTY_NOTICES.md). Dataset use and redistribution
remain subject to source terms; public download does not adjudicate underlying
marketplace rights. Aggregate evidence is not a blanket relicensing of data.

The byte-preserved package README and manifests retain preparation-time access
text and flags, including `public_payload_download_verified=false`,
`publicly_archived=false` and `project_license_selected=false`, for provenance.
This **outer repository README** records current access; it does not rewrite
those snapshots or their checks. The checker's `PREPARED_PAYLOAD_INTEGRITY_VERIFIED`
status is an integrity statement, not today's public-access status.
GitHub commits identify versions. This is not a
DOI deposit, anonymous archive, journal submission or acceptance.

中文：代码与聚合结果压缩包已公开，无登录下载、解包及完整性检查已通过。
展开目录也已通过 298 文件逐字节核验，缺失、额外及差异均为零。
外层仓库 README 是当前访问说明；包内准备状态文本保留为冻结历史。
先读聚合索引，再运行完整性检查；已有研究环境可另运行合成测试。
原始车辆行和模型不公开；“检查通过”不代表完整实验复现或达到录用门槛。
子包旧标志是历史快照状态，不是新的公开访问证明。
