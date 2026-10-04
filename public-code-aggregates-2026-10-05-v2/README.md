# BVI-Val: ranking and field choice in budgeted price prediction

Research code and aggregate evidence for **Risk Ranking, Benefit Ranking, and
Field Choice in Budgeted Vehicle Advertised-Price Prediction**. This is an
application evaluation, not a claim of a new learning primitive or universal
policy superiority. The task simulates revealing specifications already
recorded in vehicle advertisements; it does not measure transactions, actual
verification costs or operational profit.

## Start here

| Directory | What it provides |
| --- | --- |
| [`aggregates/`](aggregates/) | Aggregate results and audit summaries; no vehicle-level outputs |
| [`AGGREGATE_INDEX.md`](AGGREGATE_INDEX.md) | File-by-file interpretation and historical versus follow-up status |
| [`recipes/at/`](recipes/at/) | Fixed Austrian commercial-vehicle comparison: source workbook to results |
| [`recipes/historical-primary/`](recipes/historical-primary/) | Source-directed MUCars, JUCars and AutoScout24 primary reconstruction |
| [`recipes/historical-comparators/`](recipes/historical-comparators/) | Fixed-field benefit and neighbor controls, using privately rebuilt primary artifacts |
| [`recipes/source-inspection/`](recipes/source-inspection/) | Historical implementation and synthetic-test inspection, including Turkey; not an independent fitting recipe |
| [`SOURCE_UNITS.md`](SOURCE_UNITS.md) | Source-specific units and currency-evidence limits |

The historical joint policy improved MAE over risk routing on MUCars and
AutoScout24, but not clearly on JUCars or Turkey. Stronger retrospective
fixed-field benefit controls absorb most or all of the apparent field-choice
increment on MUCars and JUCars. In the locally prospectively fixed AT follow-up,
joint benefit and fixed-power benefit policies make **identical actions** on
3,251 evaluation records (326 actions each). Their MAE is EUR 7,364.4627;
the neighbor policy yields EUR 7,318.0484. These results do not establish
superiority of the fitted benefit head. AT belongs to the AutoScout platform
family and includes new/used commercial vehicles: it is not an independent
platform replication. Intervals condition on fitted models and allocations.

## Check the download without data or dependencies

From this repository root, with Python 3.12 or later:

```sh
python3 -I -B -S CHECK_RELEASE.py
```

This verifies the exact released payload hashes and frozen recipe packages.
It does **not** train models, access targets, verify scientific novelty, or
reproduce empirical MAEs. The `.git` directory of a clone is ignored; other
unexpected files are rejected. Keep generated outputs outside the download.

For empirical reconstruction, follow the selected recipe README and obtain
the specified dataset version directly from its depositor. AT and the three
historical primary/core-control recipes have been exercised locally on the
authors' host; this repository does not claim independent third-party or
cross-platform empirical reproduction. Its hash-pinned installation recipe
targets macOS arm64/CPython 3.12.14. Historical tests and AT evaluation have
already been opened: rerunning them is reconstruction, not new confirmation.
The additional source-inspection snapshot makes the historical Turkey feature,
fitting, scoring and evaluation dependencies inspectable. Its reconstructed
inspection plans still require omitted original artifact bindings and do not
constitute a standalone empirical run. Turkey and additional development
diagnostics have aggregate evidence, but **no standalone full Turkey/GDFS
training recipe in this release**. New AT cohort and frozen-score summaries
are post-result diagnostics without new target access, fitting or loss tests.

## Access, versions and rights

Repository: <https://github.com/T0Q0R/bvival-evaluation>. A commit identifies an
exact version. This is a prepared code/aggregate release. Its payload has not
yet been verified as publicly downloadable; the existing repository landing
page alone is not an execution release. It is not a journal submission, DOI
deposit, anonymous repository or acceptance.
No raw marketplace records, row-level predictions/actions/labels, fitted
models, private execution trees, full manuscript or author declaration forms
are included. Source access identifiers, versions and hashes are in each
recipe. Source availability does not adjudicate underlying marketplace rights.
Do not redistribute reconstructed records under this software licence.

Project-original software is under [MIT](LICENSE); the precise exclusions
are in [LICENSING_SCOPE.md](LICENSING_SCOPE.md). Dependencies and source datasets
retain their own terms. Aggregate outputs are supplied as research evidence,
not relicensed third-party data. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**Frozen snapshot status:** the `recipes/*/manifest.json` files and their
READMEs retain creation-time flags such as `publicly_archived=false` and
`project_license_selected=false`. These files are byte-preserved historical
records, not today's repository status. The root README/licence describe this
release. Preserving them avoids rewriting protocol or reconstruction history.
Earlier subpackage checks intentionally test these creation-time flags.

中文说明：本包准备发布代码与聚合证据，载荷尚未验证为公开可下载；
保留阳性、阴性及强简单基线结果；
不公开逐行车辆数据或模型。下载校验不是完整重训练，代码公开也不等于
已达到期刊录用标准。历史子包保留当时状态，以根目录说明为当前发布范围。
