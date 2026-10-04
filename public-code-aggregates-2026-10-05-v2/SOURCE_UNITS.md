# Price-unit evidence: historical sources and AT follow-up, 4 October 2026

This audit does not change any frozen protocol, price parser, model, allocation,
bootstrap, or result file. The historical evidence below remains documentary;
the separate AT source qualification/evaluation is identified explicitly.

| Source | Reporting convention | Evidence boundary |
|---|---|---|
| MUCars-2024 v2 | MAD | Historical source/manuscript convention; no currency conversion or cross-market MAE pooling. |
| JUCars-2024 v2 | JOD | Historical source/manuscript convention; no currency conversion or cross-market MAE pooling. |
| AutoScout24 2025 | EUR | Historical source/manuscript convention; no currency conversion or cross-market MAE pooling. |
| Turkey December 2025 | Source price units; author documentation reports TL (Turkish lira) | Pinned upstream code explicitly labels the December `fiyat` mean TL; no independent listing-level currency or scale-consistency verification. |
| AT 2023 commercial-vehicle follow-up | EUR | The fixed eligible cohort requires explicit EUR in the hash-verified source workbook; evaluation targets were decoded only after policy freezing. Not a currency conversion, pooling rule or independent vehicle-price audit. |

The [pinned preparation script](https://github.com/masoudmaleki/used-car-temporal-ml/blob/df15151fc655a54a43e47ea6917c400eaa7f8670/prepare_data.py)
declares `DEC_RAW = Aralık_2025.xlsx`, passes its cleaned `fiyat` column to
`report("December", ...)`, and labels the resulting mean as TL (lines 37–43,
88–95, 111–115). No currency conversion is performed in that preparation
script. The [same-version README](https://github.com/masoudmaleki/used-car-temporal-ml/blob/df15151fc655a54a43e47ea6917c400eaa7f8670/README.md)
identifies the workbook as December arabam.com listings and mentions raw TL
among its modelling references. These are depositor assertions, not an
independent audit of individual advertisements.

The current manuscript conservatively retains "source price units".
A future wording revision may identify **author-documented TL**, while
retaining the absence of independent currency/scale verification. Do not
silently rewrite frozen `source_currency_not_independently_verified` flags.
Extreme prices alone neither establish nor rule out mixed units. They were
not removed on an outcome-dependent basis.
