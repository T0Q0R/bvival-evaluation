# Third-party software evidence for release preparation

Observed 4 October 2026. This file records third-party attribution and exact
installed-package evidence. It is **not this project's LICENSE**, a legal
compatibility decision, a complete binary redistribution notice bundle, or
permission to redistribute third-party datasets. No material was uploaded.

## Original-code GDFS adaptation

The study used Ian Covert's `dynamic-selection` implementation, pinned to
commit `e2b6f7403fdac4d217ac2ec5dea96acd60240b60`:
[upstream repository](https://github.com/iancovert/dynamic-selection).
The retained upstream licence is
`experiments/third_party/gdfs-upstream/LICENSE`, labelled MIT, with
`Copyright (c) 2023 Ian Covert`. Its SHA256 is
`ba259238ad1f4575e08555a1ec22838f2ccea8dcedcb5f16dcec70c7d6e09553`.
Keep that full, unchanged copyright/permission notice with any copied upstream
source. Our task-adaptation wrappers are distinct from the upstream source;
the upstream licence does not select a licence for our own project.

The adaptation is not a reproduction of the authors' published performance or
a licence for source vehicle rows. See Supplementary S4.3 for task differences.

## Installed dependency inventories

`collect_dependency_notice_inventory.py` reads distribution metadata and hashes
included notice files without importing study packages, models or prices.

| Role | Exact inventory | Scope and hash |
| --- | --- | --- |
| Main | `dependency-notice-inventory-main-2026-10-04.json` | All 27 distributions in the original resolved macOS-arm64 wheel lock, 49 installed notice files. SHA256 `205f106cdc1f50a5a6efdd1bd5d6c107d969baff945a473c480c4ffc5af6bff0`. |
| Neural | `dependency-notice-inventory-neural-2026-10-04.json` | Four version-sensitive distributions named by the original Turkey neural-role plan, 122 installed notice files. SHA256 `0cb89a0a1bc0bba7b1b0cd01d960dfa56e4c05700938f164f358dcf1a81faa05`. This is not every transitive package visible in the neural environment. |
| Neural, full observed closure | `dependency-notice-inventory-neural-full-2026-10-04.json` | All 36 observed installed distributions including pip/setuptools, 184 notice files, checked in a new hash-installed venv against the original four-package role. SHA256 `416f922d7c8cc19e0f119fc6329b5c839083b49b0416e407e16bbb34bd688bd0`. This is not a minimal requirements list or inventory of all embedded binary components. |

Every inspected distribution had at least one installed notice file. Each
record preserves the declared licence expression/classifiers, METADATA hash,
notice-file paths, sizes and hashes. These are package-declared evidence, not
an independent determination of every bundled component's rights.

Selected declarations (do not replace the full inventories with this table):

| Dependency | Observed version | Declaration in installed metadata |
| --- | --- | --- |
| CatBoost | 1.2.10, both roles | Apache License, Version 2.0, not MIT |
| NumPy, main | 2.3.5 | BSD classifier; retained licence text and included component notices |
| NumPy, neural | 2.5.3 | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` |
| SciPy | 1.18.1, main | BSD classifier; retained licence text and included component notices |
| pandas | 2.2.3, main | BSD 3-Clause License |
| scikit-learn | 1.9.1, both inspected roles | BSD-3-Clause |
| joblib | 1.6.0, main | BSD-3-Clause |
| pytest | 9.1.1, main | MIT |
| PyTorch | 2.14.0, neural | `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` |

The inventories do not distribute dependency wheels, pretrained weights,
Python itself, or operating-system libraries. They do not verify a fresh
installation by themselves or make all dependencies “MIT”. Separate actual
same-host installation evidence is recorded in
`../notes/analysis/bvival-policy-install-verification-2026-10-04.md` (main) and
`../notes/analysis/bvival-neural-install-verification-2026-10-04.md` (neural).
Neither report claims new-environment empirical retraining. An eventual source-only release
must retain notices for source it actually includes. A binary/wheel/container
release requires a separate inventory of what that artifact distributes.

## Boundaries requiring author confirmation

1. On 5 October the user delegated code/aggregate release choices. MIT was
   selected for project-original software in `public-release/LICENSE`, with
   explicit exclusions in `public-release/LICENSING_SCOPE.md`. This does not
   relicense third-party data, dependencies or aggregates, or determine individual
   copyright shares or institutional approval. Historical manifests remain unchanged.
2. Dataset access statements and underlying marketplace/derived-file rights
   are separate from dependency licences. See `paper-bvival/availability-audit.md`.
3. A software package is not automatically a public deposit. The repository
   https://github.com/T0Q0R/bvival-evaluation has been created, but the prepared
   payload has not yet been uploaded: integration write permission and browser
   file-upload permission were unavailable. There is no project DOI or verified
   external payload download yet. Raw rows, row-level outputs, models and private
   execution trees remain outside the authorised release scope.
4. Both main and full neural inventories are platform-specific. The earlier
   neural inventory remains a four-package historical subset. None is a
   cross-platform compatibility certification or a complete SPDX SBOM.

中文核对：已找到并保留 GDFS 原 MIT 许可，实查了主环境 27 个锁定依赖和
neural 原计划指定的四个依赖；后续清单已在新安装环境覆盖 neural 的完整
36 个观察分发包/184 个 notice 文件。CatBoost 和 PyTorch 不应概括成 MIT。
这不代替作者授予本项目许可，也不授权车辆数据/逐行衍生文件或对外上传。
