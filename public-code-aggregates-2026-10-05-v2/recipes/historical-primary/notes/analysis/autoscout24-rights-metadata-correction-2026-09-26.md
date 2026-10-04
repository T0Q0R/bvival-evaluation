# AutoScout24 2025 record-rights metadata correction

Checked 26 September 2026 against the original [Zenodo record](https://zenodo.org/records/17643343), its [record API](https://zenodo.org/api/records/17643343), and the [DataCite DOI metadata](https://api.datacite.org/dois/10.5281/zenodo.17643343). This is a metadata and author-governance audit, not legal advice or permission from AutoScout24.

| Surface | Observed field | Interpretation |
|---|---|---|
| Zenodo human landing page | Rights / License heading appeared blank in the rendered page | The display alone is not evidence that no machine-readable license exists. |
| Zenodo record API | `metadata.license.id` was `mit-license`; `metadata.rights` was null | The deposited record has a machine-readable MIT label. |
| DataCite DOI metadata | `rightsList[0].rights` was `MIT License`, SPDX identifier `mit` | Independent metadata export agrees on the record-level MIT label. |
| Depositor description | States research/educational/analytical use and identifies the CSV as derived from public AutoScout24 listings | A depositor statement does not independently establish control of the underlying marketplace listing rights. |

**Correction.** Earlier local notes and frozen audit JSON said that the record API had no machine-readable license. That statement is incorrect as checked today. The old audits are preserved as historical artifacts, including their hashes. Current manuscript, code comments, and release-readme wording now distinguish the blank landing-page display from the MIT API/DataCite metadata. Do not silently rewrite pretest audit files or treat metadata correction as a new experiment.

**Unresolved decision.** The MIT label describes the repository record but does not, by itself, prove the depositor could grant downstream rights to raw marketplace-derived listings or this project's row-level derivatives. Keep raw and row-level AutoScout24 material out of the shareable package until the human authors/institution complete a rights review. Aggregate-only tables, code, citation and source-directed acquisition remain the current conservative design. No contacting or republication was performed here.
