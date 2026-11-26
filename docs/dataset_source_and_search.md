# Dataset source, quality issues, and search evidence

## 1. Source and permission record

| Item | Verified project record |
|---|---|
| Dataset | `humanoid_robot_tasks_200k_messy.csv` |
| Origin | Synthetic humanoid-robot task data supplied with the CDS4003 project materials. |
| Real-world status | Not real Agility Robotics data; the supplied injector explicitly states this. |
| Clean-data generator | Not supplied in the source materials. |
| Quality-issue injector | `data/make_dataset_messy.py`, supplied with the project materials. |
| External licence | None supplied. The team must confirm licence/permission with the instructor or provider before public redistribution. |
| Synthetic-data approval | Required by the brief at Milestone 2. Add the written instructor approval; do not claim it has been obtained without evidence. |

The full CSV is included to make this technical project reproducible. Its inclusion in a **public** repository remains conditional on the provider's permission.

## 2. Intentional quality issues in the supplied injector

The following behaviours are visible in `make_dataset_messy.py`; they are documented here rather than presented as naturally occurring real-world defects.

| Quality issue | Injector behaviour | Technical treatment |
|---|---|---|
| Missing values | Approximately 2% random missingness in each unprotected column. | Deterministic audit first; train-only mode/median imputation inside model pipelines. |
| Inconsistent category labels | Approximately 2.5% of non-missing values in `shift`, `warehouse_zone`, and `task_type` receive case, whitespace, spelling, or separator variants. | Trim, lowercase, convert spaces/hyphens to underscores, and correct `shippping` to `shipping`. |
| Impossible numeric values | Approximately 0.15% of candidates in five numeric fields receive negative or extreme values such as 999 kg, 150% battery, or −100°C. | Convert values outside inclusive domain ranges to missing; do not treat legitimate statistical extremes as automatic errors. |
| Mixed binary formats | Approximately 0.1% of candidates in binary fields receive `yes`, `No`, `TRUE`, or `unknown`. | Normalise recognised labels to 0/1; set unknown labels to missing. Unknown supervised targets are excluded from fitting/evaluation. |
| Fully blank rows | Inserts approximately 0.1% blank records. | Remove 200 blank rows during training-data structural cleaning; preserve and score rows during batch inference. |
| Duplicate records | Exact duplicate blank records occur in the raw file. No non-empty exact duplicates were found. | Training code removes exact duplicates and repeated IDs if present. Confirm with the instructor whether blank-record duplicates satisfy the assessment requirement. |

## 3. Measured quality audit

The reproducible full-data run produced:

| Measure | Result |
|---|---:|
| Raw rows | 200,200 |
| Raw columns | 30 |
| Fully blank rows removed | 200 |
| Rows after structural cleaning | 200,000 |
| Raw missing cells | 113,967 |
| Domain-invalid values converted to missing | 1,467 |
| Missing cells after structural/domain cleaning | 109,630 |
| Valid labeled modelling rows | 199,950 |
| Overall failure rate among labeled rows | 52.39% |

After blank rows are removed, no non-empty exact duplicates or duplicate `task_id` values are found. The pipeline still contains duplicate-row and duplicate-ID handling so the rule is explicit and reusable.

Detailed column-level results:

- `outputs/tables/data_quality_audit.csv`
- `outputs/tables/domain_invalid_values.csv`
- `outputs/tables/cleaning_summary.csv`
- `outputs/figures/01_missingness_before_cleaning.png`

## 4. Student-owned dataset search evidence

The brief requires evidence of a real-dataset search before synthetic data is approved. No genuine alternative-dataset search results were supplied for this repository, so none are invented here. Each student team must complete the table from its own browser history, downloads, notes, and instructor discussions.

Suggested starting points named in the brief include Kaggle, UCI Machine Learning Repository, Google Dataset Search, and government/open-data portals such as Bayanat. A search entry should record what was actually checked, not what sounds plausible.

| Date | Source/portal | Exact search terms | Dataset/result checked | Rows / columns | Suitable target? | Licence checked | Accept / reject and why | Evidence link or screenshot filename |
|---|---|---|---|---|---|---|---|---|
| [STUDENT ENTRY] |  |  |  |  |  |  |  |  |
| [STUDENT ENTRY] |  |  |  |  |  |  |  |  |
| [STUDENT ENTRY] |  |  |  |  |  |  |  |  |

Before Milestone 2 submission, verify:

- [ ] The business problem was approved before dataset selection.
- [ ] Search terms, dates, sources, and rejected alternatives are recorded truthfully.
- [ ] Licence/permission evidence is saved and linked.
- [ ] The accepted dataset meets the row and column requirements.
- [ ] The target matches the approved business question.
- [ ] The dataset has not been prohibited as previous coursework.
- [ ] Written synthetic-data approval is attached.
- [ ] The instructor has confirmed whether the supplied duplicate-blank-record pattern meets the duplicate requirement.

## 5. Limitations and bias risks

- **Construct validity:** synthetic fields and target relationships may not correspond to real robot physics or warehouse operations.
- **Representativeness:** no real sampling frame, fleet, site, or time period is available, so population coverage cannot be established.
- **Temporal validity:** no timestamp is supplied; temporal holdout, seasonality, and drift cannot be tested.
- **Label meaning:** `task_failure` is generated synthetically. Its relationship to real safety incidents or maintenance outcomes is unknown.
- **Fairness and worker impact:** shift and zone associations may be artefacts of the generator. Do not use them to evaluate workers or sites.
- **Leakage risk:** fields derived from the target or observed after execution are present in the raw file and must remain excluded.
- **Financial validity:** AED error costs are illustrative assumptions, not accounting observations.
- **Operational validity:** discrimination metrics do not demonstrate safe robot control. Any real deployment needs a separate safety case, real-data validation, human oversight, and monitoring.
