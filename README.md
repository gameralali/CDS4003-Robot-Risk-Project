# Humanoid Robot Task Failure-Risk Prediction

CDS4003 business data science project using Python, CRISP-DM, scikit-learn, Jupyter, and Streamlit. The technical workflow predicts whether a warehouse robot task is likely to fail **before the task starts**, so that a supervisor can review higher-risk tasks.

> **Status:** the code, trained model, tests, notebook, and local Streamlit app are complete. The public GitHub repository and Streamlit Community Cloud URL must be created by the student team. Do not describe the app as live until that deployment is finished.

## Student completion fields

Replace every item in this section before submission.

- **Team members and roles:** [ADD NAMES, STUDENT IDs, AND VERIFIABLE ROLES]
- **GitHub repository:** [ADD PUBLIC REPOSITORY URL]
- **Live Streamlit app:** [ADD STREAMLIT COMMUNITY CLOUD URL]
- **Instructor approval for synthetic data:** [ADD DATE AND LINK TO WRITTEN EVIDENCE]
- **Dataset licence/permission:** [CONFIRM WITH THE INSTRUCTOR OR DATA PROVIDER]
- **AI-use declaration:** review the declaration below and make it match the team's actual use.

## 1. Business problem and intended use

Unplanned robot-task failures can interrupt warehouse workflows and require supervisor review. This project frames a binary classification question: **using information available before execution, what is the probability that a task will fail?**

The app offers three decision thresholds:

- **Balanced (F1):** a practical demonstration trade-off between false positives and false negatives.
- **Standard (0.50):** the conventional probability cutoff.
- **High recall (cost-based):** an aggressive review policy selected using illustrative error costs.

The output is decision support, not a robot-control command. All costs are scenario assumptions, not observed financial data. The data and model are synthetic and are not validated for physical robot operation.

## 2. Dataset source, licence, and suitability

- **File:** `data/humanoid_robot_tasks_200k_messy.csv`
- **Origin:** course-project synthetic data supplied with the assessment materials. The original clean-data generator was not supplied; `data/make_dataset_messy.py` is the supplied quality-issue injector.
- **Real-world status:** synthetic; it does **not** represent real Agility Robotics operations or measurements.
- **Licence:** no external licence was supplied. Confirm permission with the instructor/data provider before public redistribution. Do not invent a licence.
- **Approval:** the project brief requires prior written instructor approval for synthetic data at Milestone 2. Add that evidence before submission.
- **Size:** 200,200 raw rows and 30 columns; 200 fully blank rows are removed during structural cleaning, leaving 200,000 records. There are 199,950 rows with a valid binary target for modelling.
- **Target:** `task_failure` (1 = failure, 0 = success).
- **Suitability:** mixes categorical, binary, and numeric pre-task variables and contains missing values, inconsistent labels, mixed binary formats, impossible values, and blank records. Exact duplicate blank records occur in the raw file; no non-empty exact duplicates were found. Confirm with the instructor that this satisfies the brief's duplicate-record requirement.

See `docs/data_dictionary.md` and `docs/dataset_source_and_search.md` for the schema, intentional quality issues, limitations, and the search-evidence table the students must complete honestly.

## 3. Technical approach

The executed notebook and source code follow the seven required stages:

1. **Business understanding:** pre-task failure classification and threshold-based supervisor review.
2. **Data collection:** supplied synthetic data and quality-issue generation code.
3. **Data understanding and cleaning:** quality audits, category normalisation, binary-label conversion, domain validation, blank/duplicate handling, and documented missing-data treatment.
4. **EDA and feature engineering:** outcome distributions, group failure rates, correlations, numeric distributions, and seven engineered features.
5. **Model development:** dummy prior baseline, logistic regression, and random forest.
6. **Tuning and evaluation:** three-fold `RandomizedSearchCV` on a stratified training-only sample, validation-based model/threshold selection, and one untouched test evaluation.
7. **Deployment:** a Streamlit app for single predictions, batch CSV scoring, and model evidence.

### Leakage controls

The following fields are excluded from model inputs:

- `task_success` and `failure_probability_simulated`: direct target leakage.
- `completion_time_sec` and `operator_intervention`: post-task information unavailable at pre-task decision time.
- `task_id`: identifier, not a generalisable risk factor.

Missing-value imputation, scaling, and one-hot encoding occur inside model pipelines. Numeric medians, categorical modes, scaling parameters, and encoding categories are learned from training folds only. The split is stratified: 70% training, 15% validation, and 15% untouched test data. The selected model is refit on training plus validation data after selection, then evaluated once on the test set.

## 4. Results summary

### Validation model comparison at each cost-selected threshold

| Model | ROC AUC | PR AUC | Recall | F1 | Illustrative error cost (AED) |
|---|---:|---:|---:|---:|---:|
| Random forest | 0.767 | 0.782 | 0.9997 | 0.689 | 569,220 |
| Logistic regression | 0.774 | 0.790 | 0.9992 | 0.690 | 569,340 |
| Dummy prior | 0.500 | 0.524 | 1.0000 | 0.688 | 571,160 |

Random forest was selected using the lowest validation cost, but its advantage over logistic regression is negligible. Students should critically discuss whether the simpler logistic model would be preferable; do not claim a large performance advantage.

### Untouched test results for the selected random forest

| Operating point | Threshold | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC | Cost (AED) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 0.50 | 0.699 | 0.714 | 0.712 | 0.713 | 0.770 | 0.786 | 2,443,140 |
| Balanced F1 | 0.35 | 0.662 | 0.625 | 0.891 | 0.734 | 0.770 | 0.786 | 1,193,680 |
| High recall | 0.11 | 0.527 | 0.525 | 0.9998 | 0.689 | 0.770 | 0.786 | 569,420 |

The high-recall policy produces 14,198 false positives and 3 false negatives on 29,993 test records. A flag-all baseline would cost AED 571,200 under the same assumptions, so the high-recall model saves only AED 1,780 (about 0.31%). This small advantage is a key business limitation, not evidence of a large operational saving.

The leading pipeline-level permutation-importance features are `thermal_load_index`, `battery_deficit`, `joint_load_pct`, `battery_level_pct`, and `load_distance_index`. Importance is associative and does not prove causation.

Full results are in `outputs/tables/`, `outputs/figures/`, `models/model_metadata.json`, and `outputs/Project_Results.xlsx`.

## 5. Repository structure

```text
.
├── app/
│   └── streamlit_app.py          # single, batch, and evidence views
├── data/
│   ├── humanoid_robot_tasks_200k_messy.csv
│   ├── sample_humanoid_robot_tasks.csv
│   └── make_dataset_messy.py
├── docs/
│   ├── data_dictionary.md
│   ├── dataset_source_and_search.md
│   ├── deployment.md
│   ├── student_writing_guide.md
│   └── verification.md
├── models/
│   ├── best_model.joblib
│   └── model_metadata.json
├── notebooks/
│   └── 01_end_to_end_project.ipynb
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── run_summary.json
│   └── Project_Results.xlsx
├── src/
│   ├── config.py
│   ├── build_workbook.py
│   ├── data_processing.py
│   ├── modeling.py
│   ├── reporting.py
│   ├── run_project.py
│   └── scoring.py
├── tests/
│   └── test_pipeline.py
├── .streamlit/config.toml
├── Dockerfile
├── index.html
├── requirements.txt
└── README.md
```

The full CSV is about 31 MB and the model about 18 MB, both below GitHub's 100 MB single-file limit. If the data provider does not permit public redistribution, remove the full CSV before publishing and provide an approved access method instead; keep the sample only if permission covers it.

## 6. Run locally

Use Python 3.11 or 3.12. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Launch the app

```bash
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit. The app loads the included trained artifacts; no retraining is needed.

### Run tests

```bash
python -m unittest discover -s tests -v
```

The tests cover deterministic cleaning, invalid values, feature engineering, exact feature contracts, leakage exclusions, missing-column errors, and row-preserving scoring of blank and duplicate records.

### Reproduce the analysis

```bash
python -m src.run_project
```

This rebuilds the audits, figures, tuning histories, model comparison, test metrics, and model artifacts. Runtime depends on hardware. The configured experiment uses a fixed random seed (`4003`), a 30,000-row training-only tuning sample, three-fold searches, and full training-data refits.

To rebuild the formula-driven results workbook from saved outputs:

```bash
python -m src.build_workbook
```

Open a locally rebuilt workbook in Excel or LibreOffice to recalculate formula caches. The delivered workbook has already been recalculated and validated.

To execute the notebook in a Jupyter-compatible environment, open `notebooks/01_end_to_end_project.ipynb` and choose **Run All**. It is already supplied with successful outputs from a top-to-bottom execution.

## 7. Deploy publicly

Detailed steps are in `docs/deployment.md`. In brief: create the GitHub repository, commit the project with meaningful student commits, sign in to Streamlit Community Cloud using GitHub, select the same repository, and set the main file to `app/streamlit_app.py`. Then add the real public URL above and test the deployed app.

## 8. AI-use declaration — review and retain accurately

The course brief permits AI tools only for coding support and requires every use to be declared. The following declaration records the assistance used for this technical project; the team must amend it if their actual use differs.

| Tool | Purpose | Scope in this repository |
|---|---|---|
| Kimi (Moonshot AI) | Coding and debugging support | Python cleaning/feature-engineering/modeling code, Streamlit implementation, tests, notebook code cells, debugging, and repository documentation scaffolding |

No AI-generated group report or individual written assessment is included. **All assessed report prose, critical reflection, and individual contribution statements must be written independently by the students in their own words.** `docs/student_writing_guide.md` contains only questions, evidence pointers, and checklists; it is not submission-ready prose.

## 9. Limitations and ethical use

- Synthetic relationships may make the prediction problem easier or different from real operations; test metrics do not establish physical-robot effectiveness.
- The supplied data has no date/time field, so temporal validation and concept-drift testing are not demonstrated.
- The AED 500 false-negative and AED 40 false-positive costs are illustrative. Real costs, review capacity, safety constraints, and worker impacts must be elicited before deployment.
- Permutation importance and rule-based input checks are not causal explanations.
- A high-recall threshold can overwhelm reviewers with false positives; monitor workload and override outcomes.
- Validate on real, representative, lawfully obtained data with appropriate safety and privacy review before any operational use.
- Keep humans accountable for final decisions; do not connect this demonstration directly to robot actuation.
