# Student writing and evidence guide — not a report

**Do not submit this document as the group report or individual assessment.** The brief requires all assessed written text to be the students' own work. This guide provides prompts, checklists, and pointers to technical evidence only. Write your own explanations, reasoning, reflection, and contribution statement.

## 1. Before writing

- Read the project brief and rubric again.
- Run the notebook and app yourself so you can explain the work without relying on generated prose.
- Agree as a group on the approved business question and intended decision-maker.
- Collect genuine Milestone 1/2 approvals, dataset-search evidence, licence information, GitHub links, and commit links.
- Decide which operating point you recommend and be prepared to justify it rather than simply quoting a metric.
- Keep a personal log of what you actually did, changed, tested, and learned.

## 2. Group report evidence map

The group report should summarise all seven stages and refer to the repository for detail. Put the real GitHub and live Streamlit links on the first page. Include the dataset source/licence and AI-tool declaration.

For each stage below, respond to the prompts in your own words and cite the relevant evidence files.

### Stage 1 — Business problem formulation

Prompts:

- Who makes which decision, at what moment, using which pre-task information?
- What exactly counts as a failure, and why is this a binary classification problem?
- What measurable objective did the team set beyond “build a model”?
- What are the operational consequences of a false negative and a false positive?
- Why is a probability useful, and what human action should follow a review flag?

Evidence: `README.md` sections 1 and 9; `src/config.py` cost assumptions; app **Decision limits**.

### Stage 2 — Data collection and justification

Prompts:

- Which real-data sources and queries did the team actually check, and why were alternatives rejected?
- Why was synthetic data necessary, and where is the instructor's written approval?
- What licence or permission applies? If none was supplied, what did the team do to resolve this?
- How do 200,200 raw rows, 30 columns, and the target meet the brief?
- Does the duplicate-blank-record pattern satisfy the instructor's duplicate requirement? Record the answer honestly.
- What population can the data legitimately represent?

Evidence: `docs/dataset_source_and_search.md`; `docs/data_dictionary.md`; `data/make_dataset_messy.py`; `outputs/tables/data_quality_audit.csv`.

### Stage 3 — Data understanding and cleaning

Prompts:

- Which quality issues were intentionally injected, and which were measured by the audit?
- Why were impossible domain values converted to missing instead of deleted or winsorised?
- Why are medians/modes fitted inside pipelines rather than on the full dataset?
- Why are unknown target labels excluded from supervised training?
- Why are batch blank/duplicate rows preserved while training structural rows are removed?
- What happened to missing-cell counts after blank-row removal, binary normalisation, and domain validation?
- Which exclusions prevent direct and post-task leakage?

Evidence: `src/data_processing.py`; `src/modeling.py`; `outputs/tables/cleaning_summary.csv`; `outputs/tables/domain_invalid_values.csv`; `outputs/figures/01_missingness_before_cleaning.png`.

### Stage 4 — EDA and feature engineering

Prompts:

- Which two or three patterns are most relevant to the business question, and what exactly do the charts show?
- Which groups have higher/lower observed failure rates, and how much uncertainty or missingness remains?
- Why might a correlation or group difference not be causal?
- What does each engineered feature represent operationally?
- Which engineered features duplicate information already present in base features, and what risk does that create for interpretation?
- How do train-only imputation and encoding affect reproducibility?

Evidence: `outputs/figures/02_…` through `06_…`; `outputs/tables/failure_rate_by_*.csv`; `src/data_processing.py`; notebook Stage 4.

### Stage 5 — Model selection and development

Prompts:

- Why are a dummy prior, logistic regression, and random forest appropriate comparators?
- What does each model assume or learn, and what are their interpretability trade-offs?
- Why use a stratified 70/15/15 split, and what role does each partition play?
- Why was random forest selected when logistic regression had slightly higher ROC AUC and PR AUC?
- Is a validation-cost difference of AED 120 meaningful? Would the simpler model be preferable under a different selection rule?

Evidence: `src/modeling.py`; `outputs/tables/model_comparison_validation.csv`; tuning-history CSVs; notebook Stage 5.

### Stage 6 — Hyperparameter tuning and evaluation

Prompts:

- What search space and iteration budget were used, and why was tuning performed on a training-only sample?
- Why is the test set not used to choose the model or threshold?
- What do precision, recall, F1, ROC AUC, and PR AUC each communicate for this problem?
- Why can accuracy be misleading when operating thresholds are aggressive?
- Compare the confusion matrices at 0.50, 0.35, and 0.11 in terms of missed failures and reviewer workload.
- Under the illustrative costs, what is the flag-all cost, and how small is the high-recall model's advantage?
- Which operating point would the team pilot, and what real information is needed before that decision?
- What do the leading permutation-importance features suggest, and why do they not establish causality?

Evidence: `models/model_metadata.json`; `outputs/tables/final_test_metrics.csv`; `outputs/tables/final_test_predictions.csv`; `outputs/figures/07_…` through `10_…`; `outputs/Project_Results.xlsx`.

### Stage 7 — Deployment and business use

Prompts:

- How do single scoring, batch scoring, and model evidence support different users?
- How does the app recreate the exact training feature contract?
- How are blank rows, duplicate rows, missing columns, and invalid values handled at inference?
- What is the difference between probability, threshold decision, rule-based input check, and causal explanation?
- Which monitoring measures would be needed after a real pilot: drift, calibration, review capacity, overrides, realised failures, or costs?
- What remains incomplete until the public app and repository links are added?

Evidence: `app/streamlit_app.py`; `src/scoring.py`; `tests/test_pipeline.py`; `docs/deployment.md`; `docs/verification.md`.

## 3. Individual Part A — 2,000–3,000 words

Write independently and in your own words. Cover all seven stages, but allocate more space to your own reasoning than to restating group text.

For each stage, consider this sequence:

1. What decision did **you** make or evaluate?
2. Which alternatives did you consider?
3. What evidence supports the decision?
4. What result followed?
5. What limitation or uncertainty remains?
6. What did you personally learn?

Self-check:

- [ ] My justification is not a paraphrase of the README or group report.
- [ ] I explain why choices fit the business decision, not only how the code works.
- [ ] I interpret both strong and weak results honestly.
- [ ] I discuss the small RF-versus-logistic validation difference.
- [ ] I discuss the high-recall policy's false-positive burden and small cost advantage.
- [ ] I include specific evidence rather than unsupported claims.

## 4. Individual Part B — 500–800 words of critical reflection

Choose the challenges and alternatives you personally understand and can evidence. Possible questions to investigate—not statements to copy—include:

- How did synthetic target generation affect the apparent ease and usefulness of the task?
- What could not be learned without timestamps or real operational outcomes?
- Was optimising AED cost alone appropriate when the cost-minimising threshold flagged almost every task?
- How would reviewer capacity change the threshold choice?
- Would logistic regression be a better final choice because the validation-cost difference is negligible?
- How might class weighting, calibration, a different cost ratio, or a capacity-constrained threshold change the result?
- What are the trade-offs of using a 30,000-row tuning sample while refitting on all training/development rows?
- How did pipeline-level permutation importance avoid the transformed-feature mismatch, and what interpretation limits remain?
- Why did `pandas.NA` require special handling in categorical object arrays?
- What additional real-data, safety, fairness, or monitoring work would be required before operational deployment?

Do not claim you personally solved an issue unless you can explain it and provide your own evidence.

## 5. Individual Part C — at least 300 words

Describe your actual contribution and make it verifiable. Include your GitHub username.

| Contribution area | Your specific action | Evidence to add |
|---|---|---|
| Business/data | [What you researched, decided, or documented] | [Approval, search note, commit, or file URL] |
| Cleaning/EDA | [Code you wrote, reviewed, or tested] | [Commit/diff/notebook cell URL] |
| Modeling/evaluation | [Experiment, review, or interpretation you performed] | [Commit, result table, or discussion URL] |
| Deployment/testing | [App change, deployment step, or test you executed] | [Commit, deployed URL, screenshot, or test log] |
| Collaboration | [Review, integration, or team coordination you performed] | [Pull request, review, meeting record] |

End your own statement by checking that your claimed role matches the repository history. General claims such as “I helped with modeling” are weaker than a precise, honest description with evidence.

## 6. Final academic-integrity check

- [ ] Every sentence of assessed prose is my/our own work.
- [ ] AI use is limited to the permitted coding support and is declared accurately in README and report.
- [ ] No guide text, generated report paragraph, or generated reflection has been pasted into an assessed document.
- [ ] Every dataset source, licence, approval, metric, and contribution claim is supported by real evidence.
- [ ] The group report contains the actual GitHub and public Streamlit URLs on its first page.
- [ ] The individual submission contains Part A, Part B, and Part C within the required lengths.
