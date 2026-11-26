# Verification record

This file records technical checks performed on the delivered project. It is not a claim that the public Streamlit deployment has occurred.

## Automated analysis checks

- The complete pipeline runs independently with `python -m src.run_project`.
- The notebook `notebooks/01_end_to_end_project.ipynb` was executed top to bottom with **8 of 8 code cells executed and 0 error outputs**.
- The final model artifact and metadata load successfully.
- The metadata feature list exactly matches the 31 model inputs in `src/config.py`.
- Direct leakage, identifier, and post-task fields are excluded by contract and regression test.

## Unit and regression tests

Command:

```bash
python -m unittest discover -s tests -v
```

Eight tests cover:

1. structural cleaning of blank and duplicate training rows;
2. category, binary-label, and impossible-value normalisation;
3. engineered-feature formulas;
4. blank/duplicate row preservation and ordering during scoring;
5. the exact model-feature contract;
6. actionable missing-column validation;
7. actionable empty-input validation;
8. artifact loading and leakage/post-task exclusions.

## Browser checks performed locally

The app was launched from the repository root with:

```bash
streamlit run app/streamlit_app.py
```

Verified in a Chromium browser:

- initial desktop render and responsive navigation;
- default single-task submission returning a failure probability of 58.3% and a review recommendation;
- Balanced, Standard, and High recall operating points displaying thresholds 0.35, 0.50, and 0.11 respectively;
- 1,000-row sample CSV upload and scoring;
- sample batch summary showing 1,000 tasks, 755 review flags at the balanced threshold, and a 52.4% mean failure probability;
- scored CSV download containing 1,000 rows, 32 columns, probabilities in [0, 1], and binary review flags;
- a four-row edge case containing an ordinary record, another record, an exact duplicate, and an all-missing record;
- all four edge-case records returned and downloaded with probabilities, proving that inference does not drop blank or duplicate rows;
- model-evidence metrics including ROC AUC and PR AUC, model configuration, exclusions, and permutation importance;
- no browser console errors or warnings after the final evidence-chart revision;
- mobile layout at 390 × 844 pixels with `documentWidth = viewportWidth = 390`, so no horizontal page overflow.

Screenshots are stored as:

- `outputs/qa_single_task_initial.png`
- `outputs/qa_model_evidence.png`
- `outputs/qa_mobile_single_task.png`

## Workbook checks

The results workbook is generated from the saved pipeline outputs. LibreOffice recalculation succeeded; static verification passed with **724 formulas, 0 missing formula caches, 0 formula errors, and 0 compatibility warnings**. Independent checks matched the saved test metrics, confirmed the AED 571,200 flag-all cost and AED 1,780 high-recall saving, and verified all 8 charts contain series. See the workbook itself for source labels and assumptions.

## Portability and package checks

- Required source, data, model, notebook, documentation, and workbook files are present.
- The notebook retains eight executed code cells and zero error outputs.
- Environment-specific absolute paths were removed from notebook text output.
- The final ZIP passes integrity testing and contains no single file over GitHub's 100 MB limit.

## Checks still requiring student accounts

- GitHub repository creation, access, and genuine student commit history.
- Streamlit Community Cloud deployment from that repository.
- Public URL access in a signed-out browser.
- Deployed-app single prediction, batch upload/download, and mobile smoke test.
- Instructor approval and dataset redistribution/licence confirmation.
