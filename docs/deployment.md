# GitHub and Streamlit Community Cloud deployment

Public deployment requires the student team's GitHub and Streamlit accounts. The repository is deployment-ready, but no public deployment has been performed by the coding assistant.

Official reference: [Streamlit Community Cloud deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## 1. Pre-publication checks

- [ ] Replace the student fields in `README.md`.
- [ ] Add written instructor approval for synthetic data.
- [ ] Confirm whether the full CSV and sample may be redistributed publicly; no licence was supplied.
- [ ] Review the AI-use declaration so it matches actual use.
- [ ] Ensure all group-report and individual-assessment prose is student-authored.
- [ ] Run `python -m unittest discover -s tests -v` successfully.
- [ ] Run the app locally and test single prediction, batch scoring, and downloads.
- [ ] Remove secrets, personal local paths, and unrelated files.

## 2. Create the repository with genuine student history

The delivered archive intentionally contains no fabricated Git history. Each student should commit their own real review, changes, and documentation with meaningful messages. Do not manufacture authorship or divide one unchanged dump into artificial commits merely to simulate participation.

From the project root:

```bash
git init
git branch -M main
git add .
git commit -m "Add reproducible robot risk analysis pipeline"
```

Create an empty repository on GitHub, then use the commands shown by GitHub. They will have this form:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Examples of genuine follow-up contributions include:

- a student adding dataset-search evidence and licence documentation;
- a student reviewing the leakage exclusions and adding a test;
- a student improving input validation or accessibility and recording browser evidence;
- a student adding the verified deployment URL and troubleshooting notes;
- pull-request reviews that discuss a real change and its outcome.

Make the repository public, or add the instructor as a collaborator if private. The app itself must also be publicly viewable to satisfy the brief.

## 3. Deploy from the same GitHub repository

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) and connect the GitHub account that owns or administers the repository.
2. In the workspace, choose **Create app**, then indicate that an app already exists.
3. Select the repository and branch (normally `main`).
4. Set the main file path to:

   ```text
   app/streamlit_app.py
   ```

5. Use Python 3.12 if the deployment settings request a version.
6. Deploy and inspect the build logs. The root `requirements.txt` and `.streamlit/config.toml` are already in the expected locations.
7. After launch, open the app's public `.streamlit.app` URL in a private/incognito window to verify viewer access.

No secrets are required. Do not add credentials to the repository or Streamlit secrets for this project.

## 4. Required deployed-app smoke test

Record the date, tester, browser, and evidence for each item.

- [ ] Home page loads without an exception or endless spinner.
- [ ] **Single task:** submit default values; a probability and recommendation appear.
- [ ] Change battery, joint load, temperature, or another input; submit again and confirm that the new input is scored.
- [ ] Switch among Balanced, Standard, and High recall; the displayed threshold changes.
- [ ] **Batch scoring:** upload `data/sample_humanoid_robot_tasks.csv`; 1,000 tasks are scored.
- [ ] Download the scored CSV; it contains `failure_probability` and `review_required` and preserves 1,000 rows.
- [ ] Upload a file with a missing required column; a clear error is shown rather than a traceback.
- [ ] **Model evidence:** all three operating points, ROC AUC, PR AUC, confusion-related costs, importance chart, and excluded fields render.
- [ ] Check a narrow/mobile viewport for overflow and readable controls.
- [ ] Browser console has no application errors.

## 5. Final links and submission

Add the exact public app URL to:

- the **Live Streamlit app** field near the top of `README.md`;
- the first page of the student-authored group report, together with the GitHub URL.

Commit that README update with a meaningful message. Re-test after any later commit because Community Cloud redeploys repository changes.

## Optional Docker run

Docker is not required for the assessment's Community Cloud route. For a separate local container check:

```bash
docker build -t robot-risk-console .
docker run --rm -p 8501:8501 robot-risk-console
```

Then open `http://localhost:8501`. Do not describe this optional container route as the required public Streamlit deployment.
