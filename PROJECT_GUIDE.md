# EPC Project Risk Scoring Demo

An end-to-end machine-learning practice project that uses fictional activity data to identify work items that may become high risk in the next 30 days. It compares logistic regression and random forest, evaluates their results, scores activities, and summarizes risk by project.

> **Data notice:** All records and labels are synthetic. They are not McDermott, client, or real project data. Metrics are for learning and do not show real-world model performance.

## Run it

Requires Python 3.10 or later.

### Windows PowerShell

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python epc_risk_demo.py
~~~

### macOS or Linux

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python epc_risk_demo.py
~~~

## Pipeline overview

1. Generate fictional activity records covering materials, procurement, schedule, and fabrication.
2. Validate required columns, identifiers, labels, and percentage ranges.
3. Prepare numeric and category features, then train logistic regression and random forest classifiers.
4. Compare ROC-AUC, precision, recall, and F1 on a held-out sample.
5. Score activities and create a project-level risk summary.

## Input fields

Each row is one fictional activity. Features include discipline, material readiness, late procurement, schedule variance in days, fabrication progress, and planned progress. The target, high_risk_next_30d, is a synthetic label sampled from an illustrative rule; it is not based on observed project outcomes.

## Generated outputs

The script writes files to artifacts/:

- synthetic_activities.csv — generated input records.
- model_comparison.csv — evaluation metrics for each model.
- activity_risk_scores.csv — activity rows with a predicted score and risk band.
- project_risk_summary.csv — activity counts and average scores by project.

## Interpreting results

Precision asks, “Of the activities flagged, how many were labeled risky?” Recall asks, “Of all activities labeled risky, how many were flagged?” ROC-AUC measures how well the model ranks risky examples above non-risky examples across thresholds. A high score is a prompt to investigate, not proof that a delay will happen.

The random train/test split and 0.5 classification threshold keep this demo simple. For actual project history, define a point-in-time prediction question and evaluate on later periods or held-out projects to reduce information leakage. Choose an operating threshold based on the cost of missed issues and the team's review capacity.

See MODEL_CARD.md for intended use and limitations.
