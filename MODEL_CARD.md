# Model Card: Synthetic Activity Risk Demo

## Purpose

This project demonstrates a basic classification workflow for ranking fictional activities by the chance of receiving a synthetic high-risk label in the next 30 days. It is an educational example for practicing data preparation, model comparison, and batch scoring.

## Intended use

- Learn how a tabular classification pipeline can be structured.
- Practice explaining ROC-AUC, precision, recall, and F1.
- Explore how activity scores can be summarized by project.

## Out of scope

- Predicting actual project delays, cost overruns, or procurement outcomes.
- Making decisions about real contractors, vendors, budgets, or schedules.
- Using this demo's metrics or score bands to guide live project work.

## Data and target

The script generates all data locally from random distributions. The target label is sampled from an illustrative hidden rule using generated fields. It is not sourced from observed project outcomes and is not a validated definition of risk. No client data is included.

## Models and evaluation

The example compares logistic regression and random forest. It uses a random stratified holdout and a 0.5 classification threshold for the displayed precision, recall, and F1. The model with the higher ROC-AUC is selected for the example batch scores.

These choices keep the code understandable, but they are not a recommended evaluation design for longitudinal project records. Real activity histories can repeat across snapshots, so a random split may leak similar records across train and test sets. A real use case should define an as-of date, measure outcomes in a future window, and evaluate on later periods or held-out projects.

## Limitations

- Synthetic data and labels do not establish real-world predictive value.
- Probabilities have not been calibrated.
- Threshold and risk-band cutoffs are illustrative, not cost-optimized.
- ROC-AUC alone is not enough to choose an operational model.
- The model is not causal: a high score does not explain which intervention will resolve the issue.
- Data drift, fairness, stakeholder review, and operational monitoring are not implemented.

## Before any real use

Agree on a measurable outcome with domain experts; use approved data with point-in-time feature definitions; validate on future periods and separate projects; review calibration and error costs; document the operating threshold; provide understandable reasons for alerts; and monitor data freshness, model performance, and user feedback. A qualified project owner must review any alert before action.
