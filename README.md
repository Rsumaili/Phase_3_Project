# SyriaTel Customer Churn Prediction

## Overview
This project predicts customer churn for SyriaTel to reduce revenue loss by identifying at-risk customers for targeted retention. A tuned Random Forest model achieves 65% recall, highlighting key churn drivers like customer service calls and daytime charges.

## Business Problem
- **Stakeholder**: SyriaTel Customer Retention Team.
- **Goal**: Predict churn to prioritize retention efforts, minimizing revenue loss.
- **Approach**: Binary classification (churned/retained) using machine learning.
- **Objective**: Maximize recall to identify most churners for cost-effective interventions.

## Dataset
- **Source**: [SyriaTel Customer Churn Dataset](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset).
- **Details**: 3,333 records, 21 features (e.g., call minutes, charges, service calls), `churn` target (~14.5% churn rate).
- **Preprocessing**:
  - Dropped: `phone number` (irrelevant), `state` (high cardinality).
  - Encoding categorical variables, dropping of high cardinality and low importance columns in the data set: `day_charge_per_minute`, `service_calls_intensity`.
  - Encoded: `international plan`, `voice mail plan` (one-hot).
  - Scaled: `StandardScaler` (train-only fit).
  - Balanced: SMOTE for training data.

## Methodology
- **Data Preparation**:
  - 80-20 train-test split (stratified).
  - Scaled features, applied SMOTE to address class imbalance.
  - Engineered features for cost and complaint insights.
- **Modeling**:
  - **Baseline**: Logistic Regression (recall: 0.63, AUC-ROC: 0.82).
  - **Final**: Random Forest with `GridSearchCV` (recall: 0.65, AUC-ROC: 0.83).
  - **Tuning**: Optimized `n_estimators=200`, `max_depth=10`, `class_weight='balanced'`.
  - **Rationale**: Random Forest captures non-linear patterns, outperforms baseline.
- **Evaluation**:
  - **Primary Metric**: Recall (churned class).
  - **Secondary Metric**: AUC-ROC.
  - **Why**: Recall prioritizes catching churners; AUC-ROC balances performance.
- **Key Findings**:
  - Top features: `customer service calls`, `total day charge`, `international plan_yes`.
  - Insights: Dissatisfaction (calls), price sensitivity (charges), plan issues drive churn.

## Recommendations
- **Retention**:
  - Offer support for customers with >4 service calls.
  - Provide discounts for high `total day charge` (>40).
  - Review `international plan` pricing/service.
- **Deployment**:
  - Integrate model into CRM for monthly churn flagging.
  - Retrain quarterly to adapt to trends.
- **Communication**:
  - Use feature importance plot to show churn drivers.
  - Highlight 65% churn detection for stakeholder impact.

## Limitations
- Moderate precision (0.35) risks false positives.
- Model may not generalize if customer behavior shifts.
- Dropping `state` may miss minor regional trends.
