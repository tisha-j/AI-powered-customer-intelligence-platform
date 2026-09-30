# AI-Powered Customer Intelligence Platform

An end-to-end machine learning platform for customer purchase-risk prediction, customer segmentation, explainability, and revenue-at-risk prioritization using the UCI Online Retail II dataset.

## Overview

This project transforms transactional e-commerce data into customer-level intelligence that can support retention and customer-prioritization decisions.

The workflow covers:

Data Cleaning → Feature Engineering → Customer Segmentation → Risk Modelling → Model Explainability → Customer Prioritization → Streamlit Dashboard

The dataset contains approximately 1.07 million transactions covering nearly 5,900 customers.

## What It Does

1. Cleans and prepares transactional e-commerce data
2. Builds customer-level RFM and behavioral features
3. Creates a behavior-derived purchase-risk target
4. Segments customers using unsupervised machine learning
5. Trains and evaluates supervised machine-learning models
6. Uses SHAP for model explainability
7. Calculates customer priority and revenue-at-risk measures
8. Presents the results through an interactive Streamlit dashboard

## Machine Learning

### Customer Risk Prediction

The project predicts whether a customer is at risk of not making a future purchase.

The risk target is derived from customer purchasing behavior rather than a ground-truth churn label. This makes the model a behavioural risk-proxy model rather than a conventional labelled churn dataset.

The final XGBoost model achieved:

* ROC-AUC: 0.816
* PR-AUC: 0.746

Temporal validation was used to separate historical customer information from future purchasing behaviour.

### Customer Segmentation

K-Means clustering is used to identify groups of customers based on behavioural and purchasing characteristics.

* Silhouette score: 0.892

The resulting segments are used alongside risk predictions to support customer prioritization.

### Explainability

SHAP is used to identify the features contributing to customer risk predictions, helping translate model outputs into interpretable customer-level insights.

## Business Outputs

The platform goes beyond a model prediction by producing:

* Customer risk probability
* Customer priority score
* Priority bands
* Revenue at risk
* Customer segments
* Risk-driver explanations

These outputs are designed to help identify customers requiring attention and prioritize retention opportunities based on both customer risk and potential business impact.

## Project Structure

```text
AI-powered-customer-intelligence-platform/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── notebooks/
├── reports/
│
├── src/
│   ├── churn_model.py
│   ├── data_pipeline.py
│   ├── explainability.py
│   ├── feature_engineering.py
│   ├── segmentation.py
│   └── config.py
│
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
└── run_pipeline.py
```

## Setup

Place the UCI Online Retail II Excel dataset inside:

```text
data/raw/
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python run_pipeline.py
```

Launch the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

Run tests:

```bash
pytest
```

## Dataset

UCI Online Retail II dataset containing transactional e-commerce records.

The project uses the transaction history to construct customer-level behavioural features and evaluate future purchasing risk.

## Limitations

The risk target is a behaviour-derived proxy rather than a ground-truth churn label.

The dataset also does not contain explicit information about marketing campaigns, customer demographics, competitor activity, or external business factors.

Therefore, model outputs should be interpreted as behavioural risk signals rather than definitive predictions of customer churn.

## Technology

Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Streamlit, Matplotlib, Seaborn

## Author

Tisha S Jain

Data Scientist | Predictive Analytics | Machine Learning | Customer Intelligence
