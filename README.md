# AI-Powered Customer Intelligence Platform

End-to-end ML project using the UCI Online Retail II dataset.

## What it does
1. Cleans transactional e-commerce data
2. Builds customer-level RFM and behavioral features
3. Creates a behavior-derived customer risk target
4. Performs customer segmentation using unsupervised ML
5. Trains and compares supervised ML models
6. Uses SHAP for model explainability
7. Produces customer-priority scores
8. Provides a Streamlit dashboard

## Setup

Put the UCI Online Retail II Excel file inside `data/raw/`.

Then:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run app/streamlit_app.py
```

Run tests with:

```bash
pytest
```

The risk label is a proxy derived from transaction recency, not a ground-truth churn label.
