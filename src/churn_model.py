import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from .config import (
    MODEL_DIR,
    REPORT_DIR,
    PROCESSED_DIR,
)


EXCLUDE = {
    "Customer ID",
    "PurchasedNext90Days",
    "Risk",
    "SnapshotDate",
}


def train_models(df):

    df = df.copy()

    df["SnapshotDate"] = pd.to_datetime(
        df["SnapshotDate"]
    )

    feature_columns = [
        column
        for column in df.select_dtypes(
            include=np.number
        ).columns
        if column not in EXCLUDE
    ]

    # ---------------------------------------------------------
    # TIME-BASED SPLIT
    # ---------------------------------------------------------

    dates = sorted(
        df["SnapshotDate"].unique()
    )

    split_index = int(
        len(dates) * 0.80
    )

    train_dates = dates[:split_index]
    validation_dates = dates[split_index:]

    train_df = df[
        df["SnapshotDate"].isin(
            train_dates
        )
    ]

    validation_df = df[
        df["SnapshotDate"].isin(
            validation_dates
        )
    ]

    X_train = (
        train_df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    y_train = train_df[
        "PurchasedNext90Days"
    ]

    X_test = (
        validation_df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    y_test = validation_df[
        "PurchasedNext90Days"
    ]

    print("\n=== TEMPORAL VALIDATION ===")

    print(
        f"Training period: "
        f"{train_dates[0]} -> "
        f"{train_dates[-1]}"
    )

    print(
        f"Validation period: "
        f"{validation_dates[0]} -> "
        f"{validation_dates[-1]}"
    )

    print(
        f"Training examples: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_df):,}"
    )

    print(
        f"Training purchase rate: "
        f"{y_train.mean():.2%}"
    )

    print(
        f"Validation purchase rate: "
        f"{y_test.mean():.2%}"
    )

    # ---------------------------------------------------------
    # MODELS
    # ---------------------------------------------------------

    models = {

        "logistic_regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                )
            )
        ]),

        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )
    }

    results = []
    fitted_models = {}

    # ---------------------------------------------------------
    # TRAIN + EVALUATE
    # ---------------------------------------------------------

    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        probability = (
            model.predict_proba(X_test)[:, 1]
        )

        prediction = (
            model.predict(X_test)
        )

        roc_auc = roc_auc_score(
            y_test,
            probability
        )

        pr_auc = average_precision_score(
            y_test,
            probability
        )

        results.append({
            "model": name,
            "roc_auc": roc_auc,
            "average_precision": pr_auc,
        })

        fitted_models[name] = model

        print(
            f"ROC-AUC: {roc_auc:.4f}"
        )

        print(
            f"PR-AUC: {pr_auc:.4f}"
        )

        print(
            classification_report(
                y_test,
                prediction,
                zero_division=0,
            )
        )

    # ---------------------------------------------------------
    # SELECT BEST MODEL
    # ---------------------------------------------------------

    result_df = (
        pd.DataFrame(results)
        .sort_values(
            "roc_auc",
            ascending=False
        )
    )

    result_df.to_csv(
        REPORT_DIR
        / "model_comparison.csv",
        index=False
    )

    best_model_name = (
        result_df.iloc[0]["model"]
    )

    best_model = fitted_models[
        best_model_name
    ]

    print(
        f"\nSelected model: "
        f"{best_model_name}"
    )

    # ---------------------------------------------------------
    # RETRAIN ON ALL HISTORICAL SNAPSHOTS
    # ---------------------------------------------------------

    X_all = (
        df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    y_all = df[
        "PurchasedNext90Days"
    ]

    best_model.fit(
        X_all,
        y_all
    )

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------

    joblib.dump(
        best_model,
        MODEL_DIR
        / "risk_model.joblib"
    )

    joblib.dump(
        feature_columns,
        MODEL_DIR
        / "feature_columns.joblib"
    )

        

    # ---------------------------------------------------------
    # BUSINESS PRIORITIZATION
    # ---------------------------------------------------------
    latest_date = (
        df["SnapshotDate"].max()
    )

    latest = df[
        df["SnapshotDate"]
        == latest_date
    ].copy()

    X_latest = (
        latest[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    latest["PurchaseProbability"] = (
        best_model
        .predict_proba(X_latest)[:, 1]
    )

    latest["RiskProbability"] = (
        1
        - latest["PurchaseProbability"]
    )

    # Expected revenue currently exposed to churn risk
    latest["ExpectedRevenueAtRisk"] = (
        latest["Monetary"]
        * latest["RiskProbability"]
    )

    # Create business priority bands
    q75 = latest["ExpectedRevenueAtRisk"].quantile(0.75)
    q40 = latest["ExpectedRevenueAtRisk"].quantile(0.40)

    latest["PriorityBand"] = np.select(
        [
            latest["ExpectedRevenueAtRisk"] >= q75,
            latest["ExpectedRevenueAtRisk"] >= q40
        ],
        [
            "High",
            "Medium"
        ],
        default="Low"
    )

    print("\n=== BUSINESS PRIORITIZATION ===")

    print(
        latest[
            [
                "Customer ID",
                "Monetary",
                "RiskProbability",
                "ExpectedRevenueAtRisk",
                "PriorityBand"
            ]
        ]
        .sort_values(
            "ExpectedRevenueAtRisk",
            ascending=False
        )
        .head(10)
    )

    latest.to_parquet(
        PROCESSED_DIR
        / "final_customer_scores.parquet",
        index=False
    )

    return {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": feature_columns,
        "results": result_df,
    }