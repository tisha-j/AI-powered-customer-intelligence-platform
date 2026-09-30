import pandas as pd
import numpy as np
from .config import PROCESSED_DIR, REPORT_DIR
import json


SNAPSHOT_FREQUENCY = 30
PREDICTION_WINDOW = 90


def build_customer_features(transactions):

    df = transactions.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    min_date = df["InvoiceDate"].min().normalize()
    max_date = df["InvoiceDate"].max().normalize()

    snapshots = []

    # Start only after enough history exists
    snapshot_date = min_date + pd.Timedelta(days=180)

    while snapshot_date + pd.Timedelta(
        days=PREDICTION_WINDOW
    ) <= max_date:

        history = df[
            df["InvoiceDate"].dt.normalize() <= snapshot_date
        ]

        future = df[
            (df["InvoiceDate"].dt.normalize() > snapshot_date)
            & (
                df["InvoiceDate"].dt.normalize()
                <= snapshot_date + pd.Timedelta(
                    days=PREDICTION_WINDOW
                )
            )
        ]

        if history.empty:
            snapshot_date += pd.Timedelta(
                days=SNAPSHOT_FREQUENCY
            )
            continue

        customers = (
            history
            .groupby("Customer ID")
            .agg(
                LastPurchase=("InvoiceDate", "max"),
                FirstPurchase=("InvoiceDate", "min"),
                Frequency=("Invoice", "nunique"),
                Monetary=("Revenue", "sum"),
                Quantity=("Quantity", "sum"),
                UniqueProducts=("StockCode", "nunique"),
                Countries=("Country", "nunique"),
            )
            .reset_index()
        )

        customers["Recency"] = (
            snapshot_date
            - customers["LastPurchase"].dt.normalize()
        ).dt.days

        customers["CustomerAgeDays"] = (
            customers["LastPurchase"].dt.normalize()
            - customers["FirstPurchase"].dt.normalize()
        ).dt.days.clip(lower=0)

        customers["AvgOrderValue"] = (
            customers["Monetary"]
            / customers["Frequency"].clip(lower=1)
        )

        customers["AvgItemsPerOrder"] = (
            customers["Quantity"]
            / customers["Frequency"].clip(lower=1)
        )

        customers["AvgRevenuePerProduct"] = (
            customers["Monetary"]
            / customers["UniqueProducts"].clip(lower=1)
        )

        # Future behaviour
        future_customers = set(
            future["Customer ID"].unique()
        )

        customers["PurchasedNext90Days"] = (
            customers["Customer ID"]
            .isin(future_customers)
            .astype(int)
        )

        customers["SnapshotDate"] = snapshot_date

        for col in [
            "Frequency",
            "Monetary",
            "Quantity",
            "UniqueProducts",
            "AvgOrderValue",
        ]:
            customers[f"Log_{col}"] = np.log1p(
                customers[col]
            )

        customers = customers.drop(
            columns=[
                "LastPurchase",
                "FirstPurchase",
            ]
        )

        snapshots.append(customers)

        print(
            f"Snapshot {snapshot_date.date()} | "
            f"Customers: {len(customers):,} | "
            f"Purchase rate: "
            f"{customers['PurchasedNext90Days'].mean():.2%}"
        )

        snapshot_date += pd.Timedelta(
            days=SNAPSHOT_FREQUENCY
        )

    result = pd.concat(
        snapshots,
        ignore_index=True
    )

    result["Risk"] = (
        1 - result["PurchasedNext90Days"]
    )

    output = (
        PROCESSED_DIR
        / "customer_snapshots.parquet"
    )

    result.to_parquet(
        output,
        index=False
    )

    metadata = {
        "snapshot_frequency_days": SNAPSHOT_FREQUENCY,
        "prediction_window_days": PREDICTION_WINDOW,
        "target": "PurchasedNext90Days",
        "risk_definition": (
            "Risk=1 when customer does not "
            "purchase in the following 90 days."
        ),
        "source": "UCI Online Retail II",
    }

    (
        REPORT_DIR / "feature_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        f"\nTotal snapshots: "
        f"{result['SnapshotDate'].nunique()}"
    )

    print(
        f"Total training examples: "
        f"{len(result):,}"
    )

    return result, metadata