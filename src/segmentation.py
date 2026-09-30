import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from .config import (
    PROCESSED_DIR,
    MODEL_DIR,
    REPORT_DIR,
)


SEGMENT_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Quantity",
    "UniqueProducts",
    "CustomerAgeDays",
    "AvgOrderValue",
    "AvgItemsPerOrder",
]


def run_segmentation(customer):

    df = customer.copy()

    # ---------------------------------------------------------
    # 1. Prepare features
    # ---------------------------------------------------------

    X = df[SEGMENT_FEATURES].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    # ---------------------------------------------------------
    # 2. Reduce effect of extreme outliers
    #
    # Log transforms are especially important for:
    # Monetary, Frequency, Quantity and product counts.
    # ---------------------------------------------------------

    log_features = [
        "Frequency",
        "Monetary",
        "Quantity",
        "UniqueProducts",
        "CustomerAgeDays",
        "AvgOrderValue",
        "AvgItemsPerOrder",
    ]

    for col in log_features:
        X[col] = np.log1p(
            X[col].clip(lower=0)
        )

    # Clip each feature to its 1st–99th percentile.
    # This prevents a handful of extreme customers
    # from dominating the clustering.
    for col in X.columns:

        lower = X[col].quantile(0.01)
        upper = X[col].quantile(0.99)

        X[col] = X[col].clip(
            lower=lower,
            upper=upper
        )

    # ---------------------------------------------------------
    # 3. Standardize
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ---------------------------------------------------------
    # 4. Test different numbers of clusters
    # ---------------------------------------------------------

    evaluation = []

    best_k = None
    best_score = -1

    for k in range(2, 7):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(
            X_scaled
        )

        silhouette = silhouette_score(
            X_scaled,
            labels
        )

        cluster_sizes = pd.Series(
            labels
        ).value_counts()

        smallest_cluster_pct = (
            cluster_sizes.min()
            / len(labels)
        )

        evaluation.append({
            "k": k,
            "silhouette_score": silhouette,
            "smallest_cluster_pct": smallest_cluster_pct,
        })

        print(
            f"k={k} | "
            f"silhouette={silhouette:.4f} | "
            f"smallest cluster="
            f"{smallest_cluster_pct:.2%}"
        )

        # Don't select a solution where one cluster
        # contains less than 2% of customers.
        if (
            silhouette > best_score
            and smallest_cluster_pct >= 0.02
        ):
            best_score = silhouette
            best_k = k

    # Fallback if every solution has a tiny cluster.
    if best_k is None:

        best_k = max(
            evaluation,
            key=lambda x: x["silhouette_score"]
        )["k"]

        print(
            "\nWarning: every tested solution "
            "contained a small cluster."
        )

    print(
        f"\nSelected k={best_k}; "
        f"silhouette={best_score:.4f}"
    )

    # ---------------------------------------------------------
    # 5. Train final segmentation model
    # ---------------------------------------------------------

    final_model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=20
    )

    df["Segment"] = final_model.fit_predict(
        X_scaled
    )

    # ---------------------------------------------------------
    # 6. Save models
    # ---------------------------------------------------------

    joblib.dump(
        scaler,
        MODEL_DIR / "segment_scaler.joblib"
    )

    joblib.dump(
        final_model,
        MODEL_DIR / "segment_model.joblib"
    )

    # ---------------------------------------------------------
    # 7. Save evaluation
    # ---------------------------------------------------------

    evaluation_df = pd.DataFrame(
        evaluation
    )

    evaluation_df.to_csv(
        REPORT_DIR / "segment_evaluation.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 8. Save segmented customers
    # ---------------------------------------------------------

    df.to_parquet(
        PROCESSED_DIR
        / "segmented_customers.parquet",
        index=False
    )

    # ---------------------------------------------------------
    # 9. Print segment sizes
    # ---------------------------------------------------------

    print("\nSegment sizes:")

    print(
        df["Segment"]
        .value_counts()
        .sort_index()
    )

    return df