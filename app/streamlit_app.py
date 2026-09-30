import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_customer_scores.parquet"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "risk_model.joblib"
)

FEATURE_PATH = (
    BASE_DIR
    / "models"
    / "feature_columns.joblib"
)

SHAP_PATH = (
    BASE_DIR
    / "reports"
    / "shap_feature_importance.png"
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    return model, feature_columns


df = load_data()
model, feature_columns = load_model()


# =========================================================
# SEGMENT LABELS
# =========================================================

segment_names = {
    0: "Engaged / High-Value",
    1: "Dormant / Low-Value"
}

df["SegmentName"] = (
    df["Segment"]
    .map(segment_names)
    .fillna("Unknown")
)


# =========================================================
# HEADER
# =========================================================

st.title("AI-Powered Customer Intelligence Platform")

st.caption(
    "Customer segmentation • Purchase-risk prediction • "
    "Explainability • Revenue-at-risk prioritization"
)

st.divider()


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("Customer Filters")

selected_segments = st.sidebar.multiselect(
    "Customer Segment",
    options=sorted(
        df["SegmentName"].unique()
    ),
    default=sorted(
        df["SegmentName"].unique()
    )
)

selected_priorities = st.sidebar.multiselect(
    "Priority Band",
    options=[
        "High",
        "Medium",
        "Low"
    ],
    default=[
        "High",
        "Medium",
        "Low"
    ]
)

risk_threshold = st.sidebar.slider(
    "Minimum Risk Probability",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)


filtered = df[
    df["SegmentName"].isin(
        selected_segments
    )
    &
    df["PriorityBand"].isin(
        selected_priorities
    )
    &
    (
        df["RiskProbability"]
        >= risk_threshold
    )
].copy()


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

st.header("Executive Overview")

total_customers = len(filtered)

high_priority = (
    filtered["PriorityBand"]
    == "High"
).sum()

at_risk = (
    filtered["RiskProbability"]
    >= 0.50
).sum()

avg_risk = (
    filtered["RiskProbability"]
    .mean()
)

revenue_at_risk = (
    filtered["ExpectedRevenueAtRisk"]
    .sum()
)


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Customers",
    f"{total_customers:,}"
)

col2.metric(
    "High Priority",
    f"{high_priority:,}"
)

col3.metric(
    "At Risk",
    f"{at_risk:,}"
)

col4.metric(
    "Average Risk",
    f"{avg_risk:.1%}"
)

col5.metric(
    "Revenue at Risk",
    f"£{revenue_at_risk:,.0f}"
)


# =========================================================
# CUSTOMER SEGMENTATION
# =========================================================

st.divider()

st.header("Customer Segmentation")


seg_counts = (
    filtered["SegmentName"]
    .value_counts()
    .reset_index()
)

seg_counts.columns = [
    "Segment",
    "Customers"
]


fig = px.bar(
    seg_counts,
    x="Segment",
    y="Customers",
    text="Customers"
)

fig.update_traces(
    textposition="outside"
)

fig.update_layout(
    height=400,
    xaxis_title=None,
    yaxis_title="Customers"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# SEGMENT PROFILE
# =========================================================

st.subheader("Segment Profile")


segment_profile = (
    filtered
    .groupby("SegmentName")
    .agg(
        Customers=("Customer ID", "count"),
        Median_Recency=("Recency", "median"),
        Median_Frequency=("Frequency", "median"),
        Median_Monetary=("Monetary", "median"),
        Median_Products=("UniqueProducts", "median"),
        Median_Order_Value=("AvgOrderValue", "median")
    )
    .reset_index()
)


for _, row in segment_profile.iterrows():

    segment = row["SegmentName"]

    if segment == "Engaged / High-Value":
        st.markdown("### 🟢 Engaged / High-Value")
    else:
        st.markdown("### 🟠 Dormant / Low-Value")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Customers",
        f"{row['Customers']:,.0f}"
    )

    c2.metric(
        "Median Recency",
        f"{row['Median_Recency']:.0f} days"
    )

    c3.metric(
        "Median Orders",
        f"{row['Median_Frequency']:.0f}"
    )

    c4.metric(
        "Median Value",
        f"£{row['Median_Monetary']:,.0f}"
    )

    c5.metric(
        "Median Order",
        f"£{row['Median_Order_Value']:,.0f}"
    )


st.info(
    """
    **Engaged / High-Value:** more recent and frequent purchasing
    behavior with substantially higher historical customer value.

    **Dormant / Low-Value:** longer purchase gaps, lower purchase
    frequency and lower historical customer value.
    """
)


# =========================================================
# PURCHASE RISK
# =========================================================

st.divider()

st.header("Purchase Risk Overview")


risk_col1, risk_col2 = st.columns(2)


# ---------------------------------------------------------
# RISK DISTRIBUTION
# ---------------------------------------------------------

with risk_col1:

    st.subheader("Risk Distribution")

    risk_bins = pd.cut(
        filtered["RiskProbability"],
        bins=np.linspace(
            0,
            1,
            11
        ),
        include_lowest=True
    )

    risk_counts = (
        risk_bins
        .value_counts()
        .sort_index()
        .reset_index()
    )

    risk_counts.columns = [
        "Risk Range",
        "Customers"
    ]

    risk_counts["Risk Range"] = [
        f"{interval.left:.1f}–{interval.right:.1f}"
        for interval
        in risk_counts["Risk Range"]
    ]

    fig = px.bar(
        risk_counts,
        x="Risk Range",
        y="Customers",
        text="Customers"
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        height=400,
        xaxis_title="Risk Probability",
        yaxis_title="Customers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ---------------------------------------------------------
# PRIORITY DISTRIBUTION
# ---------------------------------------------------------

with risk_col2:

    st.subheader("Priority Distribution")

    priority_counts = (
        filtered["PriorityBand"]
        .value_counts()
        .reindex(
            [
                "High",
                "Medium",
                "Low"
            ],
            fill_value=0
        )
        .reset_index()
    )

    priority_counts.columns = [
        "Priority",
        "Customers"
    ]

    fig = px.bar(
        priority_counts,
        x="Priority",
        y="Customers",
        text="Customers"
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        height=400,
        xaxis_title=None,
        yaxis_title="Customers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# EXPECTED REVENUE AT RISK
# =========================================================

st.divider()

st.header("Expected Revenue at Risk")


revenue_segment = (
    filtered
    .groupby("SegmentName")[
        "ExpectedRevenueAtRisk"
    ]
    .sum()
    .reset_index()
)

revenue_segment.columns = [
    "Segment",
    "RevenueAtRisk"
]


fig = px.bar(
    revenue_segment,
    x="Segment",
    y="RevenueAtRisk",
    text="RevenueAtRisk"
)

fig.update_traces(
    texttemplate="£%{text:,.0f}",
    textposition="outside"
)

fig.update_layout(
    height=450,
    xaxis_title=None,
    yaxis_title="Expected Revenue at Risk (£)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# PRIORITY CUSTOMER QUEUE
# =========================================================

st.divider()

st.header("Priority Customer Queue")

st.caption(
    "Customers ranked by estimated historical revenue currently "
    "exposed to purchase risk."
)


priority = (
    filtered
    .sort_values(
        "ExpectedRevenueAtRisk",
        ascending=False
    )
    .head(50)
    .copy()
)


# Use Streamlit dataframe only for the main queue.
# This is the one place where a table is useful.

display_priority = priority[
    [
        "Customer ID",
        "SegmentName",
        "Recency",
        "Frequency",
        "Monetary",
        "RiskProbability",
        "ExpectedRevenueAtRisk",
        "PriorityBand"
    ]
].copy()


display_priority.columns = [
    "Customer",
    "Segment",
    "Recency (days)",
    "Orders",
    "Historical Value",
    "Risk",
    "Revenue at Risk",
    "Priority"
]


st.dataframe(
    display_priority,
    use_container_width=True,
    hide_index=True,
    height=500
)


# =========================================================
# CUSTOMER 360
# =========================================================

st.divider()

st.header("Customer 360")


# Highest revenue-at-risk customer becomes default.

if len(filtered) > 0:

    default_customer = int(
        filtered
        .sort_values(
            "ExpectedRevenueAtRisk",
            ascending=False
        )
        .iloc[0]["Customer ID"]
    )

else:

    default_customer = int(
        df
        .sort_values(
            "ExpectedRevenueAtRisk",
            ascending=False
        )
        .iloc[0]["Customer ID"]
    )


customer_ids = sorted(
    df["Customer ID"]
    .dropna()
    .astype(int)
    .unique()
)


default_index = customer_ids.index(
    default_customer
)


selected_customer = st.selectbox(
    "Select Customer",
    customer_ids,
    index=default_index
)


customer = (
    df[
        df["Customer ID"]
        == selected_customer
    ]
    .iloc[0]
)


# =========================================================
# CUSTOMER KPI
# =========================================================

c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Segment",
    customer["SegmentName"]
)

c2.metric(
    "Purchase Risk",
    f"{customer['RiskProbability']:.1%}"
)

c3.metric(
    "Historical Value",
    f"£{customer['Monetary']:,.0f}"
)

c4.metric(
    "Estimated Value at Risk",
    f"£{customer['ExpectedRevenueAtRisk']:,.0f}"
)


# =========================================================
# CUSTOMER BEHAVIOR
# =========================================================

profile_col1, profile_col2 = st.columns(2)


with profile_col1:

    st.subheader("Behavior Profile")

    st.markdown(
        f"**Recency:** "
        f"{customer['Recency']:.0f} days"
    )

    st.markdown(
        f"**Purchase Frequency:** "
        f"{customer['Frequency']:.0f} orders"
    )

    st.markdown(
        f"**Historical Monetary Value:** "
        f"£{customer['Monetary']:,.2f}"
    )

    st.markdown(
        f"**Unique Products:** "
        f"{customer['UniqueProducts']:.0f}"
    )

    st.markdown(
        f"**Average Order Value:** "
        f"£{customer['AvgOrderValue']:,.2f}"
    )

    st.markdown(
        f"**Customer Age:** "
        f"{customer['CustomerAgeDays']:.0f} days"
    )



    # =========================================================
    # CUSTOMER SHAP
    # =========================================================

    with profile_col2:

        st.subheader("Top Risk Drivers")

        try:

            X_customer = (
                pd.DataFrame(
                    [
                        customer[
                            feature_columns
                        ].values
                    ],
                    columns=feature_columns
                )
                .replace(
                    [
                        np.inf,
                        -np.inf
                    ],
                    np.nan
                )
                .fillna(0)
            )

            explainer = shap.TreeExplainer(
                model
            )

            shap_values = (
                explainer
                .shap_values(
                    X_customer
                )
            )

            if isinstance(
                shap_values,
                list
            ):

                shap_values = (
                    shap_values[1]
                )

            shap_values = np.asarray(
                shap_values
            )

            if shap_values.ndim == 3:

                shap_values = (
                    shap_values[:, :, 1]
                )

            # -------------------------------------------------
            # IMPORTANT:
            # The model predicts PurchaseProbability.
            # Positive SHAP = higher purchase probability.
            # RiskProbability = 1 - PurchaseProbability.
            #
            # Therefore, invert SHAP direction when explaining
            # customer risk.
            # -------------------------------------------------

            risk_shap_values = -shap_values[0]

            contributions = pd.DataFrame({
                "Feature": feature_columns,
                "Impact": risk_shap_values
            })

            contributions[
                "AbsoluteImpact"
            ] = (
                contributions[
                    "Impact"
                ].abs()
            )

            contributions = (
                contributions
                .sort_values(
                    "AbsoluteImpact",
                    ascending=False
                )
                .head(5)
            )

            for _, row in contributions.iterrows():

                feature = row["Feature"]
                impact = row["Impact"]

                if impact > 0:

                    st.error(
                        f"🔴 **{feature}**  \n"
                        f"Increases predicted risk  "
                        f"({impact:+.3f})"
                    )

                else:

                    st.success(
                        f"🟢 **{feature}**  \n"
                        f"Reduces predicted risk  "
                        f"({impact:+.3f})"
                    )

        except Exception as e:

            st.warning(
                "Customer-level SHAP explanation "
                "could not be generated."
            )

            st.caption(
                str(e)
            )



# =========================================================
# BUSINESS ACTION
# =========================================================

st.divider()

st.subheader("Recommended Business Action")


risk = customer["RiskProbability"]

segment = customer["Segment"]


if (
    risk >= 0.75
    and segment == 0
):

    st.error(
        """
        **HIGH-VALUE RETENTION**

        This customer combines substantial historical value
        with high predicted purchase risk.

        Recommended action: targeted retention outreach,
        personalized offers or re-engagement.
        """
    )


elif (
    risk >= 0.75
    and segment == 1
):

    st.warning(
        """
        **REACTIVATION**

        This customer shows high predicted purchase risk
        and a dormant purchasing profile.

        Recommended action: low-cost reactivation campaign.
        """
    )


elif risk >= 0.50:

    st.warning(
        """
        **PROACTIVE ENGAGEMENT**

        Customer shows elevated predicted purchase risk.

        Recommended action: personalized engagement and
        continued behavioral monitoring.
        """
    )


else:

    st.success(
        """
        **MAINTAIN ENGAGEMENT**

        Customer currently shows relatively lower predicted
        purchase risk.

        Recommended action: maintain regular engagement.
        """
    )


# =========================================================
# GLOBAL MODEL EXPLAINABILITY
# =========================================================

st.divider()

st.header("Global Model Explainability — Purchase Probability")

st.caption(
    "Overall feature importance showing which features "
    "most influence predicted purchase probability."
)


if SHAP_PATH.exists():

    st.image(
        str(SHAP_PATH),
        use_container_width=True
    )

else:

    st.warning(
        "SHAP feature-importance report was not found."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Risk represents predicted probability that a customer "
    "will not purchase within the next 90-day prediction window. "
    "It is a model-based behavioral risk signal, not confirmed churn."
)