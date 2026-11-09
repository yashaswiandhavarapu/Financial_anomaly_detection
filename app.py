import streamlit as st
import pandas as pd
import numpy as np


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Financial Audit Intelligence",
    page_icon="📊",
    layout="wide"
)


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "audit_priority.csv"
    )

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    ).fillna(0)

    df["audit_priority_score"] = pd.to_numeric(
        df["audit_priority_score"],
        errors="coerce"
    ).fillna(0)

    df["risk_probability"] = pd.to_numeric(
        df["risk_probability"],
        errors="coerce"
    ).fillna(0)

    return df


df = load_data()


# ==========================================================
# HEADER
# ==========================================================

st.title("📊 Financial Audit Intelligence System")

st.markdown(
    """
    **Detect unusual expense patterns for audit prioritization
    and financial governance.**
    """
)

st.divider()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🔎 Audit Filters")

priority_options = [
    "All",
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW"
]

selected_priority = st.sidebar.selectbox(
    "Audit Priority",
    priority_options
)


if selected_priority != "All":

    filtered_df = df[
        df["audit_priority"] == selected_priority
    ].copy()

else:

    filtered_df = df.copy()


# ==========================================================
# KPI CARDS
# ==========================================================

st.subheader("Audit Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Transactions",
        f"{len(df):,}"
    )


with col2:

    critical = (
        df["audit_priority"]
        .eq("CRITICAL")
        .sum()
    )

    st.metric(
        "Critical",
        f"{critical:,}"
    )


with col3:

    high = (
        df["audit_priority"]
        .eq("HIGH")
        .sum()
    )

    st.metric(
        "High Priority",
        f"{high:,}"
    )


with col4:

    avg_score = (
        df["audit_priority_score"]
        .mean()
    )

    st.metric(
        "Average Risk Score",
        f"{avg_score:.2f}"
    )


# ==========================================================
# PRIORITY DISTRIBUTION
# ==========================================================

st.subheader("Audit Priority Distribution")

priority_counts = (
    df["audit_priority"]
    .value_counts()
    .reindex(
        [
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW"
        ],
        fill_value=0
    )
)

st.bar_chart(
    priority_counts
)


# ==========================================================
# AUDIT PRIORITY TABLE
# ==========================================================

st.subheader("🚨 Audit Priority Queue")

display_columns = [
    "audit_rank",
    "id",
    "client_id",
    "amount",
    "merchant_category",
    "risk_probability",
    "audit_priority_score",
    "audit_priority",
    "audit_reason",
    "recommended_action"
]

display_columns = [
    col
    for col in display_columns
    if col in filtered_df.columns
]

st.dataframe(
    filtered_df[display_columns],
    width="stretch",
    height=500
)


# ==========================================================
# TOP 10
# ==========================================================

st.subheader("🔥 Highest Priority Transactions")

top10 = (
    df
    .sort_values(
        "audit_priority_score",
        ascending=False
    )
    .head(10)
)

st.dataframe(
    top10[display_columns],
    width="stretch"
)


# ==========================================================
# MERCHANT ANALYSIS
# ==========================================================

st.subheader("🏪 High-Risk Merchant Categories")

merchant_risk = (
    df.groupby("merchant_category")
    .agg(
        transactions=("id", "count"),
        average_score=(
            "audit_priority_score",
            "mean"
        )
    )
    .sort_values(
        "average_score",
        ascending=False
    )
    .head(15)
)

st.bar_chart(
    merchant_risk["average_score"]
)


# ==========================================================
# AUDIT SUMMARY
# ==========================================================

st.subheader("📋 Audit Summary")

critical_count = (
    df["audit_priority"]
    .eq("CRITICAL")
    .sum()
)

high_count = (
    df["audit_priority"]
    .eq("HIGH")
    .sum()
)

st.info(
    f"""
    **Audit Recommendation**

    • {critical_count:,} transactions require immediate investigation.

    • {high_count:,} transactions should receive priority review.

    • Lower-priority transactions can remain under routine monitoring.

    The system is designed to **prioritize auditor attention**, not
    automatically declare a transaction fraudulent.
    """
)


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Financial Audit Intelligence System | "
    "Random Forest + Anomaly Detection | "
    "Audit Prioritization"
)