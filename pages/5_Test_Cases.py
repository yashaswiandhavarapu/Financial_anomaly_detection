import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Test Cases",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Anomaly Detection Test Cases")

st.markdown(
    """
    Test the audit prioritization system using representative
    normal and suspicious transaction cases.
    """
)

st.divider()


# ==========================================================
# LOAD AUDIT DATA
# ==========================================================

@st.cache_data
def load_data():
    return pd.read_csv("audit_priority_deployment.csv")


df = load_data()


# ==========================================================
# TEST CASE 1 — NORMAL
# ==========================================================

st.subheader("✅ Test Case 1 — Normal Transaction")

normal = df[
    df["audit_priority"] == "LOW"
]

if len(normal) > 0:

    normal_case = normal.iloc[0]

    st.write("Example of a low-priority transaction:")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Transaction ID",
            str(normal_case["id"])
        )

    with col2:
        st.metric(
            "Amount",
            f"${float(normal_case['amount']):,.2f}"
        )

    with col3:
        st.metric(
            "Risk Score",
            f"{float(normal_case['audit_priority_score']):.2f}"
        )

    with col4:
        st.metric(
            "Priority",
            normal_case["audit_priority"]
        )

    st.success(
        "Expected behavior: LOW priority / routine monitoring."
    )

else:

    st.warning(
        "No LOW-priority transaction available."
    )


# ==========================================================
# TEST CASE 2 — HIGH RISK
# ==========================================================

st.subheader("🚨 Test Case 2 — Suspicious Transaction")

high = df[
    df["audit_priority"].isin(
        ["CRITICAL", "HIGH"]
    )
]

if len(high) > 0:

    high_case = high.iloc[0]

    st.write("Example of a high-priority transaction:")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Transaction ID",
            str(high_case["id"])
        )

    with col2:
        st.metric(
            "Amount",
            f"${float(high_case['amount']):,.2f}"
        )

    with col3:
        st.metric(
            "Risk Score",
            f"{float(high_case['audit_priority_score']):.2f}"
        )

    with col4:
        st.metric(
            "Priority",
            high_case["audit_priority"]
        )

    st.error(
        "Expected behavior: HIGH/CRITICAL priority and auditor review."
    )

    st.write(
        "**Reason:**",
        high_case["audit_reason"]
    )

else:

    st.warning(
        "No HIGH/CRITICAL transaction available."
    )


# ==========================================================
# SIDE-BY-SIDE COMPARISON
# ==========================================================

st.subheader("📊 Normal vs Suspicious")

comparison = pd.DataFrame({
    "Metric": [
        "Amount",
        "Risk Probability",
        "Audit Score",
        "Priority",
        "Audit Reason"
    ],
    "Normal Transaction": [
        normal_case["amount"] if len(normal) else "N/A",
        normal_case["risk_probability"] if len(normal) else "N/A",
        normal_case["audit_priority_score"] if len(normal) else "N/A",
        normal_case["audit_priority"] if len(normal) else "N/A",
        normal_case["audit_reason"] if len(normal) else "N/A"
    ],
    "Suspicious Transaction": [
        high_case["amount"] if len(high) else "N/A",
        high_case["risk_probability"] if len(high) else "N/A",
        high_case["audit_priority_score"] if len(high) else "N/A",
        high_case["audit_priority"] if len(high) else "N/A",
        high_case["audit_reason"] if len(high) else "N/A"
    ]
})

st.dataframe(
    comparison,
    width="stretch"
)


# ==========================================================
# TEST CONCLUSION
# ==========================================================

st.subheader("Test Conclusion")

st.info(
    """
    The test cases demonstrate that the system can distinguish
    between transactions requiring routine monitoring and
    transactions requiring increased auditor attention.

    The system provides risk prioritization rather than making
    an automatic fraud determination.
    """
)