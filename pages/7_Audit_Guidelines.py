import streamlit as st

st.set_page_config(
    page_title="Audit Guidelines",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Audit Interpretation Guidelines")

st.markdown(
    """
    These guidelines explain how auditors should interpret the
    risk scores and use the system during financial review.
    """
)

st.divider()

# ==========================================================
# PRIORITY LEVELS
# ==========================================================

st.subheader("🚦 Audit Priority Levels")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("CRITICAL", "80–100")

with col2:
    st.metric("HIGH", "60–79")

with col3:
    st.metric("MEDIUM", "30–59")

with col4:
    st.metric("LOW", "0–29")


# ==========================================================
# CRITICAL
# ==========================================================

st.subheader("🔴 CRITICAL — Immediate Investigation")

st.write(
    """
    Transactions with very high audit priority should receive
    immediate auditor attention.

    Possible indicators include:

    • Significant deviation from historical spending
    • High transaction amount relative to income
    • High credit-limit utilization
    • Unusual merchant activity
    • Rapid successive transactions
    • Online or unusual transaction behavior
    """
)

# ==========================================================
# HIGH
# ==========================================================

st.subheader("🟠 HIGH — Priority Review")

st.write(
    """
    High-priority transactions should be reviewed after critical
    cases. Auditors should examine the transaction details,
    customer history, merchant information, and model explanation.
    """
)

# ==========================================================
# MEDIUM
# ==========================================================

st.subheader("🟡 MEDIUM — Secondary Review")

st.write(
    """
    Medium-priority transactions show some unusual characteristics
    but do not necessarily require immediate investigation.
    They can be reviewed as part of routine audit procedures.
    """
)

# ==========================================================
# LOW
# ==========================================================

st.subheader("🟢 LOW — Routine Monitoring")

st.write(
    """
    Low-priority transactions are relatively consistent with the
    observed spending patterns and can remain under routine
    monitoring.
    """
)


# ==========================================================
# IMPORTANT MODEL LIMITATION
# ==========================================================

st.subheader("⚠️ Important Model Limitation")

st.warning(
    """
    The system does NOT automatically determine that a transaction
    is fraudulent.

    The Random Forest and anomaly detection models are decision-
    support tools designed to prioritize auditor attention.

    Final decisions should be made by a qualified human auditor
    after reviewing the available financial evidence.
    """
)


# ==========================================================
# RECOMMENDED AUDIT WORKFLOW
# ==========================================================

st.subheader("🔎 Recommended Audit Workflow")

st.markdown(
    """
    **1. Identify**

    Start with Critical and High priority transactions.

    **2. Investigate**

    Review transaction amount, date, merchant, customer history,
    and transaction behavior.

    **3. Explain**

    Use the Explainable AI page to understand the factors that
    influenced the model prediction.

    **4. Verify**

    Compare the transaction against supporting financial records.

    **5. Record**

    Document the auditor's decision and investigation notes.

    **6. Escalate**

    Escalate confirmed issues according to the organization's
    financial governance procedures.
    """
)


# ==========================================================
# MODEL INTERPRETATION
# ==========================================================

st.subheader("🤖 Model Interpretation")

st.info(
    """
    The model produces a risk score based on historical transaction
    patterns and engineered financial/behavioral features.

    A higher score indicates greater similarity to patterns that
    the system considers unusual.

    It should therefore be interpreted as an audit-prioritization
    signal rather than proof of misconduct.
    """
)


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Financial Audit Intelligence System | "
    "Audit Interpretation Guidelines"
)