import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Transaction Investigation",
    page_icon="🔍",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv("audit_priority_deployment.csv")

    df["amount"] = pd.to_numeric(
        df["amount"], errors="coerce"
    ).fillna(0)

    df["audit_priority_score"] = pd.to_numeric(
        df["audit_priority_score"], errors="coerce"
    ).fillna(0)

    df["risk_probability"] = pd.to_numeric(
        df["risk_probability"], errors="coerce"
    ).fillna(0)

    return df


df = load_data()

st.title("🔍 Transaction Investigation")

st.markdown(
    "Investigate individual transactions and understand why "
    "they were prioritized for audit."
)

st.divider()


# ==========================================================
# SELECT TRANSACTION
# ==========================================================

transaction_ids = df["id"].astype(str).tolist()

selected_id = st.selectbox(
    "Select Transaction ID",
    transaction_ids
)

transaction = df[
    df["id"].astype(str) == selected_id
].iloc[0]


# ==========================================================
# TRANSACTION SUMMARY
# ==========================================================

st.subheader("Transaction Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Audit Score",
        f"{transaction['audit_priority_score']:.2f}"
    )

with col2:
    st.metric(
        "Risk Probability",
        f"{transaction['risk_probability']:.2%}"
    )

with col3:
    st.metric(
        "Amount",
        f"${transaction['amount']:,.2f}"
    )

with col4:
    st.metric(
        "Priority",
        transaction["audit_priority"]
    )


# ==========================================================
# TRANSACTION DETAILS
# ==========================================================

st.subheader("Transaction Details")

details = {
    "Transaction ID": transaction.get("id", "N/A"),
    "Date": transaction.get("date", "N/A"),
    "Client ID": transaction.get("client_id", "N/A"),
    "Card ID": transaction.get("card_id", "N/A"),
    "Merchant ID": transaction.get("merchant_id", "N/A"),
    "Merchant Category": transaction.get(
        "merchant_category", "N/A"
    ),
    "Merchant City": transaction.get(
        "merchant_city", "N/A"
    ),
    "Merchant State": transaction.get(
        "merchant_state", "N/A"
    ),
    "Transaction Type": transaction.get(
        "use_chip", "Not available"
    ),
}

details_df = pd.DataFrame(
    details.items(),
    columns=["Field", "Value"]
)

st.table(details_df)


# ==========================================================
# WHY WAS IT FLAGGED?
# ==========================================================

st.subheader("⚠️ Why Was This Transaction Flagged?")

reason = transaction.get(
    "audit_reason",
    "Unusual transaction pattern"
)

st.warning(reason)


# ==========================================================
# RECOMMENDED ACTION
# ==========================================================

st.subheader("📋 Recommended Audit Action")

st.info(
    transaction["recommended_action"]
)


# ==========================================================
# SUPPORTING RISK FACTORS
# ==========================================================

st.subheader("Risk Factors")

risk_factors = {}

if "amount_deviation" in transaction.index:
    risk_factors["Amount Deviation"] = (
        transaction["amount_deviation"]
    )

if "historical_amount_deviation" in transaction.index:
    risk_factors["Historical Amount Deviation"] = (
        transaction["historical_amount_deviation"]
    )

if "amount_to_income" in transaction.index:
    risk_factors["Amount / Income"] = (
        transaction["amount_to_income"]
    )

if "amount_to_credit_limit" in transaction.index:
    risk_factors["Amount / Credit Limit"] = (
        transaction["amount_to_credit_limit"]
    )

if "client_merchant_count" in transaction.index:
    risk_factors["Client-Merchant Frequency"] = (
        transaction["client_merchant_count"]
    )

if risk_factors:

    factors_df = pd.DataFrame(
        list(risk_factors.items()),
        columns=["Feature", "Value"]
    )

    st.dataframe(
        factors_df,
        width="stretch"
    )


# ==========================================================
# AUDITOR DECISION
# ==========================================================

st.subheader("👤 Auditor Review")

decision = st.radio(
    "Audit Decision",
    [
        "Needs Investigation",
        "Reviewed - No Issue",
        "Potential Issue Confirmed"
    ]
)

notes = st.text_area(
    "Auditor Notes",
    placeholder="Enter investigation notes..."
)

if st.button("Save Review"):

    st.success(
        f"Review recorded: {decision}"
    )