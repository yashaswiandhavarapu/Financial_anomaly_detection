import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Explainable AI",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Explainable AI")
st.markdown(
    """
    Understand which transaction characteristics influenced
    the Random Forest risk prediction.
    """
)

st.divider()


# ==========================================================
# LOAD DATA AND MODEL
# ==========================================================

@st.cache_data
def load_data():
    return pd.read_csv("audit_priority_deployment.csv")


@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")


df = load_data()
model = load_model()


# ==========================================================
# SELECT TRANSACTION
# ==========================================================

st.subheader("Select a Transaction")

transaction_ids = df["id"].astype(str).tolist()

selected_id = st.selectbox(
    "Transaction ID",
    transaction_ids
)

selected = df[
    df["id"].astype(str) == selected_id
].iloc[0]


# ==========================================================
# TRANSACTION SUMMARY
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Amount",
        f"${float(selected['amount']):,.2f}"
    )

with col2:
    st.metric(
        "Risk Probability",
        f"{float(selected['risk_probability']):.2%}"
    )

with col3:
    st.metric(
        "Audit Score",
        f"{float(selected['audit_priority_score']):.2f}"
    )

with col4:
    st.metric(
        "Priority",
        selected["audit_priority"]
    )


# ==========================================================
# EXISTING AUDIT REASON
# ==========================================================

st.subheader("🔎 Initial Audit Explanation")

st.warning(
    selected["audit_reason"]
)


# ==========================================================
# PREPARE FEATURES
# ==========================================================

numeric_features = [
    "amount",
    "absolute_amount",
    "transaction_hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_night",
    "amount_to_income",
    "amount_to_credit_limit",
    "debt_to_income",
    "client_transaction_count",
    "client_avg_amount",
    "client_std_amount",
    "amount_deviation",
    "merchant_transaction_count",
    "client_merchant_count",
    "client_category_count",
    "is_online",
    "is_chip",
    "is_swipe",
    "previous_amount",
    "previous_avg_amount",
    "previous_transaction_count",
    "minutes_since_previous_transaction",
    "historical_amount_deviation",
    "is_first_transaction",
    "has_error",
    "current_age",
    "credit_score",
    "num_credit_cards",
    "num_cards_issued",
    "credit_limit",
    "yearly_income",
    "total_debt",
    "per_capita_income"
]

categorical_features = [
    "use_chip",
    "gender",
    "card_brand",
    "card_type",
    "has_chip"
]

model_features = list(model.feature_names_in_)


# ==========================================================
# CREATE FEATURE ROW
# ==========================================================

feature_row = pd.DataFrame(
    index=[0],
    columns=model_features
)

for feature in model_features:

    if feature in selected.index:

        feature_row.loc[0, feature] = selected[feature]

    else:

        feature_row.loc[0, feature] = 0


# Convert numeric columns

for feature in numeric_features:

    if feature in feature_row.columns:

        feature_row[feature] = pd.to_numeric(
            feature_row[feature],
            errors="coerce"
        ).fillna(0)


# Encode categorical columns

for feature in categorical_features:

    if feature in feature_row.columns:

        feature_row[feature] = (
            feature_row[feature]
            .astype(str)
            .astype("category")
            .cat.codes
        )


feature_row = feature_row.fillna(0)


# ==========================================================
# SHAP
# ==========================================================

st.subheader("🧠 SHAP Feature Explanation")

try:

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        feature_row
    )

    # Handle different SHAP output formats
    if isinstance(shap_values, list):

        values = shap_values[1][0]

    elif len(np.shape(shap_values)) == 3:

        values = shap_values[0, :, 1]

    else:

        values = shap_values[0]


    explanation = pd.DataFrame({
        "Feature": model_features,
        "SHAP Impact": values,
        "Absolute Impact": np.abs(values)
    })

    explanation = explanation.sort_values(
        "Absolute Impact",
        ascending=False
    )

    top_features = explanation.head(10)


    # ------------------------------------------------------
    # Chart
    # ------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        top_features["Feature"][::-1],
        top_features["SHAP Impact"][::-1]
    )

    ax.set_xlabel(
        "SHAP Impact on Prediction"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Top Factors Influencing Risk Prediction"
    )

    plt.tight_layout()

    st.pyplot(fig)


    # ------------------------------------------------------
    # Table
    # ------------------------------------------------------

    st.dataframe(
        top_features[
            [
                "Feature",
                "SHAP Impact"
            ]
        ],
        width="stretch"
    )


    # ------------------------------------------------------
    # Plain-language explanation
    # ------------------------------------------------------

    st.subheader(
        "📝 Auditor Interpretation"
    )

    positive = explanation[
        explanation["SHAP Impact"] > 0
    ].head(3)

    if len(positive) > 0:

        features = ", ".join(
            positive["Feature"].tolist()
        )

        st.info(
            f"""
            The model's risk prediction was influenced most strongly
            by the following factors: **{features}**.

            Positive SHAP values indicate that these features pushed
            the prediction toward the higher-risk class.
            """
        )

    else:

        st.info(
            "No strong positive feature contribution was identified "
            "for this transaction."
        )


except Exception as e:

    st.error(
        "SHAP explanation could not be generated."
    )

    st.code(str(e))
