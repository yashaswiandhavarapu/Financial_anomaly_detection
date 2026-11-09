import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Model Evaluation")

st.markdown(
    """
    Evaluate the performance of the Random Forest anomaly
    classification model and examine the trade-off between
    detecting risky transactions and reducing false alarms.
    """
)

st.divider()


# ==========================================================
# LOAD THRESHOLD RESULTS
# ==========================================================

@st.cache_data
def load_threshold_data():
    return pd.read_csv("threshold_analysis.csv")


df = load_threshold_data()


# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

st.subheader("Overall Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "ROC-AUC",
        "0.8893"
    )

with col2:
    st.metric(
        "PR-AUC",
        "0.0154"
    )


st.info(
    """
    ROC-AUC measures the model's ability to distinguish between
    risky and non-risky transactions across different thresholds.

    PR-AUC is particularly important here because the dataset is
    highly imbalanced, with very few risky transactions.
    """
)


# ==========================================================
# THRESHOLD ANALYSIS
# ==========================================================

st.subheader("Threshold Analysis")

st.dataframe(
    df,
    width="stretch"
)


# ==========================================================
# PRECISION VS RECALL
# ==========================================================

st.subheader("Precision vs Recall")

fig, ax = plt.subplots()

ax.plot(
    df["threshold"],
    df["precision"],
    marker="o",
    label="Precision"
)

ax.plot(
    df["threshold"],
    df["recall"],
    marker="o",
    label="Recall"
)

ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision-Recall Trade-off")
ax.legend()
ax.grid(True)

st.pyplot(fig)


# ==========================================================
# F1 SCORE
# ==========================================================

st.subheader("F1-Score by Threshold")

fig, ax = plt.subplots()

ax.plot(
    df["threshold"],
    df["f1"],
    marker="o"
)

ax.set_xlabel("Classification Threshold")
ax.set_ylabel("F1-Score")
ax.set_title("F1-Score Across Thresholds")
ax.grid(True)

st.pyplot(fig)


# ==========================================================
# FALSE POSITIVE RATE
# ==========================================================

st.subheader("False Positive Rate")

fig, ax = plt.subplots()

ax.plot(
    df["threshold"],
    df["false_positive_rate"],
    marker="o"
)

ax.set_xlabel("Classification Threshold")
ax.set_ylabel("False Positive Rate")
ax.set_title("False Positive Rate Across Thresholds")
ax.grid(True)

st.pyplot(fig)


# ==========================================================
# RECOMMENDED THRESHOLD
# ==========================================================

best_row = df[
    df["recall"] >= 0.20
].sort_values(
    "precision",
    ascending=False
).iloc[0]


st.subheader("Recommended Audit Threshold")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Threshold",
        f"{best_row['threshold']:.2f}"
    )

with col2:
    st.metric(
        "Precision",
        f"{best_row['precision']:.2%}"
    )

with col3:
    st.metric(
        "Recall",
        f"{best_row['recall']:.2%}"
    )


st.success(
    f"""
    For the current audit system, a threshold of
    **{best_row['threshold']:.2f}** provides the highest precision
    among the evaluated thresholds while maintaining at least
    20% recall.
    """
)


# ==========================================================
# IMPORTANT INTERPRETATION
# ==========================================================

st.subheader("Audit Interpretation")

st.warning(
    """
    The model should not be interpreted as automatically declaring
    a transaction fraudulent.

    Because the dataset contains very few positive risk labels,
    false positives are substantial. Therefore, the model is used
    primarily to prioritize transactions for human auditor review.
    """
)


# ==========================================================
# MODEL RESULTS
# ==========================================================

st.subheader("Key Findings")

st.markdown(
    """
    **Random Forest results**

    - ROC-AUC: **0.8893**
    - PR-AUC: **0.0154**
    - Best F1 threshold: **0.70**
    - Recommended audit threshold: **0.60**
    - Recall at 0.60: **24.54%**
    - Precision at 0.60: **1.83%**

    The results demonstrate that threshold selection is important
    for controlling false alarms in a highly imbalanced audit
    dataset.
    """
)