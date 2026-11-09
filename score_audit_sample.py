import pandas as pd
import numpy as np
import joblib

INPUT_FILE = "audit_sample.csv"
RF_MODEL = "random_forest_model.pkl"
IF_MODEL = "isolation_forest_model.pkl"
OUTPUT_FILE = "audit_risk_scores.csv"

print("Loading audit sample...")
df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))

# Same features used during model training
numeric_features = [
    "amount", "absolute_amount",
    "transaction_hour", "day_of_week", "month",
    "is_weekend", "is_night",
    "amount_to_income", "amount_to_credit_limit",
    "debt_to_income",
    "client_transaction_count", "client_avg_amount",
    "client_std_amount", "amount_deviation",
    "merchant_transaction_count", "client_merchant_count",
    "client_category_count",
    "is_online", "is_chip", "is_swipe",
    "previous_amount", "previous_avg_amount",
    "previous_transaction_count",
    "minutes_since_previous_transaction",
    "historical_amount_deviation", "is_first_transaction",
    "has_error", "current_age", "credit_score",
    "num_credit_cards", "num_cards_issued",
    "credit_limit", "yearly_income",
    "total_debt", "per_capita_income"
]

categorical_features = [
    "use_chip",
    "gender",
    "card_brand",
    "card_type",
    "has_chip"
]

all_features = numeric_features + categorical_features

# Prepare features
X = df[all_features].copy()

for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    X[col] = X[col].fillna(0)

# IMPORTANT:
# Use the same category encoding approach as training.
# For this sample, categories are converted consistently
# according to their sorted unique values.

for col in categorical_features:
    X[col] = X[col].astype(str)

    categories = sorted(X[col].unique())

    mapping = {
        value: i for i, value in enumerate(categories)
    }

    X[col] = (
        X[col]
        .map(mapping)
        .fillna(-1)
        .astype("int16")
    )

print("Features prepared:", X.shape)

# Load models
print("\nLoading models...")

rf_model = joblib.load(RF_MODEL)
if_model = joblib.load(IF_MODEL)

print("Models loaded.")

# ---------------------------------------------------------
# Random Forest probability
# ---------------------------------------------------------

print("\nCalculating risk probability...")

risk_probability = (
    rf_model.predict_proba(X)[:, 1]
)

# ---------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------

print("Calculating anomaly score...")

if_features = [
    "absolute_amount",
    "amount_to_income",
    "amount_to_credit_limit",
    "debt_to_income",
    "transaction_hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_night",
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
    "credit_score",
    "num_credit_cards",
    "credit_limit",
    "yearly_income",
    "total_debt"
]

X_if = X[if_features]

raw_anomaly = -if_model.decision_function(X_if)

# Normalize anomaly score
minimum = raw_anomaly.min()
maximum = raw_anomaly.max()

if maximum != minimum:
    anomaly_score = (
        (raw_anomaly - minimum) /
        (maximum - minimum)
    )
else:
    anomaly_score = np.zeros(len(raw_anomaly))

# ---------------------------------------------------------
# Combined audit score
# ---------------------------------------------------------

audit_score = (
    0.6 * risk_probability +
    0.4 * anomaly_score
) * 100

# Risk level
risk_level = np.select(
    [
        audit_score >= 60,
        audit_score >= 30
    ],
    [
        "HIGH",
        "MEDIUM"
    ],
    default="LOW"
)

# ---------------------------------------------------------
# Audit reason
# ---------------------------------------------------------

audit_reason = np.where(
    df["amount_deviation"] > 3,
    "Unusual spending amount",
    np.where(
        df["is_night"] == 1,
        "Night transaction",
        np.where(
            df["is_online"] == 1,
            "Online transaction",
            np.where(
                df["amount_to_income"] > 0.10,
                "High amount relative to income",
                np.where(
                    df["amount_to_credit_limit"] > 0.50,
                    "High credit utilization",
                    np.where(
                        df["client_merchant_count"] <= 1,
                        "Unusual merchant",
                        "Unusual transaction pattern"
                    )
                )
            )
        )
    )
)

# ---------------------------------------------------------
# Final audit dataset
# ---------------------------------------------------------

result = df[
    [
        "id",
        "date",
        "client_id",
        "card_id",
        "amount",
        "use_chip",
        "merchant_id",
        "merchant_city",
        "merchant_state",
        "merchant_category",
        "risk_label"
    ]
].copy()

result["risk_probability"] = risk_probability
result["anomaly_score"] = anomaly_score
result["audit_risk_score"] = audit_score
result["risk_level"] = risk_level
result["audit_reason"] = audit_reason

# Highest risk first
result = result.sort_values(
    "audit_risk_score",
    ascending=False
)

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n======================================")
print("AUDIT SCORING COMPLETE")
print("======================================")

print("Rows:", len(result))

print("\nRisk level distribution:")
print(result["risk_level"].value_counts())

print("\nTop 10 audit priorities:")

print(
    result[
        [
            "id",
            "amount",
            "merchant_category",
            "audit_risk_score",
            "risk_level",
            "audit_reason"
        ]
    ].head(10).to_string(index=False)
)

print("\nSaved as:", OUTPUT_FILE)