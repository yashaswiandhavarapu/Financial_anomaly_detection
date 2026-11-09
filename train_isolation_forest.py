import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest

INPUT_FILE = "financial_expense_time_features.csv"
MODEL_FILE = "isolation_forest_model.pkl"

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))


# ==========================================================
# FEATURES FOR ANOMALY DETECTION
# ==========================================================

features = [
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


print("\nSelecting features...")

X = df[features].copy()


# ==========================================================
# CLEAN NUMERIC DATA
# ==========================================================

print("Cleaning data...")

for column in X.columns:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median()
)


# ==========================================================
# SAMPLE DATA FOR TRAINING
# ==========================================================

# Isolation Forest does not need millions of rows.
# Use a representative sample to train efficiently.

sample_size = min(300000, len(X))

print("\nSampling", sample_size, "transactions...")

X_sample = X.sample(
    n=sample_size,
    random_state=42
)


# ==========================================================
# TRAIN ISOLATION FOREST
# ==========================================================

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=150,
    contamination=0.01,
    max_samples=10000,
    random_state=42,
    n_jobs=-1
)

model.fit(X_sample)

print("Training completed.")


# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    model,
    MODEL_FILE
)

print("\n==========================================")
print("ISOLATION FOREST CREATED")
print("==========================================")

print("Training samples:", sample_size)
print("Trees:", 150)
print("Contamination:", 0.01)
print("Saved as:", MODEL_FILE)