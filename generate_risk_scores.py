import pandas as pd
import numpy as np
import joblib
import os

INPUT_FILE = "financial_expense_time_features.csv"

RF_MODEL = "random_forest_model.pkl"
IF_MODEL = "isolation_forest_model.pkl"

OUTPUT_FILE = "audit_risk_scores.csv"

CHUNK_SIZE = 250000


# ==========================================================
# FEATURES USED BY RANDOM FOREST
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

all_features = numeric_features + categorical_features


# ==========================================================
# ISOLATION FOREST FEATURES
# ==========================================================

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


# ==========================================================
# LOAD MODELS
# ==========================================================

print("Loading models...")

rf_model = joblib.load(RF_MODEL)
if_model = joblib.load(IF_MODEL)

print("Models loaded.")


# ==========================================================
# CREATE CATEGORY MAPPINGS
# ==========================================================

print("\nCreating categorical mappings from training data...")

train_reader = pd.read_csv(
    "train_data.csv",
    usecols=categorical_features,
    chunksize=250000
)

category_values = {
    column: []
    for column in categorical_features
}

for chunk in train_reader:

    for column in categorical_features:

        values = chunk[column].astype(str).unique()

        for value in values:

            if value not in category_values[column]:
                category_values[column].append(value)


category_maps = {}

for column in categorical_features:

    category_maps[column] = {
        value: index
        for index, value
        in enumerate(category_values[column])
    }

    print(
        column,
        "categories:",
        len(category_maps[column])
    )


# ==========================================================
# PREPARE OUTPUT
# ==========================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

first_chunk = True

total_processed = 0


# ==========================================================
# PROCESS DATA IN CHUNKS
# ==========================================================

print("\nStarting chunk processing...")

reader = pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE
)

for chunk_number, df in enumerate(reader, start=1):

    print(
        f"\nProcessing chunk {chunk_number}..."
    )

    # ------------------------------------------------------
    # Prepare ML features
    # ------------------------------------------------------

    X = df[all_features].copy()

    # Numeric
    for column in numeric_features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Replace infinity
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill numerical missing values
    for column in numeric_features:

        X[column] = X[column].fillna(0)

    # ------------------------------------------------------
    # Categorical encoding
    # ------------------------------------------------------

    for column in categorical_features:

        mapping = category_maps[column]

        X[column] = (
            X[column]
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype("int16")
        )

    # ------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------

    rf_probability = (
        rf_model
        .predict_proba(X)[:, 1]
    )

    # ------------------------------------------------------
    # Isolation Forest
    # ------------------------------------------------------

    X_if = X[if_features]

    raw_anomaly = (
        -if_model.decision_function(X_if)
    )

    # Normalize within chunk
    min_score = raw_anomaly.min()
    max_score = raw_anomaly.max()

    if max_score != min_score:

        anomaly_score = (
            (raw_anomaly - min_score)
            / (max_score - min_score)
        )

    else:

        anomaly_score = np.zeros(
            len(raw_anomaly)
        )

    # ------------------------------------------------------
    # Combined risk score
    # ------------------------------------------------------

    combined_score = (
        0.6 * rf_probability +
        0.4 * anomaly_score
    )

    risk_score = (
        combined_score * 100
    )

    # ------------------------------------------------------
    # Risk level
    # ------------------------------------------------------

    risk_level = np.select(
        [
            risk_score >= 60,
            risk_score >= 30
        ],
        [
            "HIGH",
            "MEDIUM"
        ],
        default="LOW"
    )

    # ------------------------------------------------------
    # Audit reason
    # ------------------------------------------------------

    reason = np.where(
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

                            np.where(
                                (
                                    (df["minutes_since_previous_transaction"] >= 0)
                                    &
                                    (df["minutes_since_previous_transaction"] < 5)
                                ),
                                "Rapid successive transaction",
                                "Unusual transaction pattern"
                            )
                        )
                    )
                )
            )
        )
    )

    # ------------------------------------------------------
    # Create output
    # ------------------------------------------------------

    result = pd.DataFrame({
        "id": df["id"].values,
        "date": df["date"].values,
        "client_id": df["client_id"].values,
        "card_id": df["card_id"].values,
        "amount": df["amount"].values,
        "merchant_id": df["merchant_id"].values,
        "merchant_city": df["merchant_city"].values,
        "merchant_state": df["merchant_state"].values,
        "merchant_category": df["merchant_category"].values,
        "risk_label": df["risk_label"].values,

        "risk_probability": rf_probability,
        "anomaly_score": anomaly_score,
        "audit_risk_score": risk_score,
        "risk_level": risk_level,
        "audit_reason": reason
    })

    # ------------------------------------------------------
    # Save chunk
    # ------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

    total_processed += len(df)

    print(
        "Processed:",
        total_processed
    )


# ==========================================================
# FINISHED
# ==========================================================

print("\n==========================================")
print("AUDIT RISK SCORING COMPLETE")
print("==========================================")

print(
    "Total transactions processed:",
    total_processed
)

print(
    "Output:",
    OUTPUT_FILE
)