import pandas as pd
import numpy as np
import joblib
import os

MODEL_FILE = "random_forest_model.pkl"
MODEL_DATA = "test_model_data.csv"
ORIGINAL_DATA = "test_data.csv"

OUTPUT_FILE = "flagged_anomalies.csv"

THRESHOLD = 0.60
CHUNK_SIZE = 100000

print("Loading Random Forest...")
model = joblib.load(MODEL_FILE)

model_features = list(model.feature_names_in_)

print("Model features:", len(model_features))
print("Risk threshold:", THRESHOLD)


# ==========================================================
# READ ORIGINAL DATA COLUMNS
# ==========================================================

original_header = pd.read_csv(
    ORIGINAL_DATA,
    nrows=0
).columns.tolist()

identifier_columns = [
    "id",
    "date",
    "client_id",
    "card_id",
    "amount",
    "merchant_id",
    "merchant_city",
    "merchant_state",
    "merchant_category",
    "risk_label"
]

identifier_columns = [
    col for col in identifier_columns
    if col in original_header
]

print("Identifier columns:", identifier_columns)


# ==========================================================
# REMOVE OLD OUTPUT
# ==========================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)


# ==========================================================
# PROCESS IN CHUNKS
# ==========================================================

model_reader = pd.read_csv(
    MODEL_DATA,
    chunksize=CHUNK_SIZE
)

original_reader = pd.read_csv(
    ORIGINAL_DATA,
    usecols=identifier_columns,
    chunksize=CHUNK_SIZE
)


total_rows = 0
total_flagged = 0
first_chunk = True

print("\nStarting anomaly interpretation...")


for chunk_number, (model_df, original_df) in enumerate(
    zip(model_reader, original_reader),
    start=1
):

    print(
        f"Processing chunk {chunk_number}..."
    )

    # ------------------------------------------------------
    # Prepare model features
    # ------------------------------------------------------

    X = model_df[model_features].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    # ------------------------------------------------------
    # Predict risk probability
    # ------------------------------------------------------

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    # ------------------------------------------------------
    # Select flagged transactions
    # ------------------------------------------------------

    flagged = probabilities >= THRESHOLD

    if not flagged.any():

        total_rows += len(model_df)
        continue

    flagged_model = model_df.loc[
        flagged
    ].copy()

    flagged_original = original_df.loc[
        flagged
    ].copy()

    flagged_probability = probabilities[
        flagged
    ]

    # ------------------------------------------------------
    # Create result
    # ------------------------------------------------------

    result = flagged_original.reset_index(
        drop=True
    )

    result["risk_probability"] = (
        flagged_probability
    )

    result["risk_level"] = "HIGH"

    # ------------------------------------------------------
    # Explain each transaction
    # ------------------------------------------------------

    reasons = []

    for index in flagged_model.index:

        row = flagged_model.loc[index]

        row_reasons = []

        # Unusual spending
        if (
            "amount_deviation" in row.index
            and row["amount_deviation"] > 3
        ):
            row_reasons.append(
                "Unusual spending amount"
            )

        # Historical deviation
        if (
            "historical_amount_deviation" in row.index
            and row["historical_amount_deviation"] > 3
        ):
            row_reasons.append(
                "Large deviation from historical spending"
            )

        # Online transaction
        if (
            "is_online" in row.index
            and row["is_online"] == 1
        ):
            row_reasons.append(
                "Online transaction"
            )

        # Night transaction
        if (
            "is_night" in row.index
            and row["is_night"] == 1
        ):
            row_reasons.append(
                "Night transaction"
            )

        # Income relationship
        if (
            "amount_to_income" in row.index
            and row["amount_to_income"] > 0.10
        ):
            row_reasons.append(
                "High amount relative to income"
            )

        # Credit utilization
        if (
            "amount_to_credit_limit" in row.index
            and row["amount_to_credit_limit"] > 0.50
        ):
            row_reasons.append(
                "High amount relative to credit limit"
            )

        # Unusual merchant
        if (
            "client_merchant_count" in row.index
            and row["client_merchant_count"] <= 1
        ):
            row_reasons.append(
                "Unusual merchant for customer"
            )

        # Rapid transaction
        if (
            "minutes_since_previous_transaction" in row.index
            and
            row["minutes_since_previous_transaction"] >= 0
            and
            row["minutes_since_previous_transaction"] < 5
        ):
            row_reasons.append(
                "Rapid successive transaction"
            )

        # Transaction error
        if (
            "has_error" in row.index
            and row["has_error"] == 1
        ):
            row_reasons.append(
                "Transaction contains an error"
            )

        # Default explanation
        if not row_reasons:

            row_reasons.append(
                "Model detected an unusual transaction pattern"
            )

        # Keep maximum three explanations
        reasons.append(
            "; ".join(row_reasons[:3])
        )

    result["audit_reason"] = reasons

    # ------------------------------------------------------
    # Add useful feature values
    # ------------------------------------------------------

    useful_features = [
        "amount_deviation",
        "historical_amount_deviation",
        "amount_to_income",
        "amount_to_credit_limit",
        "is_online",
        "is_night",
        "client_merchant_count",
        "minutes_since_previous_transaction"
    ]

    for feature in useful_features:

        if feature in flagged_model.columns:

            result[feature] = (
                flagged_model[feature]
                .values
            )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

    total_rows += len(model_df)
    total_flagged += len(result)

    print(
        f"Flagged in chunk: {len(result)}"
    )

    print(
        f"Total flagged so far: {total_flagged}"
    )


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n==========================================")
print("STEP 11 COMPLETE")
print("==========================================")

print(
    "Transactions examined:",
    total_rows
)

print(
    "Flagged transactions:",
    total_flagged
)

print(
    "Flag rate:",
    f"{(total_flagged / total_rows) * 100:.2f}%"
)

print(
    "Saved as:",
    OUTPUT_FILE
)