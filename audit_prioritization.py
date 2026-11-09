import pandas as pd
import numpy as np

INPUT_FILE = "flagged_anomalies.csv"
OUTPUT_FILE = "audit_priority.csv"

print("Loading flagged anomalies...")

df = pd.read_csv(INPUT_FILE)

print("Flagged transactions:", len(df))


# ==========================================================
# CONVERT IMPORTANT FEATURES
# ==========================================================

numeric_columns = [
    "risk_probability",
    "amount_deviation",
    "historical_amount_deviation",
    "amount_to_income",
    "amount_to_credit_limit",
    "client_merchant_count",
    "minutes_since_previous_transaction"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)


# ==========================================================
# NORMALIZE RISK PROBABILITY
# ==========================================================

risk_component = (
    df["risk_probability"] * 100
)


# ==========================================================
# BEHAVIORAL RISK COMPONENT
# ==========================================================

behavior_score = np.zeros(
    len(df)
)


# Unusual spending
behavior_score += np.where(
    df["amount_deviation"] > 3,
    20,
    0
)


# Historical spending deviation
behavior_score += np.where(
    df["historical_amount_deviation"] > 3,
    20,
    0
)


# High amount relative to income
behavior_score += np.where(
    df["amount_to_income"] > 0.10,
    15,
    0
)


# High credit utilization
behavior_score += np.where(
    df["amount_to_credit_limit"] > 0.50,
    15,
    0
)


# Unusual merchant
behavior_score += np.where(
    df["client_merchant_count"] <= 1,
    10,
    0
)


# Night transaction
if "is_night" in df.columns:

    behavior_score += np.where(
        df["is_night"] == 1,
        5,
        0
    )


# Online transaction
if "is_online" in df.columns:

    behavior_score += np.where(
        df["is_online"] == 1,
        5,
        0
    )


# Rapid successive transaction
behavior_score += np.where(
    (
        (df["minutes_since_previous_transaction"] >= 0)
        &
        (df["minutes_since_previous_transaction"] < 5)
    ),
    10,
    0
)


# Limit behavior score
behavior_score = np.minimum(
    behavior_score,
    100
)


# ==========================================================
# TRANSACTION SEVERITY
# ==========================================================

amount = pd.to_numeric(
    df["amount"],
    errors="coerce"
).abs().fillna(0)

amount_score = np.clip(
    amount / 1000 * 100,
    0,
    100
)


# ==========================================================
# FINAL AUDIT PRIORITY SCORE
# ==========================================================

df["audit_priority_score"] = (
    0.50 * risk_component
    +
    0.35 * behavior_score
    +
    0.15 * amount_score
)


df["audit_priority_score"] = (
    df["audit_priority_score"]
    .clip(0, 100)
    .round(2)
)


# ==========================================================
# PRIORITY LEVEL
# ==========================================================

df["audit_priority"] = np.select(
    [
        df["audit_priority_score"] >= 80,
        df["audit_priority_score"] >= 60,
        df["audit_priority_score"] >= 30
    ],
    [
        "CRITICAL",
        "HIGH",
        "MEDIUM"
    ],
    default="LOW"
)


# ==========================================================
# AUDIT ACTION
# ==========================================================

df["recommended_action"] = np.select(
    [
        df["audit_priority"] == "CRITICAL",
        df["audit_priority"] == "HIGH",
        df["audit_priority"] == "MEDIUM"
    ],
    [
        "Immediate auditor investigation",
        "Priority auditor review",
        "Secondary review"
    ],
    default="Routine monitoring"
)


# ==========================================================
# RANK TRANSACTIONS
# ==========================================================

df = df.sort_values(
    "audit_priority_score",
    ascending=False
).reset_index(
    drop=True
)

df["audit_rank"] = (
    df.index + 1
)


# ==========================================================
# SAVE
# ==========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================================
# SUMMARY
# ==========================================================

print("\n==========================================")
print("STEP 12 COMPLETE")
print("==========================================")

print("\nPriority distribution:")

print(
    df["audit_priority"].value_counts()
)


print("\nTop 20 audit priorities:")

display_columns = [
    "audit_rank",
    "id",
    "amount",
    "risk_probability",
    "audit_priority_score",
    "audit_priority",
    "audit_reason",
    "recommended_action"
]

print(
    df[display_columns]
    .head(20)
    .to_string(index=False)
)


print("\nSaved as:")
print(OUTPUT_FILE)