import pandas as pd
import numpy as np

INPUT_FILE = "financial_expense_cleaned.csv"
OUTPUT_FILE = "financial_expense_features.csv"

print("Loading cleaned dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date"]
)

print("Rows loaded:", len(df))


# ============================================================
# 1. TIME-BASED FEATURES
# ============================================================

print("\nCreating time features...")

df["transaction_hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

# Saturday = 5, Sunday = 6
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype("int8")

# Consider 10 PM - 5 AM as night
df["is_night"] = (
    (df["transaction_hour"] >= 22) |
    (df["transaction_hour"] <= 5)
).astype("int8")


# ============================================================
# 2. ABSOLUTE TRANSACTION AMOUNT
# ============================================================

print("Creating amount features...")

df["absolute_amount"] = df["amount"].abs()


# ============================================================
# 3. FINANCIAL RATIO FEATURES
# ============================================================

print("Creating financial ratios...")

# Amount compared with yearly income
df["amount_to_income"] = (
    df["absolute_amount"] /
    (df["yearly_income"] + 1)
)

# Amount compared with credit limit
df["amount_to_credit_limit"] = (
    df["absolute_amount"] /
    (df["credit_limit"] + 1)
)

# Debt compared with yearly income
df["debt_to_income"] = (
    df["total_debt"] /
    (df["yearly_income"] + 1)
)


# ============================================================
# 4. CUSTOMER TRANSACTION FREQUENCY
# ============================================================

print("Calculating customer transaction frequency...")

client_transaction_count = (
    df.groupby("client_id")["id"]
    .transform("count")
)

df["client_transaction_count"] = (
    client_transaction_count
    .astype("int32")
)


# ============================================================
# 5. CUSTOMER SPENDING STATISTICS
# ============================================================

print("Calculating customer spending statistics...")

client_avg_amount = (
    df.groupby("client_id")["absolute_amount"]
    .transform("mean")
)

client_std_amount = (
    df.groupby("client_id")["absolute_amount"]
    .transform("std")
)

df["client_avg_amount"] = client_avg_amount

df["client_std_amount"] = (
    client_std_amount.fillna(0)
)


# ============================================================
# 6. DEVIATION FROM CUSTOMER'S NORMAL SPENDING
# ============================================================

print("Calculating spending deviation...")

df["amount_deviation"] = (
    (
        df["absolute_amount"] -
        df["client_avg_amount"]
    )
    /
    (df["client_std_amount"] + 1)
)


# ============================================================
# 7. MERCHANT FREQUENCY
# ============================================================

print("Calculating merchant frequency...")

df["merchant_transaction_count"] = (
    df.groupby("merchant_id")["id"]
    .transform("count")
    .astype("int32")
)


# ============================================================
# 8. CLIENT-MERCHANT FREQUENCY
# ============================================================

print("Calculating client-merchant frequency...")

df["client_merchant_count"] = (
    df.groupby(
        ["client_id", "merchant_id"]
    )["id"]
    .transform("count")
    .astype("int32")
)


# ============================================================
# 9. CLIENT-CATEGORY FREQUENCY
# ============================================================

print("Calculating client-category frequency...")

df["client_category_count"] = (
    df.groupby(
        ["client_id", "merchant_category"]
    )["id"]
    .transform("count")
    .astype("int32")
)


# ============================================================
# 10. ONLINE TRANSACTION FLAG
# ============================================================

print("Creating online transaction flag...")

df["is_online"] = (
    df["use_chip"]
    .eq("Online Transaction")
    .astype("int8")
)


# ============================================================
# 11. CHIP TRANSACTION FLAG
# ============================================================

df["is_chip"] = (
    df["use_chip"]
    .eq("Chip Transaction")
    .astype("int8")
)


# ============================================================
# 12. SWIPE TRANSACTION FLAG
# ============================================================

df["is_swipe"] = (
    df["use_chip"]
    .eq("Swipe Transaction")
    .astype("int8")
)


# ============================================================
# 13. SAVE FEATURE DATASET
# ============================================================

print("\nSaving feature dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 14. FINAL INFORMATION
# ============================================================

print("\n==========================================")
print("FEATURE DATASET CREATED SUCCESSFULLY")
print("==========================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nNew feature columns:")

new_features = [
    "transaction_hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_night",
    "absolute_amount",
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
    "is_swipe"
]

for feature in new_features:
    print(feature)

print("\nSaved as:")
print(OUTPUT_FILE)