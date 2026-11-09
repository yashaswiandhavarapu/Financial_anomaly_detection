import pandas as pd
import numpy as np

INPUT_FILE = "financial_expense_features.csv"
OUTPUT_FILE = "financial_expense_time_features.csv"

print("Loading feature dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date"]
)

print("Rows loaded:", len(df))

# ---------------------------------------------------------
# 1. SORT CHRONOLOGICALLY
# ---------------------------------------------------------

print("\nSorting transactions by client and date...")

df = df.sort_values(
    ["client_id", "date"]
).reset_index(drop=True)


# ---------------------------------------------------------
# 2. PREVIOUS TRANSACTION AMOUNT
# ---------------------------------------------------------

print("Creating previous transaction amount...")

df["previous_amount"] = (
    df.groupby("client_id")["absolute_amount"]
    .shift(1)
)


# ---------------------------------------------------------
# 3. PREVIOUS CUSTOMER AVERAGE
# ---------------------------------------------------------

print("Creating historical customer average...")

group = df.groupby("client_id")["absolute_amount"]

# Cumulative sum BEFORE current transaction
cumulative_sum = group.cumsum() - df["absolute_amount"]

# Number of previous transactions
previous_count = (
    df.groupby("client_id")
    .cumcount()
)

df["previous_avg_amount"] = (
    cumulative_sum /
    previous_count.replace(0, np.nan)
)


# ---------------------------------------------------------
# 4. PREVIOUS CUSTOMER TRANSACTION COUNT
# ---------------------------------------------------------

df["previous_transaction_count"] = (
    df.groupby("client_id")
    .cumcount()
)


# ---------------------------------------------------------
# 5. TIME SINCE PREVIOUS TRANSACTION
# ---------------------------------------------------------

print("Creating time since previous transaction...")

previous_date = (
    df.groupby("client_id")["date"]
    .shift(1)
)

df["minutes_since_previous_transaction"] = (
    (df["date"] - previous_date)
    .dt.total_seconds()
    / 60
)


# ---------------------------------------------------------
# 6. DEVIATION FROM HISTORICAL AVERAGE
# ---------------------------------------------------------

print("Creating historical amount deviation...")

df["historical_amount_deviation"] = (
    df["absolute_amount"] /
    (df["previous_avg_amount"] + 1)
)


# ---------------------------------------------------------
# 7. FIRST TRANSACTION FLAG
# ---------------------------------------------------------

df["is_first_transaction"] = (
    df["previous_transaction_count"] == 0
).astype("int8")


# ---------------------------------------------------------
# 8. FILL FIRST-TRANSACTION VALUES
# ---------------------------------------------------------

df["previous_amount"] = (
    df["previous_amount"]
    .fillna(0)
)

df["previous_avg_amount"] = (
    df["previous_avg_amount"]
    .fillna(df["absolute_amount"])
)

df["minutes_since_previous_transaction"] = (
    df["minutes_since_previous_transaction"]
    .fillna(-1)
)


# ---------------------------------------------------------
# 9. RESTORE ORIGINAL TRANSACTION ORDER
# ---------------------------------------------------------

print("Restoring transaction order...")

df = df.sort_values("id").reset_index(drop=True)


# ---------------------------------------------------------
# 10. SAVE
# ---------------------------------------------------------

print("\nSaving time-aware feature dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# 11. SUMMARY
# ---------------------------------------------------------

print("\n==========================================")
print("TIME-AWARE FEATURES CREATED")
print("==========================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nNew features:")

features = [
    "previous_amount",
    "previous_avg_amount",
    "previous_transaction_count",
    "minutes_since_previous_transaction",
    "historical_amount_deviation",
    "is_first_transaction"
]

for feature in features:
    print(feature)

print("\nSaved as:")
print(OUTPUT_FILE)