import pandas as pd

INPUT_FILE = "test_data.csv"
OUTPUT_FILE = "audit_sample.csv"

print("Creating audit sample...")

# Read only the columns needed for the dashboard/model
columns = [
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
    "risk_label",

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

# Read only 100,000 rows
df = pd.read_csv(
    INPUT_FILE,
    usecols=columns,
    nrows=100000
)

print("Rows:", len(df))

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Saved:", OUTPUT_FILE)