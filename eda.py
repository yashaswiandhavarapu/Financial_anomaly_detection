import pandas as pd

FILE = "financial_expense_cleaned.csv"

print("Loading cleaned dataset...")
df = pd.read_csv(FILE)

print("\n========== DATASET OVERVIEW ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n========== DATE RANGE ==========")
print("Start:", df["date"].min())
print("End:", df["date"].max())

print("\n========== AMOUNT STATISTICS ==========")
print(df["amount"].describe())

print("\n========== NEGATIVE TRANSACTIONS ==========")
print("Negative amounts:", (df["amount"] < 0).sum())
print("Positive amounts:", (df["amount"] > 0).sum())
print("Zero amounts:", (df["amount"] == 0).sum())

print("\n========== TRANSACTION TYPE ==========")
print(df["use_chip"].value_counts())

print("\n========== RISK LABEL DISTRIBUTION ==========")
print(df["risk_label"].value_counts())

print("\n========== LABELED DATA DISTRIBUTION ==========")
labeled = df[df["risk_label"] != "Unknown"]

print("Labeled transactions:", len(labeled))
print(labeled["risk_label"].value_counts())

print("\n========== MERCHANT CATEGORIES ==========")
print(
    df["merchant_category"]
    .value_counts()
    .head(20)
)

print("\n========== CLIENT STATISTICS ==========")
print("Unique clients:", df["client_id"].nunique())
print("Unique cards:", df["card_id"].nunique())
print("Unique merchants:", df["merchant_id"].nunique())

print("\n========== ERROR STATISTICS ==========")
print(df["has_error"].value_counts())

print("\n========== FINANCIAL STATISTICS ==========")

financial_columns = [
    "yearly_income",
    "total_debt",
    "credit_score",
    "credit_limit"
]

print(df[financial_columns].describe())

print("\n========== TOP 10 LARGEST TRANSACTIONS ==========")
print(
    df.nlargest(10, "amount")[
        ["id", "client_id", "amount",
         "merchant_category", "risk_label"]
    ].to_string(index=False)
)

print("\n========== TOP 10 HIGHEST ABSOLUTE TRANSACTIONS ==========")
temp = df.copy()
temp["absolute_amount"] = temp["amount"].abs()

print(
    temp.nlargest(10, "absolute_amount")[
        ["id", "client_id", "amount",
         "merchant_category", "risk_label"]
    ].to_string(index=False)
)