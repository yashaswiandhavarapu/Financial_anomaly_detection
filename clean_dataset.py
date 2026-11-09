import pandas as pd

input_file = "financial_expense_dataset.csv"
output_file = "financial_expense_cleaned.csv"

print("Loading dataset...")

df = pd.read_csv(input_file)

print("Original shape:", df.shape)


# ==========================================
# 1. CLEAN MONEY COLUMNS
# ==========================================

money_columns = [
    "amount",
    "per_capita_income",
    "yearly_income",
    "total_debt",
    "credit_limit"
]

for column in money_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ==========================================
# 2. CONVERT DATE
# ==========================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# ==========================================
# 3. CREATE ERROR FLAG
# ==========================================

df["has_error"] = (
    df["errors"]
    .notna()
    .astype(int)
)

# Remove original errors column
df = df.drop(columns=["errors"])


# ==========================================
# 4. HANDLE LOCATION VALUES
# ==========================================

df["merchant_city"] = df["merchant_city"].fillna("UNKNOWN")

df["merchant_state"] = df["merchant_state"].fillna("UNKNOWN")

df["zip"] = df["zip"].fillna(-1)


# ==========================================
# 5. HANDLE OTHER CATEGORICAL MISSING VALUES
# ==========================================

categorical_columns = [
    "gender",
    "card_brand",
    "card_type",
    "has_chip",
    "merchant_category",
    "use_chip"
]

for column in categorical_columns:
    df[column] = df[column].fillna("UNKNOWN")


# ==========================================
# 6. CHECK MISSING VALUES
# ==========================================

print("\n===== MISSING VALUES AFTER CLEANING =====")

missing = df.isnull().sum()

print(
    missing[missing > 0]
    .sort_values(ascending=False)
)


# ==========================================
# 7. CHECK DUPLICATES
# ==========================================

print("\n===== DUPLICATES =====")

print(
    "Duplicate rows:",
    df.duplicated().sum()
)


# ==========================================
# 8. SAVE CLEANED DATASET
# ==========================================

print("\nSaving cleaned dataset...")

df.to_csv(
    output_file,
    index=False
)

print("\n================================")
print("CLEANED DATASET CREATED")
print("================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Saved as:", output_file)