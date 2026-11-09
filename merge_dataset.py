import pandas as pd

file = "financial_expense_dataset.csv"

df = pd.read_csv(file)

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
for column in df.columns:
    print(column)

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
missing = df.isnull().sum()

print(missing[missing > 0].sort_values(ascending=False))

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== SAMPLE DATA =====")
print(df.head(5).to_string())

print("\n===== RISK LABELS =====")
print(df["risk_label"].value_counts(dropna=False))

print("\n===== AMOUNT SAMPLE =====")
print(df["amount"].head(10).to_string())