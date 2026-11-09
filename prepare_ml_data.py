import pandas as pd

INPUT_FILE = "financial_expense_time_features.csv"
OUTPUT_FILE = "ml_labeled_data.csv"

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total rows:", len(df))

# --------------------------------------------------
# KEEP ONLY LABELED TRANSACTIONS
# --------------------------------------------------

labeled = df[df["risk_label"].isin(["Yes", "No"])].copy()

print("\nLabeled rows:", len(labeled))

print("\nRisk distribution:")
print(labeled["risk_label"].value_counts())

# --------------------------------------------------
# CREATE TARGET
# --------------------------------------------------

labeled["target"] = (
    labeled["risk_label"] == "Yes"
).astype("int8")

# --------------------------------------------------
# REMOVE UNKNOWN LABELS
# --------------------------------------------------

print("\nTarget distribution:")
print(labeled["target"].value_counts())

# --------------------------------------------------
# SAVE
# --------------------------------------------------

print("\nSaving labeled ML dataset...")

labeled.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n================================")
print("ML DATASET CREATED")
print("================================")

print("Rows:", len(labeled))
print("Columns:", len(labeled.columns))
print("Saved as:", OUTPUT_FILE)