import pandas as pd

INPUT_FILE = "ml_labeled_data.csv"

TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data.csv"

print("Loading labeled dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date"]
)

print("Total rows:", len(df))

# Sort chronologically
df = df.sort_values("date").reset_index(drop=True)

# --------------------------------------------------
# TIME-BASED SPLIT
# --------------------------------------------------

split_date = df["date"].quantile(0.80)

print("\nSplit date:", split_date)

train = df[df["date"] < split_date].copy()
test = df[df["date"] >= split_date].copy()

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n========== TRAINING DATA ==========")
print("Rows:", len(train))
print(train["target"].value_counts())

print("\n========== TEST DATA ==========")
print("Rows:", len(test))
print(test["target"].value_counts())

print("\n========== DATE RANGES ==========")
print("Training:", train["date"].min(), "to", train["date"].max())
print("Testing :", test["date"].min(), "to", test["date"].max())

# --------------------------------------------------
# SAVE
# --------------------------------------------------

print("\nSaving files...")

train.to_csv(TRAIN_FILE, index=False)
test.to_csv(TEST_FILE, index=False)

print("\n================================")
print("TRAIN / TEST SPLIT COMPLETE")
print("================================")
print("Training file:", TRAIN_FILE)
print("Testing file:", TEST_FILE)