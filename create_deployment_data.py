import pandas as pd

INPUT_FILE = "audit_priority.csv"
OUTPUT_FILE = "audit_priority_deployment.csv"

print("Loading audit priority data...")

df = pd.read_csv(INPUT_FILE)

print("Original rows:", len(df))

# Keep the highest-priority transactions
df = df.sort_values(
    "audit_priority_score",
    ascending=False
)

# Keep top 10,000
df = df.head(10000)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n================================")
print("DEPLOYMENT DATA CREATED")
print("================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Saved as:", OUTPUT_FILE)