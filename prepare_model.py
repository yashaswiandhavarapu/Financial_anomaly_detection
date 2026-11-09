import pandas as pd
import numpy as np

TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data.csv"

TRAIN_OUTPUT = "train_model_data.csv"
TEST_OUTPUT = "test_model_data.csv"

print("Loading training data...")

train = pd.read_csv(TRAIN_FILE)
print("Training rows:", len(train))

print("Loading testing data...")

test = pd.read_csv(TEST_FILE)
print("Testing rows:", len(test))


# ==========================================================
# FEATURES WE WANT TO USE
# ==========================================================

numeric_features = [
    "amount",
    "absolute_amount",

    "transaction_hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_night",

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


categorical_features = [
    "use_chip",
    "gender",
    "card_brand",
    "card_type",
    "has_chip"
]


# ==========================================================
# CHECK FEATURES
# ==========================================================

print("\nChecking feature columns...")

for column in numeric_features + categorical_features:
    if column not in train.columns:
        print("WARNING - Missing column:", column)


# ==========================================================
# COPY DATA
# ==========================================================

X_train = train[
    numeric_features + categorical_features
].copy()

X_test = test[
    numeric_features + categorical_features
].copy()

y_train = train["target"].copy()
y_test = test["target"].copy()


# ==========================================================
# HANDLE NUMERIC VALUES
# ==========================================================

print("\nCleaning numerical features...")

for column in numeric_features:

    X_train[column] = pd.to_numeric(
        X_train[column],
        errors="coerce"
    )

    X_test[column] = pd.to_numeric(
        X_test[column],
        errors="coerce"
    )

    # Use training median for missing values
    median_value = X_train[column].median()

    X_train[column] = X_train[column].fillna(
        median_value
    )

    X_test[column] = X_test[column].fillna(
        median_value
    )


# ==========================================================
# COMBINE CATEGORICAL VALUES
# ==========================================================

print("\nEncoding categorical features...")

for column in categorical_features:

    # Convert to string
    X_train[column] = (
        X_train[column]
        .astype(str)
        .fillna("UNKNOWN")
    )

    X_test[column] = (
        X_test[column]
        .astype(str)
        .fillna("UNKNOWN")
    )

    # Create mapping using training data only
    categories = X_train[column].unique()

    mapping = {
        value: index
        for index, value in enumerate(categories)
    }

    # Unknown categories in test get -1
    X_train[column] = (
        X_train[column]
        .map(mapping)
        .fillna(-1)
        .astype("int16")
    )

    X_test[column] = (
        X_test[column]
        .map(mapping)
        .fillna(-1)
        .astype("int16")
    )


# ==========================================================
# CHECK FOR INFINITE VALUES
# ==========================================================

print("\nChecking infinite values...")

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

# Fill any resulting NaN
for column in X_train.columns:

    median_value = X_train[column].median()

    X_train[column] = X_train[column].fillna(
        median_value
    )

    X_test[column] = X_test[column].fillna(
        median_value
    )


# ==========================================================
# SAVE
# ==========================================================

print("\nSaving model-ready data...")

train_model = X_train.copy()
train_model["target"] = y_train.values

test_model = X_test.copy()
test_model["target"] = y_test.values

train_model.to_csv(
    TRAIN_OUTPUT,
    index=False
)

test_model.to_csv(
    TEST_OUTPUT,
    index=False
)


# ==========================================================
# SUMMARY
# ==========================================================

print("\n==========================================")
print("MODEL DATA PREPARATION COMPLETE")
print("==========================================")

print("\nTraining shape:")
print(train_model.shape)

print("\nTesting shape:")
print(test_model.shape)

print("\nTraining target:")
print(train_model["target"].value_counts())

print("\nTesting target:")
print(test_model["target"].value_counts())

print("\nSaved:")
print(TRAIN_OUTPUT)
print(TEST_OUTPUT)