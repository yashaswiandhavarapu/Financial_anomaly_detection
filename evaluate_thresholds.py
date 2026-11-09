import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

MODEL_FILE = "random_forest_model.pkl"
TEST_FILE = "test_model_data.csv"

print("Loading test data...")

df = pd.read_csv(TEST_FILE)

print("Test rows:", len(df))


# ==========================================================
# SEPARATE FEATURES AND TARGET
# ==========================================================

X = df.drop(columns=["target"])
y = df["target"]


# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading Random Forest...")

model = joblib.load(MODEL_FILE)

print("Model loaded.")


# ==========================================================
# CHECK FEATURE MATCH
# ==========================================================

print("\nChecking model features...")

model_features = model.feature_names_in_

missing = [
    col for col in model_features
    if col not in X.columns
]

extra = [
    col for col in X.columns
    if col not in model_features
]

print("Model features:", len(model_features))
print("Test features:", len(X.columns))

if missing:
    print("\nMissing features:")
    print(missing)

    raise ValueError(
        "Test dataset is missing model features."
    )

# Keep exactly the features used during training
X = X[model_features]


# ==========================================================
# CLEAN NUMERICAL VALUES
# ==========================================================

print("\nCleaning data...")

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# ==========================================================
# PREDICT PROBABILITIES
# ==========================================================

print("\nGenerating probabilities...")

probabilities = model.predict_proba(X)[:, 1]

print("Predictions generated.")


# ==========================================================
# OVERALL METRICS
# ==========================================================

print("\n==========================================")
print("OVERALL MODEL PERFORMANCE")
print("==========================================")

roc_auc = roc_auc_score(
    y,
    probabilities
)

pr_auc = average_precision_score(
    y,
    probabilities
)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC : {pr_auc:.4f}")


# ==========================================================
# THRESHOLD ANALYSIS
# ==========================================================

thresholds = [
    0.01,
    0.02,
    0.03,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


results = []


print("\n==========================================")
print("THRESHOLD ANALYSIS")
print("==========================================")

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions
    ).ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn
    })


results_df = pd.DataFrame(results)


print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ==========================================================
# BEST F1 THRESHOLD
# ==========================================================

best_f1_row = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n==========================================")
print("BEST F1 THRESHOLD")
print("==========================================")

print(
    f"Threshold : "
    f"{best_f1_row['threshold']:.2f}"
)

print(
    f"Precision  : "
    f"{best_f1_row['precision']:.4f}"
)

print(
    f"Recall     : "
    f"{best_f1_row['recall']:.4f}"
)

print(
    f"F1-score   : "
    f"{best_f1_row['f1']:.4f}"
)


# ==========================================================
# BEST PRECISION WITH REASONABLE RECALL
# ==========================================================

# For audit systems, we don't want extremely low recall.
# Look for thresholds with at least 20% recall.

reasonable = results_df[
    results_df["recall"] >= 0.20
]

if len(reasonable) > 0:

    best_precision_row = reasonable.loc[
        reasonable["precision"].idxmax()
    ]

    print("\n==========================================")
    print("BEST PRECISION WITH RECALL >= 20%")
    print("==========================================")

    print(
        f"Threshold : "
        f"{best_precision_row['threshold']:.2f}"
    )

    print(
        f"Precision  : "
        f"{best_precision_row['precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{best_precision_row['recall']:.4f}"
    )

    print(
        f"F1-score   : "
        f"{best_precision_row['f1']:.4f}"
    )


# ==========================================================
# SAVE RESULTS
# ==========================================================

results_df.to_csv(
    "threshold_analysis.csv",
    index=False
)

print("\n==========================================")
print("STEP 10 COMPLETE")
print("==========================================")

print(
    "Saved as: threshold_analysis.csv"
)