import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

TRAIN_FILE = "train_model_data.csv"
TEST_FILE = "test_model_data.csv"
MODEL_FILE = "random_forest_model.pkl"

print("Loading model data...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print("Training rows:", len(train))
print("Testing rows:", len(test))


# ==========================================================
# 1. SEPARATE FEATURES AND TARGET
# ==========================================================

X_train_full = train.drop(columns=["target"])
y_train_full = train["target"]

X_test = test.drop(columns=["target"])
y_test = test["target"]


# ==========================================================
# 2. CREATE A MANAGEABLE BALANCED TRAINING SAMPLE
# ==========================================================

print("\nCreating balanced training sample...")

normal_train = train[train["target"] == 0]
risk_train = train[train["target"] == 1]

# Keep all risky transactions
# Use 5 normal transactions for every risky transaction
normal_sample_size = min(
    len(normal_train),
    len(risk_train) * 5
)

normal_sample = normal_train.sample(
    n=normal_sample_size,
    random_state=42
)

balanced_train = pd.concat(
    [normal_sample, risk_train]
).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X_train = balanced_train.drop(columns=["target"])
y_train = balanced_train["target"]

print("Balanced training rows:", len(balanced_train))
print("\nTraining class distribution:")
print(y_train.value_counts())


# ==========================================================
# 3. TRAIN RANDOM FOREST
# ==========================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed.")


# ==========================================================
# 4. PREDICTIONS
# ==========================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

# Probability of class 1
y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================================
# 5. CLASSIFICATION REPORT
# ==========================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# ==========================================================
# 6. CONFUSION MATRIX
# ==========================================================

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ==========================================================
# 7. ROC-AUC
# ==========================================================

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nROC-AUC:", round(roc_auc, 4))


# ==========================================================
# 8. PR-AUC
# ==========================================================

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print("PR-AUC:", round(pr_auc, 4))


# ==========================================================
# 9. FEATURE IMPORTANCE
# ==========================================================

print("\n==========================================")
print("TOP 15 FEATURE IMPORTANCES")
print("==========================================")

importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.head(15).to_string(
        index=False
    )
)


# ==========================================================
# 10. SAVE MODEL
# ==========================================================

print("\nSaving model...")

joblib.dump(
    model,
    MODEL_FILE
)

print("\n==========================================")
print("RANDOM FOREST MODEL CREATED")
print("==========================================")

print("Saved as:", MODEL_FILE)