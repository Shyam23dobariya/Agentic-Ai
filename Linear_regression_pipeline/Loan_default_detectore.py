# Python version : 3.11+
# Install        : pip install scikit-learn numpy pandas
# Run            : python loan_default_detector.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve
)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Generate Dataset
# ═════════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(seed=42)
n = 600

monthly_income     = rng.uniform(10, 100,  size=n)
loan_amount        = rng.uniform(50, 500,  size=n)
credit_score       = rng.uniform(300, 850, size=n)
num_existing_loans = rng.integers(0, 6,    size=n)   # 0–5 inclusive

risk_score = (
    -0.05  * monthly_income
    + 0.008 * loan_amount
    - 0.012 * credit_score
    + 1.5   * num_existing_loans
    + rng.normal(0, 2, size=n)
)

will_default = (risk_score > 0).astype(int)

X = np.column_stack([monthly_income, loan_amount, credit_score, num_existing_loans])
y = will_default

feature_names = ["monthly_income", "loan_amount", "credit_score", "num_existing_loans"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 65)
print("STEP 1 — Dataset Summary")
print("=" * 65)
print(f"  Total samples   : {n}")
print(f"  Train samples   : {len(X_train)}")
print(f"  Test  samples   : {len(X_test)}")
print(f"  Default (1)     : {y.sum()}  ({y.mean()*100:.1f}%)")
print(f"  No Default (0)  : {(y==0).sum()}  ({(y==0).mean()*100:.1f}%)")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Train Both Models
# ═════════════════════════════════════════════════════════════════════════════
dt  = DecisionTreeClassifier(max_depth=5, random_state=42)
rf  = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

print("\n" + "=" * 65)
print("STEP 2 — Models Trained")
print("=" * 65)
print("  ✔ Decision Tree  (max_depth=5)")
print("  ✔ Random Forest  (n_estimators=100, max_depth=5)")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Evaluate Both Models
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, name):
    """Return a dict of all evaluation metrics for one model."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm      = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "Model"    : name,
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall"   : recall_score(y_test, y_pred, zero_division=0),
        "F1"       : f1_score(y_test, y_pred, zero_division=0),
        "AUC"      : roc_auc_score(y_test, y_proba),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }

dt_res = evaluate_model(dt, X_test, y_test, "Decision Tree")
rf_res = evaluate_model(rf, X_test, y_test, "Random Forest")

# Individual model detail
for res in [dt_res, rf_res]:
    print("\n" + "=" * 65)
    print(f"STEP 3 — {res['Model']} Results")
    print("=" * 65)
    print(f"  Accuracy  : {res['Accuracy']:.4f}")
    print(f"  Precision : {res['Precision']:.4f}")
    print(f"  Recall    : {res['Recall']:.4f}")
    print(f"  F1 Score  : {res['F1']:.4f}")
    print(f"  AUC       : {res['AUC']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted 0   Predicted 1")
    print(f"  Actual 0  :     TN={res['TN']:>3}       FP={res['FP']:>3}")
    print(f"  Actual 1  :     FN={res['FN']:>3}       TP={res['TP']:>3}")

# Side-by-side comparison table
print("\n" + "=" * 65)
print("STEP 3 — Side-by-Side Comparison")
print("=" * 65)
col_w = 18
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
print(f"{'Metric':<14} {'Decision Tree':>{col_w}} {'Random Forest':>{col_w}}")
print("-" * (14 + col_w * 2 + 2))
for m in metrics:
    print(f"{m:<14} {dt_res[m]:>{col_w}.4f} {rf_res[m]:>{col_w}.4f}")
print("-" * (14 + col_w * 2 + 2))
cm_rows = [("TN","TN"), ("FP","FP"), ("FN","FN"), ("TP","TP")]
for label, key in cm_rows:
    print(f"{label:<14} {dt_res[key]:>{col_w}} {rf_res[key]:>{col_w}}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Find Best Threshold via Precision-Recall Curve (Random Forest)
# ═════════════════════════════════════════════════════════════════════════════
rf_proba = rf.predict_proba(X_test)[:, 1]

precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, rf_proba)

# F1 at each threshold (thresholds array is 1 shorter than precision/recall)
f1_vals = np.where(
    (precision_vals[:-1] + recall_vals[:-1]) == 0,
    0,
    2 * precision_vals[:-1] * recall_vals[:-1] / (precision_vals[:-1] + recall_vals[:-1])
)

best_idx       = np.argmax(f1_vals)
best_threshold = thresholds[best_idx]
best_precision = precision_vals[best_idx]
best_recall    = recall_vals[best_idx]
best_f1        = f1_vals[best_idx]

print("\n" + "=" * 65)
print("STEP 4 — Optimal Threshold (Random Forest, Precision-Recall Curve)")
print("=" * 65)
print(f"  Best Threshold : {best_threshold:.4f}")
print(f"  Precision      : {best_precision:.4f}")
print(f"  Recall         : {best_recall:.4f}")
print(f"  F1 Score       : {best_f1:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Compare Default (0.5) vs Optimised Threshold
# ═════════════════════════════════════════════════════════════════════════════
def metrics_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
        "F1"       : f1_score(y_true, y_pred, zero_division=0),
    }

default_metrics  = metrics_at_threshold(y_test, rf_proba, 0.5)
optimised_metrics = metrics_at_threshold(y_test, rf_proba, best_threshold)

print("\n" + "=" * 65)
print("STEP 5 — Default (0.5) vs Optimised Threshold Comparison")
print("=" * 65)
col_w = 16
print(f"{'Metric':<12} {'Threshold = 0.50':>{col_w}} {'Threshold = ' + f'{best_threshold:.4f}':>{col_w}}")
print("-" * (12 + col_w * 2 + 2))
for m in ["Precision", "Recall", "F1"]:
    print(f"{m:<12} {default_metrics[m]:>{col_w}.4f} {optimised_metrics[m]:>{col_w}.4f}")
print("-" * (12 + col_w * 2 + 2))

# Conclusion
improved = optimised_metrics["F1"] > default_metrics["F1"]
direction = "improved" if improved else "did not improve"
print(
    f"\n  Conclusion: The optimised threshold ({best_threshold:.4f}) {direction} the F1 Score "
    f"compared to the default (0.5) — "
    f"{optimised_metrics['F1']:.4f} vs {default_metrics['F1']:.4f}."
)

print("\n" + "=" * 65)
print("Done.")
print("=" * 65)