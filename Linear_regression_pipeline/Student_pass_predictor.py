# Python version : 3.11+
# Install        : pip install scikit-learn numpy
# Run            : python student_pass_predictor.py

import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ── Provided dataset (unchanged) ─────────────────────────────────────────────
rng = default_rng(seed=99)
n = 500

study_hours        = rng.uniform(1, 10, size=n)
attendance_percent = rng.uniform(40, 100, size=n)
assignments_done   = rng.uniform(0, 10, size=n)

scores = (
    20
    + 5.5  * study_hours
    + 0.4  * attendance_percent
    + 3.0  * assignments_done
    + rng.normal(0, 8, size=n)
)

y = (scores >= 70).astype(int)
X = np.column_stack([study_hours, attendance_percent, assignments_done])

# ═════════════════════════════════════════════════════════════════════════════
# TASK 1 — Train a Logistic Regression model
# ═════════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=99
)

model = LogisticRegression()
model.fit(X_train, y_train)

pass_count = (y == 1).sum()
fail_count = (y == 0).sum()

print("=" * 60)
print("TASK 1 — Class Distribution (full dataset, n=500)")
print("=" * 60)
print(f"  Pass (1) : {pass_count} students")
print(f"  Fail (0) : {fail_count} students")
print(f"  Total    : {n} students")

# ═════════════════════════════════════════════════════════════════════════════
# TASK 2 — Predict and display results
# ═════════════════════════════════════════════════════════════════════════════
y_pred       = model.predict(X_test)
y_proba      = model.predict_proba(X_test)          # shape (n_test, 2)
p_pass       = y_proba[:, 1]                        # P(Pass) for each student

print("\n" + "=" * 60)
print("TASK 2 — First 10 Test Students")
print("=" * 60)
header = f"{'#':>3} | {'Study Hrs':>9} | {'Attendance %':>12} | {'Assignments':>11} | {'Actual':>6} | {'Predicted':>9} | {'P(Pass)':>7} | {'Correct?':>8}"
print(header)
print("-" * len(header))
for i in range(10):
    sh  = X_test[i, 0]
    att = X_test[i, 1]
    asg = X_test[i, 2]
    act = y_test[i]
    prd = y_pred[i]
    pp  = p_pass[i]
    ok  = "Yes" if act == prd else "No"
    print(f"{i+1:>3} | {sh:>9.2f} | {att:>12.2f} | {asg:>11.2f} | {act:>6} | {prd:>9} | {pp:>7.4f} | {ok:>8}")

# ═════════════════════════════════════════════════════════════════════════════
# TASK 3 — Confusion Matrix & Manual Accuracy
# ═════════════════════════════════════════════════════════════════════════════
cm = confusion_matrix(y_test, y_pred)
# sklearn layout: rows = actual, cols = predicted
# [[TN, FP],
#  [FN, TP]]
TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

accuracy = (TP + TN) / (TP + TN + FP + FN)

print("\n" + "=" * 60)
print("TASK 3 — Confusion Matrix Results")
print("=" * 60)
print(f"  True  Positives (TP) — Predicted Pass, Actually Pass : {TP}")
print(f"  True  Negatives (TN) — Predicted Fail, Actually Fail : {TN}")
print(f"  False Positives (FP) — Predicted Pass, Actually Fail : {FP}")
print(f"  False Negatives (FN) — Predicted Fail, Actually Pass : {FN}")
print(f"\n  Manual Accuracy = (TP + TN) / Total")
print(f"                  = ({TP} + {TN}) / ({TP} + {TN} + {FP} + {FN})")
print(f"                  = {TP + TN} / {TP + TN + FP + FN}")
print(f"                  = {accuracy:.4f}  ({accuracy*100:.2f}%)")

# ═════════════════════════════════════════════════════════════════════════════
# TASK 4 — Compare Two Decision Thresholds (0.5 vs 0.6)
# ═════════════════════════════════════════════════════════════════════════════
p_pass_test = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("TASK 4 — Threshold Comparison")
print("=" * 60)

for threshold in [0.5, 0.6]:
    pred_thresh  = (p_pass_test >= threshold).astype(int)
    n_pass       = (pred_thresh == 1).sum()
    n_fail       = (pred_thresh == 0).sum()
    acc          = (pred_thresh == y_test).sum() / len(y_test)

    print(f"\n  Threshold = {threshold}")
    print(f"    Predicted Pass : {n_pass} students")
    print(f"    Predicted Fail : {n_fail} students")
    print(f"    Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)