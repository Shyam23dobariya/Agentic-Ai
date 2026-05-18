import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Provided code (unchanged) ────────────────────────────────────────────────
rng = np.random.default_rng(seed=21)
n_samples = 400
study_hours = rng.uniform(1, 10, size=n_samples)
exam_score = 40 + 7.5 * study_hours + rng.normal(0, 6, size=n_samples)

df = pd.DataFrame({"study_hours": study_hours, "exam_score": exam_score})

# ── 1. Features & Target ─────────────────────────────────────────────────────
X = df[["study_hours"]]   # 2-D array required by sklearn
y = df["exam_score"]

# ── 2. Train / Test Split (80 / 20, stratification not needed for regression)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Fit model on training data only ───────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

print(f"Intercept : {model.intercept_:.4f}  (true ≈ 40)")
print(f"Coefficient: {model.coef_[0]:.4f}  (true ≈ 7.5)")

# ── 4. Predict on held-out test set ──────────────────────────────────────────
y_pred = model.predict(X_test)

# Aligned actual-vs-predicted DataFrame (index preserved from the split)
results_df = pd.DataFrame({
    "actual":    y_test.values,
    "predicted": y_pred
}, index=y_test.index)

print("\nActual vs Predicted (first 10 rows of test set):")
print(results_df.head(10).round(2))

# ── 5. Metrics function ───────────────────────────────────────────────────────
def regression_metrics_from_df(df: pd.DataFrame,
                                y_col: str,
                                y_pred_col: str) -> dict:
    """
    Return a dict with keys: 'mae', 'rmse', 'r2'
    using sklearn.metrics (numpy only for sqrt of MSE).
    """
    y_true = df[y_col]
    y_hat  = df[y_pred_col]

    mae  = mean_absolute_error(y_true, y_hat)
    mse  = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_hat)

    return {"mae": mae, "rmse": rmse, "r2": r2}

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
metrics = regression_metrics_from_df(results_df, y_col="actual", y_pred_col="predicted")

print("\nTest-set Regression Metrics")
print("=" * 32)
print(f"  MAE  : {metrics['mae']:.4f}")
print(f"  RMSE : {metrics['rmse']:.4f}")
print(f"  R²   : {metrics['r2']:.4f}")