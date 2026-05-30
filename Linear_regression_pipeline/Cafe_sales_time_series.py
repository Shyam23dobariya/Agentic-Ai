# Python version : 3.11+
# Install        : pip install pandas numpy
# Run            : python cafe_sales_time_series.py

import pandas as pd
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Create the Daily Sales Dataset (unchanged)
# ═════════════════════════════════════════════════════════════════════════════
data = {
    "date": pd.date_range(start="2024-01-01", periods=30, freq="D"),
    "sales": [
        200, 220, 215, 230, 250, 245, 260, 270, 265, 280,
        300, 295, 310, 330, 325, 340, 360, 355, 370, 390,
        410, 405, 420, 440, 435, 450, 470, 465, 480, 500
    ]
}

df = pd.DataFrame(data)

print("=" * 65)
print("STEP 1 — Raw Dataset")
print("=" * 65)
print(df.to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Prepare the Time Series
# ═════════════════════════════════════════════════════════════════════════════
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

print("\n" + "=" * 65)
print("STEP 2 — Time Series Prepared")
print("=" * 65)
print(f"  Index type : {type(df.index)}")
print(f"  Date range : {df.index.min().date()}  →  {df.index.max().date()}")
print(f"  Sorted     : {df.index.is_monotonic_increasing}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Rolling Window Features (3-day window)
# ═════════════════════════════════════════════════════════════════════════════
df["rolling_mean_3"] = df["sales"].rolling(window=3).mean()
df["rolling_std_3"]  = df["sales"].rolling(window=3).std()
df["rolling_max_3"]  = df["sales"].rolling(window=3).max()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Lag Features
# ═════════════════════════════════════════════════════════════════════════════
df["lag_1"] = df["sales"].shift(1)    # yesterday's sales
df["lag_7"] = df["sales"].shift(7)    # sales from 7 days ago

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Chronological Train-Test Split (80 / 20, no shuffle)
# ═════════════════════════════════════════════════════════════════════════════
split_idx = int(len(df) * 0.80)       # 30 * 0.8 = 24 rows → train

train = df.iloc[:split_idx]
test  = df.iloc[split_idx:]

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Baseline Forecasts on Test Set
# ═════════════════════════════════════════════════════════════════════════════
# naive_forecast      : use lag_1 (yesterday's actual sales)
# rolling_mean_forecast: use rolling_mean_3 computed from full df
test = test.copy()
test["naive_forecast"]        = test["lag_1"]
test["rolling_mean_forecast"] = test["rolling_mean_3"]

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — MAPE Function & Evaluation
# ═════════════════════════════════════════════════════════════════════════════
def mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean Absolute Percentage Error (excludes rows where actual == 0)."""
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

# Drop rows where either forecast is NaN before evaluating
test_eval = test.dropna(subset=["naive_forecast", "rolling_mean_forecast"])

mape_naive   = mape(test_eval["sales"], test_eval["naive_forecast"])
mape_rolling = mape(test_eval["sales"], test_eval["rolling_mean_forecast"])

# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — Print Final Output
# ═════════════════════════════════════════════════════════════════════════════

# 8a — First 10 rows after feature creation
print("\n" + "=" * 65)
print("STEP 8a — First 10 Rows After Feature Creation")
print("=" * 65)
print(df.head(10).round(2).to_string())

# 8b — Train and test date ranges
print("\n" + "=" * 65)
print("STEP 8b — Train / Test Date Ranges")
print("=" * 65)
print(f"  Train : {train.index.min().date()}  →  {train.index.max().date()}"
      f"  ({len(train)} rows)")
print(f"  Test  : {test.index.min().date()}   →  {test.index.max().date()}"
      f"  ({len(test)} rows)")

# 8c — MAPE values
print("\n" + "=" * 65)
print("STEP 8c — Baseline Forecast MAPE on Test Set")
print("=" * 65)
print(f"  Naive Forecast MAPE        (lag_1)          : {mape_naive:.4f}%")
print(f"  Rolling Mean Forecast MAPE (rolling_mean_3) : {mape_rolling:.4f}%")

# 8d — Test set detail table
print("\n" + "=" * 65)
print("STEP 8c — Test Set Predictions Detail")
print("=" * 65)
display_cols = ["sales", "naive_forecast", "rolling_mean_forecast"]
print(test[display_cols].round(2).to_string())

# 8e — Comparison sentence
print("\n" + "=" * 65)
print("STEP 8d — Comparison")
print("=" * 65)
if mape_naive < mape_rolling:
    better, worse = "Naive Forecast (lag_1)", "Rolling Mean Forecast"
    better_mape, worse_mape = mape_naive, mape_rolling
else:
    better, worse = "Rolling Mean Forecast (rolling_mean_3)", "Naive Forecast"
    better_mape, worse_mape = mape_rolling, mape_naive

print(
    f"  The {better} performed better with a MAPE of "
    f"{better_mape:.4f}% compared to {worse_mape:.4f}% "
    f"for the {worse}."
)

print("\n" + "=" * 65)
print("Pipeline Complete.")
print("=" * 65)