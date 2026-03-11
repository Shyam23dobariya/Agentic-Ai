import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# 1. Create a sample CSV file
# ─────────────────────────────────────────────
np.random.seed(42)

sample_data = {
    "StudentID": range(1, 21),
    "Name": [
        "Alice", "Bob", "Carol", "David", "Eva",
        "Frank", "Grace", "Henry", "Isla", "Jack",
        "Karen", "Leo", "Mia", "Noah", "Olivia",
        "Paul", "Quinn", "Rachel", "Sam", "Tina"
    ],
    "Age":   np.random.randint(18, 30, size=20),
    "Score": np.random.randint(50, 100, size=20),
    "Label": np.random.choice(["Pass", "Fail"], size=20, p=[0.75, 0.25]),
    "Attendance": np.round(np.random.uniform(60, 100, size=20), 1),
}

csv_path = "students.csv"
df_raw = pd.DataFrame(sample_data)
df_raw.to_csv(csv_path, index=False)
print(f"Sample CSV created → '{csv_path}'  ({len(df_raw)} rows)\n")

# ─────────────────────────────────────────────
# 2. Load the dataset using pd.read_csv()
# ─────────────────────────────────────────────
df = pd.read_csv(csv_path)

print("=" * 55)
print("  AI Dataset Inspection Pipeline")
print("=" * 55)

# ─────────────────────────────────────────────
# 3a. First 5 rows
# ─────────────────────────────────────────────
print("\n── First 5 Rows (head()) ──────────────────────")
print(df.head())

# ─────────────────────────────────────────────
# 3b. Last 5 rows
# ─────────────────────────────────────────────
print("\n── Last 5 Rows (tail()) ───────────────────────")
print(df.tail())

# ─────────────────────────────────────────────
# 3c. Structural information
# ─────────────────────────────────────────────
print("\n── Dataset Info (info()) ──────────────────────")
df.info()

# ─────────────────────────────────────────────
# 3d. Summary statistics
# ─────────────────────────────────────────────
print("\n── Summary Statistics (describe()) ────────────")
print(df.describe().round(2))

# ─────────────────────────────────────────────
# 4. Select a single column
# ─────────────────────────────────────────────
print("\n── Single Column Selection → 'Score' ──────────")
score_column = df["Score"]
print(score_column)
print(f"Type : {type(score_column).__name__}")

# ─────────────────────────────────────────────
# 5. Select multiple columns
# ─────────────────────────────────────────────
print("\n── Multiple Column Selection → Name, Score, Label ──")
subset_df = df[["Name", "Score", "Label"]]
print(subset_df)
print(f"Shape : {subset_df.shape}")

# ─────────────────────────────────────────────
# 6. Filter rows based on a numerical condition
# ─────────────────────────────────────────────
print("\n── Filtered Rows (Score > 80) ─────────────────")
high_scorers = df[df["Score"] > 80]
print(high_scorers)
print(f"\nTotal students with Score > 80 : {len(high_scorers)}")

print("\n── Filtered Rows (Age < 22 AND Label == 'Pass') ──")
young_passers = df[(df["Age"] < 22) & (df["Label"] == "Pass")]
print(young_passers)
print(f"\nTotal young passers (Age < 22 & Pass) : {len(young_passers)}")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Summary")
print("=" * 55)
print(f"  Dataset shape          : {df.shape}")
print(f"  Columns                : {list(df.columns)}")
print(f"  Score column — mean    : {df['Score'].mean():.2f}")
print(f"  Score column — max     : {df['Score'].max()}")
print(f"  Rows with Score > 80   : {len(high_scorers)}")
print(f"  Pass / Fail counts     :\n{df['Label'].value_counts().to_string()}")
print("=" * 55)
