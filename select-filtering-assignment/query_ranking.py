import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. Create sample dataset
# ─────────────────────────────────────────────
data = {
    "Name":     ["Alice", "Bob", "Carol", "David", "Eva",
                 "Frank", "Grace", "Henry", "Isla", "Jack"],
    "Score":    [95, 82, 91, 78, 88, 60, 97, 73, 85, 92],
    "Passed":   [True, True, True, False, True,
                 False, True, False, True, True],
    "Category": ["A", "B", "A", "B", "A",
                 "B", "A", "B", "A", "A"],
    "Attempts": [1, 2, 1, 3, 1, 4, 1, 2, 2, 1],
}

df = pd.DataFrame(data)

print("=" * 55)
print("  AI Data Query and Ranking System")
print("=" * 55)

print("\n── Full Dataset ───────────────────────────────")
print(df.to_string(index=True))

# ─────────────────────────────────────────────
# 2a. Select a single column
# ─────────────────────────────────────────────
print("\n── Single Column → 'Score' ────────────────────")
score_series = df["Score"]
print(score_series.to_string())
print(f"Type : {type(score_series).__name__}")

# ─────────────────────────────────────────────
# 2b. Select multiple columns → new DataFrame
# ─────────────────────────────────────────────
print("\n── Multiple Columns → Name, Score, Passed ─────")
subset_df = df[["Name", "Score", "Passed"]].copy()
print(subset_df.to_string(index=False))
print(f"Shape : {subset_df.shape}")

# ─────────────────────────────────────────────
# 2c. iloc — position-based: first 3 rows
# ─────────────────────────────────────────────
print("\n── iloc → First 3 Rows (rows 0, 1, 2) ─────────")
first_three = df.iloc[:3]
print(first_three.to_string(index=True))

# ─────────────────────────────────────────────
# 2d. loc — label-based after setting index
# ─────────────────────────────────────────────
df_indexed = df.set_index("Name")          # Name becomes the row label

print("\n── loc → Rows by Name Index ────────────────────")
selected_by_name = df_indexed.loc[["Alice", "Grace", "Jack"]]
print(selected_by_name.to_string())

# ─────────────────────────────────────────────
# 2e. Filter: Score > 85
# ─────────────────────────────────────────────
print("\n── Filter: Score > 85 ──────────────────────────")
high_score = df[df["Score"] > 85].copy()
print(high_score.to_string(index=False))
print(f"Count : {len(high_score)} students")

# ─────────────────────────────────────────────
# 2f. Filter: Score > 85 AND Passed is True
# ─────────────────────────────────────────────
print("\n── Filter: Score > 85 AND Passed == True ───────")
high_and_passed = df[(df["Score"] > 85) & (df["Passed"] == True)].copy()
print(high_and_passed.to_string(index=False))
print(f"Count : {len(high_and_passed)} students")

# ─────────────────────────────────────────────
# 2g. Sort filtered result descending by Score
# ─────────────────────────────────────────────
print("\n── Filtered + Sorted Descending by Score ───────")
sorted_result = high_and_passed.sort_values("Score", ascending=False)
print(sorted_result.to_string(index=False))

# ─────────────────────────────────────────────
# 3. Chained filter + sort in one expression
# ─────────────────────────────────────────────
print("\n── Chained: Category A, Score > 80, Sorted ────")
chained = (
    df[(df["Category"] == "A") & (df["Score"] > 80)]
    .sort_values("Score", ascending=False)
    .reset_index(drop=True)
)
print(chained.to_string(index=True))
print(f"Count : {len(chained)} students")

# ─────────────────────────────────────────────
# Bonus: NOT operator (~) — students who failed
# ─────────────────────────────────────────────
print("\n── Bonus Filter: NOT Passed (~) ────────────────")
failed = df[~df["Passed"]].copy()
print(failed.to_string(index=False))
print(f"Count : {len(failed)} students who did not pass")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Summary")
print("=" * 55)
print(f"  Total students          : {len(df)}")
print(f"  Score > 85              : {len(high_score)}")
print(f"  Score > 85 & Passed     : {len(high_and_passed)}")
print(f"  Failed (Passed = False) : {len(failed)}")
print(f"  Top scorer              : {df.loc[df['Score'].idxmax(), 'Name']} "
      f"({df['Score'].max()})")
print(f"  Average score           : {df['Score'].mean():.2f}")
print("=" * 55)
