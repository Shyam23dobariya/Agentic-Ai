import pandas as pd
import numpy as np

# ── 1. Create sample DataFrame ───────────────────────────────────────────────
data = {
    "Employee": [
        "Amit", "Neha", "Rahul", "Sneha",
        "Vikram", "Priya", "Arjun", "Divya"
    ],
    "Department": [
        "IT", "HR", "IT", "Finance",
        "HR", "Finance", "IT", "HR"
    ],
    "Salary": [
        600000, 500000, np.nan, 700000,
        520000, np.nan, 650000, 480000
    ],
    "Temporary_Notes": [
        "On probation", "Contract",
        "Pending docs", "Verified",
        "Intern", "New joiner",
        "On leave", "Temporary role"
    ]
}

df = pd.DataFrame(data)

print("=" * 50)
print("ORIGINAL DATASET")
print("=" * 50)
print(df)

# ── 2. Detect and print missing values ───────────────────────────────────────
print("\n" + "=" * 50)
print("MISSING VALUES DETECTED")
print("=" * 50)
print(df.isnull().sum())

# ── 3. Fill missing Salary values with column mean ───────────────────────────
salary_mean = df["Salary"].mean()
df["Salary"] = df["Salary"].fillna(salary_mean)

print(f"\nMean Salary used to fill missing values: ₹{salary_mean:,.0f}")

# ── 4. Drop the Temporary_Notes column ───────────────────────────────────────
df = df.drop(columns=["Temporary_Notes"])

# ── 5. Rename Salary to Annual_Salary ────────────────────────────────────────
df = df.rename(columns={"Salary": "Annual_Salary"})

print("\n" + "=" * 50)
print("CLEANED DATASET")
print("=" * 50)
print(df)

# ── 6. Group by Department: mean salary & employee count ─────────────────────
summary = df.groupby("Department").agg(
    Mean_Annual_Salary=("Annual_Salary", "mean"),
    Employee_Count=("Employee", "count")
).reset_index()

summary["Mean_Annual_Salary"] = summary["Mean_Annual_Salary"].round(2)

# ── 7. Print final summary table ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("DEPARTMENT SUMMARY")
print("=" * 50)
print(summary.to_string(index=False))