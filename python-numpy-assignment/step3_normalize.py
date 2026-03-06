import numpy as np

data = np.array([10, 20, 30, 40, 50, 60, 70, 80])

mean = np.mean(data)
std  = np.std(data)

# Normalize using vectorized broadcasting
normalized = (data - mean) / std

print("=" * 40)
print("  Step 3 — Normalize the Data")
print("=" * 40)
print(f"Original data  : {data}")
print(f"Mean           : {mean}")
print(f"Std deviation  : {std:.2f}")
print(f"\nNormalized data: {normalized.round(2)}")
print(f"\nSanity checks (after normalization):")
print(f"  Post-norm mean (≈ 0) : {normalized.mean():.10f}")
print(f"  Post-norm std  (≈ 1) : {normalized.std():.10f}")
