import numpy as np

# ── Pipeline ──────────────────────────────────
data       = np.array([10, 20, 30, 40, 50, 60, 70, 80])
mean       = np.mean(data)
std        = np.std(data)
normalized = (data - mean) / std
reshaped   = normalized.reshape(2, 4)

# ── Print all results ─────────────────────────
print("=" * 45)
print("  NumPy Preprocessing Pipeline — Summary")
print("=" * 45)

print(f"\nOriginal data       : {data}")
print(f"Mean                : {mean}")
print(f"Standard Deviation  : {std:.2f}")
print(f"\nNormalized data     : {normalized.round(2)}")
print(f"\nReshaped array (2 rows × 4 cols):")
print(reshaped.round(2))
print(f"\nReshaped data shape : {reshaped.shape}")

print("\n" + "=" * 45)
