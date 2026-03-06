import numpy as np

data       = np.array([10, 20, 30, 40, 50, 60, 70, 80])
mean       = np.mean(data)
std        = np.std(data)
normalized = (data - mean) / std

# Reshape 1D (8,) → 2D (2, 4)
reshaped = normalized.reshape(2, 4)

print("=" * 40)
print("  Step 4 — Reshape to 2D Array")
print("=" * 40)
print(f"Normalized (1D) shape : {normalized.shape}")
print(f"Reshaped   (2D) shape : {reshaped.shape}")
print(f"\nReshaped array (2 rows × 4 cols):")
print(reshaped.round(2))
print(f"\nRow 0 : {reshaped[0].round(2)}")
print(f"Row 1 : {reshaped[1].round(2)}")