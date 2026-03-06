import numpy as np

np.random.seed(42)

data = np.random.randn(100, 5) * [10, 0.5, 100, 3, 50] \
     + [50, 1.5, 200, 0, -30]

feature_mean = data.mean(axis=0)
feature_std  = data.std(axis=0)

# Broadcasting: (100,5) - (5,) → (100,5)
normalized = (data - feature_mean) / feature_std

print(f"Original data shape  : {data.shape}")
print(f"Normalized data shape: {normalized.shape}")
print(f"\nPost-normalization mean per feature (≈ 0): {normalized.mean(axis=0).round(8)}")
print(f"Post-normalization std  per feature (≈ 1): {normalized.std(axis=0).round(8)}")
print(f"\nFirst 3 normalized rows:\n{normalized[:3].round(4)}")