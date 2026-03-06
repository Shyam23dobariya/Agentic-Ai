import numpy as np

np.random.seed(42)

data = np.random.randn(100, 5) * [10, 0.5, 100, 3, 50] \
     + [50, 1.5, 200, 0, -30]

feature_mean = data.mean(axis=0)   # shape: (5,)
feature_std  = data.std(axis=0)    # shape: (5,)

print(f"Data shape        : {data.shape}")
print(f"Mean shape        : {feature_mean.shape}  ← one value per feature")
print(f"Std  shape        : {feature_std.shape}  ← one value per feature")
print(f"\nPer-feature means : {feature_mean.round(4)}")
print(f"Per-feature stds  : {feature_std.round(4)}")