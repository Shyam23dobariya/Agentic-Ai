import numpy as np

np.random.seed(42)

data = np.random.randn(100, 5) * [10, 0.5, 100, 3, 50] \
     + [50, 1.5, 200, 0, -30]

feature_mean = data.mean(axis=0)
feature_std  = data.std(axis=0)
normalized   = (data - feature_mean) / feature_std

split_index = int(len(normalized) * 0.80)
train_set   = normalized[:split_index]   # VIEW
train_copy  = normalized[:split_index].copy()  # independent COPY

# ── Demonstrate VIEW behaviour ──────────────────────────
original_value = normalized[0, 0]
sentinel       = 9999.0

train_set[0, 0] = sentinel   # write through the view

print("=== VIEW (train_set = normalized[:80]) ===")
print(f"  Before modification  : normalized[0,0] = {original_value:.6f}")
print(f"  Set train_set[0,0]   = {sentinel}")
print(f"  After  modification  : normalized[0,0] = {normalized[0,0]}")
print(f"  Change reflected?    : {np.isclose(normalized[0,0], sentinel)}")

# Restore
train_set[0, 0] = original_value

# ── Demonstrate COPY behaviour ──────────────────────────
original_copy_val = train_copy[0, 0]
train_copy[0, 0]  = sentinel

print("\n=== COPY (train_copy = normalized[:80].copy()) ===")
print(f"  Before modification  : normalized[0,0] = {normalized[0,0]:.6f}")
print(f"  Set train_copy[0,0]  = {sentinel}")
print(f"  After  modification  : normalized[0,0] = {normalized[0,0]:.6f}")
print(f"  Change reflected?    : {np.isclose(normalized[0,0], sentinel)}")

print("\n→ Conclusion: slices share memory (view); .copy() does not.")