import numpy as np

# ── 1. Seed ────────────────────────────────────────────
np.random.seed(42)

# ── 2. Generate dataset ────────────────────────────────
data = np.random.randn(100, 5) * [10, 0.5, 100, 3, 50] \
     + [50, 1.5, 200, 0, -30]

# ── 3. Per-feature statistics ─────────────────────────
feature_mean = data.mean(axis=0)
feature_std  = data.std(axis=0)

# ── 4. Normalize ──────────────────────────────────────
normalized = (data - feature_mean) / feature_std

# ── 5. Train / test split ─────────────────────────────
split_index = int(len(normalized) * 0.80)
train_set   = normalized[:split_index]
test_set    = normalized[split_index:]

# ── 6. View demonstration ─────────────────────────────
original_value  = normalized[0, 0]
train_set[0, 0] = 9999.0
view_confirmed  = np.isclose(normalized[0, 0], 9999.0)
train_set[0, 0] = original_value   # restore

# ── 7. Summary output ─────────────────────────────────
print("=" * 50)
print("  NumPy Pipeline — Full Summary")
print("=" * 50)
print(f"Original data shape  : {data.shape}")
print(f"Mean shape           : {feature_mean.shape}")
print(f"Std  shape           : {feature_std.shape}")
print(f"Normalized shape     : {normalized.shape}")
print(f"Training set shape   : {train_set.shape}")
print(f"Test     set shape   : {test_set.shape}")
print()
print(f"View behaviour confirmed : {view_confirmed}")
print("Note: Modifying the slice affected the original array.")
print("      Use .copy() to prevent this when needed.")
print("=" * 50)