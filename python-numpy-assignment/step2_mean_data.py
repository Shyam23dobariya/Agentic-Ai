import numpy as np

data = np.array([10, 20, 30, 40, 50, 60, 70, 80])

# Vectorized statistics
mean = np.mean(data)
std  = np.std(data)

print("=" * 40)
print("  Step 2 — Mean & Standard Deviation")
print("=" * 40)
print(f"Original data     : {data}")
print(f"Mean              : {mean}")
print(f"Standard deviation: {std:.2f}")
print(f"Min value         : {np.min(data)}")
print(f"Max value         : {np.max(data)}")