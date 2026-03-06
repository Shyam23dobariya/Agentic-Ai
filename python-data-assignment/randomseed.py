import numpy as np

np.random.seed(42)

print("Random seed set to 42.")
print("Every numpy random call after this will produce the same values.")

# Quick proof
sample = np.random.randn(3)
print(f"Sample random values (always the same): {sample.round(6)}")