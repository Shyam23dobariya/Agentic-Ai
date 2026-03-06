import numpy as np

np.random.seed(42)

NUM_SAMPLES  = 100
NUM_FEATURES = 5

# Each feature gets a unique scale (std) and offset (mean)
scales  = np.array([10,  0.5, 100,  3,  50])
offsets = np.array([50,  1.5, 200,  0, -30])

data = np.random.randn(NUM_SAMPLES, NUM_FEATURES) * scales + offsets

print(f"Dataset generated.")
print(f"Shape  : {data.shape}  →  {NUM_SAMPLES} samples × {NUM_FEATURES} features")
print(f"\nFirst 5 rows:\n{data[:5].round(3)}")
print(f"\nMin per feature : {data.min(axis=0).round(3)}")
print(f"Max per feature : {data.max(axis=0).round(3)}")
