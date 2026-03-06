import numpy as np

np.random.seed(42)

data = np.random.randn(100, 5) * [10, 0.5, 100, 3, 50] \
     + [50, 1.5, 200, 0, -30]

feature_mean = data.mean(axis=0)
feature_std  = data.std(axis=0)
normalized   = (data - feature_mean) / feature_std

split_index = int(len(normalized) * 0.80)   # 80

train_set = normalized[:split_index]         # rows 0 – 79
test_set  = normalized[split_index:]         # rows 80 – 99

print(f"Total samples       : {len(normalized)}")
print(f"Split index         : {split_index}  (80 %)")
print(f"\nTraining set shape  : {train_set.shape}")
print(f"Test     set shape  : {test_set.shape}")
print(f"\nTrain set — first row : {train_set[0].round(4)}")
print(f"Test  set — first row : {test_set[0].round(4)}")