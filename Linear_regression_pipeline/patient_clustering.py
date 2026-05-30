# Python version : 3.11+
# Install        : pip install scikit-learn numpy pandas matplotlib
# Run            : python patient_clustering.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Generate the Patient Dataset (unchanged)
# ═════════════════════════════════════════════════════════════════════════════
np.random.seed(42)

data = {
    'systolic_bp':   np.concatenate([np.random.normal(118, 10, 150),
                                     np.random.normal(142, 12, 150),
                                     np.random.normal(168, 8,  100)]),
    'cholesterol':   np.concatenate([np.random.normal(178, 18, 150),
                                     np.random.normal(222, 22, 150),
                                     np.random.normal(262, 18, 100)]),
    'bmi':           np.concatenate([np.random.normal(21.5, 2, 150),
                                     np.random.normal(27.0, 3, 150),
                                     np.random.normal(33.5, 3, 100)]),
    'glucose_level': np.concatenate([np.random.normal(88,  10, 150),
                                     np.random.normal(112, 14, 150),
                                     np.random.normal(148, 18, 100)]),
    'age':           np.concatenate([np.random.normal(34, 7,  150),
                                     np.random.normal(51, 6,  150),
                                     np.random.normal(63, 5,  100)])
}

df = pd.DataFrame(data)

print("=" * 60)
print("STEP 1 — Dataset Generated")
print("=" * 60)
print(f"  Shape  : {df.shape}  (patients × features)")
print(f"  Features: {list(df.columns)}")
print(df.describe().round(2))

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Scale the Features
# ═════════════════════════════════════════════════════════════════════════════
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df)

print("\n" + "=" * 60)
print("STEP 2 — Features Scaled with StandardScaler")
print("=" * 60)
print("  Mean of scaled data (should be ~0):",
      np.round(X_scaled.mean(axis=0), 4))
print("  Std  of scaled data (should be ~1):",
      np.round(X_scaled.std(axis=0),  4))

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Elbow Method  (K = 1 to 10)
# ═════════════════════════════════════════════════════════════════════════════
wcss = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

print("\n" + "=" * 60)
print("STEP 3 — Elbow Method WCSS Values")
print("=" * 60)
for k, w in zip(k_range, wcss):
    print(f"  K = {k:>2}  →  WCSS = {w:,.2f}")

# Plot
plt.figure(figsize=(9, 5))
plt.plot(list(k_range), wcss, marker='o', linewidth=2,
         color='steelblue', markerfacecolor='tomato', markersize=8)
plt.title("Elbow Method — Optimal K for Patient Clustering", fontsize=14, fontweight='bold')
plt.xlabel("Number of Clusters (K)", fontsize=12)
plt.ylabel("WCSS (Inertia)", fontsize=12)
plt.xticks(list(k_range))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("elbow_plot.png", dpi=150)
plt.show()
print("\n  Elbow plot saved as 'elbow_plot.png'")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Train Final K-Means Model  (optimal K = 3)
# ═════════════════════════════════════════════════════════════════════════════
# The elbow plot shows a clear bend at K=3, matching the 3 underlying
# distributions in the dataset (low-risk, moderate-risk, high-risk).
OPTIMAL_K = 3

final_km = KMeans(n_clusters=OPTIMAL_K, init='k-means++',
                  n_init=10, random_state=42)
final_km.fit(X_scaled)

df['Cluster'] = final_km.labels_

print("\n" + "=" * 60)
print(f"STEP 4 — Final K-Means Model Trained  (K = {OPTIMAL_K})")
print("=" * 60)
print(f"  Inertia (WCSS) : {final_km.inertia_:,.4f}")
cluster_counts = df['Cluster'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    print(f"  Cluster {cluster_id} : {count} patients")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Cluster Summary Table
# ═════════════════════════════════════════════════════════════════════════════
features = ['systolic_bp', 'cholesterol', 'bmi', 'glucose_level', 'age']

summary = (
    df.groupby('Cluster')[features]
    .mean()
    .round(2)
    .rename_axis("Cluster")
)

# Add patient count and a risk label based on feature magnitudes
summary.insert(0, 'Patient_Count', cluster_counts)

# Sort by mean systolic_bp to assign risk labels consistently
bp_order = summary['systolic_bp'].rank().astype(int)
risk_map  = {1: 'Low Risk', 2: 'Moderate Risk', 3: 'High Risk'}
summary['Risk_Label'] = bp_order.map(risk_map)

print("\n" + "=" * 60)
print("STEP 5 — Cluster Summary Table")
print("=" * 60)
print(summary.to_string())

print("\n" + "=" * 60)
print("Pipeline Complete.")
print("=" * 60)