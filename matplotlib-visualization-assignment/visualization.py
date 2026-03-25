import numpy as np
import matplotlib.pyplot as plt
 
# ─────────────────────────────────────────
# 1. Create a list of 10 epochs (1 to 10)
# ─────────────────────────────────────────
epochs = list(range(1, 11))
 
# ─────────────────────────────────────────
# 2. Generate synthetic training loss values
# ─────────────────────────────────────────
np.random.seed(42)
loss_values = np.array([1.0 / (i + np.random.uniform(0.5, 1.5)) for i in range(1, 11)])
 
# ─────────────────────────────────────────
# 3a. Line Plot — Loss vs Epoch
# ─────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_values, marker='o', color='steelblue',
         linewidth=2, markersize=7, label='Training Loss')
plt.title('Training Loss vs Epoch (Line Plot)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('line_plot.png', dpi=150)
plt.show()
print(" Line Plot saved as line_plot.png")
 
# ─────────────────────────────────────────
# 3b. Scatter Plot — Epoch vs Loss
# ─────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(epochs, loss_values, color='tomato', s=100, zorder=5, label='Loss per Epoch')
plt.title('Epoch vs Loss (Scatter Plot)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('scatter_plot.png', dpi=150)
plt.show()
print(" Scatter Plot saved as scatter_plot.png")
 
# ─────────────────────────────────────────
# 4. Bar Chart — Model Accuracy Comparison
# ─────────────────────────────────────────
models    = ['Model A', 'Model B', 'Model C']
accuracy  = [0.85, 0.90, 0.88]
colors    = ['cornflowerblue', 'mediumseagreen', 'sandybrown']
 
plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracy, color=colors, width=0.5, edgecolor='black')
 
# Add value labels on top of each bar
for bar, acc in zip(bars, accuracy):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.005,
             f'{acc:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
 
plt.title('Model Accuracy Comparison (Bar Chart)', fontsize=14, fontweight='bold')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0.80, 0.95)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('bar_chart.png', dpi=150)
plt.show()
print("Bar Chart saved as bar_chart.png")
 
