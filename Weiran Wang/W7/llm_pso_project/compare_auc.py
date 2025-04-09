import numpy as np
from visualization_utils import plot_auc_comparison

# Load AUC histories
auc_gpt = np.load("logs/auc_gpt.npy")
auc_pso = np.load("logs/auc_pso.npy")

# Plot comparison
plot_auc_comparison(auc_gpt, auc_pso, save_path="logs/auc_comparison.png")

print("AUC comparison chart saved to logs/auc_comparison.png")
