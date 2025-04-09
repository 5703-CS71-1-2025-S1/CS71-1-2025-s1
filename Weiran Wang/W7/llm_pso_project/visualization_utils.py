import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

def plot_auc_trend(auc_history, save_path="logs/auc_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(auc_history) + 1), auc_history, marker='o')
    plt.title("AUC Convergence Trend")
    plt.xlabel("Iteration")
    plt.ylabel("Validation AUC")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    # Save AUC history to .npy for future comparison
    if "auc_gpt" in save_path:
        np.save("logs/auc_gpt.npy", np.array(auc_history))
    elif "auc_pso" in save_path:
        np.save("logs/auc_pso.npy", np.array(auc_history))

def save_best_result(params, auc, save_path="logs/best_result.txt"):
    with open(save_path, "w") as f:
        f.write(f"Best AUC: {auc:.4f}\n")
        f.write("Best Parameters:\n")
        f.write(str(params))

def plot_auc_comparison(auc_gpt, auc_pso, save_path="logs/auc_comparison.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(auc_gpt) + 1), auc_gpt, marker='o', label="GPT + PSO")
    plt.plot(range(1, len(auc_pso) + 1), auc_pso, marker='x', label="PSO Only")
    plt.title("AUC Convergence Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Validation AUC")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Example usage:
if __name__ == '__main__':
    plot_auc_trend([0.71, 0.73, 0.76, 0.77, 0.78], save_path="logs/auc_curve_gpt.png")
    save_best_result([64, 0.001, 0.2, 1e-4], 0.78, save_path="logs/best_result_gpt.txt")
    plot_auc_comparison([0.71, 0.75, 0.78], [0.70, 0.73, 0.75])
