import numpy as np
import matplotlib.pyplot as plt

def load_auc_data(path):
    try:
        return np.load(path)
    except:
        return None

def plot_auc_comparison(auc_pso, auc_llm, save_path="logs/auc_comparison_v2.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(auc_pso)+1), auc_pso, marker='o', label="PSO Only")
    plt.plot(range(1, len(auc_llm)+1), auc_llm, marker='s', label="LLM + PSO")
    plt.title("AUC Convergence Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Validation AUC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    auc_pso = load_auc_data("logs/auc_pso.npy")
    auc_llm = load_auc_data("logs/auc_llm.npy")
    if auc_pso is not None and auc_llm is not None:
        plot_auc_comparison(auc_pso, auc_llm)

if __name__ == '__main__':
    main()