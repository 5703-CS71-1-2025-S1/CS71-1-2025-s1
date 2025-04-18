import matplotlib.pyplot as plt
import numpy as np
import os

def plot_auc_trend(auc_history, save_path="logs/auc_curve.png", npy_path="logs/auc_history.npy"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(auc_history)+1), auc_history, marker='o')
    plt.title("AUC Convergence Trend")
    plt.xlabel("Iteration")
    plt.ylabel("Validation AUC")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    np.save(npy_path, np.array(auc_history))
    plt.close()

def save_best_result(position, auc_score, save_path="logs/best_result.txt"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("Best AUC: {:.4f}\n".format(auc_score))
        f.write("Best Parameters:\n")
        f.write("Hidden: {:.2f}\n".format(position[0]))
        f.write("Learning Rate: {:.6f}\n".format(position[1]))
        f.write("Dropout: {:.3f}\n".format(position[2]))
        f.write("L2: {:.6f}\n".format(position[3]))