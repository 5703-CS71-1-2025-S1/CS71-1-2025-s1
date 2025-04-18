import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

N_RUNS = 3
LLM_PATH = "logs/box_llm_iter.npy"
PSO_PATH = "logs/box_pso_iter.npy"

def batch_run(script_name):
    iters = []
    for _ in range(N_RUNS):
        result = subprocess.run(["python", script_name], capture_output=True, text=True)
        lines = result.stdout.split("\n")
        for line in reversed(lines):
            if "Best Iteration" in line:
                iter_num = int(line.strip().split()[-1])
                iters.append(iter_num)
                break
    return np.array(iters)

def plot_boxplot(data1, data2, labels, save_path="logs/box_iter_v2.png"):
    plt.figure(figsize=(7, 5))
    plt.boxplot([data1, data2], labels=labels, patch_artist=True, showmeans=True)
    plt.ylabel("Iterations to Best AUC")
    plt.title("Convergence Speed: LLM+PSO vs PSO Only")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    os.makedirs("logs", exist_ok=True)
    iters_pso = batch_run("main_pso_only.py")
    np.save(PSO_PATH, iters_pso)
    iters_llm = batch_run("main_controller_v2.py")
    np.save(LLM_PATH, iters_llm)
    plot_boxplot(iters_pso, iters_llm, labels=["PSO", "LLM+PSO"])

if __name__ == '__main__':
    main()
