# visualization_utils.py
import json
import matplotlib.pyplot as plt


def plot_auc_history(gpt_file="experiment_history.json", baseline_file="baseline_result.json"):
    with open(gpt_file, "r") as f:
        gpt_history = json.load(f)

    try:
        with open(baseline_file, "r") as f:
            baseline_entries = json.load(f)
            if not isinstance(baseline_entries, list):
                baseline_entries = [baseline_entries]
    except FileNotFoundError:
        baseline_entries = []

    # Parse GPT and baseline
    gpt_aucs = [round(entry["auc"], 4) for entry in gpt_history]
    gpt_rounds = list(range(1, len(gpt_aucs) + 1))

    baseline_aucs = [round(entry["auc"], 4) for entry in baseline_entries]

    plt.figure(figsize=(8, 5))

    # Plot GPT line
    plt.plot(gpt_rounds, gpt_aucs, color="gray", linestyle="--", linewidth=1, label="GPT Line")
    plt.scatter(gpt_rounds, gpt_aucs, c="blue", s=100, edgecolors="black", label="GPT")

    # Plot baseline separately
    if baseline_aucs:
        baseline_x = [max(gpt_rounds) + 1 + i for i in range(len(baseline_aucs))]
        plt.scatter(baseline_x, baseline_aucs, c="red", s=100, edgecolors="black", label="Baseline")
        for r, auc in zip(baseline_x, baseline_aucs):
            plt.text(r, auc, f"{auc:.4f}", ha='center', va='bottom')

    # Annotate GPT points
    for r, auc in zip(gpt_rounds, gpt_aucs):
        plt.text(r, auc, f"{auc:.4f}", ha='center', va='bottom')

    plt.title("AUC over Optimization Rounds (GPT vs Baseline)")
    plt.xlabel("Round")
    plt.ylabel("Validation AUC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_auc_history()
