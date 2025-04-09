import matplotlib.pyplot as plt

def plot_auc_trend(meta_history):
    aucs = [entry["auc"] for entry in meta_history]
    iters = [entry["iter"] for entry in meta_history]

    plt.figure(figsize=(8, 4))
    plt.plot(iters, aucs, marker='o', label="Best AUC")
    plt.title("AUC Trend Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Best AUC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_meta_param_trend(meta_history):
    iters = [entry["iter"] for entry in meta_history]
    inertia = [entry["inertia_weight"] for entry in meta_history]
    c1 = [entry["cognitive_coeff"] for entry in meta_history]
    c2 = [entry["social_coeff"] for entry in meta_history]

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax[0].plot(iters, inertia, label="Inertia Weight", marker='o')
    ax[0].set_ylabel("Inertia Weight")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(iters, c1, label="Cognitive Coeff (c1)", marker='o')
    ax[1].plot(iters, c2, label="Social Coeff (c2)", marker='x')
    ax[1].set_ylabel("Learning Factors")
    ax[1].set_xlabel("Iteration")
    ax[1].grid(True)
    ax[1].legend()

    plt.suptitle("GPT-Suggested Meta-Parameter Trend")
    plt.tight_layout()
    plt.show()

def plot_auc_comparison(gpt_results, fixed_results):
    import numpy as np

    def get_stats_by_round(results):
        stats = []
        for round_data in results:
            aucs = [x["auc"] for x in round_data]
            stats.append({
                "best": np.max(aucs),
                "mean": np.mean(aucs),
                "std": np.std(aucs),
            })
        return stats

    gpt_stats = get_stats_by_round(gpt_results)
    fixed_stats = get_stats_by_round(fixed_results)
    rounds = range(1, len(gpt_stats) + 1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(rounds, [s["best"] for s in gpt_stats], label="GPT-Driven PSO", marker='o')
    axs[0].plot(rounds, [s["best"] for s in fixed_stats], label="Fixed PSO", marker='x')
    axs[0].set_title("Best AUC per Iteration")
    axs[0].set_ylabel("Best AUC")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(rounds, [s["mean"] for s in gpt_stats], label="GPT-Driven PSO", marker='o')
    axs[1].plot(rounds, [s["mean"] for s in fixed_stats], label="Fixed PSO", marker='x')
    axs[1].set_title("Average AUC per Iteration")
    axs[1].set_ylabel("Mean AUC")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(rounds, [s["std"] for s in gpt_stats], label="GPT-Driven PSO", marker='o')
    axs[2].plot(rounds, [s["std"] for s in fixed_stats], label="Fixed PSO", marker='x')
    axs[2].set_title("AUC Std Deviation per Iteration")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Std Dev of AUC")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
