import numpy as np
from pso_core import Particle
from model_objective import evaluate_model_auc
from visualization_utils import plot_auc_trend, save_best_result

# Hyperparameter bounds
BOUNDS = [
    (16, 256),       # hidden_size
    (1e-5, 1e-1),     # learning_rate
    (0.0, 0.5),       # dropout
    (1e-6, 1e-2)      # l2_regularization
]

N_PARTICLES = 10
N_ITER = 10

# === PSO-only optimization loop ===
def run_pso_only():
    print("Initializing PSO swarm (no GPT)...")
    swarm = [Particle(BOUNDS) for _ in range(N_PARTICLES)]
    global_best_score = -np.inf
    global_best_position = None
    auc_history = []

    for iter in range(N_ITER):
        print(f"\n=== Iteration {iter+1}/{N_ITER} ===")
        for i, p in enumerate(swarm):
            hidden = int(p.position[0])
            lr = p.position[1]
            dropout = p.position[2]
            l2 = p.position[3]
            score = evaluate_model_auc(hidden, lr, dropout, l2)
            p.score = score

            if score > p.best_score:
                p.best_score = score
                p.best_position = p.position.copy()

            if score > global_best_score:
                global_best_score = score
                global_best_position = p.position.copy()

            print(f"Particle {i}: AUC={score:.4f}, Pos={p.position}")

        auc_history.append(global_best_score)

        # Update particle positions
        for p in swarm:
            r1, r2 = np.random.rand(p.dim), np.random.rand(p.dim)
            cognitive = 1.5 * r1 * (p.best_position - p.position)
            social = 1.5 * r2 * (global_best_position - p.position)
            p.velocity = 0.5 * p.velocity + cognitive + social
            p.position += p.velocity
            p.clip_position(BOUNDS)

    print("\nOptimization complete (PSO only).")
    print(f"Best AUC = {global_best_score:.4f}")
    print(f"Best Parameters = {global_best_position}")

    plot_auc_trend(auc_history, save_path="logs/auc_curve_pso.png", npy_path="logs/auc_pso.npy")
    save_best_result(global_best_position, global_best_score, save_path="logs/best_result_pso.txt")

if __name__ == '__main__':
    run_pso_only()
