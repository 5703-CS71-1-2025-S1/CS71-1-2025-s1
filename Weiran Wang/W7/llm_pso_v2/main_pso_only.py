import numpy as np
from pso_core_v2 import Particle
from model_objective import evaluate_model_auc
from visualization_utils import plot_auc_trend, save_best_result
import os

BOUNDS = [
    (16, 256),
    (1e-5, 1e-1),
    (0.0, 0.5),
    (1e-6, 1e-2)
]
N_PARTICLES = 10
N_ITER = 10

def run_pso():
    swarm = [Particle(BOUNDS) for _ in range(N_PARTICLES)]
    global_best_score = -np.inf
    global_best_position = None
    auc_history = []
    best_iter = 0

    for iter in range(N_ITER):
        for p in swarm:
            score = evaluate_model_auc(*p.decode())
            p.score = score
            if score > p.best_score:
                p.best_score = score
                p.best_position = p.position.copy()
            if score > global_best_score:
                global_best_score = score
                global_best_position = p.position.copy()
                best_iter = iter
        auc_history.append(global_best_score)
        for p in swarm:
            p.update_velocity(global_best_position)
            p.update_position(BOUNDS)

    os.makedirs("logs", exist_ok=True)
    print(f"[PSO-Only] Best AUC = {global_best_score:.4f}")
    print(f"Best Iteration = {best_iter + 1}")
    plot_auc_trend(auc_history, save_path="logs/auc_curve_pso.png", npy_path="logs/auc_pso.npy")
    save_best_result(global_best_position, global_best_score, save_path="logs/best_result_pso.txt")

def main():
    run_pso()

if __name__ == '__main__':
    main()