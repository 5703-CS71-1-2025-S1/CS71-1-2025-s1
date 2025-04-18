import numpy as np
from pso_core_v2 import Particle
from model_objective import evaluate_model_auc
from llm_interface import build_prompt, call_llm
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
MAX_REPLACE = 3
PATIENCE = 5
THRESHOLD = 0.0005

def run_baseline_pso():
    swarm = [Particle(BOUNDS) for _ in range(N_PARTICLES)]
    global_best_score = -np.inf
    global_best_position = None
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
        for p in swarm:
            p.update_velocity(global_best_position)
            p.update_position(BOUNDS)

    print(f"[Baseline PSO] Best AUC = {global_best_score:.4f}")
    print(f"Best Iteration = {best_iter + 1}")
    return global_best_score

def run_llm_pso(test_score):
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

        if iter > PATIENCE:
            recent = auc_history[-PATIENCE:]
            if max(recent) - min(recent) < THRESHOLD:
                break

        if global_best_score < test_score:
            best_particles = sorted(swarm, key=lambda p: p.score, reverse=True)[:MAX_REPLACE]
            worst_particles = sorted(swarm, key=lambda p: p.score)[:MAX_REPLACE]
            prompt = build_prompt(best_particles)
            suggestions = call_llm(prompt, n=MAX_REPLACE)
            for wp, new_pos in zip(worst_particles, suggestions):
                wp.position[:] = new_pos

        for p in swarm:
            p.update_velocity(global_best_position)
            p.update_position(BOUNDS)

    print(f"[LLM+PSO] Best AUC = {global_best_score:.4f}")
    print(f"Best Iteration = {best_iter + 1}")
    plot_auc_trend(auc_history, save_path="logs/auc_curve_llm.png", npy_path="logs/auc_llm.npy")
    save_best_result(global_best_position, global_best_score, save_path="logs/best_result_llm.txt")

def main():
    test_score = run_baseline_pso()
    run_llm_pso(test_score)

if __name__ == '__main__':
    main()
