import numpy as np
from pso_core import run_pso, Particle
from model_objective import evaluate_model_auc
from llm_interface import build_prompt, call_llm
from visualization_utils import plot_auc_trend, save_best_result

# Hyperparameter bounds: hidden_size, learning_rate, dropout, l2
BOUNDS = [
    (16, 256),       # hidden_size
    (1e-5, 1e-1),     # learning_rate
    (0.0, 0.5),       # dropout
    (1e-6, 1e-2)      # l2_regularization
]

N_PARTICLES = 10
N_ITER = 10
GPT_REPLACE_EVERY = 3  # Call GPT every N iterations
N_REPLACE = 2           # Replace the worst N particles

# === Main Control Loop ===
def run_llm_pso():
    print("Initializing swarm...")
    swarm = [Particle(BOUNDS) for _ in range(N_PARTICLES)]
    global_best_score = -np.inf
    global_best_position = None
    auc_history = []

    for iter in range(N_ITER):
        print(f"\n=== Iteration {iter+1}/{N_ITER} ===")
        scores = []
        for i, p in enumerate(swarm):
            hidden = int(p.position[0])
            lr = p.position[1]
            dropout = p.position[2]
            l2 = p.position[3]
            score = evaluate_model_auc(hidden, lr, dropout, l2)
            p.score = score
            scores.append(score)

            if score > p.best_score:
                p.best_score = score
                p.best_position = p.position.copy()

            if score > global_best_score:
                global_best_score = score
                global_best_position = p.position.copy()

            print(f"Particle {i}: AUC={score:.4f}, Pos={p.position}")

        auc_history.append(global_best_score)

        # === GPT replacement of worst particles ===
        if (iter + 1) % GPT_REPLACE_EVERY == 0:
            print("Calling GPT to improve worst particles...")
            sorted_particles = sorted(swarm, key=lambda x: x.score)
            worst_particles = sorted_particles[:N_REPLACE]
            best_particles = sorted_particles[-N_REPLACE:]
            prompt = build_prompt(
                particles=[p.position[:2] for p in best_particles],
                scores=[p.score for p in best_particles],
                velocities=[p.velocity[:2] for p in best_particles]
            )
            suggestions = call_llm(prompt)
            for i, new_pos in enumerate(suggestions):
                wp = worst_particles[i % N_REPLACE]
                wp.position[:2] = new_pos  # Replace first two dimensions
                wp.velocity[:2] = np.random.uniform(-1, 1, 2)  # Random initial velocity
                print(f"Replaced particle {i}: New pos={wp.position}")

        # === PSO position update ===
        for p in swarm:
            r1, r2 = np.random.rand(p.dim), np.random.rand(p.dim)
            cognitive = 1.5 * r1 * (p.best_position - p.position)
            social = 1.5 * r2 * (global_best_position - p.position)
            p.velocity = 0.5 * p.velocity + cognitive + social
            p.position += p.velocity
            p.clip_position(BOUNDS)

    print("\nOptimization complete.")
    print(f"Best AUC = {global_best_score:.4f}")
    print(f"Best Parameters = {global_best_position}")

    # Save results
    plot_auc_trend(auc_history, save_path="logs/auc_curve_gpt.png", npy_path="logs/auc_gpt.npy")
    save_best_result(global_best_position, global_best_score)

if __name__ == '__main__':
    run_llm_pso()
