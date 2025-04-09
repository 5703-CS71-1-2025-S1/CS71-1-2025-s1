import numpy as np
from model_objective import evaluate_model_auc

# === 1. Class of particles ===
class Particle:
    def __init__(self, bounds):
        self.dim = len(bounds)
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.zeros(self.dim)
        self.best_position = self.position.copy()
        self.best_score = -np.inf  # 因为我们最大化AUC
        self.score = -np.inf

    def clip_position(self, bounds):
        for i, (low, high) in enumerate(bounds):
            self.position[i] = np.clip(self.position[i], low, high)

# === 2. PSO core logic ===
def run_pso(objective_func, bounds, n_particles=10, n_iterations=5, inertia=0.5, cognitive=1.5, social=1.5):
    swarm = [Particle(bounds) for _ in range(n_particles)]
    gbest_score = -np.inf
    gbest_position = None

    for iter in range(n_iterations):
        print(f"\n=== Iteration {iter+1}/{n_iterations} ===")
        for i, particle in enumerate(swarm):

            hidden = int(particle.position[0])
            lr = particle.position[1]
            dropout = particle.position[2]
            l2 = particle.position[3]

            try:
                score = objective_func(hidden, lr, dropout, l2)
            except Exception as e:
                print(f"Error in evaluating particle {i}: {e}")
                score = 0.0

            particle.score = score
            print(f"Particle {i}: AUC = {score:.4f}, Position = {particle.position}")

            if score > particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()

            if score > gbest_score:
                gbest_score = score
                gbest_position = particle.position.copy()

        for particle in swarm:
            r1, r2 = np.random.rand(particle.dim), np.random.rand(particle.dim)
            cognitive_term = cognitive * r1 * (particle.best_position - particle.position)
            social_term = social * r2 * (gbest_position - particle.position)
            particle.velocity = inertia * particle.velocity + cognitive_term + social_term
            particle.position += particle.velocity
            particle.clip_position(bounds)

    return gbest_position, gbest_score
