import numpy as np
from model_objective import evaluate_model_auc

# === 1. Class of particles ===
class Particle:
    def __init__(self, bounds):
        self.dim = len(bounds)
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.zeros(self.dim)
        self.best_position = self.position.copy()
        self.best_score = -np.inf  
        self.score = -np.inf

    def decode(self):
        hidden = int(self.position[0])
        lr = self.position[1]
        dropout = self.position[2]
        l2 = self.position[3]
        return hidden, lr, dropout, l2

    def clip_position(self, bounds):
        for i, (low, high) in enumerate(bounds):
            self.position[i] = np.clip(self.position[i], low, high)

    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        self.clip_position(bounds)

# === 2. PSO core logic (standalone test) ===
def run_pso(objective_func, bounds, n_particles=10, n_iterations=5):
    swarm = [Particle(bounds) for _ in range(n_particles)]
    gbest_score = -np.inf
    gbest_position = None

    for iter in range(n_iterations):
        print(f"=== Iteration {iter+1}/{n_iterations} ===")
        for i, particle in enumerate(swarm):
            hidden, lr, dropout, l2 = particle.decode()
            try:
                score = objective_func(hidden, lr, dropout, l2)
            except Exception as e:
                print(f"Error in evaluating particle {i}: {e}")
                score = 0.0

            particle.score = score
            print(f"Particle {i}: AUC = {score:.4f}, Pos = {particle.position}")

            if score > particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()

            if score > gbest_score:
                gbest_score = score
                gbest_position = particle.position.copy()

        for particle in swarm:
            particle.update_velocity(gbest_position)
            particle.update_position(bounds)

    return gbest_position, gbest_score