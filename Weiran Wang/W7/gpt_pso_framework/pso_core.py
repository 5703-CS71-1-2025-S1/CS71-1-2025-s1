# pso_core.py

import numpy as np

def update_particles(particles, param_bounds, global_best_position, w, c1, c2):
    """
    Perform PSO particle update step using current global best.

    Args:
        particles (list of lists): Current particle positions
        param_bounds (list of tuples): [(min1, max1), ...]
        global_best_position (list): Best position found so far
        w, c1, c2: PSO meta-parameters

    Returns:
        new_particles (list of lists): Updated particle positions
    """
    new_particles = []
    for particle in particles:
        new_position = []

        for i in range(len(particle)):
            r1 = np.random.rand()
            r2 = np.random.rand()

            inertia = w * particle[i]
            cognitive = c1 * r1 * (particle[i] - particle[i])  # optional: personal best
            social = c2 * r2 * (global_best_position[i] - particle[i])

            updated = inertia + cognitive + social

            # clip to bounds
            min_val, max_val = param_bounds[i]
            updated = np.clip(updated, min_val, max_val)

            new_position.append(updated)

        new_particles.append(new_position)

    return new_particles
