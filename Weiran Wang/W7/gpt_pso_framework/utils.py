# utils.py

import numpy as np

def generate_fixed_particles(swarmsize, param_bounds, seed=42):
    """
    Generate reproducible initial particles for PSO search.

    Args:
        swarmsize (int): Number of particles to generate
        param_bounds (list of tuples): [(min1, max1), (min2, max2), ...]
        seed (int): Random seed for reproducibility

    Returns:
        List[List[float]]: List of particles, each particle is a list of parameters
    """
    np.random.seed(seed)
    particles = []

    for _ in range(swarmsize):
        particle = [np.random.uniform(low, high) for (low, high) in param_bounds]
        particles.append(particle)

    return particles
