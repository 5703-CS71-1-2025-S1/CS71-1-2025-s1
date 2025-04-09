# evaluator.py

def evaluate_particle_set(particles, eval_fn):
    results = []

    for i, position in enumerate(particles):
        auc = eval_fn(position)

        results.append({
            "id": i,
            "position": position,
            "auc": auc
        })

    return results
