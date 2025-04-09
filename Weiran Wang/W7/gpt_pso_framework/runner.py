import copy
from gpt_prompt import build_prompt_for_iteration, call_gpt_for_iteration
from pso_core import update_particles
from evaluator import evaluate_particle_set

def run_dynamic_pso_loop(initial_particles, param_bounds, eval_fn, max_iter=10):
    swarmsize = len(initial_particles)
    particle_dim = len(initial_particles[0])

    particles = copy.deepcopy(initial_particles)
    gbest = None
    gbest_score = -float("inf")

    # Initial meta-parameters
    inertia_weight = 0.8
    cognitive_coeff = 1.5
    social_coeff = 1.0

    meta_history = []
    all_iterations = []
    last_meta = None
    last_auc = None

    for iteration in range(max_iter):
        print(f"\n[Iter {iteration+1}] Running PSO with w={inertia_weight}, c1={cognitive_coeff}, c2={social_coeff}")

        # Step 1: Evaluate all particles
        iteration_result = evaluate_particle_set(particles, eval_fn)
        all_iterations.append(iteration_result)

        best_particle = max(iteration_result, key=lambda x: x["auc"])
        print(f"Best AUC this iteration: {best_particle['auc']:.4f}")

        should_update = False
        if best_particle["auc"] > gbest_score + 1e-4:
            gbest = copy.deepcopy(best_particle)
            gbest_score = best_particle["auc"]
            should_update = True
        else:
            print("No improvement in AUC.")

        # Step 2: Ask GPT for new meta-parameters
prompt = build_prompt_for_iteration(
    iteration_result,
    iteration,
    current_meta={
        "inertia_weight": inertia_weight,
        "cognitive_coeff": cognitive_coeff,
        "social_coeff": social_coeff
    },
    improvement=should_update,
    last_meta=last_meta,
    last_auc=last_auc,
    new_auc=best_particle["auc"]
)
        new_meta = call_gpt_for_iteration(prompt, iteration)
        print(f"GPT Reason: {new_meta['reason']}")

        # === ✅ Key modification here ===
        if iteration < 3 or should_update:
            inertia_weight = float(new_meta["inertia_weight"])
            cognitive_coeff = float(new_meta["cognitive_coeff"])
            social_coeff = float(new_meta["social_coeff"])
            last_meta = new_meta 
            last_auc = best_particle["auc"]
            print("Using new meta-parameters from GPT.")
        else:
            print("Retaining previous meta-parameters.")

        # Step 3: Store and update particle positions
        meta_history.append({
            "iter": iteration + 1,
            "inertia_weight": inertia_weight,
            "cognitive_coeff": cognitive_coeff,
            "social_coeff": social_coeff,
            "auc": best_particle["auc"],
            "reason": new_meta["reason"]
        })

        particles = update_particles(
            particles, param_bounds, best_particle["position"],
            inertia_weight, cognitive_coeff, social_coeff
        )

    return all_iterations, meta_history, gbest

def run_fixed_pso_loop(initial_particles, param_bounds, eval_fn, max_iter=10):
    import copy

    swarmsize = len(initial_particles)
    particle_dim = len(initial_particles[0])
    particles = copy.deepcopy(initial_particles)

    inertia_weight = 0.8
    cognitive_coeff = 1.5
    social_coeff = 1.0

    all_iterations = []
    meta_history = []
    gbest = None
    gbest_score = -float("inf")

    for iteration in range(max_iter):
        print(f"\n[Fixed PSO] Iteration {iteration + 1}")
        print(f"Running with fixed meta-parameters: w={inertia_weight}, c1={cognitive_coeff}, c2={social_coeff}")

        iteration_result = evaluate_particle_set(particles, eval_fn)
        all_iterations.append(iteration_result)

        best_particle = max(iteration_result, key=lambda x: x["auc"])
        print(f"Best AUC this iteration: {best_particle['auc']:.4f}")

        if best_particle["auc"] > gbest_score + 1e-4:
            gbest = copy.deepcopy(best_particle)
            gbest_score = best_particle["auc"]

        meta_history.append({
            "iter": iteration + 1,
            "inertia_weight": inertia_weight,
            "cognitive_coeff": cognitive_coeff,
            "social_coeff": social_coeff,
            "auc": best_particle["auc"],
            "reason": "Fixed PSO baseline (no GPT feedback)"
        })

        from pso_core import update_particles
        particles = update_particles(
            particles, param_bounds, best_particle["position"],
            inertia_weight, cognitive_coeff, social_coeff
        )

    return all_iterations, meta_history, gbest

