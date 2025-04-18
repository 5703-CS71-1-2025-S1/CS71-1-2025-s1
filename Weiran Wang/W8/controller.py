# controller.py
from data_loader import load_credit_data
from executor import train_and_evaluate
from creator import build_prompt, ask_gpt_for_params
import json

# Load dataset
(X_train, y_train), (X_val, y_val), (X_test, y_test), _ = load_credit_data()

# Initialize history
history = []

# Number of optimization rounds
n_rounds = 3

for i in range(n_rounds):
    print(f"\n========== Round {i+1} ==========")

    # Build prompt and get GPT suggestion
    prompt = build_prompt(history)
    suggestion = ask_gpt_for_params(prompt)

    # Run model with suggested parameters
    auc, _ = train_and_evaluate(
        X_train, y_train, X_val, y_val,
        hidden_dim=suggestion['hidden_dim'],
        lr=suggestion['learning_rate_init'],
        dropout=suggestion['dropout'],
        l2=suggestion['l2'],
        batch_size=suggestion['batch_size']
    )

    # Log result
    result = {"auc": auc, "params": suggestion, "source": "gpt"}
    history.append(result)

    print(f"Suggested Params: {json.dumps(suggestion)}")
    print(f"Validation AUC: {auc:.4f}")

# Save GPT results only
with open("experiment_history.json", "w") as f:
    json.dump(history, f, indent=2)

# Baseline test for comparison
print("\n========== Baseline Test ==========")
baseline_params = {
    "hidden_dim": 64,
    "learning_rate_init": 0.01,
    "dropout": 0.3,
    "l2": 1e-4,
    "batch_size": 64
}
baseline_auc, _ = train_and_evaluate(
    X_train, y_train, X_val, y_val,
    hidden_dim=baseline_params['hidden_dim'],
    lr=baseline_params['learning_rate_init'],
    dropout=baseline_params['dropout'],
    l2=baseline_params['l2'],
    batch_size=baseline_params['batch_size']
)

baseline_result = {"auc": baseline_auc, "params": baseline_params, "source": "baseline"}
with open("baseline_result.json", "w") as f:
    json.dump(baseline_result, f, indent=2)

print(f"Baseline Params: {json.dumps(baseline_params)}")
print(f"Baseline AUC: {baseline_auc:.4f}")
print("\nAll rounds complete. Results saved to experiment_history.json and baseline_result.json")