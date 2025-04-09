# gpt_prompt.py

import json
from openai import OpenAI

client = OpenAI() 

def build_prompt_for_iteration(history, round_id, current_meta=None, improvement=False, last_meta=None, last_auc=None, new_auc=None):
    prompt = f"""You are helping optimize the PSO algorithm's meta-parameters.
This is iteration {round_id+1} of PSO optimization.

Below are the top 5 performing particle configurations (MLP hyperparameters and AUC scores):
"""
    top = sorted(history, key=lambda x: -x["auc"])[:5]
    for i, row in enumerate(top):
        prompt += f"#{i+1}: hidden={row['hidden']}, lr={row['lr']:.5f}, dropout={row['dropout']:.2f}, l2={row['l2']:.5f}, AUC={row['auc']:.4f}\n"

    if round_id > 0:
        prompt += f"""
In the previous iteration, GPT suggested the following PSO meta-parameters:
- inertia_weight = {last_meta['inertia_weight']:.2f}
- cognitive_coeff (c1) = {last_meta['cognitive_coeff']:.2f}
- social_coeff (c2) = {last_meta['social_coeff']:.2f}

This configuration resulted in validation AUC changing from {last_auc:.4f} to {new_auc:.4f}.
"""

        if improvement:
            prompt += "Since this was an improvement, you may fine-tune around these values.\n"
        else:
            prompt += "Since this led to a performance drop, you should explore significantly different values.\n"

    prompt += f"""
The current PSO meta-parameters in use are:
- inertia_weight = {current_meta['inertia_weight']:.2f}
- cognitive_coeff (c1) = {current_meta['cognitive_coeff']:.2f}
- social_coeff (c2) = {current_meta['social_coeff']:.2f}

Please recommend the next PSO meta-parameters to try:
- inertia_weight (float between 0.1 and 1.2)
- cognitive_coeff (float between 0.5 and 2.5)
- social_coeff (float between 0.5 and 2.5)

Please include a brief explanation for your suggestions.

Respond ONLY in this JSON format:
{{
  "inertia_weight": float,
  "cognitive_coeff": float,
  "social_coeff": float,
  "reason": "..."
}}
"""
    return prompt
