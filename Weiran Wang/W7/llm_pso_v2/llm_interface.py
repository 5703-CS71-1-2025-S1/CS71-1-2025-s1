import openai
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv("OPENAI_API_KEY.env")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Build prompt with top-performing particles ===
def build_prompt(particles):
    prompt = (
        "Below are the top-performing particles from a particle swarm optimization process. "
        "Each line contains 4 hyperparameters: hidden layer size, learning rate, dropout rate, and L2 regularization, "
        "followed by the current performance score (1 - AUC).\n"
    )
    for p in particles:
        hidden, lr, dropout, l2 = p.position
        score = 1 - p.score
        prompt += f"{hidden:.2f},{lr:.6f},{dropout:.3f},{l2:.6f},{score:.6f}\n"
    prompt += (
        "\nBased on these examples, suggest the same number of new 4D hyperparameter vectors (hidden, lr, dropout, l2) "
        "that are likely to further reduce the cost (1 - AUC). Only output values separated by commas, one set per line."
    )
    return prompt

# === Call GPT model and parse suggestions ===
def call_llm(prompt, n=2, model="gpt-4o"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a hyperparameter optimization assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        reply = response.choices[0].message.content
        suggestions = []
        for line in reply.strip().split("\n"):
            values = [float(x.strip()) for x in line.strip().split(",")]
            if len(values) == 4:
                suggestions.append(np.array(values))
            if len(suggestions) >= n:
                break
        return suggestions
    except Exception:
        return []
