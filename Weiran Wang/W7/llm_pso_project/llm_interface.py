from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from typing import List

# === Load API key ===
load_dotenv("OPENAI_API_KEY.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === 1. prompt ===
def build_prompt(particles: List[np.ndarray], scores: List[float], velocities: List[np.ndarray]) -> str:
    prompt = (
        "Below is the string showing the best number of neurons as the first entry and best number of layers as the "
        "second entry of the DL model for {Npop} particles with their corresponding cost as the fifth entry, "
        "while dynamically updating the number of neurons and layers to reduce the cost for the same model using "
        "Particle Swarm Optimization. The third and fourth entries are the neurons velocities and layers velocities, respectively.\n"
    )
    particle_lines = []
    for p, v, s in zip(particles, velocities, scores):
        line = f"{p[0]:.4f},{p[1]:.6f},{v[0]:.4f},{v[1]:.4f},{1 - s:.6f}"
        particle_lines.append(line)
    prompt += "\n".join(particle_lines)
    prompt += ("\n\nGive me exactly {Npop} more number of neurons and layers for the same model in order to reduce "
               "the cost further. Your response must be exactly in the same format as input and must contain only values. "
               "Your response must not contain the cost values.")
    return prompt.replace("{Npop}", str(len(particles)))

# === 2. Call OpenAI to get particle suggestions  ===
def call_llm(prompt: str, model="gpt-4o") -> List[np.ndarray]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI particle optimizer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        reply = response.choices[0].message.content
        suggestions = []
        for line in reply.strip().split("\n"):
            values = [float(x) for x in line.strip().split(",")[:2]]
            suggestions.append(np.array(values))
        return suggestions
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return []


if __name__ == '__main__':
    particles = [np.array([64.0, 0.001]), np.array([128.0, 0.01])]
    velocities = [np.array([1.5, 0.01]), np.array([-0.5, 0.005])]
    scores = [0.75, 0.72]
    prompt = build_prompt(particles, scores, velocities)
    print("Prompt:\n", prompt)
    suggestions = call_llm(prompt)
    print("LLM Suggestions:", suggestions)
