# creator.py
import openai  
import os
import json
import re
from dotenv import load_dotenv

# 修正：OPENAT_API_KEY → OPENAI_API_KEY
load_dotenv("OPENAI_API_KEY.env")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_prompt(history_records, metric="AUC"):
    prompt = """You are an expert in machine learning optimization. Your task is to suggest the next hyperparameter combination for training a multilayer perceptron (MLP) model on a credit scoring dataset. The goal is to maximize the validation (metric).

Here are the recent experimental results:"""
    for i, rec in enumerate(history_records):
        prompt += f"\nRun {i+1}: AUC={rec['auc']:.4f}, Params={rec['params']}"
    prompt += """\n\nBased on the above, suggest a new hyperparameter configuration with the following fields:
- hidden_dim (int, 32-128)
- learning_rate_init (float, 1e-4 ~ 1e-1)
- dropout (float, 0.0 ~ 0.5)
- l2 (float, 1e-5 ~ 1e-1)
- batch_size (int, one of 32, 64, 128)

Respond ONLY with a valid JSON object. No explanation or labels."""
    return prompt

def ask_gpt_for_params(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[{"role": "user", "content": prompt}]  
    )
    reply = response.choices[0].message.content.strip()  
    try:

        result = json.loads(reply)
        return result
    except Exception:
        raise ValueError("Failed to parse GPT response. Content was:\n" + reply)

if __name__ == "__main__":
    # 修正：history 中的 "12" → "l2"
    history = [
        {"auc": 0.7412, "params": {"hidden_dim": 64, "learning_rate_init": 0.001, "dropout": 0.2, "l2": 1e-4, "batch_size": 64}},
        {"auc": 0.7550, "params": {"hidden_dim": 64, "learning_rate_init": 0.001, "dropout": 0.3, "l2": 1e-4, "batch_size": 64}}
    ]
    prompt = build_prompt(history)
    suggestion = ask_gpt_for_params(prompt)
    print(json.dumps(suggestion, indent=2)) 