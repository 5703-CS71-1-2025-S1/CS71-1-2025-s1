# GPT + PSO Optimization Framework

This project implements a hybrid optimization framework using **Particle Swarm Optimization (PSO)** enhanced by **GPT-4o** to tune hyperparameters of a deep learning model (MLP) on the **UCI Credit Card Default Dataset**.

---

## 🚀 Features
- Deep learning model: Multi-Layer Perceptron (MLP)
- Evaluation metric: AUC (Area Under ROC Curve)
- PSO to explore parameter space
- GPT-4o to suggest new particle positions (every N iterations)
- Visual tracking of AUC convergence and comparison using `.npy` logs

---

## 📁 Project Structure
```bash
├── model_objective.py       # MLP + AUC objective function
├── pso_core.py              # PSO implementation and particle structure
├── llm_interface.py         # GPT-4o prompt construction and suggestions
├── main_controller.py       # Main optimization loop (PSO + GPT)
├── main_pso_only.py         # Baseline PSO-only loop (no GPT)
├── compare_auc.py           # Script to visualize AUC convergence comparison
├── visualization_utils.py   # AUC plotting + results export
├── OPENAI_API_KEY.env       # Your local OpenAI API key (not committed)
├── default of credit card clients.xls  # UCI dataset
├── logs/                    # Stores plots and result files (.png, .txt, .npy)
```

---

## ⚙️ Requirements
```bash
pip install openai python-dotenv scikit-learn torch matplotlib
```

---

## 🔑 Setup OpenAI API
1. Create a `.env` file named `OPENAI_API_KEY.env`
2. Add the following line:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 🧪 Run Optimization
### ▶️ GPT + PSO Hybrid:
```bash
python main_controller.py
```
Generates:
- `logs/auc_curve_gpt.png`
- `logs/auc_gpt.npy`

### ▶️ PSO Only (baseline):
```bash
python main_pso_only.py
```
Generates:
- `logs/auc_curve_pso.png`
- `logs/auc_pso.npy`

---

## 📊 Compare AUC Performance (using saved `.npy` data)
After running both versions:
```bash
python compare_auc.py
```
Output saved to:
```
logs/auc_comparison.png
```

---

## 📂 Output Summary
- `logs/auc_curve_gpt.png`: AUC trend (GPT+PSO)
- `logs/auc_curve_pso.png`: AUC trend (PSO only)
- `logs/auc_comparison.png`: Comparison plot
- `logs/best_result_gpt.txt` / `best_result_pso.txt`: Final best results
- `logs/auc_gpt.npy` / `logs/auc_pso.npy`: AUC history arrays

---

## 📚 Reference
Based on the method proposed in:
**"Large Language Model Enhanced Particle Swarm Optimization for Hyperparameter Optimization of Deep Learning Models"**

---

Enjoy optimization with LLMs 🤖 + 🐦!
