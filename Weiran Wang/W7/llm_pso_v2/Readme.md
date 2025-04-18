# 📌 LLM-Enhanced PSO for MLP Hyperparameter Optimization (v2)

This project implements a GPT-assisted Particle Swarm Optimization framework to optimize the hyperparameters of a Multi-Layer Perceptron (MLP) model using the UCI Credit Default dataset.

---

## ✅ Key Improvements over v1

1. **Baseline Phase**: Run a full PSO-only phase to get a benchmark `test_score`.
2. **Conditional LLM Trigger**: Only call GPT if LLM-PSO score underperforms baseline.
3. **Early Stopping**: Stop iterations early if no AUC improvement after several rounds.
4. **Richer Prompt**: Full 4D hyperparameter + score history sent to GPT.
5. **Full Position Replacement**: GPT returns entire `[hidden, lr, dropout, l2]` vectors.
6. **Numpy Tracking**: Save `auc_pso.npy` and `auc_llm.npy` for trend comparison.

---

## 📁 File Overview

| File | Description |
|------|-------------|
| `main_controller_v2.py` | Main program controlling both PSO-only and LLM-PSO runs |
| `main_pso_only.py` | Baseline: run pure PSO optimization without GPT |
| `llm_interface.py` | Prompt builder + GPT caller for generating suggestions |
| `pso_core_v2.py` | PSO particle class and velocity update logic (with decode) |
| `model_objective.py` | Contains MLP evaluation logic based on AUC |
| `visualization_utils.py` | Utility for plotting AUC and saving best results |
| `compare_auc_v2.py` | Plots convergence trends for PSO vs LLM-PSO |
| `boxplot_compare_v2.py` | Runs multiple trials and creates performance boxplots |
| `logs/` | Contains `.npy` data, `.png` images and result files |

---

## 🚀 How to Run (in Jupyter or Terminal)

### ✅ Baseline (PSO-only)
```bash
python main_pso_only.py
```

### ✅ LLM-enhanced PSO
```bash
python main_controller_v2.py
```

### ✅ Compare single-run trends
```bash
python compare_auc_v2.py
```

### ✅ BoxPlot over 3 runs
```bash
python boxplot_compare_v2.py
```

---

## 📊 Output Logs
- Best AUC values
- Saved `.npy` AUC histories
- Visual comparison of convergence trends and performance distribution

---

## 🧠 Notes
- You can change `N_ITER`, `N_PARTICLES`, and prompt logic in the controller.
- Prompt can be extended to include velocity and gbest history for further experiments.

---

Happy optimizing 🎯
