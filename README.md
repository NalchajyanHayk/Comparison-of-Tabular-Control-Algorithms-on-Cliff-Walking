## Comparison of Tabular Control Algorithms on CliffWalking-v1

This project implements and compares six **tabular reinforcement learning control algorithms** from scratch on the **CliffWalking-v1** environment:

- Monte Carlo Control
- SARSA
- Q-Learning
- Expected SARSA
- 4-step SARSA
- 4-step Q-Learning

The assignment requires all algorithms to be implemented from scratch, use tabular \(Q(s,a)\), apply \(\epsilon\)-greedy action selection where appropriate, and compare them fairly on the same environment with multiple random seeds, reward tracking, learning curves, and final performance comparison. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

---

# 1. Project Structure

```text
rl_cliffwalking/
│
├── main.py
├── config.py
├── logger_config.py
├── env_manager.py
├── policies.py
├── utils.py
├── evaluator.py
├── plotting.py
├── experiment_manager.py
│
├── algorithms/
│   ├── __init__.py
│   ├── base.py
│   ├── monte_carlo.py
│   ├── sarsa.py
│   ├── q_learning.py
│   ├── expected_sarsa.py
│   ├── n_step_sarsa.py
│   └── n_step_q_learning.py
│
├── outputs/
│   ├── learning_curves.png
│   ├── final_comparison.png
│   ├── per_seed_results.csv
│   ├── summary_results.csv
│   ├── training_rewards.csv
│   ├── hyperparameters.json
│   └── run.log
│
├── requirements.txt
└── README.md
```

# 2. How to Run the Project

### 2.1 How to build and run with Docker

#### Build image
```bash id="docker_run_101"
docker build -t rl-cliffwalking .
```
#### Build container
```bash
docker run --rm -v $(pwd)/outputs:/app/outputs rl-cliffwalking
```

#### On Windows PowerShell, use:
```bash
docker run --rm -v ${PWD}/outputs:/app/outputs rl-cliffwalking
```

### 2.2 Run the Project using virtual environment

#### On macOS / Linux
```bash id="local_run_001"
python3 -m venv venv
source venv/bin/activate
```

#### On Windows PowerShell
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

#### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Run the project
```bash
python main.py
```

#3. Output Files

After execution, the following files are generated inside the `outputs/` folder:

- **learning_curves.png** — learning curves averaged across seeds
- **final_comparison.png** — final performance comparison
- **per_seed_results.csv** — results for each algorithm and seed
- **summary_results.csv** — aggregated results across seeds
- **training_rewards.csv** — episodic reward history for all runs
- **hyperparameters.json** — full experiment configuration
- **run.log** — execution logs produced with Python logging

# 4. Experiment Setup

The experiment was run with the following hyperparameters:

- **Environment**: CliffWalking-v1
- **Episodes**: 800
- **Learning rate (α)**: 0.5
- **Discount factor (γ)**: 1.0
- **Exploration rate (ε)**: 0.1
- **n for multi-step methods**: 4
- **Random seeds**: 5 seeds → [0, 1, 2, 3, 4]
- **Evaluation episodes**: 100
- **Smoothing window**: 20
- **Maximum steps per episode**: 500

All algorithms were trained on the same environment with the same experimental setup to ensure a fair comparison.

# 5. Project Goal

The goal of this project is to compare different tabular control methods in reinforcement learning and understand how algorithm design affects:

- learning speed
- stability
- policy quality
- and behavior near the cliff in the Cliff Walking task

This comparison is especially useful for understanding:

- on-policy vs off-policy learning
- one-step vs multi-step methods
- and sample efficiency vs stability trade-offs

---

# 6. Results

#### 6.1 Final Performance Summary

| Algorithm              | Avg Train Reward (Last 100) | Std     | Avg Greedy Eval Return | Std    |
|------------------------|-----------------------------|---------|------------------------|--------|
| **Q-Learning**         | -47.62                      | 45.68   | **-13.0**              | 0.00   |
| **Expected SARSA**     | -21.76                      | 40.88   | **-15.0**              | 0.00   |
| **SARSA**              | -25.83                      | 101.64  | **-17.0**              | 0.00   |
| **4-step Q-Learning**  | -42.66                      | 218.82  | -210.6                 | 264.19 |
| **4-step SARSA**       | -52.39                      | 223.02  | -211.0                 | 263.83 |
| **Monte Carlo**        | -499.04                     | 251.20  | -500.0                 | 0.00   |

#### 6.2 Interpretation of Results

**Best-performing algorithms**

The strongest final results were obtained by:

1. **Q-Learning** — average greedy evaluation return **-13.0**
2. **Expected SARSA** — **-15.0**
3. **SARSA** — **-17.0**

These three algorithms learned strong policies and converged to much better final behavior than the remaining methods.

**Multi-step methods**

Both 4-step Q-Learning and 4-step SARSA performed much worse in final evaluation:

- 4-step Q-Learning: **-210.6**
- 4-step SARSA: **-211.0**

Their very large standard deviations also show that they were highly unstable across seeds.

**Monte Carlo**

Monte Carlo Control performed the worst overall:

- Average last-100 training reward: **-499.04**
- Final greedy evaluation return: **-500.0**

This indicates that it did not learn a useful policy for this setup and remained far behind all TD-based methods.

# 7. Learning Curve Discussion

The learning curves show several important patterns:

- Q-Learning, Expected SARSA, and SARSA quickly moved toward much higher rewards and stabilized near the top of the plot.
- Monte Carlo stayed dramatically below the other methods throughout training, confirming poor learning efficiency in this environment.
- 4-step SARSA and 4-step Q-Learning showed much higher instability and weaker final policies compared with the one-step methods.

**Conclusion from plots**: The one-step TD methods clearly outperformed both Monte Carlo and the multi-step implementations in this experiment.

# 8. Conclusions

This experiment shows that for the current CliffWalking-v1 setup:

- **Q-Learning** achieved the best final greedy policy.
- **Expected SARSA** and **SARSA** also performed very well and were close to Q-Learning.
- **4-step methods** were unstable and significantly worse in final performance.
- **Monte Carlo Control** was the weakest method by a large margin.

**Overall**, the results suggest that in this implementation and hyperparameter setting, **one-step temporal-difference methods** were the most effective and reliable.

# 9. Notes

- All algorithms were implemented from scratch.
- No external RL libraries such as Stable-Baselines, RLlib, or CleanRL were used.
- Logging was implemented using Python’s built-in logging library.
- The project was designed in a modular OOP style so that each algorithm can be trained and evaluated separately.

---
**Prepared by:** Hayk Nalchajyan
**Date:** April 2026