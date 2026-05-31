# Bayesian Optimization of DAMM Scheduling GA

This repository contains the code for our final group project on Bayesian Optimization.
We adapted the original DAMM hackathon problem to use **Optuna (Tree-structured Parzen Estimator, TPE)** for hyperparameter tuning of a Genetic Algorithm.

## Contents
- **ga_optimizer.py**: The Genetic Algorithm with hyperparameters exposed for tuning (`mut_prob`, `elitism`, `tour_size`, `w_cap`, `w_inc`, `w_urg`, `pop_size`, `n_gen`).
- **optuna_optimizer.py**: The Optuna study setup that optimizes the GA parameters.
- **app.py**: A Streamlit dashboard where you can see the Bayesian network of changeovers AND run the Optuna Hyperparameter Tuning dynamically on the GA.

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```
