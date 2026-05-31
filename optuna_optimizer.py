import optuna
from ga_optimizer import OptimizerContext, evolve, evaluate_schedule

def run_optuna_study(ctx: OptimizerContext, pop_size: int = 60, n_gen: int = 100, n_trials: int = 50, seed: int = 42):
    def objective(trial: optuna.Trial) -> float:
        # Fixed hyperparameters to avoid excessive computation times
        
        # 3 GA parameters to tune
        mut_prob = trial.suggest_float("mut_prob", 0.1, 1.0)
        elitism = trial.suggest_int("elitism", 1, 10)
        tour_size = trial.suggest_int("tour_size", 2, 7)
        
        # Fixed Penalty weights
        ctx.w_cap = 100.0
        ctx.w_inc = 10000.0
        ctx.w_urg = 500.0

        ctx.mut_prob = mut_prob
        ctx.tour_size = tour_size

        best_ind, _ = evolve(
            ctx,
            pop_size=pop_size,
            n_gen=n_gen,
            elitism=elitism,
            seed=seed
        )
        
        # Return the actual hours + hard penalties (maybe a separate standard eval)
        fitness = evaluate_schedule(ctx, best_ind)[0]
        return fitness

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study
