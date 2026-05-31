import optuna
from ga_optimizer import OptimizerContext, evolve, evaluate_schedule

def run_optuna_study(ctx: OptimizerContext, n_trials: int = 50, seed: int = 42):
    def objective(trial: optuna.Trial) -> float:
        pop_size = trial.suggest_int("pop_size", 20, 100, step=10)
        n_gen = trial.suggest_int("n_gen", 50, 200, step=50)
        mut_prob = trial.suggest_float("mut_prob", 0.1, 1.0)
        elitism = trial.suggest_int("elitism", 1, 10)
        tour_size = trial.suggest_int("tour_size", 2, 7)
        w_cap = trial.suggest_float("w_cap", 10.0, 500.0, log=True)
        w_inc = trial.suggest_float("w_inc", 1000.0, 20000.0, log=True)
        w_urg = trial.suggest_float("w_urg", 100.0, 1000.0, log=True)

        ctx.mut_prob = mut_prob
        ctx.tour_size = tour_size
        ctx.w_cap = w_cap
        ctx.w_inc = w_inc
        ctx.w_urg = w_urg

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
