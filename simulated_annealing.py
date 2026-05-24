from __future__ import annotations

import random
import time
from copy import deepcopy
from typing import Callable, Dict, List, Tuple

import numpy as np
from ga_optimizer import (
    Chromosome,
    OptimizerContext,
    evaluate_schedule,
    mutate_swap,
    mutate_migrate,
    smart_init,
)


def simulated_annealing(
    ctx: OptimizerContext,
    *,
    max_iter: int = 15_000,
    restart_every: int = 5_000,
    seed: int = 42,
    on_iter: Callable | None = None,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    current = smart_init(ctx)
    current_fit = evaluate_schedule(ctx, current)[0]
    best = deepcopy(current)
    best_fit = current_fit

    # Temperature schedule based on initial fitness
    T0 = max(10.0, current_fit * 0.15)
    T_min = 0.01
    alpha = (T_min / T0) ** (1.0 / max_iter)
    T = T0

    history: List[Dict[str, float]] = []
    accepted = 0

    for it in range(max_iter):
        # Choose move: 60% swap, 40% migrate
        if random.random() < 0.6:
            neighbor = mutate_swap(deepcopy(current))
        else:
            neighbor = mutate_migrate(ctx, deepcopy(current))

        n_fit = evaluate_schedule(ctx, neighbor)[0]
        delta = n_fit - current_fit

        if delta < 0 or random.random() < np.exp(-delta / max(T, 0.001)):
            current = neighbor
            current_fit = n_fit
            accepted += 1
            if current_fit < best_fit:
                best_fit = current_fit
                best = deepcopy(current)

        # Cooling
        T *= alpha

        # Restart if stuck
        if it > 0 and it % restart_every == 0:
            if accepted < restart_every * 0.02:
                current = smart_init(ctx)
                current_fit = evaluate_schedule(ctx, current)[0]
            accepted = 0
            T = T0 * 0.5

        if it % 200 == 0:
            history.append({"iter": it, "best": best_fit, "current": current_fit, "T": T})
            if on_iter is not None:
                on_iter(it, best_fit, current_fit)

    return {"schedule": best, "best_fit": best_fit, "history": history}


def run_sa(ctx: OptimizerContext, *, n_iter: int = 15_000, seed: int = 42,
           on_trial: Callable | None = None) -> Dict:
    t0 = time.time()

    def _cb(it, best, cur):
        if on_trial is not None:
            on_trial(it, best, it > 0)

    res = simulated_annealing(ctx, max_iter=n_iter, seed=seed, on_iter=_cb)
    res["elapsed_s"] = time.time() - t0
    return res
