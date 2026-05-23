from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd

from ga_optimizer import (
    HOURS_PER_WEEK,
    LINES,
    PRIORITY_ORDERS,
    STARTUP_HOURS,
    OptimizerContext,
    breakdown,
    changeover_hours,
    simulate_line,
    throughput_rate,
)

MAX_REPAIR_PASSES = 6
INFEASIBLE_PENALTY = 1e9
_P90_CACHE: Dict[Tuple[str, str], float] = {}


def decode_trial(ctx: OptimizerContext, trial: optuna.Trial) -> Dict[str, List[str]]:
    assigns: Dict[str, str] = {}
    orders: Dict[str, float] = {}
    for sku in ctx.skus:
        opts = ctx.eligible[sku]
        line = opts[0] if len(opts) == 1 else trial.suggest_categorical(f"line_{sku}", opts)
        assigns[sku] = line
        orders[sku] = trial.suggest_float(f"order_{sku}", 0.0, 1.0)

    schedule: Dict[str, List[str]] = {l: [] for l in LINES}
    for sku, line in assigns.items():
        schedule[line].append(sku)
    for line in LINES:
        schedule[line].sort(key=lambda s: orders[s])
        urgent = [s for s, ul in PRIORITY_ORDERS if ul == line and s in schedule[line]]
        rest = [s for s in schedule[line] if s not in urgent]
        schedule[line] = urgent + rest
    return schedule


def repair_capacity(ctx: OptimizerContext,
                    schedule: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], bool]:
    sched = {l: list(seq) for l, seq in schedule.items()}
    for _ in range(MAX_REPAIR_PASSES):
        bd = breakdown(ctx, sched)
        over = [l for l in LINES if bd[l]["total"] > HOURS_PER_WEEK[l]]
        if not over:
            return sched, True
        moved_any = False
        for src in sorted(over, key=lambda l: bd[l]["total"] - HOURS_PER_WEEK[l], reverse=True):
            candidates = sorted(
                sched[src],
                key=lambda s: ctx.volumes[s] / throughput_rate(ctx, s, src),
                reverse=True,
            )
            for sku in candidates:
                alt = [l for l in ctx.eligible[sku]
                       if l != src and bd[l]["total"] < HOURS_PER_WEEK[l] - 1.0]
                if not alt:
                    continue
                dst = min(alt, key=lambda l: bd[l]["total"])
                sched[src].remove(sku)
                sched[dst].append(sku)
                urgent = [s for s, ul in PRIORITY_ORDERS if ul == dst and s in sched[dst]]
                rest = [s for s in sched[dst] if s not in urgent]
                sched[dst] = urgent + rest
                moved_any = True
                break
            if moved_any:
                break
        if not moved_any:
            break
    bd = breakdown(ctx, sched)
    return sched, all(bd[l]["total"] <= HOURS_PER_WEEK[l] for l in LINES)


def constraints_of_trial(trial: optuna.trial.FrozenTrial):
    return trial.user_attrs.get("constraints", (0.0, 0.0, 0.0))


def make_objective(ctx: OptimizerContext) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        raw = decode_trial(ctx, trial)
        sched, feasible = repair_capacity(ctx, raw)
        bd = breakdown(ctx, sched)
        overshoots = [max(0.0, bd[l]["total"] - HOURS_PER_WEEK[l]) for l in LINES]
        total = sum(bd[l]["total"] for l in LINES)

        trial.set_user_attr("schedule", sched)
        trial.set_user_attr("breakdown", bd)
        trial.set_user_attr("constraints", tuple(overshoots))
        trial.set_user_attr("feasible", feasible)

        if not feasible:
            return INFEASIBLE_PENALTY + sum(overshoots) * 1000.0
        return total
    return objective


def run_study(
    ctx: OptimizerContext,
    *,
    n_trials: int = 600,
    seed: int = 42,
    on_trial: Callable[[int, float, bool], None] | None = None,
) -> Dict:
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        constraints_func=constraints_of_trial,
        n_startup_trials=min(30, n_trials // 4),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                 study_name="linewise_optuna")

    completed = {"n": 0}

    def _cb(study, trial):
        completed["n"] += 1
        if on_trial is not None:
            val = trial.value if trial.value is not None else float("inf")
            on_trial(completed["n"], val, trial.user_attrs.get("feasible", False))

    t0 = time.time()
    study.optimize(make_objective(ctx), n_trials=n_trials, callbacks=[_cb],
                    show_progress_bar=False, gc_after_trial=True)
    elapsed = time.time() - t0

    best = study.best_trial
    return {
        "study": study,
        "best_trial": best,
        "schedule": best.user_attrs["schedule"],
        "breakdown": best.user_attrs["breakdown"],
        "fitness": best.value,
        "feasible": best.user_attrs.get("feasible", False),
        "elapsed_s": elapsed,
    }


def load_historical_clean(clean_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(clean_dir / "historical_weeks.csv")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    return df


def weekly_sequence(historical: pd.DataFrame, week: str) -> Dict[str, pd.DataFrame]:
    wk = historical[historical["week"] == week]
    out: Dict[str, pd.DataFrame] = {}
    for line in LINES:
        sub = wk[wk["line"] == line].copy()
        sub = sub.sort_values("fecha").reset_index(drop=True)
        sub["dur_h"] = sub["h_tot"].fillna(0.0).clip(lower=0.0)
        sub["start_h"] = sub["dur_h"].cumsum() - sub["dur_h"]
        out[line] = sub[["of", "fecha", "sku", "hl", "h_tot", "oee", "dur_h", "start_h"]]
    return out


def _ideal_rate(ctx: OptimizerContext, historical: pd.DataFrame,
                sku: str, line: str) -> float:
    key = (sku, line)
    if key in _P90_CACHE:
        return _P90_CACHE[key]
    sub = historical[(historical["sku"] == sku) & (historical["line"] == line)]
    sub = sub.dropna(subset=["hl", "h_tot"])
    sub = sub[sub["h_tot"] > 0]
    if sub.empty:
        rate = throughput_rate(ctx, sku, line) * 1.4
    else:
        rate = float(np.quantile(sub["hl"] / sub["h_tot"], 0.90))
        rate = max(rate, throughput_rate(ctx, sku, line))
    _P90_CACHE[key] = rate
    return rate


def theoretical_hours_for_week(
    ctx: OptimizerContext,
    week_seqs: Dict[str, pd.DataFrame],
    historical: pd.DataFrame | None = None,
) -> Dict[str, Dict[str, float]]:
    res = {}
    for line in LINES:
        df = week_seqs[line]
        if df.empty:
            res[line] = {"theoretical": 0.0, "real": 0.0, "simulator": 0.0,
                         "n_of": 0, "changeover_theo": 0.0, "prod_theo": 0.0,
                         "prod_sim": 0.0, "changeover_sim": 0.0}
            continue
        prod_theo = 0.0
        prod_sim = 0.0
        for _, row in df.iterrows():
            hl = row["hl"] if pd.notna(row["hl"]) else 0.0
            sku = row["sku"]
            if historical is not None:
                prod_theo += hl / _ideal_rate(ctx, historical, sku, line)
            else:
                prod_theo += hl / (throughput_rate(ctx, sku, line) * 1.4)
            prod_sim += hl / throughput_rate(ctx, sku, line)
        co_lookup = 0.0
        for i in range(len(df) - 1):
            co_lookup += changeover_hours(ctx, df.iloc[i]["sku"],
                                           df.iloc[i + 1]["sku"], line)
        theoretical = STARTUP_HOURS[line] + prod_theo
        simulator = STARTUP_HOURS[line] + prod_sim + co_lookup
        real = float(df["h_tot"].fillna(0.0).sum())
        res[line] = {"theoretical": theoretical, "real": real,
                     "simulator": simulator,
                     "prod_theo": prod_theo, "prod_sim": prod_sim,
                     "changeover_theo": 0.0, "changeover_sim": co_lookup,
                     "n_of": int(len(df))}
    return res
