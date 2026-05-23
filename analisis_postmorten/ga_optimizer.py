"""LineWise GA optimizer — shared module used by both 05_ga_optimizer.ipynb and
the Streamlit app.

Centralises the mentor constants, the historical lookup tables (throughput and
changeover matrices), the chromosome representation and the genetic algorithm
itself so that the notebook and the app are always solving the same problem
with the same simulator.
"""

from __future__ import annotations

import random
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_loaders import (
    LINES,
    load_all_operations,
    load_diario_hl,
    weekly_demand_from_diario,
)


# ---------------------------------------------------------------------------
# Mentor constants
# ---------------------------------------------------------------------------

HOURS_PER_WEEK: Dict[str, float] = {"14": 110.0, "17": 115.0, "19": 115.0}

PHYSICAL_FORMAT_BY_LINE: Dict[str, set] = {
    "14": {"1/2", "1/3"},
    "17": {"1/3"},
    "19": {"1/2", "1/3", "2/5"},
}

PRIORITY_ORDERS: List[Tuple[str, str]] = [("VI1324MY", "17")]

STARTUP_HOURS: Dict[str, float] = {"14": 1.0, "17": 1.5, "19": 1.5}

PENALTY_INCOMPATIBLE = 10_000.0
PENALTY_CAPACITY_BASE = 100.0
PENALTY_PRIORITY = 500.0


# ---------------------------------------------------------------------------
# Helpers — SKU parsing
# ---------------------------------------------------------------------------

_FORMAT_RE = re.compile(r"(33|50|44)")
_CL_TO_FMT = {33: "1/3", 50: "1/2", 44: "2/5"}


def parse_format(sku: str) -> str:
    m = _FORMAT_RE.search(str(sku).upper())
    cl = int(m.group(1)) if m else 33
    return _CL_TO_FMT[cl]


# ---------------------------------------------------------------------------
# Data context: loads everything the GA needs in one shot.
# ---------------------------------------------------------------------------

Chromosome = Dict[str, List[str]]


@dataclass
class OptimizerContext:
    """Everything the GA + visualisations need, derived from the real files."""
    weekly: pd.DataFrame
    skus: List[str]
    volumes: Dict[str, float]
    sku_format: Dict[str, str]
    eligible: Dict[str, List[str]]
    fallback_skus: List[str]
    throughput: Dict[Tuple[str, str], float]
    sku_global_rate: Dict[str, float]
    plant_mean_rate: float
    changeover: Dict[str, Dict[Tuple[str, str], float]]
    line_mean_co: Dict[str, float]
    hist_pairs: set


def build_context(data_dir: Path) -> OptimizerContext:
    """Load operations + diary and pre-compute every lookup table the GA needs."""
    from post_mortem import PostMortemAnalyzer

    ops = load_all_operations(data_dir)
    df_oee, df_cam, df_mant, df_tiem, df_vol = (
        ops["oee"], ops["cam"], ops["mant"], ops["tiem"], ops["vol"]
    )

    weekly = weekly_demand_from_diario(load_diario_hl(data_dir / "Diario Hl_Planif.xlsx"))
    weekly["original_line"] = weekly["original_tren"].str.split(",").str[0]
    skus = weekly["sku"].tolist()
    volumes = dict(zip(weekly["sku"], weekly["hl_total"]))
    sku_format = {sku: parse_format(sku) for sku in skus}

    # Historical (sku, line) pairs from 2025.
    hist_pairs = set()
    for df in (df_oee, df_vol, df_tiem):
        if "sku" in df.columns and "tren" in df.columns:
            for sku, tren in df[["sku", "tren"]].dropna().itertuples(index=False):
                hist_pairs.add((str(sku), str(tren)))

    # Throughput: real median HL/h per (sku, line).
    hl_per_h = (
        df_vol.merge(df_tiem[["of", "h_tot"]], on="of", how="left")
              .dropna(subset=["sku", "tren", "hl", "h_tot"])
    )
    hl_per_h = hl_per_h[hl_per_h["h_tot"] > 0]
    hl_per_h["rate"] = hl_per_h["hl"] / hl_per_h["h_tot"]
    hl_per_h = hl_per_h[hl_per_h["rate"].between(20, 800)]

    throughput = hl_per_h.groupby(["sku", "tren"])["rate"].median().to_dict()
    sku_global_rate = hl_per_h.groupby("sku")["rate"].median().to_dict()
    plant_mean_rate = float(hl_per_h["rate"].median())

    # Changeover matrix from the post-mortem pipeline.
    pm = PostMortemAnalyzer(
        df_oee=df_oee, df_cambios=df_cam,
        df_mantenimiento=df_mant, df_tiempo=df_tiem, df_volumen=df_vol,
    )
    pm.clean_and_isolate_maintenance()
    transitions = pm.build_transition_matrices()

    changeover: Dict[str, Dict[Tuple[str, str], float]] = {}
    line_mean_co: Dict[str, float] = {}
    for line in LINES:
        mat = transitions.get(line, {}).get("changeover_h")
        line_map: Dict[Tuple[str, str], float] = {}
        if mat is not None and not mat.empty:
            for (prev, nxt), hours in mat.stack(dropna=True).items():
                if pd.notna(hours):
                    line_map[(prev, nxt)] = (
                        float(hours) / 60.0 if hours > 30 else float(hours)
                    )
        changeover[line] = line_map
        line_mean_co[line] = (
            float(np.mean(list(line_map.values()))) if line_map else 1.0
        )

    # Eligibility: physical formats ∩ 2025 history (with diary fallback).
    eligible: Dict[str, List[str]] = {}
    fallback = []
    for sku in skus:
        fmt = sku_format[sku]
        opts = [l for l in LINES
                if fmt in PHYSICAL_FORMAT_BY_LINE[l] and (sku, l) in hist_pairs]
        if not opts:
            opts = [weekly.loc[weekly["sku"] == sku, "original_line"].iloc[0]]
            fallback.append(sku)
        eligible[sku] = opts

    return OptimizerContext(
        weekly=weekly, skus=skus, volumes=volumes, sku_format=sku_format,
        eligible=eligible, fallback_skus=fallback, throughput=throughput,
        sku_global_rate=sku_global_rate, plant_mean_rate=plant_mean_rate,
        changeover=changeover, line_mean_co=line_mean_co, hist_pairs=hist_pairs,
    )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def throughput_rate(ctx: OptimizerContext, sku: str, line: str) -> float:
    rate = ctx.throughput.get((sku, line))
    if rate is None or not np.isfinite(rate):
        rate = ctx.sku_global_rate.get(sku, ctx.plant_mean_rate)
    return float(max(20.0, rate))


def changeover_hours(ctx: OptimizerContext, prev_sku: str, next_sku: str,
                     line: str) -> float:
    if prev_sku == next_sku:
        return 0.0
    val = ctx.changeover[line].get((prev_sku, next_sku))
    if val is None or not np.isfinite(val):
        val = ctx.changeover[line].get((next_sku, prev_sku))
    if val is None or not np.isfinite(val):
        base = ctx.line_mean_co[line]
        if ctx.sku_format.get(prev_sku) != ctx.sku_format.get(next_sku):
            base *= 2.5
        val = base
    return float(max(0.0, min(12.0, val)))


def simulate_line(ctx: OptimizerContext, line: str,
                  sequence: List[str]) -> Dict[str, float]:
    if not sequence:
        return {"total": 0.0, "prod": 0.0, "changeover": 0.0, "startup": 0.0}
    prod = sum(ctx.volumes[s] / throughput_rate(ctx, s, line) for s in sequence)
    co = sum(changeover_hours(ctx, sequence[i], sequence[i + 1], line)
             for i in range(len(sequence) - 1))
    startup = STARTUP_HOURS[line]
    return {"prod": prod, "changeover": co, "startup": startup,
            "total": prod + co + startup}


def breakdown(ctx: OptimizerContext, individual: Chromosome) -> Dict[str, Dict[str, float]]:
    return {l: simulate_line(ctx, l, individual.get(l, [])) for l in LINES}


def evaluate_schedule(ctx: OptimizerContext, individual: Chromosome) -> Tuple[float]:
    flat = [s for line in LINES for s in individual.get(line, [])]
    if len(flat) != len(ctx.skus) or set(flat) != set(ctx.skus):
        return (PENALTY_INCOMPATIBLE * 10,)

    total = 0.0
    penalty = 0.0
    for line in LINES:
        seq = individual.get(line, [])
        for sku in seq:
            if ctx.sku_format.get(sku) not in PHYSICAL_FORMAT_BY_LINE[line]:
                penalty += PENALTY_INCOMPATIBLE
            elif (sku, line) not in ctx.hist_pairs and sku not in ctx.fallback_skus:
                penalty += PENALTY_INCOMPATIBLE / 2

        sim = simulate_line(ctx, line, seq)
        total += sim["total"]

        if sim["total"] > HOURS_PER_WEEK[line]:
            over = sim["total"] - HOURS_PER_WEEK[line]
            penalty += PENALTY_CAPACITY_BASE * (np.exp(over / 5.0) - 1.0)

        for sku, urgent_line in PRIORITY_ORDERS:
            if urgent_line == line and sku in seq:
                pos = seq.index(sku)
                cutoff = max(0, int(0.25 * len(seq)))
                if pos > cutoff:
                    penalty += PENALTY_PRIORITY * ((pos - cutoff) / max(1, len(seq)))

    return (total + penalty,)


# ---------------------------------------------------------------------------
# Baseline derived from the planner's diary
# ---------------------------------------------------------------------------

def baseline_individual(ctx: OptimizerContext) -> Chromosome:
    ind: Chromosome = {line: [] for line in LINES}
    for _, row in ctx.weekly.sort_values(["first_fecha", "row_order"]).iterrows():
        ind[row.original_line].append(row.sku)
    return ind


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def smart_init(ctx: OptimizerContext) -> Chromosome:
    ind: Chromosome = {line: [] for line in LINES}
    placed = set()
    for sku, line in PRIORITY_ORDERS:
        if sku in ctx.skus and line in ctx.eligible[sku]:
            ind[line].append(sku)
            placed.add(sku)
    for sku in ctx.skus:
        if sku in placed:
            continue
        ind[random.choice(ctx.eligible[sku])].append(sku)
    for line in LINES:
        head = [s for s, l in PRIORITY_ORDERS if l == line and s in ind[line]]
        rest = [s for s in ind[line] if s not in head]
        random.shuffle(rest)
        ind[line] = head + rest
    return ind


def crossover(ctx: OptimizerContext, p1: Chromosome, p2: Chromosome) -> Chromosome:
    child: Chromosome = {line: [] for line in LINES}
    p1_assign = {s: l for l in LINES for s in p1[l]}
    p2_assign = {s: l for l in LINES for s in p2[l]}

    for sku in ctx.skus:
        a, b = p1_assign.get(sku), p2_assign.get(sku)
        chosen = a if random.random() < 0.5 else b
        if chosen not in ctx.eligible[sku]:
            chosen = (a if a in ctx.eligible[sku]
                      else b if b in ctx.eligible[sku]
                      else random.choice(ctx.eligible[sku]))
        child[chosen].append(sku)

    for line in LINES:
        members = child[line]
        if len(members) <= 2:
            continue
        p1_order = [s for s in p1[line] if s in members]
        p2_order = [s for s in p2[line] if s in members]
        for s in members:
            if s not in p1_order:
                p1_order.append(s)
            if s not in p2_order:
                p2_order.append(s)
        n = len(members)
        a, b = sorted(random.sample(range(n), 2))
        segment = p1_order[a:b + 1]
        rest = [s for s in p2_order if s not in segment]
        child[line] = rest[:a] + segment + rest[a:]
    return child


def mutate_swap(ind: Chromosome) -> Chromosome:
    candidates = [l for l in LINES if len(ind[l]) >= 2]
    if not candidates:
        return ind
    line = random.choice(candidates)
    seq = ind[line]
    i, j = random.sample(range(len(seq)), 2)
    seq[i], seq[j] = seq[j], seq[i]
    return ind


def mutate_migrate(ctx: OptimizerContext, ind: Chromosome) -> Chromosome:
    bd = breakdown(ctx, ind)
    over = sorted(LINES, key=lambda l: bd[l]["total"] - HOURS_PER_WEEK[l], reverse=True)
    under = sorted(LINES, key=lambda l: bd[l]["total"] - HOURS_PER_WEEK[l])
    src = over[0]
    if not ind[src]:
        return ind
    movable = [s for s in ind[src] if len(ctx.eligible[s]) > 1]
    random.shuffle(movable)
    for sku in movable:
        cands = [l for l in ctx.eligible[sku] if l != src and l in under[:2]]
        if not cands:
            cands = [l for l in ctx.eligible[sku] if l != src]
        if not cands:
            continue
        dst = cands[0]
        ind[src].remove(sku)
        ind[dst].insert(random.randint(0, len(ind[dst])), sku)
        return ind
    return ind


def mutate(ctx: OptimizerContext, ind: Chromosome) -> Chromosome:
    ind = deepcopy(ind)
    if random.random() < 0.5:
        return mutate_swap(ind)
    return mutate_migrate(ctx, ind)


def _tournament(pop_fit, k: int = 3) -> Chromosome:
    return min(random.sample(pop_fit, k), key=lambda x: x[1])[0]


def evolve(
    ctx: OptimizerContext,
    *,
    pop_size: int = 60,
    n_gen: int = 150,
    elitism: int = 4,
    seed: int = 42,
    on_generation=None,
) -> Tuple[Chromosome, List[Dict[str, float]]]:
    """Run the GA. ``on_generation(gen, best_fit, median_fit)`` is called once
    per generation so callers (e.g. the Streamlit progress bar) can react."""
    random.seed(seed)
    np.random.seed(seed)

    population = [smart_init(ctx) for _ in range(pop_size)]
    fitnesses = [evaluate_schedule(ctx, ind)[0] for ind in population]
    best_ind = deepcopy(population[int(np.argmin(fitnesses))])
    best_fit = min(fitnesses)
    history: List[Dict[str, float]] = []

    for gen in range(n_gen):
        pop_fit = sorted(zip(population, fitnesses), key=lambda x: x[1])
        new_pop = [deepcopy(ind) for ind, _ in pop_fit[:elitism]]
        while len(new_pop) < pop_size:
            p1 = _tournament(pop_fit)
            p2 = _tournament(pop_fit)
            child = crossover(ctx, p1, p2)
            if random.random() < 0.85:
                child = mutate(ctx, child)
            new_pop.append(child)
        population = new_pop
        fitnesses = [evaluate_schedule(ctx, ind)[0] for ind in population]
        gen_best = min(fitnesses)
        if gen_best < best_fit:
            best_fit = gen_best
            best_ind = deepcopy(population[int(np.argmin(fitnesses))])
        median = float(np.median(fitnesses))
        history.append({"gen": gen, "best": best_fit, "median": median})
        if on_generation is not None:
            on_generation(gen, best_fit, median)

    return best_ind, history


# ---------------------------------------------------------------------------
# Gantt-row helper (shared by notebook and Streamlit)
# ---------------------------------------------------------------------------

def schedule_to_gantt(ctx: OptimizerContext,
                      individual: Chromosome) -> pd.DataFrame:
    rows = []
    for line in LINES:
        cursor = STARTUP_HOURS[line]
        rows.append({"line": f"L{line}", "task": "ARRANQUE", "sku": "_arr",
                     "start_h": 0.0, "end_h": STARTUP_HOURS[line],
                     "duration_h": STARTUP_HOURS[line],
                     "type": "startup", "format": "", "hl": 0.0,
                     "rate_hl_per_h": 0.0})
        prev = None
        for sku in individual[line]:
            if prev is not None:
                co = changeover_hours(ctx, prev, sku, line)
                if co > 0:
                    rows.append({"line": f"L{line}", "task": f"CO {prev}→{sku}",
                                 "sku": sku, "start_h": cursor, "end_h": cursor + co,
                                 "duration_h": co, "type": "changeover",
                                 "format": ctx.sku_format.get(sku, ""),
                                 "hl": 0.0, "rate_hl_per_h": 0.0})
                    cursor += co
            rate = throughput_rate(ctx, sku, line)
            prod_h = ctx.volumes[sku] / rate
            rows.append({"line": f"L{line}", "task": sku, "sku": sku,
                         "start_h": cursor, "end_h": cursor + prod_h,
                         "duration_h": prod_h, "type": "production",
                         "format": ctx.sku_format.get(sku, ""),
                         "hl": ctx.volumes[sku], "rate_hl_per_h": rate})
            cursor += prod_h
            prev = sku
    return pd.DataFrame(rows)
