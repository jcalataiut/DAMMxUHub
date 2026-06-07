from __future__ import annotations

import hashlib
import json
import random
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

LINES = ["14", "17", "19"]

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

_FORMAT_RE = re.compile(r"(33|50|44)")
_CL_TO_FMT = {33: "1/3", 50: "1/2", 44: "2/5"}


def parse_format(sku: str) -> str:
    m = _FORMAT_RE.search(str(sku).upper())
    cl = int(m.group(1)) if m else 33
    return _CL_TO_FMT[cl]


Chromosome = Dict[str, List[str]]
ChangeoverMode = Literal["bayes_mean", "observed_mean", "hdi_lower", "hdi_upper"]


@dataclass
class OptimizerContext:
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
    changeover_stats: Dict[str, Dict[Tuple[str, str], Dict[str, object]]]
    sku_node: Dict[str, str]
    line_mean_co: Dict[str, float]
    hist_pairs: set
    line_prior_alpha: Dict[str, float]
    line_prior_beta: Dict[str, float]
    changeover_mode: ChangeoverMode = "bayes_mean"
    changeover_hdi_mass: float = 0.95
    changeover_cache: Dict[Tuple[str, str, str, str, float], float] = field(default_factory=dict)
    w_cap: float = 100.0
    w_inc: float = 10000.0
    w_urg: float = 500.0
    mut_prob: float = 0.85
    tour_size: int = 3


def load_clean_context(clean_dir: Path) -> OptimizerContext:
    demand = pd.read_csv(clean_dir / "demand.csv")
    throughput_df = pd.read_csv(clean_dir / "throughput_rates.csv")
    changeover_df = pd.read_csv(clean_dir / "changeovers.csv")
    sku_info = pd.read_csv(clean_dir / "sku_info.csv")
    eligibility_df = pd.read_csv(clean_dir / "sku_eligibility.csv")
    hist_pairs_df = pd.read_csv(clean_dir / "historical_pairs.csv")
    hist_weeks_path = clean_dir / "historical_weeks.csv"
    hist_weeks_df = pd.read_csv(hist_weeks_path) if hist_weeks_path.exists() else pd.DataFrame()
    with open(clean_dir / "params.json") as f:
        params = json.load(f)

    for df in (demand, throughput_df, changeover_df, eligibility_df, hist_pairs_df):
        for col in ("line", "original_line", "tren"):
            if col in df.columns:
                df[col] = df[col].astype(str)

    weekly = demand
    weekly["original_line"] = weekly["original_line"].astype(str).str.strip()
    skus = demand["sku"].tolist()
    volumes = dict(zip(demand["sku"], pd.to_numeric(demand["hl_total"], errors="coerce")))
    volumes = {k: float(v) for k, v in volumes.items() if pd.notna(v)}

    sku_format = dict(zip(sku_info["sku"], sku_info["format"]))
    if {"sku", "node"}.issubset(hist_weeks_df.columns):
        sku_node = (
            hist_weeks_df.dropna(subset=["sku", "node"])
            .groupby("sku")["node"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
            .to_dict()
        )
    else:
        sku_node = {sku: sku for sku in skus}

    eligible: Dict[str, List[str]] = {}
    fallback_skus: List[str] = []
    for _, row in eligibility_df.iterrows():
        lines = [l.strip() for l in str(row["eligible_lines"]).split(",") if l.strip()]
        eligible[row["sku"]] = lines
        if bool(row.get("is_fallback", False)):
            fallback_skus.append(row["sku"])

    hist_pairs: set[Tuple[str, str]] = set()
    for _, row in hist_pairs_df.iterrows():
        hist_pairs.add((str(row["sku"]), str(row["line"])))

    throughput: Dict[Tuple[str, str], float] = {}
    for _, row in throughput_df.iterrows():
        key = (row["sku"], str(row["line"]))
        throughput[key] = float(row["rate"]) if pd.notna(row["rate"]) else 0.0

    sku_global_rate = throughput_df.groupby("sku")["rate"].median().to_dict()
    plant_mean_rate = float(throughput_df["rate"].median())

    changeover: Dict[str, Dict[Tuple[str, str], float]] = {}
    changeover_stats: Dict[str, Dict[Tuple[str, str], Dict[str, object]]] = {}
    line_mean_co: Dict[str, float] = {}
    for line in LINES:
        line_co = changeover_df[changeover_df["line"] == line]
        co_map: Dict[Tuple[str, str], float] = {}
        stats_map: Dict[Tuple[str, str], Dict[str, object]] = {}
        for _, row in line_co.iterrows():
            prev_key = row["prev_node"] if "prev_node" in row.index else row["prev_sku"]
            next_key = row["next_node"] if "next_node" in row.index else row["next_sku"]
            key = (prev_key, next_key)
            hours = float(row["hours"])
            co_map[key] = hours
            samples = _parse_changeover_samples(row, hours)
            stats_map[key] = {
                "mean": hours,
                "std": float(row.get("std_hours", np.std(samples, ddof=1) if len(samples) > 1 else 0.0) or 0.0),
                "count": int(row.get("count", len(samples)) or len(samples)),
                "samples": samples,
            }
        changeover[line] = co_map
        changeover_stats[line] = stats_map
        line_mean_co[line] = float(line_co["hours"].mean()) if not line_co.empty else 1.0

    line_prior_alpha: Dict[str, float] = {}
    line_prior_beta: Dict[str, float] = {}
    for line in LINES:
        line_co = changeover_df[changeover_df["line"] == line]
        durations = line_co["hours"].dropna().values.astype(float)
        if len(durations) < 2:
            line_prior_alpha[line] = 1.05
            line_prior_beta[line] = max(line_mean_co[line], 0.01) * 0.05
            continue
        m = float(np.mean(durations))
        v = float(np.var(durations, ddof=1))
        alpha = max(1.05, m**2 / max(v, 0.01) + 1.0)
        beta = m * (alpha - 1.0)
        line_prior_alpha[line] = alpha
        line_prior_beta[line] = beta

    return OptimizerContext(
        weekly=weekly, skus=skus, volumes=volumes, sku_format=sku_format,
        eligible=eligible, fallback_skus=fallback_skus, throughput=throughput,
        sku_global_rate=sku_global_rate, plant_mean_rate=plant_mean_rate,
        changeover=changeover, changeover_stats=changeover_stats, sku_node=sku_node,
        line_mean_co=line_mean_co, hist_pairs=hist_pairs,
        line_prior_alpha=line_prior_alpha, line_prior_beta=line_prior_beta,
    )


def _parse_changeover_samples(row: pd.Series, fallback_hours: float) -> List[float]:
    raw_samples = row.get("samples_json")
    if pd.notna(raw_samples):
        try:
            parsed = json.loads(str(raw_samples))
            samples = [float(v) for v in parsed if pd.notna(v) and np.isfinite(float(v))]
            if samples:
                return samples
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    count = int(row.get("count", 1) or 1)
    return [float(fallback_hours)] * max(1, count)


def set_changeover_policy(
    ctx: OptimizerContext,
    *,
    mode: ChangeoverMode,
    hdi_mass: float = 0.95,
) -> None:
    ctx.changeover_mode = mode
    ctx.changeover_hdi_mass = float(np.clip(hdi_mass, 0.5, 0.995))


def _stable_seed(*parts: str) -> int:
    digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _hdi_from_samples(samples: np.ndarray, mass: float) -> Tuple[float, float]:
    if len(samples) == 0:
        return 0.0, 0.0
    ordered = np.sort(samples)
    window = max(1, int(np.floor(mass * len(ordered))))
    if window >= len(ordered):
        return float(ordered[0]), float(ordered[-1])
    widths = ordered[window:] - ordered[:len(ordered) - window]
    start = int(np.argmin(widths))
    return float(ordered[start]), float(ordered[start + window])


def _bayesian_changeover_value(
    ctx: OptimizerContext,
    line: str,
    pair: Tuple[str, str],
    stats: Dict[str, object],
) -> float:
    mode = ctx.changeover_mode
    samples = np.asarray(stats.get("samples", []), dtype=float)
    samples = samples[np.isfinite(samples) & (samples >= 0)]
    if len(samples) == 0:
        return float(stats.get("mean", ctx.line_mean_co[line]))

    if mode == "observed_mean":
        return float(stats.get("mean", np.mean(samples)))

    prior_alpha = ctx.line_prior_alpha.get(line, 1.05)
    prior_beta = ctx.line_prior_beta.get(line, ctx.line_mean_co[line] * 0.05)
    alpha = prior_alpha + len(samples)
    beta = prior_beta + float(samples.sum())

    if mode == "bayes_mean":
        return beta / max(alpha - 1.0, 0.001)

    cache_key = (
        line, pair[0], pair[1], mode,
        round(ctx.changeover_hdi_mass, 4),
    )
    if cache_key in ctx.changeover_cache:
        return ctx.changeover_cache[cache_key]

    rng = np.random.default_rng(_stable_seed(line, pair[0], pair[1], mode))
    posterior_rate = rng.gamma(shape=alpha, scale=1.0 / beta, size=4000)
    posterior_duration = 1.0 / posterior_rate
    low, high = _hdi_from_samples(posterior_duration, ctx.changeover_hdi_mass)
    value = low if mode == "hdi_lower" else high
    ctx.changeover_cache[cache_key] = float(value)
    return float(value)


def throughput_rate(ctx: OptimizerContext, sku: str, line: str) -> float:
    rate = ctx.throughput.get((sku, line))
    if rate is None or not np.isfinite(rate):
        rate = ctx.sku_global_rate.get(sku, ctx.plant_mean_rate)
    return float(max(20.0, rate))


def changeover_hours(ctx: OptimizerContext, prev_sku: str, next_sku: str,
                     line: str) -> float:
    if prev_sku == next_sku:
        return 0.0
    prev_node = ctx.sku_node.get(prev_sku, prev_sku)
    next_node = ctx.sku_node.get(next_sku, next_sku)
    if prev_node == next_node:
        return 0.0
    pair = (prev_node, next_node)
    stats = ctx.changeover_stats.get(line, {}).get(pair)
    if stats is None:
        pair = (next_node, prev_node)
        stats = ctx.changeover_stats.get(line, {}).get(pair)
    if stats is not None:
        val = _bayesian_changeover_value(ctx, line, pair, stats)
    else:
        val = ctx.changeover[line].get((prev_node, next_node))
    if val is None or not np.isfinite(val):
        val = ctx.changeover[line].get((next_node, prev_node))
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
        return (ctx.w_inc * 10,)

    total = 0.0
    penalty = 0.0
    for line in LINES:
        seq = individual.get(line, [])
        for sku in seq:
            if ctx.sku_format.get(sku) not in PHYSICAL_FORMAT_BY_LINE[line]:
                penalty += ctx.w_inc
            elif (sku, line) not in ctx.hist_pairs and sku not in ctx.fallback_skus:
                penalty += ctx.w_inc / 2

        sim = simulate_line(ctx, line, seq)
        total += sim["total"]

        if sim["total"] > HOURS_PER_WEEK[line]:
            over = sim["total"] - HOURS_PER_WEEK[line]
            # Capacity is operationally hard: a line cannot run beyond the
            # available weekly hours just because the soft objective improves.
            penalty += ctx.w_inc * (1.0 + over * over)
            penalty += ctx.w_cap * (np.exp(over / 5.0) - 1.0)

        for sku, urgent_line in PRIORITY_ORDERS:
            if urgent_line == line and sku in seq:
                pos = seq.index(sku)
                cutoff = max(0, int(0.25 * len(seq)))
                if pos > cutoff:
                    penalty += ctx.w_urg * ((pos - cutoff) / max(1, len(seq)))

    return (total + penalty,)


def baseline_individual(ctx: OptimizerContext) -> Chromosome:
    ind: Chromosome = {line: [] for line in LINES}
    for _, row in ctx.weekly.sort_values(["first_fecha", "row_order"]).iterrows():
        ind[row.original_line].append(row.sku)
    return ind


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
    on_generation: Callable | None = None,
) -> Tuple[Chromosome, List[Dict[str, float]]]:
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
            p1 = _tournament(pop_fit, k=ctx.tour_size)
            p2 = _tournament(pop_fit, k=ctx.tour_size)
            child = crossover(ctx, p1, p2)
            if random.random() < ctx.mut_prob:
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
