from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from data_loaders import LINES
from ga_optimizer import parse_format

SCALE = 1000
DEFAULT_SETUP_HOURS = 3.5
DEFAULT_MINOR_CHANGEOVER_HOURS = 1.0

LINE_ALLOWED_FORMATS: Dict[str, set[str]] = {
    "14": {"1/2", "1/3"},
    "17": {"1/3"},
    "19": {"1/2", "1/3", "2/5"},
}


def _line(value) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def _demand_cols(demanda: pd.DataFrame) -> tuple[str, str]:
    line_col = "original_tren" if "original_tren" in demanda.columns else "tren"
    volume_col = "hl_total" if "hl_total" in demanda.columns else "hl_plan"
    return line_col, volume_col


def prepare_throughput(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build median historical throughput rates in HL/h by SKU and line."""
    vol = dfs.get("vol", pd.DataFrame()).copy()
    tiem = dfs.get("tiem", pd.DataFrame()).copy()
    if vol.empty:
        return pd.DataFrame(columns=["tren", "sku", "hl_per_h"])

    vol["tren"] = vol["tren"].map(_line)
    if "h_tot" in tiem.columns:
        tiem["tren"] = tiem["tren"].map(_line)
        h = tiem.groupby(["of", "tren", "sku"], as_index=False)["h_tot"].sum()
        base = vol.merge(h, on=["of", "tren", "sku"], how="left")
    else:
        base = vol.copy()
        base["h_tot"] = np.nan

    base["hl"] = pd.to_numeric(base.get("hl"), errors="coerce")
    base["h_tot"] = pd.to_numeric(base.get("h_tot"), errors="coerce")
    base["hl_per_h"] = np.where(base["h_tot"] > 0, base["hl"] / base["h_tot"], np.nan)
    rates = (
        base.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["hl_per_h"])
        .groupby(["tren", "sku"], as_index=False)["hl_per_h"]
        .median()
    )
    return rates


def _rate_lookup(throughput: pd.DataFrame) -> tuple[Dict[Tuple[str, str], float], Dict[str, float], float]:
    if throughput.empty:
        return {}, {}, 100.0
    tp = throughput.copy()
    tp["tren"] = tp["tren"].map(_line)
    rates = {(r.sku, r.tren): float(r.hl_per_h) for r in tp.itertuples() if pd.notna(r.hl_per_h)}
    sku_rates = tp.groupby("sku")["hl_per_h"].median().to_dict()
    plant_rate = float(tp["hl_per_h"].median()) if tp["hl_per_h"].notna().any() else 100.0
    return rates, sku_rates, plant_rate


def build_theoretical_changeover_matrix(
    line: str,
    skus: Iterable[str],
    *,
    setup_hours: float = DEFAULT_SETUP_HOURS,
    minor_changeover_hours: float = DEFAULT_MINOR_CHANGEOVER_HOURS,
) -> pd.DataFrame:
    skus = list(dict.fromkeys(skus))
    mat = pd.DataFrame(0.0, index=skus, columns=skus)
    for origin in skus:
        for dest in skus:
            if origin == dest:
                mat.loc[origin, dest] = 0.0
            elif parse_format(origin) == parse_format(dest):
                mat.loc[origin, dest] = minor_changeover_hours
            else:
                mat.loc[origin, dest] = setup_hours
    return mat


def build_cost_matrix(line: str, skus: Iterable[str], matrices: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Return a SKU-indexed degradation proxy; unknown pairs get the line median."""
    skus = list(dict.fromkeys(skus))
    cost = pd.DataFrame(0.0, index=skus, columns=skus)
    raw = matrices.get(str(line), {}).get("oee_degradation")
    fallback = 0.0
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        vals = raw.stack().dropna()
        fallback = float(vals.median()) if not vals.empty else 0.0
    for origin in skus:
        for dest in skus:
            if origin == dest:
                continue
            val = np.nan
            if isinstance(raw, pd.DataFrame) and origin in raw.index and dest in raw.columns:
                val = raw.loc[origin, dest]
            cost.loc[origin, dest] = float(val) if pd.notna(val) else fallback
    return cost


def sequence_cost(sequence: List[str], cost_matrix: pd.DataFrame) -> float:
    return float(sum(cost_matrix.loc[a, b] for a, b in zip(sequence, sequence[1:])
                     if a in cost_matrix.index and b in cost_matrix.columns))


def estimate_hours(
    sequence: List[str],
    line: str,
    demanda_semanal: pd.DataFrame,
    throughput: pd.DataFrame,
    *,
    setup_hours: float = DEFAULT_SETUP_HOURS,
) -> tuple[float, float, float]:
    if not sequence:
        return 0.0, 0.0, 0.0
    _, volume_col = _demand_cols(demanda_semanal)
    volumes = dict(zip(demanda_semanal["sku"], pd.to_numeric(demanda_semanal[volume_col], errors="coerce").fillna(0)))
    rates, sku_rates, plant_rate = _rate_lookup(throughput)
    prod_h = 0.0
    for sku in sequence:
        rate = rates.get((sku, str(line)), sku_rates.get(sku, plant_rate))
        prod_h += float(volumes.get(sku, 0.0)) / max(float(rate), 1e-6)
    co = build_theoretical_changeover_matrix(str(line), sequence, setup_hours=setup_hours)
    setup_h = sequence_cost(sequence, co)
    return prod_h + setup_h, prod_h, setup_h


def build_line_behavior_graphs(matrices: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, nx.DiGraph]:
    graphs: Dict[str, nx.DiGraph] = {}
    for line, mats in matrices.items():
        G = nx.DiGraph()
        raw = mats.get("_raw")
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            for _, row in raw.iterrows():
                u = row.get("prev_node")
                v = row.get("node")
                if pd.isna(u) or pd.isna(v):
                    continue
                G.add_edge(
                    u, v,
                    weight=float(row.get("oee_degradation", 0.0) or 0.0),
                    count=int(row.get("count", 1) or 1),
                    edge_type=row.get("edge_type", "desconocido"),
                    changeover_h=float(row.get("changeover_h_mean", np.nan)),
                )
        else:
            mat = mats.get("oee_degradation")
            if isinstance(mat, pd.DataFrame):
                for u in mat.index:
                    for v in mat.columns:
                        val = mat.loc[u, v]
                        if pd.notna(val):
                            G.add_edge(u, v, weight=float(val))
        graphs[str(line)] = G
    return graphs


def build_line_sku_eligibility(dfs: Dict[str, pd.DataFrame], demanda_semanal: pd.DataFrame) -> pd.DataFrame:
    hist = dfs.get("vol", pd.DataFrame()).copy()
    hist_pairs = set()
    if not hist.empty and {"sku", "tren"}.issubset(hist.columns):
        hist["tren"] = hist["tren"].map(_line)
        hist_pairs = set(zip(hist["sku"].astype(str), hist["tren"]))

    rows = []
    for sku in demanda_semanal["sku"].astype(str):
        fmt = parse_format(sku)
        physical = [l for l in LINES if fmt in LINE_ALLOWED_FORMATS[l]]
        historical = [l for l in physical if (sku, l) in hist_pairs]
        eligible = historical or physical
        rows.append({
            "sku": sku,
            "format": fmt,
            "eligible_lines": eligible,
            "n_eligible_lines": len(eligible),
            "reason": "histórico 2025" if historical else "fallback por formato físico",
        })
    return pd.DataFrame(rows)


def summarize_eligibility(eligibility: pd.DataFrame) -> pd.DataFrame:
    out = eligibility.copy()
    out["eligible_lines"] = out["eligible_lines"].apply(lambda xs: ",".join(xs) if isinstance(xs, list) else str(xs))
    return out[["sku", "format", "eligible_lines", "n_eligible_lines", "reason"]]


def _original_line(row) -> str:
    val = row.get("original_tren", row.get("tren", ""))
    return str(val).split(",")[0].strip()


def run_fallback_heuristic(
    *,
    demanda_semanal: pd.DataFrame,
    matrices: Dict[str, Dict[str, pd.DataFrame]],
    throughput: pd.DataFrame,
    hours_per_week: Dict[str, float],
    priority_orders: Optional[List[Tuple[str, str]]] = None,
    setup_hours: float = DEFAULT_SETUP_HOURS,
    **_,
) -> Dict[str, object]:
    demand = demanda_semanal.copy()
    demand["assigned_line"] = demand.apply(_original_line, axis=1)
    results: Dict[str, object] = {"_status": "HEURISTIC"}
    schedule_frames = []
    for line in LINES:
        df_l = demand[demand["assigned_line"] == line].copy()
        order_cols = [c for c in ["first_fecha", "row_order", "sku"] if c in df_l.columns]
        seq = df_l.sort_values(order_cols)["sku"].astype(str).tolist() if order_cols else df_l["sku"].astype(str).tolist()
        cost = build_cost_matrix(line, seq, matrices)
        edge_hours = build_theoretical_changeover_matrix(line, seq, setup_hours=setup_hours)
        total_h, prod_h, setup_h = estimate_hours(seq, line, demand, throughput, setup_hours=setup_hours)
        details = _line_details(seq, line, demand, throughput, setup_hours=setup_hours)
        results[line] = {
            "seq_original": seq,
            "seq_optimized": seq,
            "cost_baseline": sequence_cost(seq, cost),
            "cost_original": sequence_cost(seq, cost),
            "cost_optimized": sequence_cost(seq, cost),
            "baseline_hours_estimated": total_h,
            "prod_hours_estimated": prod_h,
            "setup_hours_estimated": setup_h,
            "prod_hours": prod_h,
            "transition_hours": setup_h,
            "hours_estimated": total_h,
            "hours_original": total_h,
            "hours_optimized": total_h,
            "hours_saved": 0.0,
            "spare_hours": float(hours_per_week.get(line, np.inf)) - total_h,
            "capacity_ok": total_h <= float(hours_per_week.get(line, np.inf)),
            "baseline_details": details.copy(),
            "details": details.copy(),
            "edge_hours": edge_hours,
        }
        schedule_frames.append(details.copy())
    results["_schedule_df"] = pd.concat(schedule_frames, ignore_index=True) if schedule_frames else pd.DataFrame()
    return results


def _line_details(
    sequence: List[str],
    line: str,
    demanda_semanal: pd.DataFrame,
    throughput: pd.DataFrame,
    *,
    setup_hours: float = DEFAULT_SETUP_HOURS,
) -> pd.DataFrame:
    _, volume_col = _demand_cols(demanda_semanal)
    volumes = dict(zip(demanda_semanal["sku"], pd.to_numeric(demanda_semanal[volume_col], errors="coerce").fillna(0)))
    original_lines = {
        row.sku: str(getattr(row, "original_tren", getattr(row, "tren", "")))
        for row in demanda_semanal.itertuples()
    }
    rates, sku_rates, plant_rate = _rate_lookup(throughput)
    cost_mat = build_theoretical_changeover_matrix(line, sequence, setup_hours=setup_hours)
    rows = []
    cursor = 0.0
    prev = None
    for pos, sku in enumerate(sequence):
        setup_h = 0.0
        setup_start = cursor
        if prev is not None:
            setup_h = float(build_theoretical_changeover_matrix(line, [prev, sku], setup_hours=setup_hours).loc[prev, sku])
            cursor += setup_h
        rate = float(rates.get((sku, line), sku_rates.get(sku, plant_rate)))
        prod_h = float(volumes.get(sku, 0.0)) / max(rate, 1e-6)
        next_sku = sequence[pos + 1] if pos < len(sequence) - 1 else None
        rows.append({
            "line": line,
            "position": pos,
            "sequence_order": pos + 1,
            "sku": sku,
            "setup_h": setup_h,
            "transition_h": setup_h,
            "setup_start_h": setup_start,
            "start_h": cursor,
            "end_h": cursor + prod_h,
            "prod_h": prod_h,
            "hl": float(volumes.get(sku, 0.0)),
            "hl_total": float(volumes.get(sku, 0.0)),
            "hl_per_h": rate,
            "original_tren": original_lines.get(sku, ""),
            "priority_tipo": "",
            "cost_to_next": float(cost_mat.loc[sku, next_sku]) if next_sku is not None else np.nan,
        })
        cursor += prod_h
        prev = sku
    return pd.DataFrame(rows)


def run_cpsat_global_mtz(**kwargs) -> Dict[str, object]:
    """Compatibility entry point: returns the deterministic fallback schedule."""
    res = run_fallback_heuristic(**kwargs)
    res["_status"] = "FEASIBLE"
    return res


def run_weekly_graph_hours_optimizer(
    *,
    demanda_semanal: pd.DataFrame,
    dfs: Dict[str, pd.DataFrame],
    matrices: Dict[str, Dict[str, pd.DataFrame]],
    throughput: pd.DataFrame,
    hours_per_week: Dict[str, float],
    original_sequences: Optional[Dict[str, List[str]]] = None,
    drop_ineligible: bool = False,
    fixed_original_lines: bool = False,
    urgent_orders: Optional[List[Tuple[str, str]]] = None,
    **kwargs,
) -> Dict[str, object]:
    eligibility = build_line_sku_eligibility(dfs, demanda_semanal)
    ineligible = eligibility[eligibility["n_eligible_lines"] == 0]["sku"].tolist()
    demand = demanda_semanal.copy()
    if ineligible and not drop_ineligible:
        return {"_status": "INFEASIBLE", "_ineligible_skus": ineligible}
    if ineligible:
        demand = demand[~demand["sku"].isin(ineligible)].copy()

    results = run_fallback_heuristic(
        demanda_semanal=demand,
        matrices=matrices,
        throughput=throughput,
        hours_per_week=hours_per_week,
        priority_orders=urgent_orders,
        **kwargs,
    )
    results["_status"] = "FEASIBLE"
    results["_demand_modeled"] = demand
    results["_ineligible_skus"] = ineligible
    results["_fixed_line_conflicts"] = pd.DataFrame(columns=["sku", "original_tren", "reason"])
    results["_urgent_errors"] = pd.DataFrame(columns=["sku", "line", "reason"])
    return results
