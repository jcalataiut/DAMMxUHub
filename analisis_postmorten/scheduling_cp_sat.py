"""Exact simultaneous assignment and sequencing model for LineWise.

The public entry point is ``run_cpsat_global_mtz``. It keeps the notebook output
shape used by the existing Gantt code while solving assignment and ordering in
one CP-SAT model.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


LINES = ["14", "17", "19"]
SCALE = 10_000
DEFAULT_DEGRADATION_PRIOR = 0.05
DEFAULT_THROUGHPUT_HL_PER_H = 150.0
DEFAULT_SETUP_HOURS = 1.5
DEFAULT_MINOR_CHANGEOVER_HOURS = 0.5


LINE_ALLOWED_FORMATS = {
    "14": {"1/2", "1/3"},
    "17": {"1/3"},
    "19": {"1/2", "1/3", "2/5"},
}


FORMAT_CHANGE_HOURS = {
    "14": {
        ("1/3", "1/2"): 3.0,
        ("1/2", "1/3"): 3.0,
    },
    "17": {
        ("1/3", "1/2"): 8.0,
        ("1/2", "1/3"): 8.0,
    },
    "19": {
        ("1/3", "1/2"): 6.0,
        ("1/2", "1/3"): 6.0,
        ("1/3", "2/5"): 6.0,
        ("2/5", "1/3"): 6.0,
        ("1/2", "2/5"): 6.0,
        ("2/5", "1/2"): 6.0,
    },
}


def prepare_throughput(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build median throughput by (line, SKU) in HL/hour."""
    tiempo = dfs["tiem"]
    volumen = dfs["vol"]
    tiempo_agg = (
        tiempo.groupby("of")["h_tot"].sum().reset_index().rename(columns={"h_tot": "h_tot_of"})
    )
    df_thru = volumen.merge(tiempo_agg, on="of", how="left")
    df_thru = df_thru[(df_thru["h_tot_of"] > 0) & (df_thru["hl"] > 0)].copy()
    df_thru["hl_per_h"] = df_thru["hl"] / df_thru["h_tot_of"]
    throughput = df_thru.groupby(["tren", "sku"])["hl_per_h"].median().reset_index()
    throughput.columns = ["tren", "sku", "hl_per_h"]
    return throughput


def _off_diagonal_values(matrix: pd.DataFrame) -> pd.Series:
    values = matrix.copy()
    common = [sku for sku in values.index if sku in values.columns]
    for sku in common:
        values.loc[sku, sku] = np.nan
    return values.stack().dropna()


def build_cost_matrix(
    line: str,
    skus: List[str],
    matrices: Dict,
    *,
    clip_imputed_below: Optional[float] = 0.0,
    default_prior: float = DEFAULT_DEGRADATION_PRIOR,
) -> pd.DataFrame:
    """Build an OEE degradation cost matrix for one line and SKU set.

    Missing transitions are imputed with the destination column mean excluding
    the diagonal, then the line prior. Historical negative costs are preserved;
    only imputed values are optionally clipped to avoid invented OEE gains.
    """
    if not skus:
        return pd.DataFrame()

    if line not in matrices:
        cost = pd.DataFrame(default_prior, index=skus, columns=skus, dtype=float)
        np.fill_diagonal(cost.values, 0.0)
        return cost

    deg_mat = matrices[line]["oee_degradation"]
    line_values = _off_diagonal_values(deg_mat)
    line_prior = float(line_values.mean()) if not line_values.empty else default_prior
    if np.isnan(line_prior):
        line_prior = default_prior
    imputed_line_prior = line_prior
    if clip_imputed_below is not None:
        imputed_line_prior = max(clip_imputed_below, imputed_line_prior)

    cost = pd.DataFrame(np.nan, index=skus, columns=skus, dtype=float)
    for origin in skus:
        for destination in skus:
            if origin == destination:
                cost.loc[origin, destination] = 0.0
            elif origin in deg_mat.index and destination in deg_mat.columns:
                value = deg_mat.loc[origin, destination]
                if pd.notna(value):
                    cost.loc[origin, destination] = float(value)

    cost_off = cost.copy()
    np.fill_diagonal(cost_off.values, np.nan)
    col_means = cost_off.mean(axis=0)

    for destination in skus:
        col_prior = col_means[destination] if pd.notna(col_means[destination]) else imputed_line_prior
        if clip_imputed_below is not None:
            col_prior = max(clip_imputed_below, float(col_prior))
        for origin in skus:
            if origin != destination and pd.isna(cost.loc[origin, destination]):
                cost.loc[origin, destination] = float(col_prior)

    return cost.fillna(imputed_line_prior)


def build_changeover_matrix(
    line: str,
    skus: List[str],
    matrices: Dict,
    *,
    fallback_hours: float = DEFAULT_SETUP_HOURS,
) -> pd.DataFrame:
    """Build a setup/changeover hours matrix using post-mortem observations."""
    if not skus:
        return pd.DataFrame()

    if line not in matrices or "changeover_h" not in matrices[line]:
        changeover = pd.DataFrame(fallback_hours, index=skus, columns=skus, dtype=float)
        np.fill_diagonal(changeover.values, 0.0)
        return changeover

    raw_mat = matrices[line]["changeover_h"]
    line_values = _off_diagonal_values(raw_mat)
    line_prior = float(line_values.median()) if not line_values.empty else fallback_hours
    if np.isnan(line_prior) or line_prior <= 0:
        line_prior = fallback_hours

    changeover = pd.DataFrame(np.nan, index=skus, columns=skus, dtype=float)
    for origin in skus:
        for destination in skus:
            if origin == destination:
                changeover.loc[origin, destination] = 0.0
            elif origin in raw_mat.index and destination in raw_mat.columns:
                value = raw_mat.loc[origin, destination]
                if pd.notna(value) and float(value) >= 0:
                    changeover.loc[origin, destination] = float(value)

    changeover_off = changeover.copy()
    np.fill_diagonal(changeover_off.values, np.nan)
    col_means = changeover_off.mean(axis=0)
    for destination in skus:
        col_prior = col_means[destination] if pd.notna(col_means[destination]) else line_prior
        for origin in skus:
            if origin != destination and pd.isna(changeover.loc[origin, destination]):
                changeover.loc[origin, destination] = max(0.0, float(col_prior))

    return changeover.fillna(line_prior)


def infer_sku_format(sku: str) -> str:
    """Infer the can format from the SKU code used in the Damm exports."""
    sku = str(sku)
    if "12" in sku:
        return "1/2"
    if "13" in sku:
        return "1/3"
    if "25" in sku:
        return "2/5"
    return "unknown"


def theoretical_changeover_hours(
    line: str,
    origin: str,
    destination: str,
    *,
    minor_changeover_hours: float = DEFAULT_MINOR_CHANGEOVER_HOURS,
    fallback_hours: float = DEFAULT_SETUP_HOURS,
) -> float:
    """Return theoretical setup time following Tabla CF Prat semantics.

    Capacity should use theoretical format-change time, not OEE degradation.
    When the inferred can format does not change, the remaining expected setup
    is treated as a minor packaging/palletization adjustment (30 min by default).
    """
    if origin == destination:
        return 0.0
    origin_format = infer_sku_format(origin)
    destination_format = infer_sku_format(destination)
    if origin_format == destination_format and origin_format != "unknown":
        return minor_changeover_hours
    if origin_format != "unknown" and destination_format != "unknown":
        return FORMAT_CHANGE_HOURS.get(str(line), {}).get(
            (origin_format, destination_format),
            fallback_hours,
        )
    return minor_changeover_hours


def build_theoretical_changeover_matrix(
    line: str,
    skus: List[str],
    *,
    minor_changeover_hours: float = DEFAULT_MINOR_CHANGEOVER_HOURS,
    fallback_hours: float = DEFAULT_SETUP_HOURS,
) -> pd.DataFrame:
    """Build the theoretical changeover matrix used by capacity constraints."""
    changeover = pd.DataFrame(0.0, index=skus, columns=skus, dtype=float)
    for origin in skus:
        for destination in skus:
            changeover.loc[origin, destination] = theoretical_changeover_hours(
                line,
                origin,
                destination,
                minor_changeover_hours=minor_changeover_hours,
                fallback_hours=fallback_hours,
            )
    return changeover


def build_line_sku_eligibility(
    dfs: Dict[str, pd.DataFrame],
    demanda_semanal: pd.DataFrame,
    *,
    lines: List[str] = LINES,
    history_table: str = "vol",
) -> pd.DataFrame:
    """Validate which line can produce each SKU under strict 2025 evidence.

    A line is eligible only when both conditions hold:
    - the SKU format is physically allowed by the line;
    - the exact SKU appeared on that line in the 2025 historical table.
    """
    hist = dfs[history_table]
    produced = {
        line: set(hist[hist["tren"].astype(str) == line]["sku"].dropna().astype(str))
        for line in lines
    }

    rows = []
    for _, item in demanda_semanal.iterrows():
        sku = str(item["sku"])
        sku_format = infer_sku_format(sku)
        for line in lines:
            format_allowed = sku_format in LINE_ALLOWED_FORMATS[line]
            in_history = sku in produced[line]
            rows.append(
                {
                    "sku": sku,
                    "line": line,
                    "format": sku_format,
                    "format_allowed": format_allowed,
                    "produced_in_2025": in_history,
                    "eligible": format_allowed and in_history,
                }
            )
    return pd.DataFrame(rows)


def summarize_eligibility(eligibility: pd.DataFrame) -> pd.DataFrame:
    """One row per SKU with eligible lines and blocking reason."""
    rows = []
    for sku, df_sku in eligibility.groupby("sku", sort=False):
        eligible_lines = df_sku[df_sku["eligible"]]["line"].tolist()
        format_value = df_sku["format"].iloc[0]
        if eligible_lines:
            reason = "OK"
        elif not df_sku["format_allowed"].any():
            reason = "Formato no permitido en L14/L17/L19"
        else:
            reason = "SKU sin evidencia exacta de produccion en 2025"
        rows.append(
            {
                "sku": sku,
                "format": format_value,
                "eligible_lines": ",".join(eligible_lines),
                "n_eligible_lines": len(eligible_lines),
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_line_behavior_graphs(matrices: Dict) -> Dict[str, object]:
    """Build one directed graph per line from 2025 transition behavior."""
    try:
        import networkx as nx
    except ImportError:
        return {}

    graphs = {}
    for line, mats in matrices.items():
        raw = mats.get("_raw", pd.DataFrame())
        graph = nx.DiGraph(line=line)
        for _, row in raw.iterrows():
            graph.add_edge(
                row["sku_prev"],
                row["sku"],
                oee_degradation=float(row["oee_degradation"]) if pd.notna(row["oee_degradation"]) else 0.0,
                count=int(row["count"]) if pd.notna(row["count"]) else 0,
                changeover_h=float(row["changeover_h_mean"]) if pd.notna(row["changeover_h_mean"]) else np.nan,
            )
        graphs[line] = graph
    return graphs


def build_transition_hours_matrix(
    line: str,
    skus: List[str],
    matrices: Dict,
    throughput: pd.DataFrame,
    hl_by_sku: Dict[str, float],
    *,
    oee_loss_weight: float = 1.0,
    fallback_hours: float = DEFAULT_SETUP_HOURS,
    use_learned_changeover: bool = True,
) -> pd.DataFrame:
    """Build directed edge hours learned from the 2025 graph.

    Edge hours = theoretical setup/changeover + positive OEE degradation
    translated into equivalent extra hours for the destination SKU.
    """
    if not skus:
        return pd.DataFrame()
    if use_learned_changeover and line in matrices and "changeover_h" in matrices[line]:
        setup = build_changeover_matrix(line, skus, matrices, fallback_hours=fallback_hours)
    else:
        setup = build_theoretical_changeover_matrix(line, skus, fallback_hours=fallback_hours)
    degradation = build_cost_matrix(line, skus, matrices, clip_imputed_below=0.0)
    edge_hours = pd.DataFrame(0.0, index=skus, columns=skus, dtype=float)
    maps = _throughput_maps(throughput)

    for origin in skus:
        for destination in skus:
            if origin == destination:
                edge_hours.loc[origin, destination] = 0.0
                continue
            prod_h_destination = (
                float(hl_by_sku.get(destination, 0.0))
                / throughput_for(line, destination, throughput, maps)
            )
            oee_loss_h = max(0.0, float(degradation.loc[origin, destination])) * prod_h_destination
            edge_hours.loc[origin, destination] = (
                float(setup.loc[origin, destination]) + oee_loss_weight * oee_loss_h
            )
    return edge_hours


def estimate_sequence_graph_hours(
    sequence: List[str],
    hl_by_sku: Dict[str, float],
    line: str,
    throughput: pd.DataFrame,
    edge_hours: pd.DataFrame,
) -> Dict:
    """Estimate total hours for a sequence using graph edge hours."""
    maps = _throughput_maps(throughput)
    elapsed = 0.0
    rows = []
    prod_total = 0.0
    transition_total = 0.0
    for pos, sku in enumerate(sequence, start=1):
        transition_h = 0.0
        if pos > 1:
            previous = sequence[pos - 2]
            transition_h = float(edge_hours.loc[previous, sku])
        prod_h = float(hl_by_sku.get(sku, 0.0)) / throughput_for(line, sku, throughput, maps)
        start_h = elapsed + transition_h
        end_h = start_h + prod_h
        rows.append(
            {
                "line": line,
                "sequence_order": pos,
                "sku": sku,
                "hl": float(hl_by_sku.get(sku, 0.0)),
                "transition_h": round(transition_h, 4),
                "prod_h": round(prod_h, 4),
                "start_h": round(start_h, 4),
                "end_h": round(end_h, 4),
            }
        )
        elapsed = end_h
        prod_total += prod_h
        transition_total += transition_h
    details = pd.DataFrame(rows)
    return {
        "total_h": round(prod_total + transition_total, 4),
        "prod_h": round(prod_total, 4),
        "transition_h": round(transition_total, 4),
        "details": details,
    }


def sequence_cost(sequence: List[str], cost_matrix: pd.DataFrame) -> float:
    total = 0.0
    for origin, destination in zip(sequence, sequence[1:]):
        if origin in cost_matrix.index and destination in cost_matrix.columns:
            total += float(cost_matrix.loc[origin, destination])
    return total


def _throughput_maps(throughput: pd.DataFrame) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float], Dict[str, float], float]:
    line_sku = throughput.set_index(["tren", "sku"])["hl_per_h"].to_dict()
    line_median = throughput.groupby("tren")["hl_per_h"].median().to_dict()
    sku_median = throughput.groupby("sku")["hl_per_h"].median().to_dict()
    global_median = float(throughput["hl_per_h"].median()) if not throughput.empty else DEFAULT_THROUGHPUT_HL_PER_H
    if np.isnan(global_median) or global_median <= 0:
        global_median = DEFAULT_THROUGHPUT_HL_PER_H
    return line_sku, line_median, sku_median, global_median


def throughput_for(
    line: str,
    sku: str,
    throughput: pd.DataFrame,
    maps: Optional[Tuple[Dict[Tuple[str, str], float], Dict[str, float], Dict[str, float], float]] = None,
) -> float:
    if maps is None:
        maps = _throughput_maps(throughput)
    line_sku, line_median, sku_median, global_median = maps
    value = line_sku.get((line, sku))
    if value is None or value <= 0 or np.isnan(value):
        value = line_median.get(line)
    if value is None or value <= 0 or np.isnan(value):
        value = sku_median.get(sku)
    if value is None or value <= 0 or np.isnan(value):
        value = global_median
    return float(value)


def estimate_hours(
    sequence: List[str],
    hl_demand: Dict[str, float],
    line: str,
    throughput: pd.DataFrame,
    changeover_matrix: Optional[pd.DataFrame] = None,
    *,
    setup_hours: float = DEFAULT_SETUP_HOURS,
) -> Dict:
    """Estimate chronological production and setup hours for a sequence."""
    maps = _throughput_maps(throughput)
    elapsed = 0.0
    rows = []
    for pos, sku in enumerate(sequence, start=1):
        setup_h = 0.0
        if pos > 1:
            prev = sequence[pos - 2]
            if changeover_matrix is not None and prev in changeover_matrix.index and sku in changeover_matrix.columns:
                value = changeover_matrix.loc[prev, sku]
                setup_h = float(value) if pd.notna(value) else setup_hours
            else:
                setup_h = setup_hours
        hl = float(hl_demand.get(sku, 0.0))
        thr = throughput_for(line, sku, throughput, maps)
        prod_h = hl / thr if thr > 0 else 0.0
        setup_start_h = elapsed
        prod_start_h = elapsed + setup_h
        end_h = prod_start_h + prod_h
        elapsed = end_h
        rows.append(
            {
                "line": line,
                "sequence_order": pos,
                "sku": sku,
                "hl": hl,
                "throughput_hl_h": thr,
                "setup_h": round(setup_h, 4),
                "prod_h": round(prod_h, 4),
                "start_h": round(prod_start_h, 4),
                "setup_start_h": round(setup_start_h, 4),
                "end_h": round(end_h, 4),
            }
        )

    details = pd.DataFrame(rows)
    setup_total = float(details["setup_h"].sum()) if not details.empty else 0.0
    prod_total = float(details["prod_h"].sum()) if not details.empty else 0.0
    return {
        "total_h": round(prod_total + setup_total, 4),
        "prod_h": round(prod_total, 4),
        "setup_h": round(setup_total, 4),
        "details": details,
    }


def greedy_nn_sequence(skus: List[str], cost_matrix: pd.DataFrame, start: Optional[str] = None) -> List[str]:
    if not skus:
        return []
    remaining = list(skus)
    if start and start in remaining:
        current = start
    else:
        current = cost_matrix.reindex(index=remaining, columns=remaining).mean(axis=1).idxmin()
    remaining.remove(current)
    sequence = [current]
    while remaining:
        next_sku = cost_matrix.loc[current, remaining].idxmin()
        sequence.append(next_sku)
        remaining.remove(next_sku)
        current = next_sku
    return sequence


def two_opt_improve(sequence: List[str], cost_matrix: pd.DataFrame, max_iter: int = 100) -> List[str]:
    best = list(sequence)
    best_cost = sequence_cost(best, cost_matrix)
    for _ in range(max_iter):
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                candidate_cost = sequence_cost(candidate, cost_matrix)
                if candidate_cost < best_cost - 1e-9:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def apply_priority_constraints(sequence: List[str], priority_orders: List[dict], line: str) -> List[str]:
    """Heuristic fallback priority patch. CP-SAT uses hard constraints instead."""
    result = list(sequence)
    for order in priority_orders:
        if order.get("tipo") != "urgencia":
            continue
        if order.get("linea") not in (None, line):
            continue
        sku = order.get("sku")
        if sku in result:
            result.remove(sku)
            result.insert(min(1, len(result)), sku)
    return result


def _unique_weekly_demand(demanda_semanal: pd.DataFrame) -> pd.DataFrame:
    original_col = "original_tren" if "original_tren" in demanda_semanal.columns else "tren"
    original_map = (
        demanda_semanal.groupby("sku")[original_col]
        .agg(lambda values: ",".join(sorted(set(str(v) for v in values if pd.notna(v)))))
        .rename("original_tren")
    )
    demand = (
        demanda_semanal.groupby("sku", as_index=False)["hl_total"]
        .sum()
        .merge(original_map, on="sku", how="left")
        .sort_values("sku")
        .reset_index(drop=True)
    )
    return demand


def _priority_lookup(priority_orders: Iterable[dict]) -> Dict[str, dict]:
    return {order.get("sku"): order for order in priority_orders if order.get("sku")}


def _urgent_value(order: dict, *keys, default=None):
    for key in keys:
        if key in order and order[key] is not None:
            return order[key]
    return default


def _normalize_urgent_orders(urgent_orders: Optional[List[dict]]) -> List[dict]:
    """Normalize urgent orders accepted by the graph-hours optimizer."""
    normalized = []
    for idx, order in enumerate(urgent_orders or [], start=1):
        sku = str(_urgent_value(order, "sku", "material", "SKU", default="")).strip()
        if not sku:
            continue
        line = _urgent_value(order, "linea", "line", "tren", default=None)
        line = str(line).replace(".0", "") if line is not None and str(line).strip() else None
        hl_value = _urgent_value(order, "hl_total", "hl", "volume_hl", "volumen_hl", default=None)
        try:
            hl_value = float(hl_value) if hl_value is not None else None
        except (TypeError, ValueError):
            hl_value = None
        latest_position = _urgent_value(
            order,
            "latest_position",
            "max_position",
            "posicion_max",
            "u_max",
            default=None,
        )
        try:
            latest_position = int(latest_position) if latest_position is not None else None
        except (TypeError, ValueError):
            latest_position = None
        normalized.append(
            {
                "order_id": _urgent_value(order, "order_id", "id", "of", default=f"URG-{idx}"),
                "sku": sku,
                "line": line,
                "hl_total": hl_value,
                "latest_position": latest_position,
                "raw": order,
            }
        )
    return normalized


def _merge_urgent_orders_into_demand(
    demand: pd.DataFrame,
    urgent_orders: Optional[List[dict]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add urgent-order volume to demand and report invalid requests."""
    urgent = _normalize_urgent_orders(urgent_orders)
    if not urgent:
        empty = pd.DataFrame()
        return demand.copy(), empty, empty

    demand = demand.copy()
    errors = []
    applied = []
    for order in urgent:
        sku = order["sku"]
        existing = demand["sku"] == sku
        has_existing = bool(existing.any())
        if order["line"] is not None and order["line"] not in LINES:
            errors.append({**order, "reason": f"Linea urgente no reconocida: {order['line']}"})
            continue
        if not has_existing and (order["hl_total"] is None or order["hl_total"] <= 0):
            errors.append({**order, "reason": "SKU urgente no esta en demanda y no trae volumen HL"})
            continue

        if has_existing:
            if order["hl_total"] is not None and order["hl_total"] > 0:
                demand.loc[existing, "hl_total"] = demand.loc[existing, "hl_total"].astype(float) + order["hl_total"]
            if order["line"] is not None:
                demand.loc[existing, "original_tren"] = order["line"]
                demand.loc[existing, "tren"] = order["line"]
        else:
            demand = pd.concat(
                [
                    demand,
                    pd.DataFrame(
                        [
                            {
                                "tren": order["line"] or "",
                                "sku": sku,
                                "hl_total": order["hl_total"],
                                "original_tren": order["line"] or "",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        applied.append(order)

    return demand, pd.DataFrame(applied), pd.DataFrame(errors)


def _sequence_for_hint(line: str, skus: List[str], cost_mats: Dict[str, pd.DataFrame]) -> List[str]:
    if not skus:
        return []
    cost = cost_mats[line].reindex(index=skus, columns=skus)
    return two_opt_improve(greedy_nn_sequence(skus, cost), cost, max_iter=30)


def _estimated_line_hours_for_hint(
    line: str,
    skus: List[str],
    hl_by_sku: Dict[str, float],
    throughput: pd.DataFrame,
    changeover_mats: Dict[str, pd.DataFrame],
    cost_mats: Dict[str, pd.DataFrame],
) -> float:
    seq = _sequence_for_hint(line, skus, cost_mats)
    hours = estimate_hours(seq, {sku: hl_by_sku[sku] for sku in skus}, line, throughput, changeover_mats[line])
    return float(hours["total_h"])


def _build_initial_solution_hint(
    skus: List[str],
    lines: List[str],
    demand: pd.DataFrame,
    throughput: pd.DataFrame,
    changeover_mats: Dict[str, pd.DataFrame],
    cost_mats: Dict[str, pd.DataFrame],
    hours_per_week: Dict[str, float],
    priority_orders: List[dict],
) -> Dict[str, List[str]]:
    """Construct a capacity-aware feasible-ish hint for CP-SAT.

    This does not decide the final answer. It only gives CP-SAT a reasonable
    first incumbent so the exact model can spend its time improving rather than
    discovering basic feasibility from scratch.
    """
    hl_by_sku = demand.set_index("sku")["hl_total"].astype(float).to_dict()
    original_by_sku = demand.set_index("sku")["original_tren"].to_dict()
    assigned = {line: [] for line in lines}
    remaining = []

    urgency_line_by_sku = {
        order.get("sku"): str(order.get("linea"))
        for order in priority_orders
        if order.get("tipo") == "urgencia" and order.get("sku") and str(order.get("linea")) in lines
    }

    for sku in skus:
        if "12" in sku and "19" in lines:
            assigned["19"].append(sku)
        elif sku in urgency_line_by_sku:
            assigned[urgency_line_by_sku[sku]].append(sku)
        else:
            remaining.append(sku)

    # Prefer the original line if it fits; otherwise place on the line with the
    # lowest resulting load. L19 is left for strict 1/2-format work whenever
    # possible because it is the bottleneck after compatibility filtering.
    for sku in sorted(remaining, key=lambda s: hl_by_sku[s], reverse=True):
        preferred = [line for line in str(original_by_sku.get(sku, "")).split(",") if line in lines]
        candidates = preferred + [line for line in lines if line not in preferred]
        if "19" in candidates and "12" not in sku:
            candidates = [line for line in candidates if line != "19"] + ["19"]

        best_line = None
        best_hours = float("inf")
        for line in candidates:
            trial = {k: list(v) for k, v in assigned.items()}
            trial[line].append(sku)
            hours = _estimated_line_hours_for_hint(
                line, trial[line], hl_by_sku, throughput, changeover_mats, cost_mats
            )
            if hours <= float(hours_per_week[line]) + 1e-6:
                best_line = line
                best_hours = hours
                break
            if hours < best_hours:
                best_line = line
                best_hours = hours
        assigned[best_line].append(sku)

    return {
        line: _sequence_for_hint(line, assigned[line], cost_mats)
        for line in lines
    }


def run_cpsat_global_mtz(
    demanda_semanal: pd.DataFrame,
    matrices: Dict,
    throughput: pd.DataFrame,
    hours_per_week: Dict[str, float],
    priority_orders: List[dict],
    *,
    lines: List[str] = LINES,
    setup_hours: float = DEFAULT_SETUP_HOURS,
    time_limit: float = 60.0,
    scale: int = SCALE,
    log_search_progress: bool = False,
) -> Optional[Dict[str, Dict]]:
    """Solve simultaneous assignment and sequencing with CP-SAT + MTZ ordering."""
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        print("OR-Tools no disponible. Instalar con: pip install ortools")
        return None

    demand = _unique_weekly_demand(demanda_semanal)
    skus = demand["sku"].tolist()
    n = len(skus)
    if n == 0:
        return {}

    sku_to_i = {sku: i for i, sku in enumerate(skus)}
    hl_by_sku = demand.set_index("sku")["hl_total"].astype(float).to_dict()
    original_by_sku = demand.set_index("sku")["original_tren"].to_dict()
    throughput_maps = _throughput_maps(throughput)

    cost_mats = {line: build_cost_matrix(line, skus, matrices) for line in lines}
    changeover_mats = {
        line: build_theoretical_changeover_matrix(
            line,
            skus,
            minor_changeover_hours=DEFAULT_MINOR_CHANGEOVER_HOURS,
            fallback_hours=setup_hours,
        )
        for line in lines
    }

    cost_sc = {
        (l, i, j): int(round(float(cost_mats[line].loc[skus[i], skus[j]]) * scale))
        for l, line in enumerate(lines)
        for i in range(n)
        for j in range(n)
        if i != j
    }
    changeover_sc = {
        (l, i, j): int(round(float(changeover_mats[line].loc[skus[i], skus[j]]) * scale))
        for l, line in enumerate(lines)
        for i in range(n)
        for j in range(n)
        if i != j
    }
    prod_sc = {
        (l, i): int(
            round(
                hl_by_sku[sku]
                / throughput_for(line, sku, throughput, throughput_maps)
                * scale
            )
        )
        for l, line in enumerate(lines)
        for i, sku in enumerate(skus)
    }
    cap_sc = {l: int(round(float(hours_per_week[line]) * scale)) for l, line in enumerate(lines)}

    model = cp_model.CpModel()
    y = {(l, i): model.NewBoolVar(f"y_l{line}_i{i}") for l, line in enumerate(lines) for i in range(n)}
    x = {
        (l, i, j): model.NewBoolVar(f"x_l{line}_i{i}_j{j}")
        for l, line in enumerate(lines)
        for i in range(n)
        for j in range(n)
        if i != j
    }
    u = {
        (l, i): model.NewIntVar(0, n, f"u_l{line}_i{i}")
        for l, line in enumerate(lines)
        for i in range(n)
    }
    start = {(l, i): model.NewBoolVar(f"start_l{line}_i{i}") for l, line in enumerate(lines) for i in range(n)}
    end = {(l, i): model.NewBoolVar(f"end_l{line}_i{i}") for l, line in enumerate(lines) for i in range(n)}
    active = {l: model.NewBoolVar(f"active_l{line}") for l, line in enumerate(lines)}

    # Unique assignment.
    for i in range(n):
        model.Add(sum(y[(l, i)] for l in range(len(lines))) == 1)

    # Strict physical incompatibility: 12-format SKUs can only run on L19.
    for i, sku in enumerate(skus):
        if "12" in sku:
            for blocked_line in ("14", "17"):
                if blocked_line in lines:
                    model.Add(y[(lines.index(blocked_line), i)] == 0)

    # Path flow conservation and MTZ ordering per line.
    for l, line in enumerate(lines):
        assigned_count = sum(y[(l, i)] for i in range(n))
        model.Add(assigned_count >= active[l])
        model.Add(assigned_count <= n * active[l])
        model.Add(sum(start[(l, i)] for i in range(n)) == active[l])
        model.Add(sum(end[(l, i)] for i in range(n)) == active[l])

        for i in range(n):
            incoming = sum(x[(l, j, i)] for j in range(n) if j != i)
            outgoing = sum(x[(l, i, j)] for j in range(n) if j != i)
            model.Add(incoming + start[(l, i)] == y[(l, i)])
            model.Add(outgoing + end[(l, i)] == y[(l, i)])
            model.Add(u[(l, i)] >= y[(l, i)])
            model.Add(u[(l, i)] <= n * y[(l, i)])
            model.Add(u[(l, i)] == 1).OnlyEnforceIf(start[(l, i)])

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                model.Add(x[(l, i, j)] <= y[(l, i)])
                model.Add(x[(l, i, j)] <= y[(l, j)])
                # MTZ: if i immediately precedes j, j must have a later position.
                model.Add(u[(l, j)] >= u[(l, i)] + 1).OnlyEnforceIf(x[(l, i, j)])

        production_time = sum(prod_sc[(l, i)] * y[(l, i)] for i in range(n))
        setup_time = sum(changeover_sc[(l, i, j)] * x[(l, i, j)] for i in range(n) for j in range(n) if i != j)
        model.Add(production_time + setup_time <= cap_sc[l])

    # Urgency priorities: hard line assignment and early position.
    for order in priority_orders:
        if order.get("tipo") != "urgencia":
            continue
        sku = order.get("sku")
        line = str(order.get("linea"))
        if sku not in sku_to_i or line not in lines:
            continue
        l = lines.index(line)
        i = sku_to_i[sku]
        model.Add(y[(l, i)] == 1)
        model.Add(u[(l, i)] <= 2)

    objective = [
        cost_sc[(l, i, j)] * x[(l, i, j)]
        for l in range(len(lines))
        for i in range(n)
        for j in range(n)
        if i != j and cost_sc[(l, i, j)] != 0
    ]
    model.Minimize(sum(objective) if objective else 0)

    hint_sequences = _build_initial_solution_hint(
        skus,
        lines,
        demand,
        throughput,
        changeover_mats,
        cost_mats,
        hours_per_week,
        priority_orders,
    )
    hint_line_by_sku = {
        sku: line for line, seq in hint_sequences.items() for sku in seq
    }
    for l, line in enumerate(lines):
        seq = hint_sequences[line]
        seq_i = [sku_to_i[sku] for sku in seq]
        model.AddHint(active[l], 1 if seq_i else 0)
        for i, sku in enumerate(skus):
            assigned_hint = 1 if hint_line_by_sku.get(sku) == line else 0
            model.AddHint(y[(l, i)], assigned_hint)
            model.AddHint(start[(l, i)], 1 if seq_i and i == seq_i[0] else 0)
            model.AddHint(end[(l, i)], 1 if seq_i and i == seq_i[-1] else 0)
            model.AddHint(u[(l, i)], seq_i.index(i) + 1 if i in seq_i else 0)
        hint_arcs = set(zip(seq_i, seq_i[1:]))
        for i in range(n):
            for j in range(n):
                if i != j:
                    model.AddHint(x[(l, i, j)], 1 if (i, j) in hint_arcs else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 4
    solver.parameters.log_search_progress = log_search_progress

    print(f"CP-SAT MTZ: {n} SKUs x {len(lines)} lineas | limite {time_limit:.0f}s")
    status = solver.Solve(model)
    print(f"Estado: {solver.StatusName(status)} | objetivo: {solver.ObjectiveValue() / scale:+.4f}")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    priority_by_sku = _priority_lookup(priority_orders)
    results: Dict[str, Dict] = {}
    schedule_rows = []
    for l, line in enumerate(lines):
        assigned_i = [i for i in range(n) if solver.Value(y[(l, i)]) == 1]
        assigned_skus = [skus[i] for i in assigned_i]
        seq_i: List[int] = []
        if assigned_i:
            current = next((i for i in assigned_i if solver.Value(start[(l, i)]) == 1), None)
            while current is not None and current not in seq_i:
                seq_i.append(current)
                current = next(
                    (j for j in assigned_i if j != current and solver.Value(x[(l, current, j)]) == 1),
                    None,
                )
        missing = [i for i in assigned_i if i not in seq_i]
        seq_i.extend(sorted(missing, key=lambda idx: solver.Value(u[(l, idx)])))
        seq = [skus[i] for i in seq_i]

        hl_dem = {sku: hl_by_sku[sku] for sku in assigned_skus}
        h_est = estimate_hours(
            seq,
            hl_dem,
            line,
            throughput,
            changeover_mats[line],
            setup_hours=setup_hours,
        )
        cost_opt = sequence_cost(seq, cost_mats[line])

        baseline_skus = demanda_semanal[demanda_semanal["tren"].astype(str) == line]["sku"].tolist()
        baseline_hl = demanda_semanal[demanda_semanal["tren"].astype(str) == line].set_index("sku")["hl_total"].to_dict()
        baseline_cost_mat = build_cost_matrix(line, baseline_skus, matrices) if baseline_skus else pd.DataFrame()
        baseline_changeover = (
            build_theoretical_changeover_matrix(
                line,
                baseline_skus,
                minor_changeover_hours=DEFAULT_MINOR_CHANGEOVER_HOURS,
                fallback_hours=setup_hours,
            )
            if baseline_skus
            else None
        )
        baseline_h = estimate_hours(
            baseline_skus,
            baseline_hl,
            line,
            throughput,
            baseline_changeover,
            setup_hours=setup_hours,
        )
        baseline_cost = sequence_cost(baseline_skus, baseline_cost_mat) if baseline_skus else 0.0
        improvement_pct = (baseline_cost - cost_opt) / max(abs(baseline_cost), 1e-6) * 100

        details = h_est["details"]
        if not details.empty:
            details["original_tren"] = details["sku"].map(original_by_sku)
            details["priority_tipo"] = details["sku"].map(
                lambda sku: priority_by_sku.get(sku, {}).get("tipo", "")
            )
            details["cost_to_next"] = [
                float(cost_mats[line].loc[sku, seq[pos + 1]]) if pos < len(seq) - 1 else np.nan
                for pos, sku in enumerate(seq)
            ]
            schedule_rows.extend(details.to_dict("records"))

        results[line] = {
            "seq_baseline": baseline_skus,
            "seq_optimized": seq,
            "assigned_skus": assigned_skus,
            "cost_baseline": baseline_cost,
            "cost_optimized": cost_opt,
            "improvement_pct": improvement_pct,
            "hours_estimated": h_est["total_h"],
            "prod_hours_estimated": h_est["prod_h"],
            "setup_hours_estimated": h_est["setup_h"],
            "baseline_hours_estimated": baseline_h["total_h"],
            "capacity_ok": h_est["total_h"] <= float(hours_per_week[line]) + 1e-6,
            "details": details,
            "baseline_details": baseline_h["details"],
            "cost_mat": cost_mats[line].reindex(index=assigned_skus, columns=assigned_skus),
            "full_cost_mat": cost_mats[line],
            "changeover_mat": changeover_mats[line].reindex(index=assigned_skus, columns=assigned_skus),
        }

        cap_flag = "OK" if results[line]["capacity_ok"] else "EXCEDE"
        print(
            f"L{line}: {len(seq):2d} SKUs | {h_est['total_h']:6.1f}h / "
            f"{hours_per_week[line]}h {cap_flag} | OEE deg {cost_opt:+.4f}"
        )

    results["_schedule_df"] = pd.DataFrame(schedule_rows)
    return results


def run_weekly_graph_hours_optimizer(
    demanda_semanal: pd.DataFrame,
    dfs: Dict[str, pd.DataFrame],
    matrices: Dict,
    throughput: pd.DataFrame,
    hours_per_week: Dict[str, float],
    original_sequences: Dict[str, List[str]],
    *,
    lines: List[str] = LINES,
    time_limit: float = 60.0,
    scale: int = SCALE,
    drop_ineligible: bool = False,
    fixed_original_lines: bool = False,
    urgent_orders: Optional[List[dict]] = None,
    use_learned_changeover: bool = True,
) -> Optional[Dict[str, Dict]]:
    """Optimize weekly production hours with one 2025 graph per line.

    Volumes stay fixed at SKU level. Assignment is allowed only for lines where
    the exact SKU was produced in 2025 and the format is physically allowed.
    The objective minimizes production hours plus graph transition hours. When
    ``fixed_original_lines`` is true, each SKU is forced to stay on the line
    where it appears in the weekly plan, after applying the same 2025 evidence
    check. ``urgent_orders`` can add or force SKUs as hard requirements; if an
    urgent order cannot be scheduled on an eligible line, the model reports the
    infeasibility instead of silently dropping it.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        print("OR-Tools no disponible. Instalar con: pip install ortools")
        return None

    demand_all = _unique_weekly_demand(demanda_semanal)
    demand_all, urgent_applied, urgent_errors = _merge_urgent_orders_into_demand(
        demand_all,
        urgent_orders,
    )
    if not urgent_errors.empty:
        print("Urgencias no viables por definicion de entrada.")
        print(urgent_errors[["order_id", "sku", "line", "hl_total", "reason"]].to_string(index=False))
        return {
            "_status": "INFEASIBLE_URGENT",
            "_urgent_orders": urgent_applied,
            "_urgent_errors": urgent_errors,
        }
    eligibility = build_line_sku_eligibility(dfs, demand_all, lines=lines)
    eligibility_summary = summarize_eligibility(eligibility)
    ineligible_skus = eligibility_summary[eligibility_summary["n_eligible_lines"] == 0]["sku"].tolist()
    urgent_skus = set(urgent_applied["sku"].tolist()) if not urgent_applied.empty else set()
    urgent_ineligible = sorted(urgent_skus.intersection(ineligible_skus))
    if urgent_ineligible:
        urgent_errors = pd.DataFrame(
            [
                {
                    "sku": sku,
                    "reason": "SKU urgente sin linea elegible exacta en historico 2025",
                }
                for sku in urgent_ineligible
            ]
        )
        print("Urgencias no viables: no hay evidencia historica 2025 compatible.")
        print(", ".join(urgent_ineligible))
        return {
            "_status": "INFEASIBLE_URGENT",
            "_eligibility": eligibility,
            "_eligibility_summary": eligibility_summary,
            "_ineligible_skus": ineligible_skus,
            "_urgent_orders": urgent_applied,
            "_urgent_errors": urgent_errors,
        }
    if ineligible_skus and not drop_ineligible:
        print("Modelo estricto no factible: hay SKUs sin linea elegible en historico 2025.")
        print(", ".join(ineligible_skus))
        return {
            "_status": "INFEASIBLE_INPUT",
            "_eligibility": eligibility,
            "_eligibility_summary": eligibility_summary,
            "_ineligible_skus": ineligible_skus,
            "_urgent_orders": urgent_applied,
            "_urgent_errors": urgent_errors,
        }

    demand = demand_all[~demand_all["sku"].isin(ineligible_skus)].copy() if drop_ineligible else demand_all
    skus = demand["sku"].tolist()
    if not skus:
        return {
            "_status": "INFEASIBLE_INPUT",
            "_eligibility": eligibility,
            "_eligibility_summary": eligibility_summary,
            "_ineligible_skus": ineligible_skus,
            "_urgent_orders": urgent_applied,
            "_urgent_errors": urgent_errors,
        }

    sku_to_i = {sku: i for i, sku in enumerate(skus)}
    hl_by_sku = demand.set_index("sku")["hl_total"].astype(float).to_dict()
    original_by_sku = demand.set_index("sku")["original_tren"].to_dict()

    eligible_lines_by_sku = {
        sku: eligibility[(eligibility["sku"] == sku) & (eligibility["eligible"])]["line"].tolist()
        for sku in skus
    }
    urgent_line_errors = []
    if not urgent_applied.empty:
        for _, order in urgent_applied.iterrows():
            sku = order["sku"]
            if sku not in eligible_lines_by_sku:
                urgent_line_errors.append(
                    {
                        "order_id": order["order_id"],
                        "sku": sku,
                        "line": order["line"],
                        "reason": "SKU urgente no incluido en la demanda modelada",
                    }
                )
                continue
            if pd.notna(order["line"]) and str(order["line"]):
                line = str(order["line"])
                if line not in eligible_lines_by_sku[sku]:
                    urgent_line_errors.append(
                        {
                            "order_id": order["order_id"],
                            "sku": sku,
                            "line": line,
                            "reason": "La linea impuesta no es elegible por formato/historico 2025",
                        }
                    )
    if urgent_line_errors:
        urgent_errors = pd.DataFrame(urgent_line_errors)
        print("Urgencias no viables con las restricciones de linea.")
        print(urgent_errors.to_string(index=False))
        return {
            "_status": "INFEASIBLE_URGENT",
            "_eligibility": eligibility,
            "_eligibility_summary": eligibility_summary,
            "_ineligible_skus": ineligible_skus,
            "_urgent_orders": urgent_applied,
            "_urgent_errors": urgent_errors,
        }
    fixed_line_conflicts = pd.DataFrame()
    if fixed_original_lines:
        fixed_rows = []
        fixed_eligible_lines_by_sku = {}
        for sku in skus:
            planned_lines = [
                line
                for line in str(original_by_sku.get(sku, "")).split(",")
                if line in lines
            ]
            eligible_planned = [
                line for line in planned_lines if line in eligible_lines_by_sku[sku]
            ]
            if len(planned_lines) != 1 or len(eligible_planned) != 1:
                fixed_rows.append(
                    {
                        "sku": sku,
                        "original_tren": original_by_sku.get(sku, ""),
                        "eligible_lines": ",".join(eligible_lines_by_sku[sku]),
                        "reason": (
                            "SKU agregado en varias lineas originales"
                            if len(planned_lines) != 1
                            else "La linea original no tiene evidencia exacta 2025"
                        ),
                    }
                )
                continue
            fixed_eligible_lines_by_sku[sku] = eligible_planned

        if fixed_rows:
            fixed_line_conflicts = pd.DataFrame(fixed_rows)
            if not drop_ineligible:
                print("Modelo fijo no factible: hay SKUs sin evidencia 2025 en su linea original.")
                print(", ".join(fixed_line_conflicts["sku"].tolist()))
                return {
                    "_status": "INFEASIBLE_INPUT",
                    "_eligibility": eligibility,
                    "_eligibility_summary": eligibility_summary,
                    "_ineligible_skus": ineligible_skus,
                    "_fixed_line_conflicts": fixed_line_conflicts,
                    "_urgent_orders": urgent_applied,
                    "_urgent_errors": urgent_errors,
                }
            demand = demand[~demand["sku"].isin(fixed_line_conflicts["sku"])].copy()
            skus = demand["sku"].tolist()
            hl_by_sku = demand.set_index("sku")["hl_total"].astype(float).to_dict()
            original_by_sku = demand.set_index("sku")["original_tren"].to_dict()
            fixed_eligible_lines_by_sku = {
                sku: lines_
                for sku, lines_ in fixed_eligible_lines_by_sku.items()
                if sku in hl_by_sku
            }
            if not skus:
                return {
                    "_status": "INFEASIBLE_INPUT",
                    "_eligibility": eligibility,
                    "_eligibility_summary": eligibility_summary,
                    "_ineligible_skus": ineligible_skus,
                    "_fixed_line_conflicts": fixed_line_conflicts,
                    "_urgent_orders": urgent_applied,
                    "_urgent_errors": urgent_errors,
                }
        eligible_lines_by_sku = fixed_eligible_lines_by_sku

    edge_mats = {
        line: build_transition_hours_matrix(
            line, skus, matrices, throughput, hl_by_sku,
            use_learned_changeover=use_learned_changeover
        )
        for line in lines
    }
    throughput_maps = _throughput_maps(throughput)

    prod_sc = {
        (l, i): int(
            round(
                hl_by_sku[sku]
                / throughput_for(line, sku, throughput, throughput_maps)
                * scale
            )
        )
        for l, line in enumerate(lines)
        for i, sku in enumerate(skus)
    }
    edge_sc = {
        (l, i, j): int(round(float(edge_mats[line].loc[skus[i], skus[j]]) * scale))
        for l, line in enumerate(lines)
        for i in range(len(skus))
        for j in range(len(skus))
        if i != j
    }
    cap_sc = {l: int(round(float(hours_per_week[line]) * scale)) for l, line in enumerate(lines)}

    model = cp_model.CpModel()
    n = len(skus)
    y = {(l, i): model.NewBoolVar(f"y_l{line}_i{i}") for l, line in enumerate(lines) for i in range(n)}
    x = {
        (l, i, j): model.NewBoolVar(f"x_l{line}_i{i}_j{j}")
        for l, line in enumerate(lines)
        for i in range(n)
        for j in range(n)
        if i != j
    }
    u = {(l, i): model.NewIntVar(0, n, f"u_l{line}_i{i}") for l, line in enumerate(lines) for i in range(n)}
    start = {(l, i): model.NewBoolVar(f"start_l{line}_i{i}") for l, line in enumerate(lines) for i in range(n)}
    end = {(l, i): model.NewBoolVar(f"end_l{line}_i{i}") for l, line in enumerate(lines) for i in range(n)}
    active = {l: model.NewBoolVar(f"active_l{line}") for l, line in enumerate(lines)}

    for i, sku in enumerate(skus):
        model.Add(sum(y[(l, i)] for l in range(len(lines))) == 1)
        for l, line in enumerate(lines):
            if line not in eligible_lines_by_sku[sku]:
                model.Add(y[(l, i)] == 0)

    if not urgent_applied.empty:
        for _, order in urgent_applied.iterrows():
            sku = order["sku"]
            if sku not in skus:
                continue
            i = skus.index(sku)
            forced_line = str(order["line"]) if pd.notna(order["line"]) and str(order["line"]) else None
            if forced_line is not None:
                forced_l = lines.index(forced_line)
                model.Add(y[(forced_l, i)] == 1)
            if pd.notna(order.get("latest_position")):
                latest_position = int(order["latest_position"])
                for l, line in enumerate(lines):
                    model.Add(u[(l, i)] <= latest_position).OnlyEnforceIf(y[(l, i)])

    for l, line in enumerate(lines):
        assigned_count = sum(y[(l, i)] for i in range(n))
        model.Add(assigned_count >= active[l])
        model.Add(assigned_count <= n * active[l])
        model.Add(sum(start[(l, i)] for i in range(n)) == active[l])
        model.Add(sum(end[(l, i)] for i in range(n)) == active[l])

        for i in range(n):
            incoming = sum(x[(l, j, i)] for j in range(n) if j != i)
            outgoing = sum(x[(l, i, j)] for j in range(n) if j != i)
            model.Add(incoming + start[(l, i)] == y[(l, i)])
            model.Add(outgoing + end[(l, i)] == y[(l, i)])
            model.Add(u[(l, i)] >= y[(l, i)])
            model.Add(u[(l, i)] <= n * y[(l, i)])
            model.Add(u[(l, i)] == 1).OnlyEnforceIf(start[(l, i)])

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                model.Add(x[(l, i, j)] <= y[(l, i)])
                model.Add(x[(l, i, j)] <= y[(l, j)])
                model.Add(u[(l, j)] >= u[(l, i)] + 1).OnlyEnforceIf(x[(l, i, j)])

        line_hours = (
            sum(prod_sc[(l, i)] * y[(l, i)] for i in range(n))
            + sum(edge_sc[(l, i, j)] * x[(l, i, j)] for i in range(n) for j in range(n) if i != j)
        )
        model.Add(line_hours <= cap_sc[l])

    objective = (
        sum(prod_sc[(l, i)] * y[(l, i)] for l in range(len(lines)) for i in range(n))
        + sum(edge_sc[(l, i, j)] * x[(l, i, j)] for l in range(len(lines)) for i in range(n) for j in range(n) if i != j)
    )
    model.Minimize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)
    print(f"CP-SAT horas grafo: {solver.StatusName(status)} | objetivo {solver.ObjectiveValue() / scale:.2f}h")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "_status": solver.StatusName(status),
            "_eligibility": eligibility,
            "_eligibility_summary": eligibility_summary,
            "_ineligible_skus": ineligible_skus,
            "_fixed_line_conflicts": fixed_line_conflicts,
            "_urgent_orders": urgent_applied,
            "_urgent_errors": urgent_errors,
            "_demand_modeled": demand,
        }

    results: Dict[str, Dict] = {
        "_status": solver.StatusName(status),
        "_eligibility": eligibility,
        "_eligibility_summary": eligibility_summary,
        "_ineligible_skus": ineligible_skus,
        "_fixed_line_conflicts": fixed_line_conflicts,
        "_urgent_orders": urgent_applied,
        "_urgent_errors": urgent_errors,
        "_demand_modeled": demand,
    }
    schedule_rows = []
    original_rows = []
    for l, line in enumerate(lines):
        assigned_i = [i for i in range(n) if solver.Value(y[(l, i)]) == 1]
        current = next((i for i in assigned_i if solver.Value(start[(l, i)]) == 1), None)
        seq_i: List[int] = []
        while current is not None and current not in seq_i:
            seq_i.append(current)
            current = next(
                (j for j in assigned_i if j != current and solver.Value(x[(l, current, j)]) == 1),
                None,
            )
        seq_i.extend([i for i in assigned_i if i not in seq_i])
        optimized_seq = [skus[i] for i in seq_i]

        original_seq = [sku for sku in original_sequences.get(line, []) if sku in skus]
        original_h = estimate_sequence_graph_hours(
            original_seq,
            hl_by_sku,
            line,
            throughput,
            edge_mats[line].reindex(index=original_seq, columns=original_seq) if original_seq else pd.DataFrame(),
        )
        optimized_h = estimate_sequence_graph_hours(
            optimized_seq,
            hl_by_sku,
            line,
            throughput,
            edge_mats[line].reindex(index=optimized_seq, columns=optimized_seq) if optimized_seq else pd.DataFrame(),
        )
        if not optimized_h["details"].empty:
            optimized_h["details"]["original_tren"] = optimized_h["details"]["sku"].map(original_by_sku)
            schedule_rows.extend(optimized_h["details"].to_dict("records"))
        if not original_h["details"].empty:
            original_rows.extend(original_h["details"].to_dict("records"))

        cap = float(hours_per_week[line])
        results[line] = {
            "seq_original": original_seq,
            "seq_optimized": optimized_seq,
            "hours_original": original_h["total_h"],
            "hours_optimized": optimized_h["total_h"],
            "hours_saved": round(original_h["total_h"] - optimized_h["total_h"], 4),
            "spare_hours": round(cap - optimized_h["total_h"], 4),
            "capacity_ok": optimized_h["total_h"] <= cap + 1e-6,
            "prod_hours": optimized_h["prod_h"],
            "transition_hours": optimized_h["transition_h"],
            "details": optimized_h["details"],
            "original_details": original_h["details"],
            "edge_hours": edge_mats[line].reindex(index=optimized_seq, columns=optimized_seq),
        }

    results["_schedule_df"] = pd.DataFrame(schedule_rows)
    results["_original_schedule_df"] = pd.DataFrame(original_rows)
    return results


def run_fallback_heuristic(
    demanda_semanal: pd.DataFrame,
    matrices: Dict,
    throughput: pd.DataFrame,
    hours_per_week: Dict[str, float],
    priority_orders: List[dict],
    *,
    lines: List[str] = LINES,
    setup_hours: float = DEFAULT_SETUP_HOURS,
) -> Dict[str, Dict]:
    """Previous local sequencing fallback, using the corrected shared utilities."""
    results: Dict[str, Dict] = {}
    schedule_rows = []
    for line in lines:
        skus_l = demanda_semanal[demanda_semanal["tren"].astype(str) == line]["sku"].tolist()
        if not skus_l:
            continue
        hl_d = demanda_semanal[demanda_semanal["tren"].astype(str) == line].set_index("sku")["hl_total"].to_dict()
        cost_mat = build_cost_matrix(line, skus_l, matrices)
        changeover_mat = build_theoretical_changeover_matrix(
            line,
            skus_l,
            minor_changeover_hours=DEFAULT_MINOR_CHANGEOVER_HOURS,
            fallback_hours=setup_hours,
        )
        seq = two_opt_improve(greedy_nn_sequence(skus_l, cost_mat), cost_mat)
        seq = apply_priority_constraints(seq, priority_orders, line)
        cost_base = sequence_cost(skus_l, cost_mat)
        cost_opt = sequence_cost(seq, cost_mat)
        h_est = estimate_hours(seq, hl_d, line, throughput, changeover_mat, setup_hours=setup_hours)
        details = h_est["details"]
        if not details.empty:
            details["original_tren"] = line
            details["priority_tipo"] = ""
            details["cost_to_next"] = [
                float(cost_mat.loc[sku, seq[pos + 1]]) if pos < len(seq) - 1 else np.nan
                for pos, sku in enumerate(seq)
            ]
            schedule_rows.extend(details.to_dict("records"))
        results[line] = {
            "seq_baseline": skus_l,
            "seq_optimized": seq,
            "assigned_skus": skus_l,
            "cost_baseline": cost_base,
            "cost_optimized": cost_opt,
            "improvement_pct": (cost_base - cost_opt) / max(abs(cost_base), 1e-6) * 100,
            "hours_estimated": h_est["total_h"],
            "prod_hours_estimated": h_est["prod_h"],
            "setup_hours_estimated": h_est["setup_h"],
            "baseline_hours_estimated": h_est["total_h"],
            "capacity_ok": h_est["total_h"] <= float(hours_per_week[line]) + 1e-6,
            "details": details,
            "baseline_details": h_est["details"],
            "cost_mat": cost_mat,
            "full_cost_mat": cost_mat,
            "changeover_mat": changeover_mat,
        }
    results["_schedule_df"] = pd.DataFrame(schedule_rows)
    return results
