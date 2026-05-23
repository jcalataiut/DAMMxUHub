"""Scenario utilities for the LineWise graph-planning beta.

The Streamlit app and notebooks use this module as the shared source of truth:
company plan, actual production, and AI plan are all evaluated with the same
2025 graph-hours model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .data_loaders import (
        LINES,
        actual_sequences_from_production,
        load_all_operations,
        load_planificado_producciones,
        load_real_production_week,
        plan_vs_actual_comparison,
        planned_demand_from_planificado,
        planned_sequences_from_planificado,
    )
    from .post_mortem import PostMortemAnalyzer
    from .scheduling_cp_sat import (
        build_line_behavior_graphs,
        build_transition_hours_matrix,
        estimate_sequence_graph_hours,
        prepare_throughput,
        run_weekly_graph_hours_optimizer,
    )
except ImportError:
    from data_loaders import (
        LINES,
        actual_sequences_from_production,
        load_all_operations,
        load_planificado_producciones,
        load_real_production_week,
        plan_vs_actual_comparison,
        planned_demand_from_planificado,
        planned_sequences_from_planificado,
    )
    from post_mortem import PostMortemAnalyzer
    from scheduling_cp_sat import (
        build_line_behavior_graphs,
        build_transition_hours_matrix,
        estimate_sequence_graph_hours,
        prepare_throughput,
        run_weekly_graph_hours_optimizer,
    )


DEFAULT_START_DATE = "2026-05-18"
DEFAULT_END_DATE = "2026-05-22"
DEFAULT_HOURS_PER_WEEK = {"14": 110.0, "17": 115.0, "19": 115.0}


def _as_line_dict(values: Dict[str, float] | None) -> Dict[str, float]:
    values = values or DEFAULT_HOURS_PER_WEEK
    return {line: float(values.get(line, DEFAULT_HOURS_PER_WEEK[line])) for line in LINES}


def real_demand_from_production(df_real: pd.DataFrame) -> pd.DataFrame:
    demand = (
        df_real.groupby(["tren", "sku"], as_index=False)["hl_real"]
        .sum()
        .rename(columns={"hl_real": "hl_total"})
    )
    demand["original_tren"] = demand["tren"]
    return demand


def line_volume_maps(
    df: pd.DataFrame,
    *,
    value_col: str,
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for line in LINES:
        line_df = df[df["tren"].astype(str) == line]
        result[line] = line_df.groupby("sku")[value_col].sum().astype(float).to_dict()
    return result


def all_week_skus(*frames: pd.DataFrame) -> List[str]:
    skus: List[str] = []
    for frame in frames:
        if frame is None or frame.empty or "sku" not in frame:
            continue
        skus.extend(frame["sku"].dropna().astype(str).tolist())
    return sorted(set(skus))


def build_beta_context(
    data_dir: Path,
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    similarity_weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """Load data and build raw/smoothed graph matrices for the beta week."""
    data_dir = Path(data_dir)
    dfs = load_all_operations(data_dir)
    df_plan = load_planificado_producciones(
        data_dir / "Planificado - producciones 14 - 17 - 19.xlsx",
        start_date=start_date,
        end_date=end_date,
    )
    df_real = load_real_production_week(
        data_dir / "Produccion_L14,17,19_18-22.xlsx",
        start_date=start_date,
        end_date=end_date,
    )

    analyzer = PostMortemAnalyzer(
        df_oee=dfs["oee"],
        df_cambios=dfs["cam"],
        df_mantenimiento=dfs["mant"],
        df_tiempo=dfs["tiem"],
        df_volumen=dfs["vol"],
    )
    analyzer.clean_and_isolate_maintenance()
    raw_matrices = analyzer.build_transition_matrices()

    plan_demand = planned_demand_from_planificado(df_plan)
    real_demand = real_demand_from_production(df_real)
    skus = all_week_skus(df_plan, df_real, plan_demand, real_demand)
    smoothed_matrices = analyzer.build_sophisticated_matrices(
        skus,
        similarity_weights=similarity_weights,
    )
    throughput = prepare_throughput(dfs)

    return {
        "dfs": dfs,
        "df_plan": df_plan,
        "df_real": df_real,
        "plan_demand": plan_demand,
        "real_demand": real_demand,
        "plan_sequences": planned_sequences_from_planificado(df_plan),
        "real_sequences": actual_sequences_from_production(df_real),
        "comparison": plan_vs_actual_comparison(df_plan, df_real),
        "analyzer": analyzer,
        "raw_matrices": raw_matrices,
        "matrices": smoothed_matrices,
        "graphs": build_line_behavior_graphs(smoothed_matrices),
        "throughput": throughput,
        "all_skus": skus,
        "start_date": start_date,
        "end_date": end_date,
    }


def estimate_sequence_scenario(
    *,
    scenario: str,
    sequences: Dict[str, List[str]],
    hl_by_line: Dict[str, Dict[str, float]],
    matrices: Dict,
    throughput: pd.DataFrame,
    hours_per_week: Optional[Dict[str, float]] = None,
    use_learned_changeover: bool = True,
) -> Dict:
    """Estimate a fixed sequence scenario with the graph-hours model."""
    hours_per_week = _as_line_dict(hours_per_week)
    line_results: Dict[str, Dict] = {}
    rows = []

    for line in LINES:
        seq = [sku for sku in sequences.get(line, []) if float(hl_by_line.get(line, {}).get(sku, 0.0)) > 0]
        hl_by_sku = {sku: float(hl_by_line.get(line, {}).get(sku, 0.0)) for sku in seq}
        edge_mat = build_transition_hours_matrix(
            line,
            seq,
            matrices,
            throughput,
            hl_by_sku,
            use_learned_changeover=use_learned_changeover,
        )
        estimate = estimate_sequence_graph_hours(seq, hl_by_sku, line, throughput, edge_mat)
        details = estimate["details"].copy()
        if not details.empty:
            details["scenario"] = scenario
            rows.extend(details.to_dict("records"))
        cap = hours_per_week[line]
        line_results[line] = {
            "seq": seq,
            "details": details,
            "edge_hours": edge_mat,
            "hours_total": estimate["total_h"],
            "prod_hours": estimate["prod_h"],
            "transition_hours": estimate["transition_h"],
            "spare_hours": round(cap - estimate["total_h"], 4),
            "capacity_ok": estimate["total_h"] <= cap + 1e-6,
            "hl_total": round(sum(hl_by_sku.values()), 4),
        }

    return {
        "scenario": scenario,
        "line_results": line_results,
        "schedule_df": pd.DataFrame(rows),
    }


def run_ai_plan(
    context: Dict,
    *,
    hours_per_week: Optional[Dict[str, float]] = None,
    demand_source: str = "plan",
    fixed_original_lines: bool = True,
    time_limit: float = 30.0,
    urgent_orders: Optional[List[dict]] = None,
    use_learned_changeover: bool = True,
) -> Dict:
    """Run CP-SAT on either company-plan demand or actual-production demand."""
    hours_per_week = _as_line_dict(hours_per_week)
    if demand_source == "real":
        demand = context["real_demand"]
        original_sequences = context["real_sequences"]
    else:
        demand = context["plan_demand"]
        original_sequences = context["plan_sequences"]

    return run_weekly_graph_hours_optimizer(
        demanda_semanal=demand,
        dfs=context["dfs"],
        matrices=context["matrices"],
        throughput=context["throughput"],
        hours_per_week=hours_per_week,
        original_sequences=original_sequences,
        time_limit=float(time_limit),
        drop_ineligible=True,
        fixed_original_lines=fixed_original_lines,
        urgent_orders=urgent_orders,
        use_learned_changeover=use_learned_changeover,
    )


def ai_result_to_scenario(
    results: Dict,
    *,
    scenario: str = "Plan AI",
) -> Dict:
    rows = []
    line_results: Dict[str, Dict] = {}
    for line in LINES:
        if line not in results:
            continue
        r = results[line]
        details = r.get("details", pd.DataFrame()).copy()
        if not details.empty:
            details["scenario"] = scenario
            rows.extend(details.to_dict("records"))
        line_results[line] = {
            "seq": r.get("seq_optimized", []),
            "details": details,
            "edge_hours": r.get("edge_hours", pd.DataFrame()),
            "hours_total": r.get("hours_optimized", 0.0),
            "prod_hours": r.get("prod_hours", 0.0),
            "transition_hours": r.get("transition_hours", 0.0),
            "spare_hours": r.get("spare_hours", 0.0),
            "capacity_ok": r.get("capacity_ok", False),
            "hl_total": float(details["hl"].sum()) if not details.empty and "hl" in details else 0.0,
        }
    return {
        "scenario": scenario,
        "line_results": line_results,
        "schedule_df": pd.DataFrame(rows),
        "raw_results": results,
    }


def scenario_summary(
    scenarios: Iterable[Dict],
    *,
    hours_per_week: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    hours_per_week = _as_line_dict(hours_per_week)
    rows = []
    for scenario in scenarios:
        name = scenario["scenario"]
        for line in LINES:
            r = scenario.get("line_results", {}).get(line)
            if not r:
                continue
            rows.append(
                {
                    "scenario": name,
                    "line": line,
                    "hl_total": r["hl_total"],
                    "skus": len(r["seq"]),
                    "prod_h": r["prod_hours"],
                    "transition_h": r["transition_hours"],
                    "total_h": r["hours_total"],
                    "capacity_h": hours_per_week[line],
                    "spare_h": r["spare_hours"],
                    "capacity_ok": r["capacity_ok"],
                }
            )
    return pd.DataFrame(rows)


def build_beta_scenarios(
    context: Dict,
    *,
    hours_per_week: Optional[Dict[str, float]] = None,
    time_limit: float = 30.0,
    urgent_orders: Optional[List[dict]] = None,
    use_learned_changeover: bool = True,
    ai_demand_source: str = "real",
    fixed_original_lines: bool = True,
) -> Dict:
    """Build company-plan, actual-production and AI-proposed scenarios."""
    hours_per_week = _as_line_dict(hours_per_week)
    company_plan = estimate_sequence_scenario(
        scenario="Plan empresa",
        sequences=context["plan_sequences"],
        hl_by_line=line_volume_maps(context["df_plan"], value_col="hl_plan"),
        matrices=context["matrices"],
        throughput=context["throughput"],
        hours_per_week=hours_per_week,
        use_learned_changeover=use_learned_changeover,
    )
    real_production = estimate_sequence_scenario(
        scenario="Produccion real",
        sequences=context["real_sequences"],
        hl_by_line=line_volume_maps(context["df_real"], value_col="hl_real"),
        matrices=context["matrices"],
        throughput=context["throughput"],
        hours_per_week=hours_per_week,
        use_learned_changeover=use_learned_changeover,
    )
    ai_raw = run_ai_plan(
        context,
        hours_per_week=hours_per_week,
        demand_source=ai_demand_source,
        fixed_original_lines=fixed_original_lines,
        time_limit=time_limit,
        urgent_orders=urgent_orders,
        use_learned_changeover=use_learned_changeover,
    )
    ai_label = "Plan AI sobre real" if ai_demand_source == "real" else "Plan AI sobre plan"
    ai_plan = ai_result_to_scenario(ai_raw, scenario=ai_label)
    summary = scenario_summary(
        [company_plan, real_production, ai_plan],
        hours_per_week=hours_per_week,
    )
    return {
        "company_plan": company_plan,
        "real_production": real_production,
        "ai_plan": ai_plan,
        "ai_raw": ai_raw,
        "summary": summary,
    }


def schedule_blocks(scenarios: Iterable[Dict]) -> pd.DataFrame:
    """Build production and transition blocks for a Plotly Gantt bar chart."""
    blocks = []
    for scenario in scenarios:
        name = scenario["scenario"]
        for line in LINES:
            details = scenario.get("line_results", {}).get(line, {}).get("details", pd.DataFrame())
            if details is None or details.empty:
                continue
            for _, row in details.iterrows():
                transition_h = float(row.get("transition_h", row.get("setup_h", 0.0)))
                start_h = float(row.get("start_h", 0.0))
                end_h = float(row.get("end_h", 0.0))
                if transition_h > 0:
                    blocks.append(
                        {
                            "scenario": name,
                            "line": line,
                            "lane": f"L{line} · {name}",
                            "sku": "Cambio",
                            "block": "Cambio",
                            "start_h": max(0.0, start_h - transition_h),
                            "end_h": start_h,
                            "duration_h": transition_h,
                        }
                    )
                blocks.append(
                    {
                        "scenario": name,
                        "line": line,
                        "lane": f"L{line} · {name}",
                        "sku": row["sku"],
                        "block": "Produccion",
                        "start_h": start_h,
                        "end_h": end_h,
                        "duration_h": float(row.get("prod_h", 0.0)),
                    }
                )
    return pd.DataFrame(blocks)


def transition_explanations(
    ai_results: Dict,
    matrices: Dict,
    raw_matrices: Dict,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    """Explain every arc selected by the AI plan using the smoothed graph model."""
    rows = []
    if not ai_results or ai_results.get("_status") not in {"OPTIMAL", "FEASIBLE"}:
        return pd.DataFrame()

    for line in LINES:
        if line not in ai_results or line not in matrices:
            continue
        seq = ai_results[line].get("seq_optimized", [])
        edge_hours = ai_results[line].get("edge_hours", pd.DataFrame())
        model = matrices[line].get("model")
        if model is None:
            continue
        for pos, (origin, destination) in enumerate(zip(seq, seq[1:]), start=1):
            explanation = model.explain_estimated_transition(
                line,
                origin,
                destination,
                raw_matrices,
                top_k=top_k,
            )
            edge_h = float(edge_hours.loc[origin, destination]) if (
                origin in edge_hours.index and destination in edge_hours.columns
            ) else np.nan
            top = explanation.get("top_contributing_transitions", [])
            evidence = top[0]["historical_transition"] if top else ""
            rows.append(
                {
                    "line": line,
                    "position": pos,
                    "origin": origin,
                    "destination": destination,
                    "edge_h": edge_h,
                    "estimated_changeover_h": explanation.get("estimated_changeover_h", np.nan),
                    "estimated_oee_degradation": explanation.get("estimated_oee_degradation", np.nan),
                    "explanation_type": explanation.get("explanation_type", ""),
                    "direct_observations": explanation.get("direct_observations_count", 0),
                    "top_evidence": evidence,
                    "top_contributors": top,
                }
            )
    return pd.DataFrame(rows)


def graph_coverage_table(raw_matrices: Dict, smoothed_matrices: Dict, skus: List[str]) -> pd.DataFrame:
    rows = []
    possible = max(len(skus) * (len(skus) - 1), 1)
    for line in LINES:
        raw_count = raw_matrices.get(line, {}).get("count", pd.DataFrame())
        observed = 0
        if not raw_count.empty:
            sub = raw_count.reindex(index=skus, columns=skus)
            observed = int((sub.fillna(0) > 0).sum().sum())
        smoothed = smoothed_matrices.get(line, {}).get("changeover_h", pd.DataFrame())
        smoothed_cells = int(smoothed.reindex(index=skus, columns=skus).notna().sum().sum()) if not smoothed.empty else 0
        rows.append(
            {
                "line": line,
                "skus": len(skus),
                "direct_edges_2025": observed,
                "direct_coverage_pct": observed / possible,
                "smoothed_cells": smoothed_cells,
            }
        )
    return pd.DataFrame(rows)
