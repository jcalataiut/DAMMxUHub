"""LineWise beta dashboard.

Run:
    streamlit run analisis_postmorten/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from bokeh.models import Arrow, ColumnDataSource, HoverTool, LabelSet, NormalHead
from bokeh.plotting import figure

try:
    import networkx as nx
except ImportError:  # pragma: no cover - networkx is in requirements
    nx = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from planning_ai import (  # noqa: E402
    DEFAULT_END_DATE,
    DEFAULT_HOURS_PER_WEEK,
    DEFAULT_START_DATE,
    LINES,
    build_beta_context,
    build_beta_scenarios,
    graph_coverage_table,
    schedule_blocks,
    transition_explanations,
)


st.set_page_config(
    page_title="LineWise beta",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem; padding-bottom: 1rem;}
    div[data-testid="stMetricValue"] {font-size: 1.55rem;}
    div[data-testid="stMetricLabel"] {font-size: 0.85rem;}
    .small-muted {color: #667085; font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


SCENARIO_COLORS = {
    "Plan empresa": "#4C78A8",
    "Produccion real": "#F58518",
    "Plan AI sobre real": "#54A24B",
    "Plan AI sobre plan": "#54A24B",
}
CHANGE_COLOR = "#9D9DA6"
STATUS_COLORS = {
    "OK": "#54A24B",
    "BAJO_PLAN": "#E45756",
    "SOBRE_PLAN": "#72B7B2",
    "NO_PRODUCIDO": "#B279A2",
    "NO_PLANIFICADO": "#F58518",
}


@st.cache_resource(show_spinner="Cargando historico 2025 y semana beta")
def cached_context(data_dir: str, start_date: str, end_date: str) -> Dict:
    return build_beta_context(Path(data_dir), start_date=start_date, end_date=end_date)


@st.cache_data(show_spinner="Resolviendo escenarios con CP-SAT")
def cached_scenarios(
    data_dir: str,
    start_date: str,
    end_date: str,
    hours_tuple: tuple,
    time_limit: int,
    use_learned_changeover: bool,
    ai_demand_source: str,
    fixed_original_lines: bool,
    urgent_json: str,
) -> Dict:
    context = cached_context(data_dir, start_date, end_date)
    hours = dict(hours_tuple)
    urgent_orders = json.loads(urgent_json)
    return build_beta_scenarios(
        context,
        hours_per_week=hours,
        time_limit=float(time_limit),
        urgent_orders=urgent_orders,
        use_learned_changeover=use_learned_changeover,
        ai_demand_source=ai_demand_source,
        fixed_original_lines=fixed_original_lines,
    )


def clean_urgent_orders(urgent_df: pd.DataFrame) -> List[dict]:
    orders = []
    for _, row in urgent_df.iterrows():
        sku = str(row.get("sku", "")).strip()
        active = bool(row.get("active", True))
        if not active or not sku:
            continue
        line = row.get("linea")
        line = None if pd.isna(line) or str(line) in {"", "Auto"} else str(line)
        latest = row.get("latest_position")
        latest = None if pd.isna(latest) or int(latest) <= 0 else int(latest)
        hl_total = row.get("hl_total")
        hl_total = None if pd.isna(hl_total) or float(hl_total) <= 0 else float(hl_total)
        orders.append(
            {
                "order_id": str(row.get("order_id", f"URG-{len(orders) + 1:02d}")),
                "sku": sku,
                "linea": line,
                "hl_total": hl_total,
                "latest_position": latest,
            }
        )
    return orders


def scenario_metric(summary: pd.DataFrame, scenario: str) -> Dict[str, float]:
    df = summary[summary["scenario"] == scenario]
    if df.empty:
        return {"total_h": np.nan, "spare_h": np.nan, "hl_total": np.nan}
    return {
        "total_h": float(df["total_h"].sum()),
        "spare_h": float(df["spare_h"].sum()),
        "hl_total": float(df["hl_total"].sum()),
    }


def hours_bar(summary: pd.DataFrame, hours: Dict[str, float]) -> go.Figure:
    fig = go.Figure()
    scenarios = [s for s in ["Plan empresa", "Produccion real", "Plan AI sobre real", "Plan AI sobre plan"] if s in set(summary["scenario"])]
    for scenario in scenarios:
        df = summary[summary["scenario"] == scenario]
        fig.add_trace(
            go.Bar(
                name=scenario,
                x=[f"L{line}" for line in df["line"]],
                y=df["total_h"],
                marker_color=SCENARIO_COLORS.get(scenario, "#4C78A8"),
                customdata=np.stack([df["prod_h"], df["transition_h"], df["spare_h"]], axis=-1),
                hovertemplate=(
                    "%{x}<br>"
                    "total: %{y:.1f} h<br>"
                    "produccion: %{customdata[0]:.1f} h<br>"
                    "cambio/grafo: %{customdata[1]:.1f} h<br>"
                    "sobrante: %{customdata[2]:.1f} h<extra></extra>"
                ),
            )
        )
    for idx, line in enumerate(LINES):
        fig.add_shape(
            type="line",
            x0=idx - 0.45,
            x1=idx + 0.45,
            y0=hours[line],
            y1=hours[line],
            line=dict(color="#D62728", width=2, dash="dash"),
        )
    fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=40, r=20, t=35, b=45),
        yaxis_title="Horas predichas",
        xaxis_title="",
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#E6E8EB")
    return fig


def gantt_figure(blocks: pd.DataFrame, line: str, hours: Dict[str, float]) -> go.Figure:
    df = blocks[blocks["line"] == line].copy()
    fig = go.Figure()
    if df.empty:
        return fig
    lanes = [
        f"L{line} · Plan empresa",
        f"L{line} · Produccion real",
        f"L{line} · Plan AI sobre real",
        f"L{line} · Plan AI sobre plan",
    ]
    lanes = [lane for lane in lanes if lane in set(df["lane"])]
    for _, row in df.iterrows():
        color = CHANGE_COLOR if row["block"] == "Cambio" else SCENARIO_COLORS.get(row["scenario"], "#4C78A8")
        text = row["sku"] if row["block"] != "Cambio" and row["duration_h"] >= 1.2 else ""
        fig.add_trace(
            go.Bar(
                x=[row["duration_h"]],
                y=[row["lane"]],
                base=row["start_h"],
                orientation="h",
                marker=dict(color=color, line=dict(color="white", width=0.5)),
                text=text,
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=(
                    f"{row['scenario']}<br>"
                    f"SKU: {row['sku']}<br>"
                    "inicio: %{base:.2f} h<br>"
                    "duracion: %{x:.2f} h<extra></extra>"
                ),
                showlegend=False,
            )
        )
    fig.add_vline(x=hours[line], line_color="#D62728", line_dash="dash", line_width=2)
    fig.update_layout(
        height=max(320, 90 * len(lanes)),
        margin=dict(l=110, r=20, t=30, b=45),
        xaxis_title="Horas desde inicio de semana",
        yaxis=dict(categoryorder="array", categoryarray=lanes[::-1], title=""),
        barmode="overlay",
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#E6E8EB")
    return fig


def status_bar(status_df: pd.DataFrame) -> go.Figure:
    counts = status_df.groupby(["tren", "status"]).size().reset_index(name="n")
    fig = go.Figure()
    for status in counts["status"].unique():
        df = counts[counts["status"] == status]
        fig.add_trace(
            go.Bar(
                name=status,
                x=[f"L{line}" for line in df["tren"]],
                y=df["n"],
                marker_color=STATUS_COLORS.get(status, "#999999"),
            )
        )
    fig.update_layout(
        barmode="stack",
        height=300,
        margin=dict(l=40, r=20, t=30, b=40),
        yaxis_title="SKUs",
        xaxis_title="",
        legend=dict(orientation="h", y=1.15),
        plot_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#E6E8EB")
    return fig


def heatmap(matrix: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale="Viridis",
            colorbar=dict(title="h"),
            hovertemplate="origen: %{y}<br>destino: %{x}<br>%{z:.2f} h<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=430,
        margin=dict(l=70, r=20, t=50, b=90),
        xaxis_tickangle=-45,
        plot_bgcolor="white",
    )
    return fig


default_data_dir = str(ROOT / "OPERACIONS")

with st.sidebar:
    st.header("Parametros")
    data_dir = st.text_input("Directorio de datos", value=default_data_dir)
    selected_line = st.selectbox("Linea", LINES, index=2)
    start_date = st.text_input("Inicio", value=DEFAULT_START_DATE)
    end_date = st.text_input("Fin", value=DEFAULT_END_DATE)
    st.divider()
    hours = {
        "14": st.number_input("Capacidad L14", min_value=40.0, max_value=180.0, value=float(DEFAULT_HOURS_PER_WEEK["14"]), step=1.0),
        "17": st.number_input("Capacidad L17", min_value=40.0, max_value=180.0, value=float(DEFAULT_HOURS_PER_WEEK["17"]), step=1.0),
        "19": st.number_input("Capacidad L19", min_value=40.0, max_value=180.0, value=float(DEFAULT_HOURS_PER_WEEK["19"]), step=1.0),
    }
    use_learned_changeover = st.checkbox("Cambio aprendido por grafo", value=True)
    ai_demand_source = st.radio(
        "Volumen objetivo AI",
        options=["real", "plan"],
        format_func=lambda v: "Produccion real" if v == "real" else "Plan empresa",
        horizontal=False,
    )
    time_limit = st.slider("Tiempo solver", min_value=5, max_value=60, value=20, step=5)


context = cached_context(data_dir, start_date, end_date)

st.title("LineWise beta")
st.caption("Plan empresa, produccion real y propuesta AI evaluadas con el mismo grafo aprendido del historico 2025.")

default_orders = pd.DataFrame(
    [
        {
            "active": False,
            "order_id": "URG-01",
            "sku": "EX1324NB",
            "linea": "19",
            "hl_total": 200.0,
            "latest_position": 3,
        }
    ]
)

tab_overview, tab_graph, tab_urgent = st.tabs(["Comparativa", "Grafo y explicabilidad", "Urgencias"])

with tab_urgent:
    st.subheader("Ordenes urgentes")
    all_skus = context["all_skus"]
    urgent_df = st.data_editor(
        default_orders,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "active": st.column_config.CheckboxColumn("Activa"),
            "sku": st.column_config.SelectboxColumn("SKU", options=all_skus, required=False),
            "linea": st.column_config.SelectboxColumn("Linea", options=["Auto"] + LINES, required=False),
            "hl_total": st.column_config.NumberColumn("HL adicionales", min_value=0.0, step=25.0),
            "latest_position": st.column_config.NumberColumn("Posicion maxima", min_value=0, step=1),
        },
    )
    urgent_orders = clean_urgent_orders(urgent_df)
    if urgent_orders:
        st.dataframe(pd.DataFrame(urgent_orders), width="stretch")
    else:
        st.info("Sin urgencias activas.")

urgent_json = json.dumps(urgent_orders, sort_keys=True)
hours_tuple = tuple((line, hours[line]) for line in LINES)
scenarios = cached_scenarios(
    data_dir,
    start_date,
    end_date,
    hours_tuple,
    time_limit,
    use_learned_changeover,
    ai_demand_source,
    urgent_json,
)
summary = scenarios["summary"].copy()
blocks = schedule_blocks(
    [scenarios["company_plan"], scenarios["real_production"], scenarios["ai_plan"]]
)
ai_status = scenarios["ai_raw"].get("_status")

with tab_overview:
    if ai_status not in {"OPTIMAL", "FEASIBLE"}:
        st.warning(f"Plan AI no factible con la configuracion actual: {ai_status}")

    plan_m = scenario_metric(summary, "Plan empresa")
    real_m = scenario_metric(summary, "Produccion real")
    ai_name = "Plan AI sobre real" if ai_demand_source == "real" else "Plan AI sobre plan"
    ai_m = scenario_metric(summary, ai_name)
    saved_vs_real = real_m["total_h"] - ai_m["total_h"] if pd.notna(ai_m["total_h"]) else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Plan empresa", f"{plan_m['total_h']:.1f} h", f"{plan_m['hl_total']:,.0f} HL")
    c2.metric("Produccion real", f"{real_m['total_h']:.1f} h", f"{real_m['hl_total']:,.0f} HL")
    c3.metric("Plan AI", f"{ai_m['total_h']:.1f} h" if pd.notna(ai_m["total_h"]) else "n/a", f"{ai_m['hl_total']:,.0f} HL" if pd.notna(ai_m["hl_total"]) else "")
    c4.metric("Ahorro vs real", f"{saved_vs_real:+.1f} h" if pd.notna(saved_vs_real) else "n/a")

    st.plotly_chart(hours_bar(summary, hours), width="stretch")
    st.plotly_chart(gantt_figure(blocks, selected_line, hours), width="stretch")

    st.subheader("Resumen por linea")
    display_summary = summary.copy()
    for col in ["hl_total", "prod_h", "transition_h", "total_h", "spare_h"]:
        display_summary[col] = display_summary[col].round(2)
    st.dataframe(display_summary, width="stretch", hide_index=True)

    comparison = context["comparison"]
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Plan vs real por SKU")
        sku_df = comparison["by_sku"].copy()
        sku_df["attainment_pct"] = (sku_df["attainment_pct"] * 100).round(1)
        sku_df = sku_df.sort_values("abs_delta_hl", ascending=False)
        st.dataframe(
            sku_df[["tren", "sku", "status", "hl_plan", "hl_real", "delta_hl", "attainment_pct", "oee_real"]].round(2),
            width="stretch",
            hide_index=True,
            height=360,
        )
    with right:
        st.subheader("Estados de cumplimiento")
        st.plotly_chart(status_bar(comparison["by_sku"]), width="stretch")

with tab_graph:
    st.subheader("Cobertura del grafo aprendido")
    coverage = graph_coverage_table(context["raw_matrices"], context["matrices"], context["all_skus"])
    coverage["direct_coverage_pct"] = (coverage["direct_coverage_pct"] * 100).round(2)
    st.dataframe(coverage, width="stretch", hide_index=True)

    ai_results = scenarios["ai_raw"]
    explanations = transition_explanations(ai_results, context["matrices"], context["raw_matrices"])
    line_expl = explanations[explanations["line"] == selected_line].copy() if not explanations.empty else pd.DataFrame()

    seq = scenarios["ai_plan"].get("line_results", {}).get(selected_line, {}).get("seq", [])
    if seq and selected_line in context["matrices"]:
        mat = context["matrices"][selected_line]["changeover_h"].reindex(index=seq, columns=seq)
        st.plotly_chart(heatmap(mat, f"L{selected_line} matriz de cambio suavizada en secuencia AI"), width="stretch")

    st.subheader("Explicacion de arcos seleccionados")
    if line_expl.empty:
        st.info("No hay arcos AI para explicar en esta configuracion.")
    else:
        show = line_expl[
            [
                "position",
                "origin",
                "destination",
                "edge_h",
                "estimated_changeover_h",
                "estimated_oee_degradation",
                "explanation_type",
                "direct_observations",
                "top_evidence",
            ]
        ].copy()
        show["estimated_oee_degradation"] = (show["estimated_oee_degradation"] * 100).round(2)
        st.dataframe(show.round(3), width="stretch", hide_index=True)

        labels = [
            f"{row.origin} -> {row.destination}"
            for _, row in line_expl.iterrows()
        ]
        selected_arc = st.selectbox("Arco", labels)
        selected_idx = labels.index(selected_arc)
        contributors = line_expl.iloc[selected_idx]["top_contributors"]
        if contributors:
            contrib_df = pd.DataFrame(contributors)
            st.dataframe(
                contrib_df[
                    [
                        "historical_transition",
                        "sim_origin",
                        "sim_dest",
                        "bilinear_weight",
                        "effective_weight",
                        "oee_degradation",
                        "changeover_h",
                        "count",
                    ]
                ].round(4),
                width="stretch",
                hide_index=True,
            )

with tab_urgent:
    st.subheader("Validacion de ubicacion")
    if ai_status in {"OPTIMAL", "FEASIBLE"} and urgent_orders:
        rows = []
        for order in urgent_orders:
            sku = order["sku"]
            located = {"order_id": order["order_id"], "sku": sku, "line": "", "position": np.nan}
            for line in LINES:
                seq = scenarios["ai_raw"].get(line, {}).get("seq_optimized", [])
                if sku in seq:
                    located["line"] = line
                    located["position"] = seq.index(sku) + 1
            rows.append(located)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if scenarios["ai_raw"].get("_urgent_errors") is not None and not scenarios["ai_raw"].get("_urgent_errors", pd.DataFrame()).empty:
        st.error("Urgencias no viables")
        st.dataframe(scenarios["ai_raw"]["_urgent_errors"], width="stretch", hide_index=True)

    if scenarios["ai_raw"].get("_ineligible_skus"):
        st.subheader("SKUs fuera del modelo")
        st.write(", ".join(scenarios["ai_raw"]["_ineligible_skus"]))

    conflicts = scenarios["ai_raw"].get("_fixed_line_conflicts", pd.DataFrame())
    if conflicts is not None and not conflicts.empty:
        st.subheader("Conflictos de linea fija")
        st.dataframe(conflicts, width="stretch", hide_index=True)
