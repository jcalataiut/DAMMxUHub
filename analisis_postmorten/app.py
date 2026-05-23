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
import streamlit.components.v1 as components
from bokeh.embed import file_html
from bokeh.models import Arrow, ColumnDataSource, HoverTool, LabelSet, NormalHead
from bokeh.plotting import figure
from bokeh.resources import INLINE

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


def hours_decomposition_figure(summary: pd.DataFrame, hours: Dict[str, float]) -> go.Figure:
    ordered_scenarios = [
        s
        for s in ["Plan empresa", "Produccion real", "Plan AI sobre real", "Plan AI sobre plan"]
        if s in set(summary["scenario"])
    ]
    rows = []
    for line in LINES:
        for scenario in ordered_scenarios:
            item = summary[(summary["line"] == line) & (summary["scenario"] == scenario)]
            if item.empty:
                continue
            row = item.iloc[0].to_dict()
            row["label"] = f"L{line} · {scenario}"
            row["overload_h"] = max(0.0, float(row["total_h"]) - hours[line])
            row["slack_h"] = max(0.0, hours[line] - float(row["total_h"]))
            rows.append(row)

    df = pd.DataFrame(rows)
    fig = go.Figure()
    if df.empty:
        return fig

    trace_defs = [
        ("Tiempo ideal produccion", "prod_h", "#4C78A8"),
        ("Impacto cambios grafo", "transition_h", "#F58518"),
        ("Holgura", "slack_h", "#C7E5CF"),
        ("Sobrecarga", "overload_h", "#E45756"),
    ]
    for name, col, color in trace_defs:
        fig.add_trace(
            go.Bar(
                name=name,
                x=df["label"],
                y=df[col],
                marker_color=color,
                hovertemplate=f"%{{x}}<br>{name}: %{{y:.1f}} h<extra></extra>",
                text=[f"{value:.0f}h" if value >= 4 else "" for value in df[col]],
                textposition="inside",
            )
        )

    for idx, row in df.iterrows():
        fig.add_annotation(
            x=row["label"],
            y=float(row["total_h"]) + max(3.0, 0.025 * df["total_h"].max()),
            text=f"{float(row['total_h']):.1f}h",
            showarrow=False,
            font=dict(size=10, color="#344054"),
        )

    fig.update_layout(
        barmode="stack",
        height=460,
        margin=dict(l=50, r=20, t=40, b=95),
        yaxis_title="Horas",
        xaxis_title="",
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="white",
    )
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(gridcolor="#E6E8EB")
    return fig


def scenario_sequences(scenarios: Dict, line: str) -> Dict[str, List[str]]:
    sequences: Dict[str, List[str]] = {}
    for key in ["company_plan", "real_production", "ai_plan"]:
        scenario = scenarios.get(key, {})
        name = scenario.get("scenario", key)
        seq = scenario.get("line_results", {}).get(line, {}).get("seq", [])
        if seq:
            sequences[name] = list(seq)
    return sequences


def scenario_by_name(scenarios: Dict, name: str) -> Dict:
    for key in ["company_plan", "real_production", "ai_plan"]:
        scenario = scenarios.get(key, {})
        if scenario.get("scenario") == name:
            return scenario
    return {}


def black_spot_edges(raw_matrices: Dict, line: str) -> pd.DataFrame:
    raw = raw_matrices.get(line, {}).get("_raw", pd.DataFrame()).copy()
    if raw.empty or "oee_degradation" not in raw or "count" not in raw:
        return pd.DataFrame()
    raw = raw[raw["count"] >= 2].copy()
    if raw.empty:
        return pd.DataFrame()
    sigma = raw["oee_degradation"].std()
    if pd.isna(sigma):
        sigma = 0.0
    threshold = raw["oee_degradation"].mean() + 1.5 * sigma
    return raw[raw["oee_degradation"] > threshold].copy()


def scale01(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    min_v = float(values.min()) if len(values) else 0.0
    max_v = float(values.max()) if len(values) else 0.0
    if max_v <= min_v:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - min_v) / (max_v - min_v)


def learned_node_table(
    context: Dict,
    scenarios: Dict,
    line: str,
    *,
    critical_top_n: int = 8,
) -> pd.DataFrame:
    seqs = scenario_sequences(scenarios, line)
    nodes = sorted({sku for seq in seqs.values() for sku in seq})
    if not nodes or line not in context["matrices"]:
        return pd.DataFrame()

    matrix = context["matrices"][line]["changeover_h"].reindex(index=nodes, columns=nodes)
    matrix = matrix.astype(float)
    for sku in nodes:
        if sku in matrix.index and sku in matrix.columns:
            matrix.loc[sku, sku] = np.nan

    counts = context["raw_matrices"].get(line, {}).get("count", pd.DataFrame())
    counts = counts.reindex(index=nodes, columns=nodes).fillna(0) if not counts.empty else pd.DataFrame(0, index=nodes, columns=nodes)
    black_edges = black_spot_edges(context["raw_matrices"], line)
    black_nodes = set()
    if not black_edges.empty:
        black_nodes = set(black_edges["sku_prev"].astype(str)).union(set(black_edges["sku"].astype(str)))

    model = context["matrices"][line].get("model")
    rows = []
    for sku in nodes:
        props = model.get_or_create_sku_properties(sku) if model is not None else {}
        memberships = [name for name, seq in seqs.items() if sku in seq]
        direct_in = float(counts[sku].sum()) if sku in counts.columns else 0.0
        direct_out = float(counts.loc[sku].sum()) if sku in counts.index else 0.0
        rows.append(
            {
                "sku": sku,
                "format": props.get("format", "unknown"),
                "scenarios": " | ".join(memberships),
                "avg_in_h": float(matrix[sku].mean()) if sku in matrix.columns else 0.0,
                "avg_out_h": float(matrix.loc[sku].mean()) if sku in matrix.index else 0.0,
                "max_in_h": float(matrix[sku].max()) if sku in matrix.columns else 0.0,
                "max_out_h": float(matrix.loc[sku].max()) if sku in matrix.index else 0.0,
                "direct_in_count_2025": direct_in,
                "direct_out_count_2025": direct_out,
                "black_spot": sku in black_nodes,
            }
        )

    df = pd.DataFrame(rows)
    risk_avg = df["avg_in_h"].fillna(0) + df["avg_out_h"].fillna(0)
    risk_max = df["max_in_h"].fillna(0) + df["max_out_h"].fillna(0)
    direct_count = np.log1p(df["direct_in_count_2025"].fillna(0) + df["direct_out_count_2025"].fillna(0))
    df["critical_score"] = (
        0.50 * scale01(risk_avg)
        + 0.30 * scale01(risk_max)
        + 0.20 * scale01(direct_count)
    )
    df.loc[df["black_spot"], "critical_score"] = np.maximum(df.loc[df["black_spot"], "critical_score"], 0.92)
    df["critical_rank"] = df["critical_score"].rank(method="first", ascending=False).astype(int)
    df["critical"] = df["black_spot"] | (df["critical_rank"] <= critical_top_n)
    df["critical_reason"] = np.where(
        df["black_spot"],
        "Black spot 2025",
        np.where(df["critical"], "Alta centralidad/coste", "Normal"),
    )
    return df.sort_values(["critical", "critical_score"], ascending=[False, False]).reset_index(drop=True)


def learned_edge_tables(
    context: Dict,
    scenarios: Dict,
    line: str,
    graph_scenario: str,
    *,
    max_background_edges: int = 70,
    edge_mode: str = "Cambios de riesgo",
    critical_top_n: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    node_df = learned_node_table(context, scenarios, line, critical_top_n=critical_top_n)
    if node_df.empty:
        return node_df, pd.DataFrame(), pd.DataFrame()

    nodes = node_df["sku"].tolist()
    matrix = context["matrices"][line]["changeover_h"].reindex(index=nodes, columns=nodes).astype(float)
    counts = context["raw_matrices"].get(line, {}).get("count", pd.DataFrame())
    counts = counts.reindex(index=nodes, columns=nodes).fillna(0) if not counts.empty else pd.DataFrame(0, index=nodes, columns=nodes)

    rows = []
    for origin in nodes:
        for destination in nodes:
            if origin == destination:
                continue
            value = matrix.loc[origin, destination]
            if pd.isna(value) or value <= 0:
                continue
            rows.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "learned_changeover_h": float(value),
                    "direct_count_2025": int(counts.loc[origin, destination]) if origin in counts.index and destination in counts.columns else 0,
                    "kind": "aprendido",
                }
            )
    background = pd.DataFrame(rows)
    if not background.empty:
        if edge_mode == "Mejores cambios":
            background = background.nsmallest(max_background_edges, "learned_changeover_h")
        else:
            background = background.nlargest(max_background_edges, "learned_changeover_h")

    scenario_edges = []
    scenario = scenario_by_name(scenarios, graph_scenario)
    seq = scenario.get("line_results", {}).get(line, {}).get("seq", [])
    edge_hours = scenario.get("line_results", {}).get(line, {}).get("edge_hours", pd.DataFrame())
    for pos, (origin, destination) in enumerate(zip(seq, seq[1:]), start=1):
        learned_h = (
            float(matrix.loc[origin, destination])
            if origin in matrix.index and destination in matrix.columns and pd.notna(matrix.loc[origin, destination])
            else np.nan
        )
        edge_h = (
            float(edge_hours.loc[origin, destination])
            if isinstance(edge_hours, pd.DataFrame)
            and origin in edge_hours.index
            and destination in edge_hours.columns
            and pd.notna(edge_hours.loc[origin, destination])
            else learned_h
        )
        direct_count = (
            int(counts.loc[origin, destination])
            if origin in counts.index and destination in counts.columns
            else 0
        )
        scenario_edges.append(
            {
                "position": pos,
                "origin": origin,
                "destination": destination,
                "learned_changeover_h": learned_h,
                "scenario_transition_h": edge_h,
                "direct_count_2025": direct_count,
                "kind": graph_scenario,
            }
        )

    return node_df, background.reset_index(drop=True), pd.DataFrame(scenario_edges)


def bokeh_learned_graph(
    context: Dict,
    scenarios: Dict,
    line: str,
    graph_scenario: str,
    *,
    max_background_edges: int = 70,
    edge_mode: str = "Cambios de riesgo",
    critical_top_n: int = 8,
):
    node_df, background, scenario_edges = learned_edge_tables(
        context,
        scenarios,
        line,
        graph_scenario,
        max_background_edges=max_background_edges,
        edge_mode=edge_mode,
        critical_top_n=critical_top_n,
    )

    p = figure(
        title=f"L{line} · grafo aprendido 2025",
        width=920,
        height=620,
        x_axis_type=None,
        y_axis_type=None,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        background_fill_color="#FAFBFC",
        border_fill_color="white",
    )
    p.grid.visible = False
    if node_df.empty:
        return p, node_df, background, scenario_edges
    if nx is None:
        return p, node_df, background, scenario_edges

    graph = nx.DiGraph()
    for sku in node_df["sku"]:
        graph.add_node(sku)
    layout_edges = pd.concat(
        [
            background[["origin", "destination", "learned_changeover_h"]] if not background.empty else pd.DataFrame(),
            scenario_edges[["origin", "destination", "learned_changeover_h"]] if not scenario_edges.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    if not layout_edges.empty and {"origin", "destination"}.issubset(layout_edges.columns):
        for _, row in layout_edges.dropna(subset=["origin", "destination"]).iterrows():
            learned_value = pd.to_numeric(row.get("learned_changeover_h", 1.0), errors="coerce")
            if pd.isna(learned_value) or learned_value < 0:
                learned_value = 1.0
            graph.add_edge(row["origin"], row["destination"], weight=1.0 / (1.0 + float(learned_value)))

    if graph.number_of_edges() > 0:
        pos = nx.spring_layout(graph, seed=42, k=0.85, iterations=140, weight="weight")
    else:
        pos = nx.circular_layout(graph)

    def edge_source(df: pd.DataFrame, value_col: str = "learned_changeover_h") -> ColumnDataSource:
        if df.empty:
            return ColumnDataSource({"xs": [], "ys": [], "origin": [], "destination": [], "hours": [], "count": []})
        xs, ys = [], []
        for _, edge in df.iterrows():
            origin, destination = edge["origin"], edge["destination"]
            xs.append([pos[origin][0], pos[destination][0]])
            ys.append([pos[origin][1], pos[destination][1]])
        values = pd.to_numeric(df[value_col], errors="coerce").fillna(df["learned_changeover_h"]).fillna(0.0)
        return ColumnDataSource(
            {
                "xs": xs,
                "ys": ys,
                "origin": df["origin"].tolist(),
                "destination": df["destination"].tolist(),
                "hours": [f"{v:.2f}" for v in values],
                "count": df.get("direct_count_2025", pd.Series([0] * len(df))).astype(int).tolist(),
            }
        )

    bg_src = edge_source(background)
    bg_renderer = p.multi_line(
        "xs",
        "ys",
        source=bg_src,
        line_width=1.2,
        line_alpha=0.18,
        line_color="#667085",
    )

    scenario_color = SCENARIO_COLORS.get(graph_scenario, "#54A24B")
    sc_src = edge_source(scenario_edges, value_col="scenario_transition_h")
    sc_renderer = p.multi_line(
        "xs",
        "ys",
        source=sc_src,
        line_width=4.0,
        line_alpha=0.92,
        line_color=scenario_color,
    )
    for _, edge in scenario_edges.iterrows():
        origin, destination = edge["origin"], edge["destination"]
        if origin not in pos or destination not in pos:
            continue
        p.add_layout(
            Arrow(
                end=NormalHead(size=10, fill_color=scenario_color, line_color=scenario_color),
                x_start=pos[origin][0],
                y_start=pos[origin][1],
                x_end=pos[destination][0],
                y_end=pos[destination][1],
                line_color=scenario_color,
                line_width=2.0,
                line_alpha=0.65,
            )
        )

    scenario_nodes = set(scenario_edges["origin"]).union(set(scenario_edges["destination"])) if not scenario_edges.empty else set()
    node_plot = node_df.copy()
    node_plot["x"] = [pos[sku][0] for sku in node_plot["sku"]]
    node_plot["y"] = [pos[sku][1] for sku in node_plot["sku"]]
    node_plot["score_label"] = node_plot["critical_score"].map(lambda v: f"{v:.3f}")
    node_plot["avg_in_label"] = node_plot["avg_in_h"].map(lambda v: f"{v:.2f}")
    node_plot["avg_out_label"] = node_plot["avg_out_h"].map(lambda v: f"{v:.2f}")
    node_plot["direct_count"] = (node_plot["direct_in_count_2025"] + node_plot["direct_out_count_2025"]).astype(int)
    node_plot["in_selected_path"] = node_plot["sku"].isin(scenario_nodes)
    node_plot["size"] = 11 + 28 * node_plot["critical_score"].clip(0, 1)
    node_plot["fill_color"] = np.select(
        [
            node_plot["black_spot"],
            node_plot["critical"],
            node_plot["in_selected_path"],
        ],
        ["#D62728", "#F58518", scenario_color],
        default="#4C78A8",
    )
    node_plot["line_color"] = np.where(node_plot["critical"], "#101828", "white")
    node_plot["alpha"] = np.where(node_plot["in_selected_path"] | node_plot["critical"], 0.96, 0.74)
    node_plot["label"] = np.where(node_plot["critical"], node_plot["sku"], "")

    node_src = ColumnDataSource(node_plot)
    node_renderer = p.circle(
        "x",
        "y",
        source=node_src,
        size="size",
        fill_color="fill_color",
        fill_alpha="alpha",
        line_color="line_color",
        line_width=1.4,
    )
    labels = LabelSet(
        x="x",
        y="y",
        text="label",
        source=node_src,
        x_offset=8,
        y_offset=5,
        text_font_size="8pt",
        text_color="#101828",
        text_font_style="bold",
    )
    p.add_layout(labels)

    p.add_tools(
        HoverTool(
            renderers=[node_renderer],
            tooltips=[
                ("SKU", "@sku"),
                ("Estado", "@critical_reason"),
                ("Formato", "@format"),
                ("Score critico", "@score_label"),
                ("Cambio medio entrada", "@avg_in_label h"),
                ("Cambio medio salida", "@avg_out_label h"),
                ("Transiciones directas 2025", "@direct_count"),
                ("Escenarios", "@scenarios"),
            ],
        ),
        HoverTool(
            renderers=[bg_renderer],
            tooltips=[
                ("Arista aprendida", "@origin -> @destination"),
                ("Cambio aprendido", "@hours h"),
                ("Veces directas 2025", "@count"),
            ],
        ),
        HoverTool(
            renderers=[sc_renderer],
            tooltips=[
                ("Secuencia", "@origin -> @destination"),
                ("Horas transicion", "@hours h"),
                ("Veces directas 2025", "@count"),
            ],
        ),
    )
    return p, node_df, background, scenario_edges


def render_bokeh_chart(fig, *, height: int = 660) -> None:
    html = file_html(fig, INLINE, fig.title.text or "Bokeh graph")
    components.html(html, height=height, scrolling=False)


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
    fixed_original_lines = st.checkbox(
        "Mantener linea original",
        value=True,
        help="Si se desactiva, la AI puede mover SKUs entre lineas elegibles por formato e historico 2025.",
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
    fixed_original_lines,
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
    st.plotly_chart(hours_decomposition_figure(summary, hours), width="stretch")
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
    st.subheader("Grafo aprendido Bokeh")
    seq_options = list(scenario_sequences(scenarios, selected_line).keys())
    graph_options = ["Solo aprendido"] + seq_options
    default_graph = ai_name if ai_name in graph_options else (seq_options[-1] if seq_options else "Solo aprendido")
    graph_scenario = st.selectbox(
        "Secuencia marcada sobre el grafo",
        graph_options,
        index=graph_options.index(default_graph),
    )
    graph_cols = st.columns([1, 1, 1])
    with graph_cols[0]:
        edge_mode = st.radio(
            "Aristas aprendidas",
            ["Cambios de riesgo", "Mejores cambios"],
            horizontal=True,
        )
    with graph_cols[1]:
        max_background_edges = st.slider("Numero de aristas", min_value=20, max_value=160, value=70, step=10)
    with graph_cols[2]:
        critical_top_n = st.slider("Nodos criticos", min_value=3, max_value=18, value=8, step=1)

    graph_fig, node_df, background_edges, scenario_edges = bokeh_learned_graph(
        context,
        scenarios,
        selected_line,
        graph_scenario,
        max_background_edges=max_background_edges,
        edge_mode=edge_mode,
        critical_top_n=critical_top_n,
    )
    render_bokeh_chart(graph_fig)
    st.caption(
        "Rojo = black spot historico 2025; naranja = nodo critico por coste/centralidad; "
        "verde o color de escenario = SKU usado en la secuencia marcada. Las aristas grises son el grafo aprendido; "
        "las aristas gruesas indican la secuencia seleccionada."
    )

    left_graph, right_graph = st.columns([1.05, 1])
    with left_graph:
        st.subheader("Nodos criticos")
        if node_df.empty:
            st.info("Sin nodos para esta linea y configuracion.")
        else:
            node_show = node_df[
                [
                    "sku",
                    "format",
                    "critical_reason",
                    "critical_score",
                    "avg_in_h",
                    "avg_out_h",
                    "direct_in_count_2025",
                    "direct_out_count_2025",
                    "scenarios",
                ]
            ].copy()
            st.dataframe(node_show.round(3).head(18), width="stretch", hide_index=True)
    with right_graph:
        st.subheader("Aristas de secuencia")
        if scenario_edges.empty:
            st.info("Selecciona un escenario para ver el camino cronologico.")
        else:
            edge_show = scenario_edges[
                [
                    "position",
                    "origin",
                    "destination",
                    "learned_changeover_h",
                    "scenario_transition_h",
                    "direct_count_2025",
                ]
            ].copy()
            st.dataframe(edge_show.round(3), width="stretch", hide_index=True, height=360)

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
