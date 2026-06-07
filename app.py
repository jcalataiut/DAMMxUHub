from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from bokeh.embed import file_html
from bokeh.layouts import column as bk_column, row as bk_row
from bokeh.models import Arrow, Button, ColumnDataSource, CustomJS, Div, HoverTool, Slider, Span, VeeHead
from bokeh.plotting import figure
from bokeh.resources import CDN

import ga_optimizer as ga_mod
from data_loaders import (
    actual_sequences_from_production,
    load_planificado_producciones,
    load_real_production_week,
    planned_demand_from_planificado,
    planned_sequences_from_planificado,
)
from ga_optimizer import (
    HOURS_PER_WEEK,
    LINES,
    OptimizerContext,
    PHYSICAL_FORMAT_BY_LINE,
    STARTUP_HOURS,
    baseline_individual,
    breakdown,
    changeover_hours,
    evolve,
    load_clean_context,
    parse_format,
    throughput_rate,
    schedule_to_gantt,
    set_changeover_policy,
)

HERE = Path(__file__).resolve().parent
CLEAN_DIR = HERE / "clean_data"
RAW_DIR = HERE / "raw_data"
WEEK_START_2026 = "2026-05-18"
WEEK_END_2026 = "2026-05-22 23:59:59"
OPTIMIZER_RESULT_VERSION = "hard_capacity_positive_demand_v2"

st.set_page_config(page_title="LineWise", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 0.5rem;}
div[data-testid="stMetricValue"] {font-size: 1.4rem;}
div[data-testid="stMetricLabel"] {font-size: 0.8rem;}
</style>
""", unsafe_allow_html=True)

FMT_COLOR = {"1/2": "#4c72b0", "1/3": "#dd8452", "2/5": "#55a868"}
LINE_COLOR = {"14": "#1f77b4", "17": "#ff7f0e", "19": "#2ca02c"}
EDGE_TYPE_COLOR = {
    "C_brand": "#2563eb",
    "C_vol": "#d97706",
    "C_pack": "#059669",
    "C_envase": "#dc2626",
    "C0_self": "#64748b",
    "desconocido": "#6b7280",
}


def _first_available(df: pd.DataFrame, candidates: List[str], fallback: str | None = None) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    if fallback is not None:
        return fallback
    raise KeyError(f"None of these columns exist: {candidates}")


def sku_to_graph_node(ctx, sku: str) -> str:
    return str(ctx.sku_node.get(str(sku), str(sku)))


def split_graph_node(node: str) -> tuple[str, str, str, str]:
    parts = str(node).split("|")
    parts = (parts + ["UK"] * 4)[:4]
    return tuple(parts)  # marca, volumen, pack, envase


def classify_node_transition(prev_node: str, next_node: str) -> str:
    if prev_node == next_node:
        return "C0_self"
    pm, pv, pp, pe = split_graph_node(prev_node)
    nm, nv, npack, ne = split_graph_node(next_node)
    if pm != nm:
        return "C_brand"
    if pv != nv:
        return "C_vol"
    if pp != npack:
        return "C_pack"
    if pe != ne:
        return "C_envase"
    return "C0_self"


def load_frames_2025():
    frames = pd.read_csv(CLEAN_DIR / "frames_2025.csv", dtype={"line": str, "week": int})
    nodes = pd.read_csv(CLEAN_DIR / "nodes_2025.csv", dtype={"line": str, "week": int})
    try:
        spots = pd.read_csv(CLEAN_DIR / "black_spots_2025.csv")
    except FileNotFoundError:
        spots = pd.DataFrame(columns=["line", "prev_node", "next_node", "prev_sku", "next_sku", "edge_type"])

    # Canonical graph columns: network nodes are marca × volumen × pack × envase.
    # SKU columns remain available only as traceability to operational orders.
    frames["graph_prev"] = frames[_first_available(frames, ["prev_node", "prev_sku"])].astype(str)
    frames["graph_next"] = frames[_first_available(frames, ["next_node", "next_sku"])].astype(str)
    frames["edge_type"] = frames.get("edge_type", "desconocido")

    nodes["graph_node"] = nodes[_first_available(nodes, ["node", "sku"])].astype(str)
    if "sku" not in nodes.columns:
        nodes["sku"] = nodes["graph_node"]
    nodes["sku"] = nodes["sku"].astype(str)

    spots["graph_prev"] = spots[_first_available(spots, ["prev_node", "prev_sku"], "prev_sku")].astype(str)
    spots["graph_next"] = spots[_first_available(spots, ["next_node", "next_sku"], "next_sku")].astype(str)
    if "prev_sku" not in spots.columns:
        spots["prev_sku"] = spots["graph_prev"]
    if "next_sku" not in spots.columns:
        spots["next_sku"] = spots["graph_next"]
    spots["prev_sku"] = spots["prev_sku"].astype(str)
    spots["next_sku"] = spots["next_sku"].astype(str)
    spots["edge_type"] = spots.get("edge_type", "desconocido")
    return frames, nodes, spots


def load_historical_gantt():
    """Load historical 2025 production orders for the Gantt timeline.
    Adds week_idx (1..53) and end datetime per order."""
    df = pd.read_csv(CLEAN_DIR / "historical_weeks.csv",
                     dtype={"line": str, "sku": str, "of": str})
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["week_start"] = pd.to_datetime(df["week_start"])
    # Map sorted unique week_start → 1..53 (aligned with frames_2025 week numbering)
    week_map = {ws: i + 1 for i, ws in enumerate(sorted(df["week_start"].unique()))}
    df["week_idx"] = df["week_start"].map(week_map)
    df["end"] = df["fecha"] + pd.to_timedelta(df["h_tot"], unit="h")
    df = df.sort_values(["line", "fecha"]).reset_index(drop=True)
    return df


def build_combined_dashboard(gantt_df, frames_df, nodes_df, day_zero, total_days,
                              spot_skus, spot_set, global_pos, ctx, initial_day,
                              node_class):
    """Single Bokeh dashboard: sliding-window Gantt + 3 graphs sharing ONE slider+play.
    All animation is client-side via CustomJS — fluid and stays connected."""
    day_zero_ms = int(pd.Timestamp(day_zero).value // 1_000_000)
    ms_per_day = 86_400_000
    WINDOW_DAYS = 42   # visible width of the Gantt: 6 weeks
    LEAD_DAYS = 21     # cursor centered (3 weeks back, 3 weeks ahead)

    cursor_ms = day_zero_ms + (initial_day - 1) * ms_per_day
    initial_week = max(1, min(53, (initial_day - 1) // 7 + 1))

    # ════════════════════════════ GANTT ════════════════════════════
    lines_y = [f"L{l}" for l in LINES]
    p_gantt = figure(
        x_axis_type="datetime",
        y_range=lines_y[::-1],
        height=240, sizing_mode="stretch_width",
        tools="pan,wheel_zoom,reset,save",
        active_scroll="wheel_zoom",
        background_fill_color="#FAFBFC", border_fill_color="white",
        toolbar_location="right",
        title="Production 2025 — sliding flow",
    )
    p_gantt.ygrid.grid_line_color = None
    p_gantt.xgrid.grid_line_color = "#f0f0f0"
    p_gantt.outline_line_color = None
    p_gantt.yaxis.major_label_text_font_style = "bold"
    p_gantt.title.text_font_size = "11pt"

    initial_x_end = cursor_ms + LEAD_DAYS * ms_per_day
    initial_x_start = initial_x_end - WINDOW_DAYS * ms_per_day
    p_gantt.x_range.start = initial_x_start
    p_gantt.x_range.end = initial_x_end

    df = gantt_df.sort_values("fecha").reset_index(drop=True).copy()
    df["y"] = "L" + df["line"].astype(str)
    df["color"] = df["line"].map(LINE_COLOR).fillna("#888888")
    df["fecha_ms"] = (df["fecha"].astype("int64") // 1_000_000).astype("int64")
    df["end_ms"] = (df["end"].astype("int64") // 1_000_000).astype("int64")
    df["fecha_str"] = df["fecha"].dt.strftime("%Y-%m-%d %H:%M")
    df["h_tot_str"] = df["h_tot"].map(lambda v: f"{v:.1f}h")
    df["hl_str"] = df["hl"].map(lambda v: f"{v:,.0f}")
    df["oee_str"] = df["oee"].map(lambda v: f"{v*100:.0f}%")
    df["graph_node"] = df["node"].astype(str) if "node" in df.columns else df["sku"].astype(str)
    df["is_spot"] = df["graph_node"].isin(spot_skus)
    df["line_color"] = np.where(df["is_spot"], "#8b0000", "#ffffff")
    df["line_w"] = np.where(df["is_spot"], 1.5, 0.4)

    initial_right = np.where(df["fecha_ms"] > cursor_ms,
                              df["end_ms"],
                              np.minimum(df["end_ms"], cursor_ms))
    initial_alpha = np.where(df["fecha_ms"] > cursor_ms, 0.0, 0.9)
    initial_visible_h = (initial_right - df["fecha_ms"]) / 3_600_000.0
    initial_label = np.where((df["fecha_ms"] <= cursor_ms) & (initial_visible_h >= 12),
                              df["sku"].values, "")
    initial_label_x = ((df["fecha_ms"] + initial_right) / 2).astype("int64")

    g_src = ColumnDataSource(dict(
        left=df["fecha_ms"].tolist(),
        right=initial_right.tolist(),
        y=df["y"].tolist(),
        color=df["color"].tolist(),
        alpha=initial_alpha.tolist(),
        end=df["end_ms"].tolist(),
        line_color=df["line_color"].tolist(),
        line_w=df["line_w"].tolist(),
        label=initial_label.tolist(),
        label_x=initial_label_x.tolist(),
        sku=df["sku"].tolist(), of=df["of"].tolist(),
        node=df["graph_node"].tolist(),
        fecha=df["fecha_str"].tolist(), dur=df["h_tot_str"].tolist(),
        hl=df["hl_str"].tolist(), oee=df["oee_str"].tolist(),
    ))

    gr = p_gantt.hbar(y="y", left="left", right="right", height=0.62,
                       fill_color="color", fill_alpha="alpha",
                       line_color="line_color", line_width="line_w",
                       source=g_src)
    p_gantt.add_tools(HoverTool(renderers=[gr], tooltips=[
        ("SKU", "@sku"), ("Node", "@node"), ("OF", "@of"),
        ("Start", "@fecha"), ("Duration", "@dur"),
        ("HL", "@hl"), ("OEE", "@oee"),
    ]))
    p_gantt.text(x="label_x", y="y", text="label", source=g_src,
                  text_font_size="9pt", text_align="center", text_baseline="middle",
                  text_color="white", text_font_style="bold")

    cursor = Span(location=cursor_ms, dimension="height",
                  line_color="#e41a1c", line_dash="solid", line_width=3)
    p_gantt.add_layout(cursor)

    # ════════════════════════════ 3 GRAPHS ════════════════════════════
    edge_sources, node_sources, graph_figs = [], [], []

    for line in LINES:
        line_edges = frames_df[frames_df["line"] == line]
        edge_first = (
            line_edges.groupby(["graph_prev", "graph_next"], as_index=False)
            .agg(
                first_week=("week", "min"),
                edge_type=("edge_type", lambda x: x.mode().iloc[0] if len(x.mode()) else "desconocido"),
                count=("count", "sum") if "count" in line_edges.columns else ("week", "size"),
            )
        )

        line_nodes = nodes_df[nodes_df["line"] == line]
        node_first = (line_nodes.groupby("graph_node")["week"]
                                .min().reset_index().rename(columns={"week": "first_week"}))
        latest_deg = (line_nodes.sort_values("week").groupby("graph_node")["degree"]
                                 .last().reset_index())
        latest_sku = (line_nodes.sort_values("week").groupby("graph_node")["sku"]
                      .last().reset_index())
        node_first = node_first.merge(latest_deg, on="graph_node").merge(latest_sku, on="graph_node", how="left")
        line_cls = node_class.get(line, {})

        pos = global_pos.get(line, {})

        # ── Edges ──
        edge_xs, edge_ys, edge_first_w = [], [], []
        edge_color, edge_w_base, edge_alpha_base = [], [], []
        edge_pair, edge_weight_label = [], []
        edge_type_l, edge_count_l = [], []
        for _, row in edge_first.iterrows():
            o, d = row["graph_prev"], row["graph_next"]
            if o not in pos or d not in pos:
                continue
            edge_xs.append([pos[o][0], pos[d][0]])
            edge_ys.append([pos[o][1], pos[d][1]])
            is_spot = (o, d) in spot_set
            edge_type = row.get("edge_type", "desconocido")
            edge_color.append("#d62728" if is_spot else EDGE_TYPE_COLOR.get(edge_type, "#5a5a5a"))
            try:
                w_h = changeover_hours(ctx, o, d, line)
            except Exception:
                w_h = 1.0
            base_w = max(0.4, min(3.5, 0.4 + w_h * 0.5))
            edge_w_base.append(base_w if is_spot else base_w * 0.55)
            edge_alpha_base.append(0.85 if is_spot else 0.28)
            edge_first_w.append(int(row["first_week"]))
            edge_pair.append(f"{o} → {d}")
            edge_weight_label.append(f"{w_h:.2f}h")
            edge_type_l.append(edge_type)
            edge_count_l.append(int(row.get("count", 1)))

        n_e = len(edge_xs)
        edge_alpha_init = [edge_alpha_base[i] if edge_first_w[i] <= initial_week else 0.0
                           for i in range(n_e)]
        edge_w_init = [edge_w_base[i] if edge_first_w[i] <= initial_week else 0.001
                       for i in range(n_e)]

        e_src = ColumnDataSource(dict(
            xs=edge_xs, ys=edge_ys,
            color=edge_color,
            alpha_base=edge_alpha_base, alpha=edge_alpha_init,
            w_base=edge_w_base, w=edge_w_init,
            first_week=edge_first_w,
            pair=edge_pair, weight=edge_weight_label,
            edge_type=edge_type_l, count=edge_count_l,
        ))
        edge_sources.append(e_src)

        # ── Nodes ──
        node_x, node_y, node_first_w, node_label_l, node_sku_l, node_deg_l = [], [], [], [], [], []
        node_color, node_alpha_base, node_size_base = [], [], []
        node_stroke, node_stroke_w = [], []
        for _, row in node_first.iterrows():
            s = row["graph_node"]
            if s not in pos:
                continue
            node_x.append(pos[s][0])
            node_y.append(pos[s][1])
            deg = float(row["degree"])
            # Smaller nodes for cleaner visualization (range ~5-11px)
            size = 5 + min(6, deg * 0.25)
            cat = line_cls.get(s, "normal")
            if cat == "blackspot":
                node_color.append("#e41a1c")
                node_stroke.append("#8b0000")
                node_stroke_w.append(1.5)
                node_alpha_base.append(0.95)
            elif cat == "critical":
                node_color.append("#ff7f0e")
                node_stroke.append("#b35900")
                node_stroke_w.append(1.0)
                node_alpha_base.append(0.90)
            else:
                node_color.append("#4C78A8")
                node_stroke.append("white")
                node_stroke_w.append(0.6)
                node_alpha_base.append(0.78)
            node_size_base.append(size)
            node_first_w.append(int(row["first_week"]))
            node_label_l.append(s)
            node_sku_l.append(str(row.get("sku", "")))
            node_deg_l.append(int(deg))

        n_n = len(node_x)
        node_alpha_init = [node_alpha_base[i] if node_first_w[i] <= initial_week else 0.0
                           for i in range(n_n)]
        node_size_init = [node_size_base[i] if node_first_w[i] <= initial_week else 0
                          for i in range(n_n)]

        n_src = ColumnDataSource(dict(
            x=node_x, y=node_y,
            color=node_color, stroke=node_stroke, stroke_w=node_stroke_w,
            alpha_base=node_alpha_base, alpha=node_alpha_init,
            size_base=node_size_base, size=node_size_init,
            first_week=node_first_w,
            node=node_label_l, sku=node_sku_l, degree=node_deg_l,
        ))
        node_sources.append(n_src)

        pg = figure(
            title=f"L{line}",
            height=520, sizing_mode="scale_width",
            x_axis_type=None, y_axis_type=None,
            tools="pan,wheel_zoom,reset",
            active_scroll="wheel_zoom",
            background_fill_color="#FAFBFC", border_fill_color="white",
            toolbar_location=None,
            match_aspect=True,
        )
        pg.grid.visible = False
        pg.outline_line_color = "#dddddd"
        pg.title.text_font_size = "13pt"
        pg.title.text_font_style = "bold"
        pg.x_range.range_padding = 0.10
        pg.y_range.range_padding = 0.10

        edges_r = pg.multi_line("xs", "ys", source=e_src,
                                 line_color="color", line_width="w",
                                 line_alpha="alpha", line_join="round")
        pg.add_tools(HoverTool(renderers=[edges_r], tooltips=[
            ("Transition", "@pair"), ("Type", "@edge_type"),
            ("Count", "@count"), ("Changeover", "@weight"),
        ]))
        nodes_r = pg.scatter("x", "y", source=n_src, size="size",
                              fill_color="color", fill_alpha="alpha",
                              line_color="stroke", line_width="stroke_w")
        pg.add_tools(HoverTool(renderers=[nodes_r], tooltips=[
            ("Node", "@node"), ("SKU ref.", "@sku"), ("Connections", "@degree"),
        ]))
        graph_figs.append(pg)

    # ════════════════════════════ SHARED CONTROLS ════════════════════════════
    slider = Slider(start=1, end=total_days, value=initial_day, step=1,
                    title="Day of the year", sizing_mode="stretch_width", show_value=True)
    play_btn = Button(label="▶", width=50, button_type="primary")
    speed_slider = Slider(start=1, end=10, value=3, step=1,
                          title="Speed (days/tick)", width=200)
    week_label = Div(
        text=f"<div style='padding:18px 8px 0 14px;font-size:14px;'><b>Week {initial_week}/53</b></div>",
        width=140,
    )

    update_cb = CustomJS(args=dict(
        g_src=g_src, cursor=cursor, x_range=p_gantt.x_range,
        edge_srcs=edge_sources, node_srcs=node_sources,
        week_label=week_label,
        day_zero_ms=day_zero_ms, ms_per_day=ms_per_day,
        window_days=WINDOW_DAYS, lead_days=LEAD_DAYS,
    ), code="""
        const day = cb_obj.value;
        const cursor_ms = day_zero_ms + (day - 1) * ms_per_day;
        const week = Math.min(53, Math.floor((day - 1) / 7) + 1);

        // --- Gantt: cursor + sliding window ---
        cursor.location = cursor_ms;
        const win_end = cursor_ms + lead_days * ms_per_day;
        x_range.start = win_end - window_days * ms_per_day;
        x_range.end = win_end;

        // --- Gantt: bar alpha + clipping ---
        const d = g_src.data;
        for (let i = 0; i < d.left.length; i++) {
            if (d.left[i] > cursor_ms) {
                d.alpha[i] = 0.0;
                d.right[i] = d.end[i];
                d.label[i] = "";
            } else {
                d.alpha[i] = 0.9;
                const r_clip = Math.min(d.end[i], cursor_ms);
                d.right[i] = r_clip;
                const vh = (r_clip - d.left[i]) / 3600000.0;
                d.label[i] = vh >= 12 ? d.sku[i] : "";
                d.label_x[i] = (d.left[i] + r_clip) / 2;
            }
        }
        g_src.change.emit();

        // --- Graphs: edge + node visibility by first_week ---
        for (let g = 0; g < edge_srcs.length; g++) {
            const es = edge_srcs[g].data;
            for (let i = 0; i < es.first_week.length; i++) {
                if (es.first_week[i] <= week) {
                    es.alpha[i] = es.alpha_base[i];
                    es.w[i] = es.w_base[i];
                } else {
                    es.alpha[i] = 0.0;
                    es.w[i] = 0.001;
                }
            }
            edge_srcs[g].change.emit();

            const ns = node_srcs[g].data;
            for (let i = 0; i < ns.first_week.length; i++) {
                if (ns.first_week[i] <= week) {
                    ns.alpha[i] = ns.alpha_base[i];
                    ns.size[i] = ns.size_base[i];
                } else {
                    ns.alpha[i] = 0.0;
                    ns.size[i] = 0;
                }
            }
            node_srcs[g].change.emit();
        }

        week_label.text = "<div style='padding:18px 8px 0 14px;font-size:14px;'><b>Week " + week + "/53</b></div>";
    """)
    slider.js_on_change("value", update_cb)

    play_cb = CustomJS(args=dict(slider=slider, btn=play_btn, speed=speed_slider), code="""
        if (window._gantt_interval) {
            clearInterval(window._gantt_interval);
            window._gantt_interval = null;
            btn.label = "▶";
            return;
        }
        btn.label = "⏸";
        if (slider.value >= slider.end) slider.value = 1;
        window._gantt_interval = setInterval(() => {
            const next = slider.value + speed.value;
            if (next >= slider.end) {
                slider.value = slider.end;
                clearInterval(window._gantt_interval);
                window._gantt_interval = null;
                btn.label = "▶";
                return;
            }
            slider.value = next;
        }, 50);
    """)
    play_btn.js_on_click(play_cb)

    controls = bk_row(play_btn, slider, speed_slider, week_label,
                       sizing_mode="stretch_width")
    graphs_row = bk_row(*graph_figs, sizing_mode="stretch_width")
    return bk_column(controls, p_gantt, graphs_row, sizing_mode="stretch_width")


def build_bokeh_graph(line, edge_df, node_df, black_spots, *,
                      title="", path_edges=None, active_sku=None,
                      pos=None, highlight_nodes=None):
    """Build a Bokeh directed graph for a line.
    3 categories: blackspot (red), critical/hub (orange), normal (blue).
    If pos provided, use fixed positions; otherwise compute with spring_layout.
    If highlight_nodes is a set, only those nodes are shown in full color.
    """
    G = nx.DiGraph()
    node_col = _first_available(node_df, ["graph_node", "node", "sku"])
    prev_col = _first_available(edge_df, ["graph_prev", "prev_node", "prev_sku"])
    next_col = _first_available(edge_df, ["graph_next", "next_node", "next_sku"])
    for _, row in node_df.iterrows():
        G.add_node(row[node_col])
    for _, row in edge_df.iterrows():
        G.add_edge(row[prev_col], row[next_col])

    p = figure(title=title, width=380, height=380,
               x_axis_type=None, y_axis_type=None,
               tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom",
               background_fill_color="#FAFBFC", border_fill_color="white")
    p.grid.visible = False
    p.x_range.range_padding = 0.2
    p.y_range.range_padding = 0.2

    if G.number_of_nodes() == 0:
        return p

    if pos is None:
        pos = nx.spring_layout(G, seed=42, k=0.85, iterations=80)
    else:
        # Use only positions for nodes present in this graph
        pos = {s: pos[s] for s in G.nodes() if s in pos}
    max_deg = max(node_df["degree"].max(), 1)

    # Edge width by changeover time
    has_weight = "weight" in edge_df.columns
    if has_weight and len(edge_df) > 1:
        wmin, wmax = edge_df["weight"].min(), edge_df["weight"].max()
        wr = max(wmax - wmin, 0.01)
    else:
        has_weight = False

    # Edges (normal + black spots) with weighted width
    edge_xs, edge_ys = [], []
    edge_w, edge_c, edge_a = [], [], []
    edge_pair, edge_weight_label, edge_type_l = [], [], []
    for _, row in edge_df.iterrows():
        o, d = row[prev_col], row[next_col]
        if o not in pos or d not in pos:
            continue
        edge_xs.append([pos[o][0], pos[d][0]])
        edge_ys.append([pos[o][1], pos[d][1]])
        if highlight_nodes is not None:
            # Non-path edge when highlighting: faint gray background
            edge_c.append("#999999")
            edge_w.append(0.5)
            edge_a.append(0.12)
        else:
            is_bs = (o, d) in black_spots
            edge_type = row.get("edge_type", "desconocido")
            edge_c.append("#d62728" if is_bs else EDGE_TYPE_COLOR.get(edge_type, "#444444"))
            if has_weight:
                w = float(row["weight"])
                wt = (w - wmin) / wr
                edge_w.append(max(0.3, 0.3 + wt * 3.7))
            else:
                edge_w.append(2.5 if is_bs else 0.8)
            edge_a.append(0.8 if is_bs else 0.25 + (edge_w[-1] / 4.0) * 0.5)
        if has_weight:
            w = float(row["weight"])
            edge_weight_label.append(f"{w:.2f}h")
        else:
            edge_weight_label.append("")
        edge_type_l.append(row.get("edge_type", "desconocido"))
        edge_pair.append(f"{o} → {d}")

    if edge_xs:
        src_e = ColumnDataSource(dict(
            xs=edge_xs, ys=edge_ys, w=edge_w, c=edge_c, a=edge_a,
            pair=edge_pair, weight=edge_weight_label, edge_type=edge_type_l,
        ))
        r_e = p.multi_line("xs", "ys", source=src_e,
                           line_color="c", line_width="w", line_alpha="a",
                           line_join="round")
        p.add_tools(HoverTool(renderers=[r_e], tooltips=[
            ("Transition", "@pair"),
            ("Type", "@edge_type"),
            ("Changeover", "@weight"),
        ]))

    # Path edges (drawn on top, even if not in historical edge_df)
    if path_edges:
        pxs, pys = [], []
        for o, d in path_edges:
            if o in pos and d in pos:
                pxs.append([pos[o][0], pos[d][0]])
                pys.append([pos[o][1], pos[d][1]])
        if pxs:
            p_src = ColumnDataSource(dict(xs=pxs, ys=pys))
            p.multi_line("xs", "ys", source=p_src, line_color="#2ca02c",
                         line_width=5, line_alpha=0.95)

    # 3-category classification: black spot, critical (high degree), normal
    deg_threshold = node_df["degree"].quantile(0.70) if len(node_df) > 0 else 0
    CAT_COLORS = {"blackspot": "#e41a1c", "critical": "#ff7f0e", "normal": "#4C78A8"}

    node_x, node_y, node_s, node_c, node_al, node_sc, node_sl = [], [], [], [], [], [], []
    node_labels, sku_refs, node_degrees = [], [], []
    spot_skus = set(p for pair in black_spots for p in pair)
    for _, row in node_df.iterrows():
        s = row[node_col]
        if s not in pos:
            continue
        node_x.append(pos[s][0])
        node_y.append(pos[s][1])
        if highlight_nodes is not None and s not in highlight_nodes:
            sz = 6
            cat = "normal"
        else:
            sz = 14  # same size for all active nodes
            if s in spot_skus:
                cat = "blackspot"
            elif row["degree"] >= deg_threshold:
                cat = "critical"
            else:
                cat = "normal"
        node_s.append(sz)
        if active_sku is not None and s == active_sku:
            node_c.append("#2ca02c")
            node_al.append(0.95)
            node_sc.append("#1a6b1a")
            node_sl.append(2.0)
        elif highlight_nodes is not None and s not in highlight_nodes:
            node_c.append("#cccccc")
            node_al.append(0.25)
            node_sc.append("#eeeeee")
            node_sl.append(0.5)
        else:
            node_c.append(CAT_COLORS[cat])
            node_al.append(0.85)
            if cat == "blackspot":
                node_sc.append("#8b0000")
                node_sl.append(2.5)
            else:
                node_sc.append("white")
                node_sl.append(1.0)
        node_labels.append(str(s))
        sku_refs.append(str(row.get("sku", "")))
        node_degrees.append(int(row.get("degree", 0)))

    if node_x:
        src_n = ColumnDataSource(dict(
            x=node_x, y=node_y, size=node_s, color=node_c, alpha=node_al,
            stroke=node_sc, sw=node_sl,
            node=node_labels, sku=sku_refs, degree=node_degrees,
        ))
        r = p.scatter("x", "y", source=src_n, size="size",
                       fill_color="color", fill_alpha="alpha",
                       line_color="stroke", line_width="sw")
        p.add_tools(HoverTool(renderers=[r], tooltips=[
            ("Node", "@node"), ("SKU ref.", "@sku"), ("Connections", "@degree"),
        ]))

    return p


def build_optimized_sequence_graph(ctx, line: str, sequence: List[str], *, title: str = ""):
    """Visualize the actual optimized route on a line, not the historical network.

    Nodes are production orders/SKUs in sequence order. Each node carries its
    canonical product-format graph node, and edges are the actual consecutive
    changeovers executed by the optimal plan.
    """
    sequence = [sku for sku in sequence if float(ctx.volumes.get(sku, 0.0)) > 1e-9]
    p = figure(
        title=title or f"L{line}",
        width=430,
        height=360,
        x_axis_type=None,
        y_axis_type=None,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        background_fill_color="#FAFBFC",
        border_fill_color="white",
    )
    p.grid.visible = False
    p.outline_line_color = "#dddddd"
    p.title.text_font_size = "13pt"
    p.title.text_font_style = "bold"

    if not sequence:
        return p

    n = len(sequence)
    # A light zig-zag keeps labels readable while preserving left-to-right order.
    xs = list(range(n))
    ys = [0.18 if i % 2 else -0.18 for i in range(n)]
    if n == 1:
        ys = [0.0]

    node_values = [sku_to_graph_node(ctx, sku) for sku in sequence]
    node_parts = [split_graph_node(node) for node in node_values]
    formats = [parts[1] for parts in node_parts]
    node_colors = [FMT_COLOR.get(fmt, "#4C78A8") for fmt in formats]
    volumes = [float(ctx.volumes.get(sku, 0.0)) for sku in sequence]
    rates = [throughput_rate(ctx, sku, line) for sku in sequence]
    prod_hours = [vol / max(rate, 1e-6) for vol, rate in zip(volumes, rates)]

    edge_rows = {
        "xs": [], "ys": [], "color": [], "width": [], "alpha": [],
        "pair": [], "edge_type": [], "hours": [], "from_node": [], "to_node": [],
    }
    for i, (prev_sku, next_sku) in enumerate(zip(sequence, sequence[1:])):
        prev_node = node_values[i]
        next_node = node_values[i + 1]
        edge_type = classify_node_transition(prev_node, next_node)
        hours = changeover_hours(ctx, prev_sku, next_sku, line)
        color = EDGE_TYPE_COLOR.get(edge_type, "#6b7280")
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        edge_rows["xs"].append([x0, x1])
        edge_rows["ys"].append([y0, y1])
        edge_rows["color"].append(color)
        edge_rows["width"].append(2.0 + min(5.0, hours * 1.2))
        edge_rows["alpha"].append(0.82)
        edge_rows["pair"].append(f"{prev_sku} → {next_sku}")
        edge_rows["edge_type"].append(edge_type)
        edge_rows["hours"].append(hours)
        edge_rows["from_node"].append(prev_node)
        edge_rows["to_node"].append(next_node)

        p.add_layout(Arrow(
            end=VeeHead(size=9, fill_color=color, line_color=color),
            x_start=x0,
            y_start=y0,
            x_end=x1,
            y_end=y1,
            line_color=color,
            line_alpha=0.50,
            line_width=1,
        ))

    if edge_rows["xs"]:
        edge_src = ColumnDataSource(edge_rows)
        edge_r = p.multi_line(
            "xs", "ys", source=edge_src,
            line_color="color", line_width="width", line_alpha="alpha",
            line_join="round",
        )
        p.add_tools(HoverTool(renderers=[edge_r], tooltips=[
            ("Transition", "@pair"),
            ("Type", "@edge_type"),
            ("Changeover", "@hours{0.00} h"),
            ("From node", "@from_node"),
            ("To node", "@to_node"),
        ]))

    node_src = ColumnDataSource(dict(
        x=xs,
        y=ys,
        order=list(range(1, n + 1)),
        sku=sequence,
        node=node_values,
        marca=[p_[0] for p_ in node_parts],
        format=formats,
        pack=[p_[2] for p_ in node_parts],
        envase=[p_[3] for p_ in node_parts],
        color=node_colors,
        volume=volumes,
        prod_h=prod_hours,
        label=[f"{i+1}. {sku}" for i, sku in enumerate(sequence)],
    ))
    node_r = p.scatter(
        "x", "y", source=node_src,
        size=18,
        fill_color="color",
        fill_alpha=0.94,
        line_color="#111827",
        line_width=1.2,
    )
    p.text(
        "x", "y", text="label", source=node_src,
        y_offset=-30,
        text_align="center",
        text_font_size="8pt",
        text_color="#111827",
    )
    p.add_tools(HoverTool(renderers=[node_r], tooltips=[
        ("Order", "@order"),
        ("SKU", "@sku"),
        ("Node", "@node"),
        ("Marca", "@marca"),
        ("Format", "@format"),
        ("Pack", "@pack"),
        ("Envase", "@envase"),
        ("HL", "@volume{0,0}"),
        ("Prod.", "@prod_h{0.00} h"),
    ]))

    p.x_range.range_padding = 0.08
    p.y_range.range_padding = 0.40
    return p


def build_optimized_overlay_graph(
    ctx,
    line: str,
    sequence: List[str],
    edge_df: pd.DataFrame,
    node_df: pd.DataFrame,
    black_spots: set[tuple[str, str]],
    *,
    title: str = "",
    pos: Dict | None = None,
    node_class: Dict[str, str] | None = None,
):
    """Full historical line graph with the optimized route overlaid.

    This keeps the same visual language as Learning 2025 while showing the
    actual route followed by the optimizer. Optimized nodes that did not appear
    historically in this line are added around the shell so moved SKUs remain
    visible.
    """
    sequence = [sku for sku in sequence if float(ctx.volumes.get(sku, 0.0)) > 1e-9]
    route_nodes = [sku_to_graph_node(ctx, sku) for sku in sequence]
    route_node_set = set(route_nodes)
    route_edges = list(zip(route_nodes, route_nodes[1:]))
    route_edge_set = set(route_edges)

    p = figure(
        title=title or f"L{line}",
        width=430,
        height=430,
        x_axis_type=None,
        y_axis_type=None,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        background_fill_color="#FAFBFC",
        border_fill_color="white",
        match_aspect=True,
    )
    p.grid.visible = False
    p.outline_line_color = "#dddddd"
    p.title.text_font_size = "13pt"
    p.title.text_font_style = "bold"

    node_col = _first_available(node_df, ["graph_node", "node", "sku"])
    prev_col = _first_available(edge_df, ["graph_prev", "prev_node", "prev_sku"])
    next_col = _first_available(edge_df, ["graph_next", "next_node", "next_sku"])

    G = nx.DiGraph()
    for _, row in node_df.iterrows():
        G.add_node(str(row[node_col]))
    for _, row in edge_df.iterrows():
        G.add_edge(str(row[prev_col]), str(row[next_col]))
    for node in route_nodes:
        G.add_node(node)
    for u, v in route_edges:
        G.add_edge(u, v)

    if G.number_of_nodes() == 0:
        return p

    pos_all = dict(pos or {})
    missing = [n for n in G.nodes() if n not in pos_all]
    if missing:
        if pos_all:
            radius = max((float(x) ** 2 + float(y) ** 2) ** 0.5 for x, y in pos_all.values())
            radius = max(radius * 1.12, 1.15)
        else:
            radius = 1.15
        for i, node in enumerate(missing):
            angle = 2 * np.pi * (i / max(len(missing), 1)) + np.pi / 10
            pos_all[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

    # Historical background edges: full graph, very faint, colored by edge type.
    hist_rows = {
        "xs": [], "ys": [], "color": [], "alpha": [], "width": [],
        "pair": [], "edge_type": [], "count": [],
    }
    edge_lookup = {}
    for _, row in edge_df.iterrows():
        u, v = str(row[prev_col]), str(row[next_col])
        if u not in pos_all or v not in pos_all:
            continue
        edge_type = row.get("edge_type", classify_node_transition(u, v))
        is_bs = (u, v) in black_spots
        color = "#d62728" if is_bs else EDGE_TYPE_COLOR.get(edge_type, EDGE_TYPE_COLOR["desconocido"])
        count = int(row.get("count", 1) or 1)
        edge_lookup[(u, v)] = row
        hist_rows["xs"].append([pos_all[u][0], pos_all[v][0]])
        hist_rows["ys"].append([pos_all[u][1], pos_all[v][1]])
        hist_rows["color"].append(color)
        hist_rows["alpha"].append(0.18 if (u, v) in route_edge_set else 0.055)
        hist_rows["width"].append(1.2 if (u, v) in route_edge_set else 0.5)
        hist_rows["pair"].append(f"{u} → {v}")
        hist_rows["edge_type"].append(edge_type)
        hist_rows["count"].append(count)

    if hist_rows["xs"]:
        hist_src = ColumnDataSource(hist_rows)
        hist_r = p.multi_line(
            "xs", "ys", source=hist_src,
            line_color="color", line_alpha="alpha", line_width="width",
            line_join="round",
        )
        p.add_tools(HoverTool(renderers=[hist_r], tooltips=[
            ("Historical edge", "@pair"),
            ("Type", "@edge_type"),
            ("Count", "@count"),
        ]))

    # Optimized route edges: same edge-type colors, stronger and with arrows.
    route_rows = {
        "xs": [], "ys": [], "color": [], "width": [],
        "pair": [], "edge_type": [], "hours": [], "from_node": [], "to_node": [],
    }
    for prev_sku, next_sku, u, v in zip(sequence, sequence[1:], route_nodes, route_nodes[1:]):
        if u not in pos_all or v not in pos_all:
            continue
        edge_type = classify_node_transition(u, v)
        hours = changeover_hours(ctx, prev_sku, next_sku, line)
        color = EDGE_TYPE_COLOR.get(edge_type, EDGE_TYPE_COLOR["desconocido"])
        x0, y0 = pos_all[u]
        x1, y1 = pos_all[v]
        route_rows["xs"].append([x0, x1])
        route_rows["ys"].append([y0, y1])
        route_rows["color"].append(color)
        route_rows["width"].append(3.2 + min(3.8, hours * 0.8))
        route_rows["pair"].append(f"{prev_sku} → {next_sku}")
        route_rows["edge_type"].append(edge_type)
        route_rows["hours"].append(hours)
        route_rows["from_node"].append(u)
        route_rows["to_node"].append(v)
        p.add_layout(Arrow(
            end=VeeHead(size=9, fill_color=color, line_color=color),
            x_start=x0,
            y_start=y0,
            x_end=x1,
            y_end=y1,
            line_color=color,
            line_alpha=0.62,
            line_width=1.2,
        ))

    if route_rows["xs"]:
        route_src = ColumnDataSource(route_rows)
        route_r = p.multi_line(
            "xs", "ys", source=route_src,
            line_color="color", line_alpha=0.92, line_width="width",
            line_join="round",
        )
        p.add_tools(HoverTool(renderers=[route_r], tooltips=[
            ("Optimized transition", "@pair"),
            ("Type", "@edge_type"),
            ("Changeover", "@hours{0.00} h"),
            ("From node", "@from_node"),
            ("To node", "@to_node"),
        ]))

    # Nodes: same classification colors as Learning 2025.
    node_class = node_class or {}
    deg = dict(G.degree())
    all_nodes = list(G.nodes())
    route_order = {}
    route_sku = {}
    for idx, (sku, node) in enumerate(zip(sequence, route_nodes), start=1):
        route_order.setdefault(node, idx)
        route_sku.setdefault(node, sku)

    xs, ys, colors, alphas, sizes, strokes, stroke_w = [], [], [], [], [], [], []
    labels, nodes_l, sku_l, class_l, degree_l, order_l = [], [], [], [], [], []
    for node in all_nodes:
        x, y = pos_all[node]
        cat = node_class.get(node, "normal")
        is_route = node in route_node_set
        is_new = node not in set(node_df[node_col].astype(str))
        if cat == "blackspot":
            color = "#e41a1c"
            stroke = "#8b0000"
        elif cat == "critical":
            color = "#ff7f0e"
            stroke = "#b35900"
        else:
            color = "#4C78A8"
            stroke = "white"
        xs.append(x)
        ys.append(y)
        colors.append(color)
        alphas.append(0.96 if is_route else 0.18)
        sizes.append(18 if is_route else 6)
        strokes.append("#111827" if is_route else stroke)
        stroke_w.append(2.2 if is_route else 0.5)
        labels.append(str(route_order.get(node, "")))
        nodes_l.append(node)
        sku_l.append(route_sku.get(node, ""))
        class_l.append("new-in-line" if is_new and is_route else cat)
        degree_l.append(int(deg.get(node, 0)))
        order_l.append(route_order.get(node, ""))

    node_src = ColumnDataSource(dict(
        x=xs, y=ys, color=colors, alpha=alphas, size=sizes,
        stroke=strokes, stroke_w=stroke_w,
        label=labels, node=nodes_l, sku=sku_l, cls=class_l,
        degree=degree_l, order=order_l,
    ))
    node_r = p.scatter(
        "x", "y", source=node_src, size="size",
        fill_color="color", fill_alpha="alpha",
        line_color="stroke", line_width="stroke_w",
    )
    p.text(
        "x", "y", text="label", source=node_src,
        text_align="center", text_baseline="middle",
        text_font_size="8pt", text_font_style="bold",
        text_color="white",
    )
    p.add_tools(HoverTool(renderers=[node_r], tooltips=[
        ("Order", "@order"),
        ("SKU", "@sku"),
        ("Node", "@node"),
        ("Class", "@cls"),
        ("Connections", "@degree"),
    ]))

    p.x_range.range_padding = 0.14
    p.y_range.range_padding = 0.14
    return p


def gantt_figure(ctx, individual, title="", cap=None):
    gantt = schedule_to_gantt(ctx, individual)
    colors = {"changeover": "#d62728", "startup": "#bdbdbd"}
    fig = go.Figure()
    for _, r in gantt.iterrows():
        c = colors.get(r["type"], FMT_COLOR.get(r["format"], "#999"))
        hover = f"<b>{r['task']}</b><br>Start: {r['start_h']:.1f}h<br>Duration: {r['duration_h']:.2f}h"
        if r["type"] == "production":
            hover += f"<br>HL: {r['hl']:,.0f}<br>Throughput: {r['rate_hl_per_h']:.0f} HL/h"
        fig.add_trace(go.Bar(x=[r["duration_h"]], y=[r["line"]], base=r["start_h"], orientation="h",
                              marker=dict(color=c, line=dict(color="black", width=0.3)),
                              text=r["task"] if r["type"] == "production" and r["duration_h"] >= 2 else "",
                              textposition="inside", insidetextanchor="middle",
                              hovertemplate=hover + "<extra></extra>", showlegend=False))
    cap = cap or max(HOURS_PER_WEEK.values())
    for line in LINES:
        fig.add_vline(x=HOURS_PER_WEEK[line], line_dash="dash", line_color="red", opacity=0.5)
    fig.update_layout(barmode="overlay", title=title, xaxis_title="Hours",
                       yaxis=dict(categoryorder="array", categoryarray=[f"L{l}" for l in LINES]),
                       height=270, margin=dict(l=50, r=15, t=35, b=20), plot_bgcolor="white")
    fig.update_xaxes(gridcolor="#eee", range=[0, cap])
    return fig


def gantt_animation(ctx, individual, title=""):
    gantt = schedule_to_gantt(ctx, individual)
    max_h = gantt["end_h"].max()
    cap = max(max(HOURS_PER_WEEK.values()) + 20, max_h)
    step = max(1, int(max_h / 60))
    colors = {"changeover": "#d62728", "startup": "#bdbdbd"}

    frames = []
    for h in range(0, int(max_h) + 1, step):
        data = []
        for ll in [f"L{l}" for l in LINES]:
            lg = gantt[gantt["line"] == ll]
            for _, r in lg.iterrows():
                c = colors.get(r["type"], FMT_COLOR.get(r["format"], "#999"))
                dh = min(r["end_h"], h) - max(r["start_h"], 0)
                if dh > 0:
                    data.append(go.Bar(x=[dh], y=[ll], base=max(r["start_h"], 0), orientation="h",
                                        marker=dict(color=c, line=dict(color="black", width=0.3)),
                                        showlegend=False))
                if r["end_h"] > h:
                    rem = r["end_h"] - max(r["start_h"], h)
                    if rem > 0:
                        data.append(go.Bar(x=[rem], y=[ll], base=max(r["start_h"], h), orientation="h",
                                            marker=dict(color=c, opacity=0.12, line=dict(color="black", width=0.3)),
                                            showlegend=False))
        frames.append(go.Frame(data=data, name=f"{h:.0f}h"))

    fig = go.Figure(frames=frames)
    for ll in [f"L{l}" for l in LINES]:
        fig.add_trace(go.Bar(x=[0], y=[ll], orientation="h", showlegend=False))
    for line in LINES:
        fig.add_vline(x=HOURS_PER_WEEK[line], line_dash="dash", line_color="red", opacity=0.5)
    fig.update_layout(barmode="overlay", title=title, xaxis_title="Hours",
                       yaxis=dict(categoryorder="array", categoryarray=[f"L{l}" for l in LINES]),
                       height=290, plot_bgcolor="white", margin=dict(l=50, r=15, t=35, b=20),
                       updatemenus=[{"type": "buttons", "buttons": [
                           {"label": "▶", "method": "animate", "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}]},
                           {"label": "⏹", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]},
                       ], "direction": "left", "showactive": False, "x": 0, "y": 1.12}],
                       sliders=[{"steps": [{"label": f"{h:.0f}h", "method": "animate", "args": [[f"{h:.0f}h"], {}]} for h in range(0, int(max_h) + 1, step)],
                                 "currentvalue": {"prefix": "Hour: "}}])
    fig.update_xaxes(gridcolor="#eee", range=[0, cap])
    return fig



def scenario_total(ctx, individual, mode, hdi_mass):
    previous = (
        ctx.changeover_mode,
        ctx.changeover_hdi_mass,
    )
    set_changeover_policy(ctx, mode=mode, hdi_mass=hdi_mass)
    bd = breakdown(ctx, individual)
    total = sum(bd[l]["total"] for l in LINES)
    set_changeover_policy(ctx, mode=previous[0], hdi_mass=previous[1])
    return total, bd


def _safe_div(num, den, default=np.nan):
    return float(num / den) if den and np.isfinite(den) else float(default)


def _weighted_average(df, value_col, weight_col):
    if value_col not in df or weight_col not in df:
        return float("nan")
    values = pd.to_numeric(df[value_col], errors="coerce")
    weights = pd.to_numeric(df[weight_col], errors="coerce")
    valid = values.notna() & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def _volumes_by_line_sku(df, value_col):
    if df.empty or value_col not in df:
        return {}
    return {
        (str(line), str(sku)): float(volume)
        for line, sku, volume in (
            df.groupby(["tren", "sku"], as_index=False)[value_col].sum()
            [["tren", "sku", value_col]]
            .itertuples(index=False, name=None)
        )
        if pd.notna(volume) and float(volume) > 0
    }


def _line_sku_volume(volume_map, line, sku, fallback=None):
    if (line, sku) in volume_map:
        return float(volume_map[(line, sku)])
    if fallback is not None:
        return float(fallback.get(sku, 0.0))
    return 0.0


def _filter_positive_sequences(sequences, volume_map, fallback=None):
    filtered = {}
    for line, seq in sequences.items():
        filtered[line] = [
            sku for sku in seq
            if _line_sku_volume(volume_map, line, sku, fallback) > 1e-9
        ]
    return filtered


def scenario_oee(bd):
    prod = sum(v["prod"] for v in bd.values())
    total = sum(v["total"] for v in bd.values())
    return _safe_div(prod, total, 0.0)


def enrich_model_breakdown(bd):
    enriched = {}
    for line, vals in bd.items():
        vals = dict(vals)
        vals["oee"] = _safe_div(vals["prod"], vals["total"], 0.0)
        enriched[line] = vals
    return enriched


def simulate_line_with_volumes(ctx, line, sequence, volume_map, *, observed_oee=None):
    prod = sum(
        _line_sku_volume(volume_map, line, sku, ctx.volumes) / throughput_rate(ctx, sku, line)
        for sku in sequence
    )
    if observed_oee is not None and np.isfinite(observed_oee) and observed_oee > 0:
        total = prod / observed_oee
        return {
            "prod": prod,
            "changeover": max(0.0, total - prod),
            "startup": 0.0,
            "total": total,
            "oee": float(observed_oee),
        }

    co = sum(
        changeover_hours(ctx, sequence[i], sequence[i + 1], line)
        for i in range(len(sequence) - 1)
    )
    startup = STARTUP_HOURS[line] if sequence else 0.0
    total = prod + co + startup
    return {
        "prod": prod,
        "changeover": co,
        "startup": startup,
        "total": total,
        "oee": _safe_div(prod, total, 0.0),
    }


def breakdown_from_sequences(ctx, sequences, volume_map, *, oee_by_line=None):
    return {
        line: simulate_line_with_volumes(
            ctx, line, sequences.get(line, []), volume_map,
            observed_oee=None if oee_by_line is None else oee_by_line.get(line),
        )
        for line in LINES
    }


def real_oee_by_line(df_real):
    return {
        line: _weighted_average(df_real[df_real["tren"] == line], "oee", "hl_real")
        for line in LINES
    }


def scenario_totals(bd):
    return {
        "prod": sum(v["prod"] for v in bd.values()),
        "total": sum(v["total"] for v in bd.values()),
        "oee": scenario_oee(bd),
    }


def context_for_plan_week(ctx, df_plan, df_real):
    weekly = planned_demand_from_planificado(df_plan).copy()
    weekly["hl_total"] = pd.to_numeric(weekly["hl_total"], errors="coerce").fillna(0.0)
    weekly = weekly[weekly["hl_total"] > 1e-9].copy()
    first_dates = df_plan.groupby("sku")["start_ts"].min().rename("first_fecha")
    weekly = weekly.merge(first_dates, on="sku", how="left")
    weekly["original_line"] = weekly["tren"].astype(str)

    skus = weekly["sku"].astype(str).tolist()
    volumes = dict(zip(weekly["sku"], pd.to_numeric(weekly["hl_total"], errors="coerce")))
    volumes = {str(k): float(v) for k, v in volumes.items() if pd.notna(v)}

    sku_format = dict(ctx.sku_format)
    for sku in set(skus) | set(df_real["sku"].astype(str).tolist()):
        sku_format.setdefault(sku, parse_format(sku))

    original_lines = (
        df_plan.groupby("sku")["tren"]
        .agg(lambda values: sorted({str(v) for v in values if str(v) in LINES}))
        .to_dict()
    )
    eligible = dict(ctx.eligible)
    fallback_skus = set(ctx.fallback_skus)
    for sku in skus:
        fmt = sku_format.get(sku, parse_format(sku))
        physical_lines = [line for line in LINES if fmt in PHYSICAL_FORMAT_BY_LINE[line]]
        historical = [
            line for line in LINES
            if fmt in PHYSICAL_FORMAT_BY_LINE[line] and (sku, line) in ctx.hist_pairs
        ]
        planned_lines = [
            line for line in original_lines.get(sku, [])
            if fmt in PHYSICAL_FORMAT_BY_LINE.get(line, set())
        ]
        eligible[sku] = physical_lines or planned_lines
        if not historical:
            fallback_skus.add(sku)

    return replace(
        ctx,
        weekly=weekly,
        skus=skus,
        volumes=volumes,
        sku_format=sku_format,
        eligible=eligible,
        fallback_skus=list(fallback_skus),
    )


def gantt_figure_from_sequences(ctx, sequences, volume_map, title="", cap=None, oee_by_line=None):
    rows = []
    for line in LINES:
        observed_oee = oee_by_line.get(line) if oee_by_line is not None else None
        line_sequence = sequences.get(line, [])
        if not line_sequence:
            continue
            
        if observed_oee is not None and np.isfinite(observed_oee) and observed_oee > 0:
            # Scale production blocks to match observed OEE total time
            # startup + changeovers are kept at standard theoretical values
            startup_h = STARTUP_HOURS[line]
            
            # 1. Compute standard changeovers
            co_hours_list = []
            prev = None
            for sku in line_sequence:
                if prev is not None:
                    co = changeover_hours(ctx, prev, sku, line)
                    co_hours_list.append(co)
                prev = sku
            
            non_prod_h = startup_h + sum(co_hours_list)
            
            # 2. Compute theoretical production time
            prod_h_sum = sum(
                _line_sku_volume(volume_map, line, sku, ctx.volumes) / throughput_rate(ctx, sku, line)
                for sku in line_sequence
            )
            
            total_h = prod_h_sum / observed_oee
            
            # 3. Determine scale factor for production parts
            if prod_h_sum > 0:
                scale_factor = max(1.0, (total_h - non_prod_h) / prod_h_sum)
            else:
                scale_factor = 1.0
                
            # 4. Construct rows with scaled production
            cursor = startup_h
            rows.append({
                "line": f"L{line}", "task": "STARTUP", "sku": "_arr",
                "start_h": 0.0, "end_h": startup_h,
                "duration_h": startup_h, "type": "startup",
                "format": "", "hl": 0.0, "rate_hl_per_h": 0.0,
            })
            
            prev = None
            co_idx = 0
            for sku in line_sequence:
                if prev is not None:
                    co = co_hours_list[co_idx]
                    co_idx += 1
                    if co > 0:
                        rows.append({
                            "line": f"L{line}", "task": f"CO {prev}→{sku}",
                            "sku": sku, "start_h": cursor, "end_h": cursor + co,
                            "duration_h": co, "type": "changeover",
                            "format": ctx.sku_format.get(sku, ""),
                            "hl": 0.0, "rate_hl_per_h": 0.0,
                        })
                        cursor += co
                
                hl = _line_sku_volume(volume_map, line, sku, ctx.volumes)
                rate = throughput_rate(ctx, sku, line)
                prod_h = (hl / rate if rate else 0.0) * scale_factor
                rows.append({
                    "line": f"L{line}", "task": sku, "sku": sku,
                    "start_h": cursor, "end_h": cursor + prod_h,
                    "duration_h": prod_h, "type": "production",
                    "format": ctx.sku_format.get(sku, ""),
                    "hl": hl, "rate_hl_per_h": rate,
                })
                cursor += prod_h
                prev = sku
        else:
            cursor = STARTUP_HOURS[line]
            if sequences.get(line):
                rows.append({
                    "line": f"L{line}", "task": "STARTUP", "sku": "_arr",
                    "start_h": 0.0, "end_h": STARTUP_HOURS[line],
                    "duration_h": STARTUP_HOURS[line], "type": "startup",
                    "format": "", "hl": 0.0, "rate_hl_per_h": 0.0,
                })
            prev = None
            for sku in line_sequence:
                if prev is not None:
                    co = changeover_hours(ctx, prev, sku, line)
                    if co > 0:
                        rows.append({
                            "line": f"L{line}", "task": f"CO {prev}→{sku}",
                            "sku": sku, "start_h": cursor, "end_h": cursor + co,
                            "duration_h": co, "type": "changeover",
                            "format": ctx.sku_format.get(sku, ""),
                            "hl": 0.0, "rate_hl_per_h": 0.0,
                        })
                        cursor += co
                hl = _line_sku_volume(volume_map, line, sku, ctx.volumes)
                rate = throughput_rate(ctx, sku, line)
                prod_h = hl / rate if rate else 0.0
                rows.append({
                    "line": f"L{line}", "task": sku, "sku": sku,
                    "start_h": cursor, "end_h": cursor + prod_h,
                    "duration_h": prod_h, "type": "production",
                    "format": ctx.sku_format.get(sku, ""),
                    "hl": hl, "rate_hl_per_h": rate,
                })
                cursor += prod_h
                prev = sku

    if not rows:
        return go.Figure()

    gantt = pd.DataFrame(rows)
    colors = {"changeover": "#d62728", "startup": "#bdbdbd"}
    fig = go.Figure()
    for _, r in gantt.iterrows():
        c = colors.get(r["type"], FMT_COLOR.get(r["format"], "#999"))
        hover = f"<b>{r['task']}</b><br>Start: {r['start_h']:.1f}h<br>Duration: {r['duration_h']:.2f}h"
        if r["type"] == "production":
            hover += f"<br>HL: {r['hl']:,.0f}<br>Throughput: {r['rate_hl_per_h']:.0f} HL/h"
        fig.add_trace(go.Bar(
            x=[r["duration_h"]], y=[r["line"]], base=r["start_h"], orientation="h",
            marker=dict(color=c, line=dict(color="black", width=0.3)),
            text=r["task"] if r["type"] == "production" and r["duration_h"] >= 2 else "",
            textposition="inside", insidetextanchor="middle",
            hovertemplate=hover + "<extra></extra>", showlegend=False,
        ))
    cap = cap or max(max(HOURS_PER_WEEK.values()), float(gantt["end_h"].max())) + 5
    for line in LINES:
        fig.add_vline(x=HOURS_PER_WEEK[line], line_dash="dash", line_color="red", opacity=0.5)
    fig.update_layout(
        barmode="overlay", title=title, xaxis_title="Hours",
        yaxis=dict(categoryorder="array", categoryarray=[f"L{l}" for l in LINES]),
        height=300, margin=dict(l=50, r=15, t=35, b=20), plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#eee", range=[0, cap])
    return fig


def metric_card(title, value, detail="", accent="#2ca02c"):
    st.markdown(
        f"""
        <div style="border-left: 4px solid {accent}; padding: 0.55rem 0.7rem;
                    background: rgba(255,255,255,0.04); border-radius: 6px;
                    min-height: 86px;">
            <div style="font-size: 0.78rem; opacity: 0.72; margin-bottom: 0.2rem;">{title}</div>
            <div style="font-size: 1.28rem; font-weight: 700;">{value}</div>
            <div style="font-size: 0.78rem; opacity: 0.75;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading 2026 plan and real...")
def get_2026_execution_data():
    df_plan = load_planificado_producciones(
        RAW_DIR / "Planificado - producciones 14 - 17 - 19.xlsx",
        start_date=WEEK_START_2026,
        end_date=WEEK_END_2026,
    )
    df_real = load_real_production_week(
        RAW_DIR / "Produccion_L14,17,19_18-22.xlsx",
        start_date=WEEK_START_2026,
        end_date=WEEK_END_2026,
    )
    return df_plan, df_real


def clean_urgent(urgent_df):
    orders = []
    for _, row in urgent_df.iterrows():
        sku = str(row.get("sku", "")).strip()
        if not bool(row.get("active", True)) or not sku:
            continue
        orders.append({
            "order_id": str(row.get("order_id", f"URG-{len(orders)+1:02d}")),
            "sku": sku,
            "line": None if pd.isna(row.get("line")) or str(row.get("line")) in {"", "Auto"} else str(row["line"]),
            "hl_total": None if pd.isna(row.get("hl_total")) or float(row["hl_total"]) <= 0 else float(row["hl_total"]),
            "latest_position": None,
        })
    return orders


# ── Data ──
@st.cache_resource(show_spinner="Cargando datos…")
def get_ctx():
    return load_clean_context(CLEAN_DIR)

@st.cache_resource(show_spinner="Cargando frames…")
def get_frames():
    return load_frames_2025()

@st.cache_resource(show_spinner="Loading history for Gantt...")
def get_gantt_history():
    return load_historical_gantt()


ctx = get_ctx()
base_ind = baseline_individual(ctx)
base_bd = breakdown(ctx, base_ind)
baseline_total = sum(base_bd[l]["total"] for l in LINES)
frames_2025, nodes_2025, spots_2025 = get_frames()
num_weeks = int(frames_2025["week"].max())
spot_set = set(zip(spots_2025["graph_prev"], spots_2025["graph_next"]))
spot_skus_set = set(p for pair in spot_set for p in pair)
gantt_history = get_gantt_history()
gantt_day_zero = pd.Timestamp(gantt_history["week_start"].min())
gantt_total_days = num_weeks * 7


@st.cache_resource(show_spinner="Construyendo dashboard interactivo…")
def combined_dashboard_html(initial_day):
    """Built ONCE: Gantt + 3 graphs + shared slider. All further interaction is JS."""
    layout = build_combined_dashboard(
        gantt_history, frames_2025, nodes_2025,
        gantt_day_zero, gantt_total_days,
        spot_skus_set, spot_set, global_pos, ctx,
        initial_day=initial_day,
        node_class=get_node_classification(),
    )
    return file_html(layout, CDN, "")


@st.cache_resource(show_spinner="Clasificando nodos por aristas negativas…")
def get_node_classification():
    """Classify graph nodes by count of adjacent black-spot edges.
    Returns {line: {node: 'normal'|'critical'|'blackspot'}}.
    Thresholds: 0 → normal, 1-2 → critical, ≥3 → blackspot."""
    from collections import defaultdict
    cls = {}
    for line in LINES:
        line_spots = spots_2025[spots_2025["line"].astype(str) == line]
        spot_count = defaultdict(int)
        for _, r in line_spots.iterrows():
            spot_count[str(r["graph_prev"])] += 1
            spot_count[str(r["graph_next"])] += 1
        all_skus = set(nodes_2025[nodes_2025["line"] == line]["graph_node"].astype(str))
        all_skus |= set(frames_2025[frames_2025["line"] == line]["graph_prev"].astype(str))
        all_skus |= set(frames_2025[frames_2025["line"] == line]["graph_next"].astype(str))
        line_cls = {}
        for sku in all_skus:
            c = spot_count.get(sku, 0)
            if c >= 3:
                line_cls[sku] = "blackspot"
            elif c >= 1:
                line_cls[sku] = "critical"
            else:
                line_cls[sku] = "normal"
        cls[line] = line_cls
    return cls


@st.cache_resource(show_spinner="Calculating spherical layout...")
def get_global_positions():
    """Spherical (shell) layout per line: 3 concentric rings by category.
    Inner ring = black spots, middle = critical, outer = normal."""
    cls = get_node_classification()
    positions = {}
    for line in LINES:
        ef = frames_2025[frames_2025["line"] == line]
        G = nx.DiGraph()
        for _, row in ef.iterrows():
            G.add_edge(row["graph_prev"], row["graph_next"])
        if G.number_of_nodes() == 0:
            continue
        line_cls = cls.get(line, {})
        nlist = [
            [n for n in G.nodes() if line_cls.get(n) == "blackspot"],
            [n for n in G.nodes() if line_cls.get(n) == "critical"],
            [n for n in G.nodes() if line_cls.get(n) == "normal"],
        ]
        nlist = [s for s in nlist if s]
        try:
            pos = nx.shell_layout(G, nlist=nlist)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=3.0, iterations=200)
        positions[line] = pos
    return positions


global_pos = get_global_positions()

page = st.sidebar.radio("Viewer", ["Learning 2025", "Optimization 2026", "Hyperparameter Tuning (Optuna)"], label_visibility="collapsed")

# ═══════════════════════ PAGE 1: 2025 ═══════════════════════
if page == "Learning 2025":
    st.title("OPla Lab")
    st.subheader("Generative Bayesian Complex Networks")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Total Nodes", nodes_2025["graph_node"].nunique())
    mc2.metric("Unique Edges",
               int(frames_2025.groupby(["line", "graph_prev", "graph_next"]).ngroups))
    mc3.metric("2025 Orders", len(gantt_history))

    # Single combined dashboard: Gantt + 3 graphs sharing one slider+play (all client-side JS)
    # Initial cursor at week 3 (day 21) so the user sees the start of the year with context.
    components.html(combined_dashboard_html(initial_day=21),
                    height=880, scrolling=False)

# ═══════════════════════ PAGE 2: 2026 ═══════════════════════
elif page == "Optimization 2026":
    st.title("Optimization · May 18-22 2026")

    df_plan_2026, df_real_2026 = get_2026_execution_data()
    planner_ind = planned_sequences_from_planificado(df_plan_2026)
    real_ind = actual_sequences_from_production(df_real_2026)
    planner_volumes = _volumes_by_line_sku(df_plan_2026, "hl_plan")
    real_volumes = _volumes_by_line_sku(df_real_2026, "hl_real")
    planner_ind = _filter_positive_sequences(planner_ind, planner_volumes)
    real_ind = _filter_positive_sequences(real_ind, real_volumes)
    real_oee_lines = real_oee_by_line(df_real_2026)
    opt_ctx = context_for_plan_week(ctx, df_plan_2026, df_real_2026)

    with st.sidebar:
        st.divider()
        st.caption("Changeovers 2025")
        co_policy_options = {
            "Bayesian Mean": "bayes_mean",
            "Observed Mean": "observed_mean",
            "Lower HDI": "hdi_lower",
            "Upper HDI": "hdi_upper",
        }
        co_policy_label = st.selectbox(
            "Value used by the optimizer",
            list(co_policy_options.keys()),
            key="co_policy",
        )
        hdi_pct = st.slider("HDI", 80, 99, 95, 1, key="co_hdi_pct")
        co_mode = co_policy_options[co_policy_label]
        hdi_mass = hdi_pct / 100.0
        set_changeover_policy(opt_ctx, mode=co_mode, hdi_mass=hdi_mass)
        st.divider()
        ga_pop = st.slider("Population", 20, 200, 60, 10, key="ga_pop")
        ga_gen = st.slider("Generations", 30, 400, 150, 10, key="ga_gen")
        ga_seed = st.number_input("Seed GA", 42, step=1, key="ga_seed")

    run_btn = st.sidebar.button("▶ Optimize", type="primary", use_container_width=True, key="run_opt")
    planner_bd = breakdown_from_sequences(opt_ctx, planner_ind, planner_volumes)
    real_bd = breakdown_from_sequences(opt_ctx, real_ind, real_volumes, oee_by_line=real_oee_lines)
    planner_totals = scenario_totals(planner_bd)
    real_totals = scenario_totals(real_bd)
    real_oee_global = _weighted_average(df_real_2026, "oee", "hl_real")

    # urgent_orders is read from previous rerun to be available before optimizing
    urgent_orders = st.session_state.get("_urgent_orders", [])

    result_key = f"res_GA_{OPTIMIZER_RESULT_VERSION}_{co_mode}_{hdi_pct}"
    if run_btn or result_key not in st.session_state:
        # Apply urgent orders: extra volume + priority
        original_priority = list(ga_mod.PRIORITY_ORDERS)
        volumes_backup = {}
        urgent_extra: dict = {}
        if urgent_orders:
            extra = []
            for o in urgent_orders:
                sku = o["sku"]
                if o["hl_total"] is not None:
                    volumes_backup.setdefault(sku, opt_ctx.volumes.get(sku, 0))
                    opt_ctx.volumes[sku] = volumes_backup[sku] + o["hl_total"]
                    urgent_extra[sku] = o["hl_total"]
                if o["line"] is not None:
                    extra.append((sku, o["line"]))
                else:
                    for line in opt_ctx.eligible.get(sku, LINES):
                        extra.append((sku, line))
            ga_mod.PRIORITY_ORDERS = original_priority + extra

        progress = st.progress(0, "Optimizando…")
        try:
            def cb(g, b, m):
                progress.progress((g+1)/ga_gen, text=f"G {g+1}/{ga_gen} · mejor={b:.1f}h")
            t0 = time.time()
            best_ind, history = evolve(opt_ctx, pop_size=ga_pop, n_gen=ga_gen, seed=ga_seed, on_generation=cb)
            st.session_state[result_key] = {"schedule": best_ind, "elapsed": time.time()-t0, "urgent_extra": urgent_extra}
            # Compute urgent orders feedback and save to session_state for display below the expander
            active_urgent = [o for o in urgent_orders if o["line"] is not None or opt_ctx.eligible.get(o["sku"], [])]
            urgent_rows = []
            if active_urgent:
                opt_ind_latest = st.session_state[result_key]["schedule"]
                for o in active_urgent:
                    sku = o["sku"]
                    lines = [o["line"]] if o["line"] else opt_ctx.eligible.get(sku, [])
                    for ln in lines:
                        seq = opt_ind_latest.get(ln, [])
                        if sku in seq:
                            pos = seq.index(sku)
                            cutoff = max(0, int(0.25 * len(seq)))
                            vol = f"+{o['hl_total']:.0f} HL" if o["hl_total"] else ""
                            urgent_rows.append({"SKU": sku, "Line": f"L{ln}",
                                                 "Pos": f"{pos+1}/{len(seq)}",
                                                 "Extra": vol,
                                                 "Status": "✓" if pos <= cutoff else "✗"})
            st.session_state["_urgent_feedback"] = urgent_rows
        finally:
            ga_mod.PRIORITY_ORDERS = original_priority
            for sku, orig_val in volumes_backup.items():
                opt_ctx.volumes[sku] = orig_val
        progress.empty()

    result = st.session_state[result_key]
    opt_ind = result["schedule"]

    _urgent_extra = result.get("urgent_extra", {})
    if _urgent_extra:
        _ai_vols = dict(opt_ctx.volumes)
        for _sku, _extra in _urgent_extra.items():
            _ai_vols[_sku] = _ai_vols.get(_sku, 0) + _extra
        ai_ctx = replace(opt_ctx, volumes=_ai_vols)
    else:
        ai_ctx = opt_ctx

    opt_bd = enrich_model_breakdown(breakdown(ai_ctx, opt_ind))
    opt_total = sum(opt_bd[l]["total"] for l in LINES)
    saved = real_totals["total"] - opt_total
    an = "GA"
    min_total, min_bd = scenario_total(ai_ctx, opt_ind, "hdi_lower", hdi_mass)
    worst_total, worst_bd = scenario_total(ai_ctx, opt_ind, "hdi_upper", hdi_mass)
    min_bd = enrich_model_breakdown(min_bd)
    worst_bd = enrich_model_breakdown(worst_bd)
    opt_oee = scenario_oee(opt_bd)
    oee_delta = opt_oee - real_oee_global

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Real Executed", f"{real_totals['total']:.1f}h")
    c2.metric("Optimal", f"{opt_total:.1f}h",
              delta=f"{opt_total - real_totals['total']:.1f}h vs real",
              delta_color="inverse")
    c3.metric("Reduction", f"{saved:.1f}h", delta=f"{saved/real_totals['total']*100:.1f}%")
    c4.metric("Minimum Execution", f"{min_total:.1f}h")
    c5.metric("Worst Execution", f"{worst_total:.1f}h")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Theoretical Planner", f"{planner_totals['total']:.1f}h")
    o2.metric("OEE global real", f"{real_oee_global*100:.1f}%")
    o3.metric("Optimal Global OEE", f"{opt_oee*100:.1f}%", delta=f"{oee_delta*100:+.1f} pp vs real")
    o4.metric("Optimization time", f"{result['elapsed']:.1f}s")

    opt_path = {
        line: [
            (sku_to_graph_node(ai_ctx, a), sku_to_graph_node(ai_ctx, b))
            for a, b in zip(opt_ind[line], opt_ind[line][1:])
        ]
        for line in LINES if len(opt_ind[line]) > 1
    }
    opt_nodes = {
        line: {sku_to_graph_node(ai_ctx, sku) for sku in opt_ind.get(line, [])}
        for line in LINES
    }

    with st.expander("📦 Urgent Orders", expanded=False):
        urgent_df = st.data_editor(
            pd.DataFrame([{"active": False, "order_id": "URG-01", "sku": "EX1324NB",
                           "line": "Auto", "hl_total": 200.0}]),
            num_rows="dynamic", key="urgent_editor",
            column_config={
                "active":    st.column_config.CheckboxColumn("Activa"),
                "order_id":  st.column_config.TextColumn("ID orden"),
                "sku":       st.column_config.SelectboxColumn("SKU", options=opt_ctx.skus, required=False),
                "line":     st.column_config.SelectboxColumn("Line", options=["Auto"] + LINES, required=False),
                "hl_total":  st.column_config.NumberColumn("HL extra", min_value=0.0, step=25.0),
            },
        )
        st.session_state["_urgent_orders"] = clean_urgent(urgent_df) if not urgent_df.empty else []

    _urgent_fb = st.session_state.get("_urgent_feedback", [])
    if _urgent_fb:
        st.dataframe(pd.DataFrame(_urgent_fb), hide_index=True, use_container_width=True)

    _oee_cols = st.columns(3)
    for idx, l in enumerate(LINES):
        _oee_delta = opt_bd[l]["oee"] - real_bd[l]["oee"]
        _oee_cols[idx].metric(
            f"OEE L{l}",
            f"{opt_bd[l]['oee']*100:.1f}%",
            delta=f"{_oee_delta*100:+.1f} pp vs real",
        )

    # 3 Bokeh graphs: full learned graph by line + actual optimized route overlay.
    _node_cls = get_node_classification()
    cg1, cg2, cg3 = st.columns(3)
    for idx, line in enumerate(LINES):
        with [cg1, cg2, cg3][idx]:
            ef = frames_2025[(frames_2025["line"] == line) & (frames_2025["week"] == num_weeks)].copy()
            nf = nodes_2025[(nodes_2025["line"] == line) & (nodes_2025["week"] == num_weeks)].copy()
            fig = build_optimized_overlay_graph(
                ai_ctx,
                line,
                opt_ind.get(line, []),
                ef,
                nf,
                spot_set,
                title=f"L{line}",
                pos=global_pos.get(line),
                node_class=_node_cls.get(line, {}),
            )
            components.html(file_html(fig, CDN, ""), height=460, scrolling=False)

    gantt_cap = max(
        max(v["total"] for v in planner_bd.values()),
        max(v["total"] for v in real_bd.values()),
        max(v["total"] for v in opt_bd.values()),
        max(HOURS_PER_WEEK.values()),
    ) + 10

    st.subheader("Theoretical Planner")
    st.plotly_chart(
        gantt_figure_from_sequences(
            opt_ctx, planner_ind, planner_volumes, title="", cap=gantt_cap,
        ),
        key="planner_g", use_container_width=True,
    )

    st.subheader("Real Executed")
    st.plotly_chart(
        gantt_figure_from_sequences(
            opt_ctx, real_ind, real_volumes, title="", cap=gantt_cap, oee_by_line=real_oee_lines,
        ),
        key="real_g", use_container_width=True,
    )

    avail_total = sum(HOURS_PER_WEEK[l] for l in LINES)
    real_prod_t = sum(real_bd[l]["prod"] for l in LINES)
    real_dead_t = real_totals["total"] - real_prod_t
    opt_prod_t  = sum(opt_bd[l]["prod"] for l in LINES)
    opt_dead_t  = opt_total - opt_prod_t

    st.subheader("Optimal")
    st.plotly_chart(
        gantt_figure(ai_ctx, opt_ind, title="", cap=gantt_cap),
        key="opt_g", use_container_width=True,
    )

    # ── Real vs Optimal Comparison ────────────────────────────────────────────
    st.subheader("Real Executed vs Optimal")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Total horas · Real", f"{real_totals['total']:.1f}h")
    h2.metric("Total hours · Optimal", f"{opt_total:.1f}h",
              delta=f"{opt_total - real_totals['total']:.1f}h vs real",
              delta_color="inverse")
    h3.metric("Horas muertas · Real", f"{real_dead_t:.1f}h")
    h4.metric("Dead hours · Optimal",
              f"{opt_dead_t:.1f}h",
              delta=f"{opt_dead_t - real_dead_t:.1f}h vs real",
              delta_color="inverse")
    st.caption(f"Total available hours (3 lines): {avail_total:.0f}h")
    comp_rows = [
        {
            "Line": f"L{l}",
            "Total Real": f"{real_bd[l]['total']:.1f}h",
            "Muertas Real": f"{real_bd[l]['total'] - real_bd[l]['prod']:.1f}h",
            "Total Optimal": f"{opt_bd[l]['total']:.1f}h",
            "Dead Optimal": f"{opt_bd[l]['total'] - opt_bd[l]['prod']:.1f}h",
            "Disponible": f"{HOURS_PER_WEEK[l]:.0f}h",
        }
        for l in LINES
    ]
    st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

    # ── Improvement dynamics by line ──────────────────────────────────────────
    st.subheader("Improvement dynamics by line")
    _ll = [f"L{l}" for l in LINES]
    _real_h = [real_bd[l]["total"] for l in LINES]
    _opt_h  = [opt_bd[l]["total"]  for l in LINES]

    _COLOR_REAL = "#8B95A5"
    _COLOR_OPT  = "#10B981"

    _fig_dyn = go.Figure()
    _fig_dyn.add_trace(go.Bar(
        name="Real Executed", x=_ll, y=_real_h,
        marker=dict(color=_COLOR_REAL, line=dict(color="#6B7585", width=1.2)),
        text=[f"{h:.0f}h" for h in _real_h], textposition="outside",
        hovertemplate="%{x} Real: %{y:.1f}h<extra></extra>",
    ))
    _fig_dyn.add_trace(go.Bar(
        name="Optimal", x=_ll, y=_opt_h,
        marker=dict(color=_COLOR_OPT, line=dict(color="#059669", width=1.2)),
        text=[f"{h:.0f}h" for h in _opt_h], textposition="outside",
        hovertemplate="%{x} Optimal: %{y:.1f}h<extra></extra>",
    ))
    for i, l in enumerate(LINES):
        _fig_dyn.add_shape(
            type="line", xref="x", yref="y",
            x0=i - 0.4, x1=i + 0.4,
            y0=HOURS_PER_WEEK[l], y1=HOURS_PER_WEEK[l],
            line=dict(color="#EF4444", dash="dash", width=2),
        )
        _ls = real_bd[l]["total"] - opt_bd[l]["total"]
        _fig_dyn.add_annotation(
            x=f"L{l}", y=max(_real_h[i], _opt_h[i]) + 8,
            text=f"{'−' if _ls > 0 else '+'}{abs(_ls):.0f}h",
            showarrow=False,
            font=dict(size=13, color="#10B981" if _ls > 0 else "#EF4444"),
        )
    _fig_dyn.update_layout(
        barmode="group", height=300, plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=10, t=30, b=20),
        yaxis=dict(gridcolor="#eee", title="Hours"),
    )
    st.plotly_chart(_fig_dyn, use_container_width=True, key="dyn_chart")

    # ── SKU migration: Theoretical Planner → Optimal ───────────────────────────
    st.subheader("SKU migration across lines")
    st.caption("SKUs that the optimizer reassigns to a different line compared to the theoretical planner.")

    _planner_line = {sku: line for line, skus in planner_ind.items() for sku in skus}
    _opt_line = {sku: line for line, skus in opt_ind.items() for sku in skus}
    _all_skus = set(_planner_line) | set(_opt_line)
    _migrations = [
        {"SKU": sku,
         "Planner": f"L{_planner_line[sku]}" if sku in _planner_line else "—",
         "Optimal":  f"L{_opt_line[sku]}"     if sku in _opt_line     else "—",
         "Changeover":  _planner_line.get(sku) != _opt_line.get(sku)}
        for sku in sorted(_all_skus)
    ]
    _moved = [r for r in _migrations if r["Changeover"]]

    if _moved:
        # Sankey con colores vivos: nodos izq = Planner, right nodes = Optimal
        # Each line has its color; flows inherit origin line color
        def _hex_to_rgba(hex_color, alpha):
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            return f"rgba({r},{g},{b},{alpha})"

        _line_idx = {l: i for i, l in enumerate(LINES)}
        _flow: dict = {}
        for r in _moved:
            if r["Planner"] == "—" or r["Optimal"] == "—":
                continue
            src = _line_idx[r["Planner"][1:]]
            tgt = _line_idx[r["Optimal"][1:]] + len(LINES)
            _flow[(src, tgt)] = _flow.get((src, tgt), 0) + 1

        _node_labels = [f"Planner  L{l}" for l in LINES] + [f"Optimal  L{l}" for l in LINES]
        _node_colors = [_hex_to_rgba(LINE_COLOR[l], 0.85) for l in LINES] * 2

        _link_sources = [s for s, _ in _flow]
        _link_targets = [t for _, t in _flow]
        _link_values  = [v for v in _flow.values()]
        _link_colors  = [_hex_to_rgba(LINE_COLOR[LINES[s]], 0.35) for s in _link_sources]
        _link_labels  = [
            f"{v} SKU{'s' if v > 1 else ''}: "
            + ", ".join(r["SKU"] for r in _moved
                        if r["Planner"] != "—" and r["Optimal"] != "—"
                        and _line_idx[r["Planner"][1:]] == s
                        and _line_idx[r["Optimal"][1:]] + len(LINES) == t)
            for (s, t), v in _flow.items()
        ]

        _fig_sk = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=28,
                thickness=26,
                label=_node_labels,
                color=_node_colors,
                line=dict(color="rgba(255,255,255,0.6)", width=1),
            ),
            link=dict(
                source=_link_sources,
                target=_link_targets,
                value=_link_values,
                color=_link_colors,
                label=_link_labels,
                hovertemplate="%{label}<extra></extra>",
            ),
        ))
        _fig_sk.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="white",
            font=dict(size=13, color="#333"),
        )
        st.plotly_chart(_fig_sk, use_container_width=True, key="sankey_migration")

        _move_df = pd.DataFrame([{"SKU": r["SKU"], "Planner": r["Planner"], "→ Optimal": r["Optimal"]}
                                  for r in _moved])
        st.dataframe(_move_df, hide_index=True, use_container_width=True)
    else:
        st.info("The optimizer keeps all SKUs on the same line as the theoretical planner.")

    # ── Contrafactual 2025 ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Counterfactual 2025 — Potential annual savings")

    if "cf_2025" not in st.session_state:
        _ctx_cf = replace(ctx, changeover_mode="observed_mean", changeover_cache={})
        _cf_rows = []
        for _wk in range(1, num_weeks + 1):
            _wd = gantt_history[gantt_history["week_idx"] == _wk]
            _real_co, _opt_co = 0.0, 0.0
            _real_h   = float(_wd["h_tot"].sum())
            _real_hl  = float(_wd["hl"].sum())
            _real_oee = _weighted_average(_wd, "oee", "hl") if not _wd.empty else 0.0
            for _ln in LINES:
                _ld = _wd[_wd["line"] == _ln].sort_values("fecha")
                _seq = _ld["sku"].tolist()
                if len(_seq) < 2:
                    continue
                _real_co += sum(
                    changeover_hours(_ctx_cf, _seq[i], _seq[i + 1], _ln)
                    for i in range(len(_seq) - 1)
                )
                # Nearest-neighbour reorder
                _rem, _cur = list(_seq), _seq[0]
                _rem.pop(0)
                while _rem:
                    _nxt = min(_rem, key=lambda s: changeover_hours(_ctx_cf, _cur, s, _ln))
                    _opt_co += changeover_hours(_ctx_cf, _cur, _nxt, _ln)
                    _rem.remove(_nxt)
                    _cur = _nxt
            _cf_rows.append({
                "Week": _wk,
                "Total hours": round(_real_h, 1),
                "Real CO (h)":   round(_real_co, 1),
                "Optimal CO (h)": round(_opt_co, 1),
                "Savings (h)":    round(_real_co - _opt_co, 1),
                "HL":            round(_real_hl, 0),
                "Real OEE (%)":  round(_real_oee * 100, 1),
            })
        st.session_state["cf_2025"] = pd.DataFrame(_cf_rows)

    _cf = st.session_state["cf_2025"]
    _cf_total_real_co  = _cf["Real CO (h)"].sum()
    _cf_total_opt_co   = _cf["Optimal CO (h)"].sum()
    _cf_total_saving   = _cf["Savings (h)"].sum()
    _cf_total_h        = _cf["Total hours"].sum()
    _cf_real_oee_avg   = float((_cf["Real OEE (%)"] * _cf["HL"]).sum() / _cf["HL"].sum())
    # Correct OEE: h_tot is pure production hours (changeovers excluded).
    # Total horas usadas = h_tot + CO. Reducir CO baja el denominador, OEE sube.
    _cf_prod_h         = (_cf["Real OEE (%)"] / 100 * _cf["Total hours"]).sum()
    _cf_real_total_used  = _cf_total_h + _cf_total_real_co
    _cf_opt_total_used   = _cf_total_h + _cf_total_opt_co
    _cf_real_oee_corr    = _cf_prod_h / _cf_real_total_used if _cf_real_total_used > 0 else 0.0
    _cf_opt_oee          = _cf_prod_h / _cf_opt_total_used  if _cf_opt_total_used  > 0 else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Real 2025 CO",   f"{_cf_total_real_co:.0f}h")
    m2.metric("Total Optimal 2025 CO", f"{_cf_total_opt_co:.0f}h",
              delta=f"{_cf_total_opt_co - _cf_total_real_co:.0f}h ({_cf_total_saving / _cf_total_real_co * 100:.1f}%)",
              delta_color="inverse")
    m3.metric("Estimated Optimal OEE",  f"{_cf_opt_oee * 100:.1f}%",
              delta=f"{(_cf_opt_oee - _cf_real_oee_corr) * 100:+.1f} pp vs real")

    # Weekly chart: Real vs optimal CO + ahorro acumulado
    _cf_cum = _cf["Savings (h)"].cumsum()
    _fig_cf = go.Figure()
    _fig_cf.add_trace(go.Bar(
        name="Real CO", x=_cf["Week"], y=_cf["Real CO (h)"],
        marker_color="#8B95A5", hovertemplate="S%{x} · Real: %{y:.1f}h<extra></extra>",
    ))
    _fig_cf.add_trace(go.Bar(
        name="Optimal CO", x=_cf["Week"], y=_cf["Optimal CO (h)"],
        marker_color="#10B981", hovertemplate="S%{x} · Optimal: %{y:.1f}h<extra></extra>",
    ))
    _fig_cf.add_trace(go.Scatter(
        name="Cumulative savings", x=_cf["Week"], y=_cf_cum,
        mode="lines", line=dict(color="#EF4444", width=2, dash="dot"),
        yaxis="y2", hovertemplate="S%{x} · Acumulado: %{y:.1f}h<extra></extra>",
    ))
    _fig_cf.update_layout(
        barmode="group", height=340, plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(title="Week", dtick=4),
        yaxis=dict(title="Changeover hours / week", gridcolor="#eee"),
        yaxis2=dict(title="Cumulative savings (h)", overlaying="y", side="right",
                    showgrid=False, tickfont=dict(color="#EF4444")),
        margin=dict(l=50, r=60, t=30, b=30),
    )
    st.plotly_chart(_fig_cf, use_container_width=True, key="cf_2025_chart")

    with st.expander("View weekly detail"):
        st.dataframe(_cf, hide_index=True, use_container_width=True)


# ═══════════════════════ PAGE 3: Optuna ═══════════════════════
elif page == "Hyperparameter Tuning (Optuna)":
    st.title("Bayesian Optimization of GA with Optuna TPE")
    st.markdown("Optimization of the Genetic Algorithm hyperparameters (population size, mutation probability, tournament size, and penalty weights) using the Tree-structured Parzen Estimator (TPE).")
    
    with st.sidebar:
        st.divider()
        n_trials = st.number_input("Number of Optuna Trials", min_value=5, max_value=200, value=20)
        pop_size = st.number_input("Fixed Population Size", min_value=10, max_value=500, value=60)
        n_gen = st.number_input("Fixed Generations", min_value=10, max_value=1000, value=100)
        seed = st.number_input("Optuna Seed", value=42)
        run_optuna_btn = st.button("▶ Run Optuna Study", type="primary", use_container_width=True)

    opt_ctx = get_ctx()
    
    if run_optuna_btn:
        import optuna_optimizer
        with st.spinner(f"Running Optuna Study with {n_trials} trials... This may take a bit."):
            study = optuna_optimizer.run_optuna_study(opt_ctx, pop_size=pop_size, n_gen=n_gen, n_trials=n_trials, seed=seed)
        
        st.success("Optuna Study completed!")
        st.subheader("Best Hyperparameters Found")
        st.json(study.best_params)
        
        st.metric("Best Fitness (lower is better)", f"{study.best_value:,.2f}")
        
        st.subheader("Trial History")
        trials_df = study.trials_dataframe()
        st.dataframe(trials_df)
