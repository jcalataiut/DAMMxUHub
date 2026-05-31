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
from bokeh.models import Button, ColumnDataSource, CustomJS, Div, HoverTool, Slider, Span
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


def load_frames_2025():
    frames = pd.read_csv(CLEAN_DIR / "frames_2025.csv", dtype={"line": str, "week": int, "prev_sku": str, "next_sku": str})
    nodes = pd.read_csv(CLEAN_DIR / "nodes_2025.csv", dtype={"line": str, "week": int, "sku": str})
    try:
        spots = pd.read_csv(CLEAN_DIR / "black_spots_2025.csv")
    except FileNotFoundError:
        spots = pd.DataFrame(columns=["line", "prev_sku", "next_sku"])
    spots["prev_sku"] = spots["prev_sku"].astype(str)
    spots["next_sku"] = spots["next_sku"].astype(str)
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
        title="Producción 2025 — flujo deslizante",
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
    df["is_spot"] = df["sku"].isin(spot_skus)
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
        fecha=df["fecha_str"].tolist(), dur=df["h_tot_str"].tolist(),
        hl=df["hl_str"].tolist(), oee=df["oee_str"].tolist(),
    ))

    gr = p_gantt.hbar(y="y", left="left", right="right", height=0.62,
                       fill_color="color", fill_alpha="alpha",
                       line_color="line_color", line_width="line_w",
                       source=g_src)
    p_gantt.add_tools(HoverTool(renderers=[gr], tooltips=[
        ("SKU", "@sku"), ("OF", "@of"),
        ("Inicio", "@fecha"), ("Duración", "@dur"),
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
        edge_first = (line_edges.groupby(["prev_sku", "next_sku"])["week"]
                                .min().reset_index().rename(columns={"week": "first_week"}))

        line_nodes = nodes_df[nodes_df["line"] == line]
        node_first = (line_nodes.groupby("sku")["week"]
                                .min().reset_index().rename(columns={"week": "first_week"}))
        latest_deg = (line_nodes.sort_values("week").groupby("sku")["degree"]
                                 .last().reset_index())
        node_first = node_first.merge(latest_deg, on="sku")
        line_cls = node_class.get(line, {})

        pos = global_pos.get(line, {})

        # ── Edges ──
        edge_xs, edge_ys, edge_first_w = [], [], []
        edge_color, edge_w_base, edge_alpha_base = [], [], []
        edge_pair, edge_weight_label = [], []
        for _, row in edge_first.iterrows():
            o, d = row["prev_sku"], row["next_sku"]
            if o not in pos or d not in pos:
                continue
            edge_xs.append([pos[o][0], pos[d][0]])
            edge_ys.append([pos[o][1], pos[d][1]])
            is_spot = (o, d) in spot_set
            edge_color.append("#d62728" if is_spot else "#5a5a5a")
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
        ))
        edge_sources.append(e_src)

        # ── Nodes ──
        node_x, node_y, node_first_w, node_sku_l, node_deg_l = [], [], [], [], []
        node_color, node_alpha_base, node_size_base = [], [], []
        node_stroke, node_stroke_w = [], []
        for _, row in node_first.iterrows():
            s = row["sku"]
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
            node_sku_l.append(s)
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
            sku=node_sku_l, degree=node_deg_l,
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
            ("Transición", "@pair"), ("Cambio", "@weight"),
        ]))
        nodes_r = pg.scatter("x", "y", source=n_src, size="size",
                              fill_color="color", fill_alpha="alpha",
                              line_color="stroke", line_width="stroke_w")
        pg.add_tools(HoverTool(renderers=[nodes_r], tooltips=[
            ("SKU", "@sku"), ("Conexiones", "@degree"),
        ]))
        graph_figs.append(pg)

    # ════════════════════════════ SHARED CONTROLS ════════════════════════════
    slider = Slider(start=1, end=total_days, value=initial_day, step=1,
                    title="Día del año", sizing_mode="stretch_width", show_value=True)
    play_btn = Button(label="▶", width=50, button_type="primary")
    speed_slider = Slider(start=1, end=10, value=3, step=1,
                          title="Velocidad (días/tick)", width=200)
    week_label = Div(
        text=f"<div style='padding:18px 8px 0 14px;font-size:14px;'><b>Semana {initial_week}/53</b></div>",
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

        week_label.text = "<div style='padding:18px 8px 0 14px;font-size:14px;'><b>Semana " + week + "/53</b></div>";
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
    for _, row in node_df.iterrows():
        G.add_node(row["sku"])
    for _, row in edge_df.iterrows():
        G.add_edge(row["prev_sku"], row["next_sku"])

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
    edge_pair, edge_weight_label = [], []
    for _, row in edge_df.iterrows():
        o, d = row["prev_sku"], row["next_sku"]
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
            edge_c.append("#d62728" if is_bs else "#444444")
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
        edge_pair.append(f"{o} → {d}")

    if edge_xs:
        src_e = ColumnDataSource(dict(
            xs=edge_xs, ys=edge_ys, w=edge_w, c=edge_c, a=edge_a,
            pair=edge_pair, weight=edge_weight_label,
        ))
        r_e = p.multi_line("xs", "ys", source=src_e,
                           line_color="c", line_width="w", line_alpha="a",
                           line_join="round")
        p.add_tools(HoverTool(renderers=[r_e], tooltips=[
            ("Transición", "@pair"),
            ("Cambio", "@weight"),
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
    spot_skus = set(p for pair in black_spots for p in pair)
    for _, row in node_df.iterrows():
        s = row["sku"]
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

    if node_x:
        src_n = ColumnDataSource(dict(
            x=node_x, y=node_y, size=node_s, color=node_c, alpha=node_al,
            stroke=node_sc, sw=node_sl,
            sku=node_df["sku"].tolist(), degree=node_df["degree"].tolist(),
        ))
        r = p.scatter("x", "y", source=src_n, size="size",
                       fill_color="color", fill_alpha="alpha",
                       line_color="stroke", line_width="sw")
        p.add_tools(HoverTool(renderers=[r], tooltips=[
            ("SKU", "@sku"), ("Conexiones", "@degree"),
        ]))

    return p


def gantt_figure(ctx, individual, title="", cap=None):
    gantt = schedule_to_gantt(ctx, individual)
    colors = {"changeover": "#d62728", "startup": "#bdbdbd"}
    fig = go.Figure()
    for _, r in gantt.iterrows():
        c = colors.get(r["type"], FMT_COLOR.get(r["format"], "#999"))
        hover = f"<b>{r['task']}</b><br>Inicio: {r['start_h']:.1f}h<br>Duración: {r['duration_h']:.2f}h"
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
    fig.update_layout(barmode="stack", title=title, xaxis_title="Horas",
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
    fig.update_layout(barmode="stack", title=title, xaxis_title="Horas",
                       yaxis=dict(categoryorder="array", categoryarray=[f"L{l}" for l in LINES]),
                       height=290, plot_bgcolor="white", margin=dict(l=50, r=15, t=35, b=20),
                       updatemenus=[{"type": "buttons", "buttons": [
                           {"label": "▶", "method": "animate", "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}]},
                           {"label": "⏹", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]},
                       ], "direction": "left", "showactive": False, "x": 0, "y": 1.12}],
                       sliders=[{"steps": [{"label": f"{h:.0f}h", "method": "animate", "args": [[f"{h:.0f}h"], {}]} for h in range(0, int(max_h) + 1, step)],
                                 "currentvalue": {"prefix": "Hora: "}}])
    fig.update_xaxes(gridcolor="#eee", range=[0, cap])
    return fig



def scenario_total(ctx, individual, mode, hdi_mass, prior_alpha):
    previous = (
        ctx.changeover_mode,
        ctx.changeover_hdi_mass,
        ctx.changeover_prior_alpha,
    )
    set_changeover_policy(
        ctx, mode=mode, hdi_mass=hdi_mass, prior_alpha=prior_alpha,
    )
    bd = breakdown(ctx, individual)
    total = sum(bd[l]["total"] for l in LINES)
    set_changeover_policy(
        ctx,
        mode=previous[0],
        hdi_mass=previous[1],
        prior_alpha=previous[2],
    )
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
        historical = [
            line for line in LINES
            if fmt in PHYSICAL_FORMAT_BY_LINE[line] and (sku, line) in ctx.hist_pairs
        ]
        if historical:
            eligible[sku] = historical
            continue

        planned_lines = [
            line for line in original_lines.get(sku, [])
            if fmt in PHYSICAL_FORMAT_BY_LINE.get(line, set())
        ]
        if planned_lines:
            eligible[sku] = planned_lines
        else:
            eligible[sku] = [line for line in LINES if fmt in PHYSICAL_FORMAT_BY_LINE[line]]
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
                "line": f"L{line}", "task": "ARRANQUE", "sku": "_arr",
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
                    "line": f"L{line}", "task": "ARRANQUE", "sku": "_arr",
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
        hover = f"<b>{r['task']}</b><br>Inicio: {r['start_h']:.1f}h<br>Duración: {r['duration_h']:.2f}h"
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
        barmode="stack", title=title, xaxis_title="Horas",
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


@st.cache_data(show_spinner="Cargando plan y real 2026…")
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
            "linea": None if pd.isna(row.get("linea")) or str(row.get("linea")) in {"", "Auto"} else str(row["linea"]),
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

@st.cache_resource(show_spinner="Cargando histórico para Gantt…")
def get_gantt_history():
    return load_historical_gantt()


ctx = get_ctx()
base_ind = baseline_individual(ctx)
base_bd = breakdown(ctx, base_ind)
baseline_total = sum(base_bd[l]["total"] for l in LINES)
frames_2025, nodes_2025, spots_2025 = get_frames()
num_weeks = int(frames_2025["week"].max())
spot_set = set(zip(spots_2025["prev_sku"], spots_2025["next_sku"]))
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
    """Classify each (line, sku) by count of adjacent black-spot (negative) edges.
    Returns {line: {sku: 'normal'|'critical'|'blackspot'}}.
    Thresholds: 0 → normal, 1-2 → critical, ≥3 → blackspot."""
    from collections import defaultdict
    cls = {}
    for line in LINES:
        line_spots = spots_2025[spots_2025["line"].astype(str) == line]
        spot_count = defaultdict(int)
        for _, r in line_spots.iterrows():
            spot_count[str(r["prev_sku"])] += 1
            spot_count[str(r["next_sku"])] += 1
        all_skus = set(nodes_2025[nodes_2025["line"] == line]["sku"].astype(str))
        all_skus |= set(frames_2025[frames_2025["line"] == line]["prev_sku"].astype(str))
        all_skus |= set(frames_2025[frames_2025["line"] == line]["next_sku"].astype(str))
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


@st.cache_resource(show_spinner="Calculando layout esférico…")
def get_global_positions():
    """Spherical (shell) layout per line: 3 concentric rings by category.
    Inner ring = black spots, middle = critical, outer = normal."""
    cls = get_node_classification()
    positions = {}
    for line in LINES:
        ef = frames_2025[frames_2025["line"] == line]
        G = nx.DiGraph()
        for _, row in ef.iterrows():
            G.add_edge(row["prev_sku"], row["next_sku"])
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

page = st.sidebar.radio("Visor", ["Aprendizaje 2025", "Optimización 2026", "Hyperparameter Tuning (Optuna)"], label_visibility="collapsed")

# ═══════════════════════ PAGE 1: 2025 ═══════════════════════
if page == "Aprendizaje 2025":
    st.title("OPla Lab")
    st.subheader("Generative Bayesian Complex Networks")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("SKUs totales", nodes_2025["sku"].nunique())
    mc2.metric("Transiciones únicas",
               int(frames_2025.groupby(["line", "prev_sku", "next_sku"]).ngroups))
    mc3.metric("Órdenes 2025", len(gantt_history))

    # Single combined dashboard: Gantt + 3 graphs sharing one slider+play (all client-side JS)
    # Initial cursor at week 3 (day 21) so the user sees the start of the year with context.
    components.html(combined_dashboard_html(initial_day=21),
                    height=880, scrolling=False)

# ═══════════════════════ PAGE 2: 2026 ═══════════════════════
elif page == "Optimización 2026":
    st.title("Optimización · 18-22 May 2026")

    df_plan_2026, df_real_2026 = get_2026_execution_data()
    planner_ind = planned_sequences_from_planificado(df_plan_2026)
    real_ind = actual_sequences_from_production(df_real_2026)
    planner_volumes = _volumes_by_line_sku(df_plan_2026, "hl_plan")
    real_volumes = _volumes_by_line_sku(df_real_2026, "hl_real")
    real_oee_lines = real_oee_by_line(df_real_2026)
    opt_ctx = context_for_plan_week(ctx, df_plan_2026, df_real_2026)

    with st.sidebar:
        st.divider()
        st.caption("Changeovers 2025")
        co_policy_options = {
            "Media bayesiana": "bayes_mean",
            "Media observada": "observed_mean",
            "HDI inferior": "hdi_lower",
            "HDI superior": "hdi_upper",
        }
        co_policy_label = st.selectbox(
            "Valor usado por el optimizador",
            list(co_policy_options.keys()),
            key="co_policy",
        )
        hdi_pct = st.slider("HDI", 80, 99, 95, 1, key="co_hdi_pct")
        prior_alpha = st.slider(
            "Fuerza prior",
            1.1, 8.0, 2.0, 0.1,
            help="Prior Gamma sobre la tasa exponencial; 2.0 equivale a una observación previa cercana a la media de línea.",
            key="co_prior_alpha",
        )
        co_mode = co_policy_options[co_policy_label]
        hdi_mass = hdi_pct / 100.0
        set_changeover_policy(
            opt_ctx, mode=co_mode, hdi_mass=hdi_mass, prior_alpha=prior_alpha,
        )
        st.divider()
        ga_pop = st.slider("Población", 20, 200, 60, 10, key="ga_pop")
        ga_gen = st.slider("Generaciones", 30, 400, 150, 10, key="ga_gen")
        ga_seed = st.number_input("Seed GA", 42, step=1, key="ga_seed")

    run_btn = st.sidebar.button("▶ Optimizar", type="primary", use_container_width=True, key="run_opt")
    planner_bd = breakdown_from_sequences(opt_ctx, planner_ind, planner_volumes)
    real_bd = breakdown_from_sequences(opt_ctx, real_ind, real_volumes, oee_by_line=real_oee_lines)
    planner_totals = scenario_totals(planner_bd)
    real_totals = scenario_totals(real_bd)
    real_oee_global = _weighted_average(df_real_2026, "oee", "hl_real")

    # urgent_orders se lee del rerun anterior para que esté disponible antes de optimizar
    urgent_orders = st.session_state.get("_urgent_orders", [])

    result_key = f"res_GA_{co_mode}_{hdi_pct}_{prior_alpha:.1f}"
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
                if o["linea"] is not None:
                    extra.append((sku, o["linea"]))
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
            active_urgent = [o for o in urgent_orders if o["linea"] is not None or opt_ctx.eligible.get(o["sku"], [])]
            urgent_rows = []
            if active_urgent:
                opt_ind_latest = st.session_state[result_key]["schedule"]
                for o in active_urgent:
                    sku = o["sku"]
                    lines = [o["linea"]] if o["linea"] else opt_ctx.eligible.get(sku, [])
                    for ln in lines:
                        seq = opt_ind_latest.get(ln, [])
                        if sku in seq:
                            pos = seq.index(sku)
                            cutoff = max(0, int(0.25 * len(seq)))
                            vol = f"+{o['hl_total']:.0f} HL" if o["hl_total"] else ""
                            urgent_rows.append({"SKU": sku, "Línea": f"L{ln}",
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
    min_total, min_bd = scenario_total(ai_ctx, opt_ind, "hdi_lower", hdi_mass, prior_alpha)
    worst_total, worst_bd = scenario_total(ai_ctx, opt_ind, "hdi_upper", hdi_mass, prior_alpha)
    min_bd = enrich_model_breakdown(min_bd)
    worst_bd = enrich_model_breakdown(worst_bd)
    opt_oee = scenario_oee(opt_bd)
    oee_delta = opt_oee - real_oee_global

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Real ejecutado", f"{real_totals['total']:.1f}h")
    c2.metric("Óptimo", f"{opt_total:.1f}h",
              delta=f"{opt_total - real_totals['total']:.1f}h vs real",
              delta_color="inverse")
    c3.metric("Reducción", f"{saved:.1f}h", delta=f"{saved/real_totals['total']*100:.1f}%")
    c4.metric("Minimum Execution", f"{min_total:.1f}h")
    c5.metric("Worst Execution", f"{worst_total:.1f}h")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Planner teórico", f"{planner_totals['total']:.1f}h")
    o2.metric("OEE global real", f"{real_oee_global*100:.1f}%")
    o3.metric("OEE global Óptimo", f"{opt_oee*100:.1f}%", delta=f"{oee_delta*100:+.1f} pp vs real")
    o4.metric("Tiempo optimización", f"{result['elapsed']:.1f}s")

    opt_path = {line: list(zip(opt_ind[line], opt_ind[line][1:])) for line in LINES if len(opt_ind[line]) > 1}

    with st.expander("📦 Órdenes urgentes", expanded=False):
        urgent_df = st.data_editor(
            pd.DataFrame([{"active": False, "order_id": "URG-01", "sku": "EX1324NB",
                           "linea": "Auto", "hl_total": 200.0}]),
            num_rows="dynamic", key="urgent_editor",
            column_config={
                "active":    st.column_config.CheckboxColumn("Activa"),
                "order_id":  st.column_config.TextColumn("ID orden"),
                "sku":       st.column_config.SelectboxColumn("SKU", options=opt_ctx.skus, required=False),
                "linea":     st.column_config.SelectboxColumn("Línea", options=["Auto"] + LINES, required=False),
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

    # 3 Bokeh graphs: full historical network with optimized path highlighted
    cg1, cg2, cg3 = st.columns(3)
    for idx, line in enumerate(LINES):
        with [cg1, cg2, cg3][idx]:
            ef = frames_2025[(frames_2025["line"] == line) & (frames_2025["week"] == num_weeks)].copy()
            ef["weight"] = ef.apply(lambda r: changeover_hours(opt_ctx, r["prev_sku"], r["next_sku"], line), axis=1)
            nf = nodes_2025[(nodes_2025["line"] == line) & (nodes_2025["week"] == num_weeks)]
            fig = build_bokeh_graph(line, ef, nf, spot_set, title=f"L{line}",
                                    path_edges=opt_path.get(line, []),
                                    highlight_nodes=set(opt_ind.get(line, [])),
                                    pos=global_pos.get(line))
            components.html(file_html(fig, CDN, ""), height=400, scrolling=False)

    gantt_cap = max(
        max(v["total"] for v in planner_bd.values()),
        max(v["total"] for v in real_bd.values()),
        max(v["total"] for v in opt_bd.values()),
        max(HOURS_PER_WEEK.values()),
    ) + 10

    st.subheader("Planner teórico")
    st.plotly_chart(
        gantt_figure_from_sequences(
            opt_ctx, planner_ind, planner_volumes, title="", cap=gantt_cap,
        ),
        key="planner_g", use_container_width=True,
    )

    st.subheader("Real ejecutado")
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

    st.subheader("Óptimo")
    st.plotly_chart(
        gantt_figure(ai_ctx, opt_ind, title="", cap=gantt_cap),
        key="opt_g", use_container_width=True,
    )

    # ── Comparativa Real vs Óptimo ────────────────────────────────────────────
    st.subheader("Real ejecutado vs Óptimo")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Total horas · Real", f"{real_totals['total']:.1f}h")
    h2.metric("Total horas · Óptimo", f"{opt_total:.1f}h",
              delta=f"{opt_total - real_totals['total']:.1f}h vs real",
              delta_color="inverse")
    h3.metric("Horas muertas · Real", f"{real_dead_t:.1f}h")
    h4.metric("Horas muertas · Óptimo",
              f"{opt_dead_t:.1f}h",
              delta=f"{opt_dead_t - real_dead_t:.1f}h vs real",
              delta_color="inverse")
    st.caption(f"Horas disponibles totales (3 líneas): {avail_total:.0f}h")
    comp_rows = [
        {
            "Línea": f"L{l}",
            "Total Real": f"{real_bd[l]['total']:.1f}h",
            "Muertas Real": f"{real_bd[l]['total'] - real_bd[l]['prod']:.1f}h",
            "Total Óptimo": f"{opt_bd[l]['total']:.1f}h",
            "Muertas Óptimo": f"{opt_bd[l]['total'] - opt_bd[l]['prod']:.1f}h",
            "Disponible": f"{HOURS_PER_WEEK[l]:.0f}h",
        }
        for l in LINES
    ]
    st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

    # ── Dinámica de mejora por línea ──────────────────────────────────────────
    st.subheader("Dinámica de mejora por línea")
    _ll = [f"L{l}" for l in LINES]
    _real_h = [real_bd[l]["total"] for l in LINES]
    _opt_h  = [opt_bd[l]["total"]  for l in LINES]

    _COLOR_REAL = "#8B95A5"
    _COLOR_OPT  = "#10B981"

    _fig_dyn = go.Figure()
    _fig_dyn.add_trace(go.Bar(
        name="Real ejecutado", x=_ll, y=_real_h,
        marker=dict(color=_COLOR_REAL, line=dict(color="#6B7585", width=1.2)),
        text=[f"{h:.0f}h" for h in _real_h], textposition="outside",
        hovertemplate="%{x} Real: %{y:.1f}h<extra></extra>",
    ))
    _fig_dyn.add_trace(go.Bar(
        name="Óptimo", x=_ll, y=_opt_h,
        marker=dict(color=_COLOR_OPT, line=dict(color="#059669", width=1.2)),
        text=[f"{h:.0f}h" for h in _opt_h], textposition="outside",
        hovertemplate="%{x} Óptimo: %{y:.1f}h<extra></extra>",
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
        yaxis=dict(gridcolor="#eee", title="Horas"),
    )
    st.plotly_chart(_fig_dyn, use_container_width=True, key="dyn_chart")

    # ── Migración de SKUs: Planner teórico → Óptimo ───────────────────────────
    st.subheader("Migración de SKUs entre líneas")
    st.caption("SKUs que el optimizador reasigna a una línea distinta respecto al planner teórico.")

    _planner_line = {sku: line for line, skus in planner_ind.items() for sku in skus}
    _opt_line = {sku: line for line, skus in opt_ind.items() for sku in skus}
    _all_skus = set(_planner_line) | set(_opt_line)
    _migrations = [
        {"SKU": sku,
         "Planner": f"L{_planner_line[sku]}" if sku in _planner_line else "—",
         "Óptimo":  f"L{_opt_line[sku]}"     if sku in _opt_line     else "—",
         "Cambio":  _planner_line.get(sku) != _opt_line.get(sku)}
        for sku in sorted(_all_skus)
    ]
    _moved = [r for r in _migrations if r["Cambio"]]

    if _moved:
        # Sankey con colores vivos: nodos izq = Planner, nodos der = Óptimo
        # Cada línea tiene su color; los flujos heredan el color de la línea origen
        def _hex_to_rgba(hex_color, alpha):
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            return f"rgba({r},{g},{b},{alpha})"

        _line_idx = {l: i for i, l in enumerate(LINES)}
        _flow: dict = {}
        for r in _moved:
            if r["Planner"] == "—" or r["Óptimo"] == "—":
                continue
            src = _line_idx[r["Planner"][1:]]
            tgt = _line_idx[r["Óptimo"][1:]] + len(LINES)
            _flow[(src, tgt)] = _flow.get((src, tgt), 0) + 1

        _node_labels = [f"Planner  L{l}" for l in LINES] + [f"Óptimo  L{l}" for l in LINES]
        _node_colors = [_hex_to_rgba(LINE_COLOR[l], 0.85) for l in LINES] * 2

        _link_sources = [s for s, _ in _flow]
        _link_targets = [t for _, t in _flow]
        _link_values  = [v for v in _flow.values()]
        _link_colors  = [_hex_to_rgba(LINE_COLOR[LINES[s]], 0.35) for s in _link_sources]
        _link_labels  = [
            f"{v} SKU{'s' if v > 1 else ''}: "
            + ", ".join(r["SKU"] for r in _moved
                        if r["Planner"] != "—" and r["Óptimo"] != "—"
                        and _line_idx[r["Planner"][1:]] == s
                        and _line_idx[r["Óptimo"][1:]] + len(LINES) == t)
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

        _move_df = pd.DataFrame([{"SKU": r["SKU"], "Planner": r["Planner"], "→ Óptimo": r["Óptimo"]}
                                  for r in _moved])
        st.dataframe(_move_df, hide_index=True, use_container_width=True)
    else:
        st.info("El optimizador mantiene todos los SKUs en la misma línea que el planner teórico.")

    # ── Contrafactual 2025 ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Contrafactual 2025 — Potencial ahorro anual")

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
                "Semana": _wk,
                "Horas totales": round(_real_h, 1),
                "CO real (h)":   round(_real_co, 1),
                "CO óptimo (h)": round(_opt_co, 1),
                "Ahorro (h)":    round(_real_co - _opt_co, 1),
                "HL":            round(_real_hl, 0),
                "OEE real (%)":  round(_real_oee * 100, 1),
            })
        st.session_state["cf_2025"] = pd.DataFrame(_cf_rows)

    _cf = st.session_state["cf_2025"]
    _cf_total_real_co  = _cf["CO real (h)"].sum()
    _cf_total_opt_co   = _cf["CO óptimo (h)"].sum()
    _cf_total_saving   = _cf["Ahorro (h)"].sum()
    _cf_total_h        = _cf["Horas totales"].sum()
    _cf_real_oee_avg   = float((_cf["OEE real (%)"] * _cf["HL"]).sum() / _cf["HL"].sum())
    # OEE correcto: h_tot son horas de producción pura (changeovers están fuera).
    # Total horas usadas = h_tot + CO. Reducir CO baja el denominador, OEE sube.
    _cf_prod_h         = (_cf["OEE real (%)"] / 100 * _cf["Horas totales"]).sum()
    _cf_real_total_used  = _cf_total_h + _cf_total_real_co
    _cf_opt_total_used   = _cf_total_h + _cf_total_opt_co
    _cf_real_oee_corr    = _cf_prod_h / _cf_real_total_used if _cf_real_total_used > 0 else 0.0
    _cf_opt_oee          = _cf_prod_h / _cf_opt_total_used  if _cf_opt_total_used  > 0 else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("CO total real 2025",   f"{_cf_total_real_co:.0f}h")
    m2.metric("CO total óptimo 2025", f"{_cf_total_opt_co:.0f}h",
              delta=f"{_cf_total_opt_co - _cf_total_real_co:.0f}h ({_cf_total_saving / _cf_total_real_co * 100:.1f}%)",
              delta_color="inverse")
    m3.metric("OEE estimado óptimo",  f"{_cf_opt_oee * 100:.1f}%",
              delta=f"{(_cf_opt_oee - _cf_real_oee_corr) * 100:+.1f} pp vs real")

    # Weekly chart: CO real vs óptimo + ahorro acumulado
    _cf_cum = _cf["Ahorro (h)"].cumsum()
    _fig_cf = go.Figure()
    _fig_cf.add_trace(go.Bar(
        name="CO real", x=_cf["Semana"], y=_cf["CO real (h)"],
        marker_color="#8B95A5", hovertemplate="S%{x} · Real: %{y:.1f}h<extra></extra>",
    ))
    _fig_cf.add_trace(go.Bar(
        name="CO óptimo", x=_cf["Semana"], y=_cf["CO óptimo (h)"],
        marker_color="#10B981", hovertemplate="S%{x} · Óptimo: %{y:.1f}h<extra></extra>",
    ))
    _fig_cf.add_trace(go.Scatter(
        name="Ahorro acumulado", x=_cf["Semana"], y=_cf_cum,
        mode="lines", line=dict(color="#EF4444", width=2, dash="dot"),
        yaxis="y2", hovertemplate="S%{x} · Acumulado: %{y:.1f}h<extra></extra>",
    ))
    _fig_cf.update_layout(
        barmode="group", height=340, plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(title="Semana", dtick=4),
        yaxis=dict(title="Horas changeover / semana", gridcolor="#eee"),
        yaxis2=dict(title="Ahorro acumulado (h)", overlaying="y", side="right",
                    showgrid=False, tickfont=dict(color="#EF4444")),
        margin=dict(l=50, r=60, t=30, b=30),
    )
    st.plotly_chart(_fig_cf, use_container_width=True, key="cf_2025_chart")

    with st.expander("Ver detalle semanal"):
        st.dataframe(_cf, hide_index=True, use_container_width=True)


# ═══════════════════════ PAGE 3: Optuna ═══════════════════════
elif page == "Hyperparameter Tuning (Optuna)":
    st.title("Bayesian Optimization of GA with Optuna TPE")
    st.markdown("Optimization of the Genetic Algorithm hyperparameters (population size, mutation probability, tournament size, and penalty weights) using the Tree-structured Parzen Estimator (TPE).")
    
    with st.sidebar:
        st.divider()
        n_trials = st.number_input("Number of Optuna Trials", min_value=5, max_value=200, value=20)
        seed = st.number_input("Optuna Seed", value=42)
        run_optuna_btn = st.button("▶ Run Optuna Study", type="primary", use_container_width=True)

    opt_ctx = get_ctx()
    
    if run_optuna_btn:
        import optuna_optimizer
        with st.spinner(f"Running Optuna Study with {n_trials} trials... This may take a bit."):
            study = optuna_optimizer.run_optuna_study(opt_ctx, n_trials=n_trials, seed=seed)
        
        st.success("Optuna Study completed!")
        st.subheader("Best Hyperparameters Found")
        st.json(study.best_params)
        
        st.metric("Best Fitness (lower is better)", f"{study.best_value:,.2f}")
        
        st.subheader("Trial History")
        trials_df = study.trials_dataframe()
        st.dataframe(trials_df)
