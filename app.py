from __future__ import annotations

import colorsys
import json
import time
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
from ga_optimizer import (
    HOURS_PER_WEEK,
    LINES,
    OptimizerContext,
    baseline_individual,
    breakdown,
    changeover_hours,
    evolve,
    load_clean_context,
    schedule_to_gantt,
)
from simulated_annealing import run_sa

HERE = Path(__file__).resolve().parent
CLEAN_DIR = HERE / "clean_data"

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


def stacked_hours(base_bd, opt_bd, label=""):
    cats, parts = [], {"Prod.": [], "Changeover": [], "Arranque": [], "Holgura": []}
    cls = {"Prod.": "#4c72b0", "Changeover": "#dd8452", "Arranque": "#8c8c8c", "Holgura": "#cfe2f3"}
    for line in LINES:
        for sc, bd in (("Plan", base_bd), (label, opt_bd)):
            cats.append(f"L{line}·{sc}")
            parts["Prod."].append(bd[line]["prod"])
            parts["Changeover"].append(bd[line]["changeover"])
            parts["Arranque"].append(bd[line]["startup"])
            parts["Holgura"].append(max(0, HOURS_PER_WEEK[line] - bd[line]["total"]))
    fig = go.Figure()
    for k, vs in parts.items():
        fig.add_trace(go.Bar(name=k, x=cats, y=vs, marker_color=cls[k],
                              text=[f"{v:.0f}h" if v > 4 else "" for v in vs], textposition="inside"))
    for i, ln in enumerate(LINES):
        fig.add_shape(type="line", x0=i*2-.45, x1=i*2+1.45, y0=HOURS_PER_WEEK[ln], y1=HOURS_PER_WEEK[ln],
                       line=dict(color="red", dash="dash", width=1.5))
    fig.update_layout(barmode="stack", height=300, plot_bgcolor="white",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02),
                       margin=dict(l=40, r=10, t=15, b=30))
    fig.update_yaxes(gridcolor="#eee", range=[0, max(HOURS_PER_WEEK.values())+20])
    return fig


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
            "latest_position": None if pd.isna(row.get("latest_position")) or int(row["latest_position"]) <= 0 else int(row["latest_position"]),
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

page = st.sidebar.radio("Visor", ["Aprendizaje 2025", "Optimización 2026"], label_visibility="collapsed")

# ═══════════════════════ PAGE 1: 2025 ═══════════════════════
if page == "Aprendizaje 2025":
    st.title("Operation Lab")
    st.subheader("Generative Bayesian Graphs")

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
else:
    st.title("Optimización · 18-22 May 2026")
    st.caption("Elige optimizador, añade urgencias, visualiza el plan.")

    algo = st.sidebar.selectbox("Algoritmo", ["GA (Genético)", "SA (Enfriamiento simulado)"], key="algo")
    with st.sidebar:
        if algo == "GA (Genético)":
            ga_pop = st.slider("Población", 20, 200, 60, 10, key="ga_pop")
            ga_gen = st.slider("Generaciones", 30, 400, 150, 10, key="ga_gen")
            ga_seed = st.number_input("Seed GA", 42, step=1, key="ga_seed")
        else:
            sa_iter = st.slider("Iteraciones", 2_000, 50_000, 15_000, 1_000, key="sa_iter")
            sa_seed = st.number_input("Seed SA", 42, step=1, key="sa_seed")

    run_btn = st.sidebar.button("▶ Optimizar", type="primary", use_container_width=True, key="run_opt")

    with st.expander("📦 Órdenes urgentes", expanded=False):
        urgent_df = st.data_editor(
            pd.DataFrame([{"active": False, "order_id": "URG-01", "sku": "EX1324NB",
                           "linea": "Auto", "hl_total": 200.0, "latest_position": 3}]),
            num_rows="dynamic", key="urgent_editor",
            column_config={
                "active": st.column_config.CheckboxColumn("Activa"),
                "sku": st.column_config.SelectboxColumn("SKU", options=ctx.skus, required=False),
                "linea": st.column_config.SelectboxColumn("Línea", options=["Auto"] + LINES, required=False),
                "hl_total": st.column_config.NumberColumn("HL extra", min_value=0.0, step=25.0),
                "latest_position": st.column_config.NumberColumn("Posición", min_value=0, step=1),
            },
        )
        urgent_orders = clean_urgent(urgent_df) if not urgent_df.empty else []

    result_key = f"res_{algo}"
    if run_btn or result_key not in st.session_state:
        # Apply urgent orders: extra volume + priority
        original_priority = list(ga_mod.PRIORITY_ORDERS)
        volumes_backup = {}
        if urgent_orders:
            extra = []
            for o in urgent_orders:
                sku = o["sku"]
                if o["hl_total"] is not None:
                    volumes_backup.setdefault(sku, ctx.volumes.get(sku, 0))
                    ctx.volumes[sku] = volumes_backup[sku] + o["hl_total"]
                if o["linea"] is not None:
                    extra.append((sku, o["linea"]))
                else:
                    for line in ctx.eligible.get(sku, LINES):
                        extra.append((sku, line))
            ga_mod.PRIORITY_ORDERS = original_priority + extra

        progress = st.progress(0, "Optimizando…")
        try:
            if algo == "GA (Genético)":
                def cb(g, b, m):
                    progress.progress((g+1)/ga_gen, text=f"G {g+1}/{ga_gen} · mejor={b:.1f}h")
                t0 = time.time()
                best_ind, history = evolve(ctx, pop_size=ga_pop, n_gen=ga_gen, seed=ga_seed, on_generation=cb)
                st.session_state[result_key] = {"schedule": best_ind, "elapsed": time.time()-t0}
            else:
                def cb(n, best, _):
                    progress.progress(min(n / sa_iter, 1.0), text=f"SA {n}/{sa_iter} · mejor={best:.1f}h")
                res = run_sa(ctx, n_iter=sa_iter, seed=sa_seed, on_trial=cb)
                st.session_state[result_key] = {"schedule": res["schedule"], "elapsed": res["elapsed_s"]}
            # Urgent orders feedback (shown right after optimization)
            active_urgent = [o for o in urgent_orders if o["linea"] is not None or ctx.eligible.get(o["sku"], [])]
            if active_urgent:
                opt_ind_latest = st.session_state[result_key]["schedule"]
                urgent_rows = []
                for o in active_urgent:
                    sku = o["sku"]
                    lines = [o["linea"]] if o["linea"] else ctx.eligible.get(sku, [])
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
                if urgent_rows:
                    st.dataframe(pd.DataFrame(urgent_rows), hide_index=True, use_container_width=True)
        finally:
            ga_mod.PRIORITY_ORDERS = original_priority
            for sku, orig_val in volumes_backup.items():
                ctx.volumes[sku] = orig_val
        progress.empty()

    result = st.session_state[result_key]
    opt_ind = result["schedule"]
    opt_bd = breakdown(ctx, opt_ind)
    opt_total = sum(opt_bd[l]["total"] for l in LINES)
    saved = baseline_total - opt_total
    an = algo.split(" ")[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Plan empresa", f"{baseline_total:.1f}h")
    c2.metric(f"{an}", f"{opt_total:.1f}h", delta=f"{-saved:.1f}h", delta_color="inverse")
    c3.metric("Ahorro", f"{saved:+.1f}h", delta=f"{saved/baseline_total*100:+.1f}%")
    c4.metric("Tiempo", f"{result['elapsed']:.1f}s")

    opt_path = {line: list(zip(opt_ind[line], opt_ind[line][1:])) for line in LINES if len(opt_ind[line]) > 1}

    # 3 Bokeh graphs: full historical network with optimized path highlighted
    cg1, cg2, cg3 = st.columns(3)
    for idx, line in enumerate(LINES):
        with [cg1, cg2, cg3][idx]:
            # Show ALL nodes and edges from week 53; highlight only optimized ones
            ef = frames_2025[(frames_2025["line"] == line) & (frames_2025["week"] == num_weeks)].copy()
            ef["weight"] = ef.apply(lambda r: changeover_hours(ctx, r["prev_sku"], r["next_sku"], line), axis=1)
            nf = nodes_2025[(nodes_2025["line"] == line) & (nodes_2025["week"] == num_weeks)]
            fig = build_bokeh_graph(line, ef, nf, spot_set, title=f"L{line}",
                                    path_edges=opt_path.get(line, []),
                                    highlight_nodes=set(opt_ind.get(line, [])),
                                    pos=global_pos.get(line))
            components.html(file_html(fig, CDN, ""), height=400, scrolling=False)

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Plan del planner")
        st.plotly_chart(gantt_figure(ctx, base_ind), key="base_g")
    with g2:
        st.subheader(f"{an} optimizado")
        st.plotly_chart(gantt_figure(ctx, opt_ind, title=f"{an} optimizado"), key="opt_g")

    mc1, mc2 = st.columns(2)
    mc1.plotly_chart(stacked_hours(base_bd, opt_bd, an), key="stacked")
    with mc2:
        rows = [{"Línea": f"L{l}", "Plan": f"{base_bd[l]['total']:.1f}h",
                  an: f"{opt_bd[l]['total']:.1f}h",
                  "Ahorro": f"{base_bd[l]['total']-opt_bd[l]['total']:+.1f}h",
                  "OK": "✓" if opt_bd[l]["total"] <= HOURS_PER_WEEK[l] else "✗"} for l in LINES]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
