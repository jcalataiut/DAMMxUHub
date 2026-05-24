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
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.resources import INLINE

from ga_optimizer import (
    HOURS_PER_WEEK,
    LINES,
    PRIORITY_ORDERS,
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

    # Determine which edges are in the optimized path
    path_set = set(path_edges) if path_edges else set()

    # Edges (normal + black spots) with weighted width
    edge_xs, edge_ys = [], []
    edge_w, edge_c, edge_a = [], [], []
    edge_pair, edge_weight_label = [], []
    for _, row in edge_df.iterrows():
        o, d = row["prev_sku"], row["next_sku"]
        if o not in pos or d not in pos:
            continue
        is_path = (o, d) in path_set
        edge_xs.append([pos[o][0], pos[d][0]])
        edge_ys.append([pos[o][1], pos[d][1]])
        if is_path:
            edge_c.append("#2ca02c")
            edge_w.append(4.0)
            edge_a.append(0.9)
        elif highlight_nodes is not None:
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


ctx = get_ctx()
base_ind = baseline_individual(ctx)
base_bd = breakdown(ctx, base_ind)
baseline_total = sum(base_bd[l]["total"] for l in LINES)
frames_2025, nodes_2025, spots_2025 = get_frames()
num_weeks = int(frames_2025["week"].max())
spot_set = set(zip(spots_2025["prev_sku"], spots_2025["next_sku"]))
spot_skus_set = set(p for pair in spot_set for p in pair)


@st.cache_resource(show_spinner="Calculando layout fijo…")
def get_global_positions():
    """Fixed spherical layout per line using all weeks."""
    positions = {}
    for line in LINES:
        ef = frames_2025[frames_2025["line"] == line]
        G = nx.DiGraph()
        for _, row in ef.iterrows():
            G.add_edge(row["prev_sku"], row["next_sku"])
        if G.number_of_nodes() == 0:
            continue
        pos = nx.spring_layout(G, seed=42, k=3.0, iterations=100)
        positions[line] = pos
    return positions


global_pos = get_global_positions()

page = st.sidebar.radio("Visor", ["Aprendizaje 2025", "Optimización 2026"], label_visibility="collapsed")

# ═══════════════════════ PAGE 1: 2025 ═══════════════════════
if page == "Aprendizaje 2025":
    st.title("Aprendizaje 2025 · Grafo de transiciones")
    st.caption("Cada semana aparecen nuevas transiciones. Los nodos crecen con las conexiones. Rojo = black spot. ▶ reproduce la evolución.")

    # ── Streamlit animation: state at top, advance at bottom after render ──
    if "week_2025" not in st.session_state:
        st.session_state.week_2025 = num_weeks
    if "playing_2025" not in st.session_state:
        st.session_state.playing_2025 = False

    col_play, col_week = st.columns([1, 6])
    with col_play:
        btn_label = "⏸" if st.session_state.playing_2025 else "▶"
        if st.button(btn_label, key="play_btn_25"):
            was_playing = st.session_state.playing_2025
            st.session_state.playing_2025 = not was_playing
            if not was_playing:
                st.session_state.week_2025 = 1
            st.rerun()
    with col_week:
        week_idx = st.slider(
            "Semana", 1, num_weeks,
            value=st.session_state.week_2025,
            disabled=st.session_state.playing_2025)
        if not st.session_state.playing_2025:
            st.session_state.week_2025 = week_idx

    c1, c2, c3 = st.columns(3)
    for idx, line in enumerate(LINES):
        with [c1, c2, c3][idx]:
            ef = frames_2025[(frames_2025["week"] == week_idx) & (frames_2025["line"] == line)].copy()
            ef["weight"] = ef.apply(lambda r: changeover_hours(ctx, r["prev_sku"], r["next_sku"], line), axis=1)
            nf = nodes_2025[(nodes_2025["week"] == week_idx) & (nodes_2025["line"] == line)]
            fig = build_bokeh_graph(line, ef, nf, spot_set, title=f"L{line}",
                                    pos=global_pos.get(line))
            components.html(file_html(fig, INLINE, ""), height=410, scrolling=False)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("SKUs totales", nodes_2025["sku"].nunique())
    mc2.metric("Transiciones", len(frames_2025[frames_2025["week"] == num_weeks]))
    mc3.metric("Semanas", num_weeks)

    # Optimized path overlay (from last GA/SA run in Optimización 2026)
    opt_res = next((st.session_state[k] for k in st.session_state if k.startswith("res_")), None)
    if opt_res is not None:
        oi = opt_res["schedule"]
        opt_path = {line: list(zip(oi[line], oi[line][1:])) for line in LINES if len(oi[line]) > 1}
        st.markdown("---")
        st.subheader("Ruta optimizada")
        cg1, cg2, cg3 = st.columns(3)
        for idx, line in enumerate(LINES):
            with [cg1, cg2, cg3][idx]:
                ef = frames_2025[(frames_2025["line"] == line) & (frames_2025["week"] == num_weeks)].copy()
                ef["weight"] = ef.apply(lambda r: changeover_hours(ctx, r["prev_sku"], r["next_sku"], line), axis=1)
                nf = nodes_2025[(nodes_2025["line"] == line) & (nodes_2025["week"] == num_weeks)]
                fig = build_bokeh_graph(line, ef, nf, spot_set, title=f"L{line}",

                                        path_edges=opt_path.get(line,[]),
                                        highlight_nodes=set(oi.get(line, [])),
                                        pos=global_pos.get(line))
                components.html(file_html(fig, INLINE, ""), height=410, scrolling=False)

    # Advance AFTER all widgets have rendered so the user sees the frame
    if st.session_state.playing_2025:
        if st.session_state.week_2025 >= num_weeks:
            st.session_state.playing_2025 = False
        else:
            st.session_state.week_2025 += 1
            time.sleep(0.15)
            st.rerun()

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
        progress = st.progress(0, "Optimizando…")
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
            components.html(file_html(fig, INLINE, ""), height=400, scrolling=False)

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
