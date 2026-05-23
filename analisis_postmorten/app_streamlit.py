"""LineWise · Streamlit dashboard for the GA scheduler.

Run with:
    streamlit run app_streamlit.py

Reuses every primitive defined in ``ga_optimizer.py`` so the dashboard never
drifts from the notebook (05_ga_optimizer.ipynb).
"""

from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow `streamlit run` from anywhere — make sibling modules importable.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ga_optimizer import (  # noqa: E402
    HOURS_PER_WEEK,
    LINES,
    PRIORITY_ORDERS,
    STARTUP_HOURS,
    OptimizerContext,
    baseline_individual,
    breakdown,
    build_context,
    changeover_hours,
    evaluate_schedule,
    evolve,
    schedule_to_gantt,
    throughput_rate,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LineWise · GA Scheduler",
    page_icon="🍺",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_DIR = Path("/Users/josecalatayud/DAMMxEHub/DAMMxUHub/OPERACIONS")

LINE_COLOR_HEX = {"14": "#1f77b4", "17": "#ff7f0e", "19": "#2ca02c"}
FMT_COLOR_HEX = {"1/2": "#4c72b0", "1/3": "#dd8452", "2/5": "#55a868"}


# ---------------------------------------------------------------------------
# Cached pipeline
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando datos históricos y matrices…")
def get_context(data_dir: str) -> OptimizerContext:
    return build_context(Path(data_dir))


@st.cache_data(show_spinner=False)
def cached_evolve(_ctx_id: str, pop_size: int, n_gen: int, seed: int,
                  data_dir: str) -> Dict:
    """Cache by the *user-controlled* hyperparameters so re-runs are free."""
    ctx = get_context(data_dir)
    best_ind, history = evolve(ctx, pop_size=pop_size, n_gen=n_gen, seed=seed)
    return {
        "best_ind": best_ind,
        "history": history,
        "fitness": evaluate_schedule(ctx, best_ind)[0],
    }


# ---------------------------------------------------------------------------
# Plotly figure builders
# ---------------------------------------------------------------------------

def gantt_figure(ctx: OptimizerContext, individual, title: str) -> go.Figure:
    gantt = schedule_to_gantt(ctx, individual)
    color_map = {**{f"L{l}_prod": FMT_COLOR_HEX["1/3"] for l in LINES}}

    def _color(row):
        if row["type"] == "changeover":
            return "#d62728"
        if row["type"] == "startup":
            return "#bdbdbd"
        return FMT_COLOR_HEX.get(row["format"], "#999")

    gantt["color"] = gantt.apply(_color, axis=1)
    gantt["hover"] = gantt.apply(
        lambda r: (
            f"<b>{r['task']}</b><br>"
            f"Línea: {r['line']}<br>"
            f"Inicio: t+{r['start_h']:.1f}h<br>"
            f"Duración: {r['duration_h']:.2f}h<br>"
            + (f"HL: {r['hl']:,.0f}<br>"
               f"Throughput: {r['rate_hl_per_h']:.0f} HL/h<br>"
               f"Formato: {r['format']}" if r['type'] == 'production' else
               "Cambio de formato" if r['type'] == 'changeover' else "Arranque línea")
        ), axis=1,
    )

    fig = go.Figure()
    for _, row in gantt.iterrows():
        fig.add_trace(go.Bar(
            x=[row["duration_h"]],
            y=[row["line"]],
            base=row["start_h"],
            orientation="h",
            marker=dict(color=row["color"],
                        line=dict(color="black", width=0.4)),
            text=row["task"] if row["type"] == "production" and row["duration_h"] >= 2 else "",
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=row["hover"] + "<extra></extra>",
            showlegend=False,
        ))

    # Capacity markers
    for line in LINES:
        fig.add_vline(x=HOURS_PER_WEEK[line], line_dash="dash", line_color="red",
                      opacity=0.4)

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Horas desde inicio de semana",
        yaxis=dict(categoryorder="array",
                   categoryarray=[f"L{l}" for l in LINES]),
        height=350,
        margin=dict(l=60, r=20, t=60, b=40),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#eee")
    return fig


def stacked_hours_figure(base_bd, opt_bd) -> go.Figure:
    """Side-by-side stacked bars for each line, broken into prod/CO/startup/slack."""
    categories = []
    parts = {"Producción": [], "Changeover": [], "Arranque": [], "Holgura": []}
    colors = {"Producción": "#4c72b0", "Changeover": "#dd8452",
              "Arranque": "#8c8c8c", "Holgura": "#cfe2f3"}
    for line in LINES:
        for scenario, bd in (("Plan", base_bd), ("GA", opt_bd)):
            categories.append(f"L{line} · {scenario}")
            parts["Producción"].append(bd[line]["prod"])
            parts["Changeover"].append(bd[line]["changeover"])
            parts["Arranque"].append(bd[line]["startup"])
            total = bd[line]["total"]
            parts["Holgura"].append(max(0.0, HOURS_PER_WEEK[line] - total))

    fig = go.Figure()
    for label, values in parts.items():
        fig.add_trace(go.Bar(
            name=label, x=categories, y=values, marker_color=colors[label],
            hovertemplate="%{x}<br>" + label + ": %{y:.1f}h<extra></extra>",
            text=[f"{v:.0f}h" if v > 4 else "" for v in values],
            textposition="inside",
        ))

    # Capacity lines per line group (2 categories per line)
    for i, line in enumerate(LINES):
        fig.add_shape(type="line",
                      x0=i * 2 - 0.45, x1=i * 2 + 1.45,
                      y0=HOURS_PER_WEEK[line], y1=HOURS_PER_WEEK[line],
                      line=dict(color="red", dash="dash", width=1.6))
        fig.add_annotation(x=i * 2 + 1.45, y=HOURS_PER_WEEK[line],
                           text=f"cap {HOURS_PER_WEEK[line]:.0f}h",
                           showarrow=False, font=dict(color="red", size=10),
                           xanchor="left", yanchor="middle")

    # Totals above each bar
    for cat_i, line_scen in enumerate(categories):
        total = sum(parts[p][cat_i] for p in ("Producción", "Changeover", "Arranque"))
        fig.add_annotation(x=line_scen, y=total + 3,
                           text=f"<b>{total:.1f}h</b>",
                           showarrow=False, font=dict(size=11))

    fig.update_layout(
        barmode="stack",
        title="¿En qué se gasta cada hora? · Plan del planner vs GA",
        xaxis_title="", yaxis_title="Horas",
        height=480, plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=120, t=70, b=60),
    )
    fig.update_yaxes(gridcolor="#eee", range=[0, max(HOURS_PER_WEEK.values()) + 25])
    return fig


def convergence_figure(history: List[Dict[str, float]], baseline_total: float,
                       optimised_total: float) -> go.Figure:
    hist_df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["gen"], y=hist_df["median"], mode="lines",
        name="mediana población", line=dict(color="#1f77b4", width=1.5),
        opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=hist_df["gen"], y=hist_df["best"], mode="lines",
        name="mejor individuo", line=dict(color="#2ca02c", width=3),
    ))
    fig.add_hline(y=baseline_total, line_dash="dash", line_color="#d62728",
                  annotation_text=f"baseline {baseline_total:.1f}h",
                  annotation_position="top right")
    fig.add_hline(y=optimised_total, line_dash="dot", line_color="#2ca02c",
                  annotation_text=f"optimizado {optimised_total:.1f}h",
                  annotation_position="bottom right")
    fig.update_layout(
        title=f"Convergencia GA · ahorro neto {baseline_total - optimised_total:+.1f}h "
              f"({(baseline_total - optimised_total) / baseline_total * 100:+.1f}%)",
        xaxis_title="generación", yaxis_title="fitness (horas + penalizaciones)",
        height=400, plot_bgcolor="white",
        yaxis_range=[min(hist_df["best"].min(), optimised_total) * 0.95,
                     baseline_total * 1.15],
    )
    fig.update_yaxes(gridcolor="#eee")
    return fig


def changeover_heatmap(ctx: OptimizerContext, individual, line: str) -> go.Figure:
    skus = individual[line]
    n = len(skus)
    mat = np.full((n, n), np.nan)
    for i, p in enumerate(skus):
        for j, q in enumerate(skus):
            if i == j:
                continue
            v = ctx.changeover[line].get((p, q))
            if v is None:
                v = ctx.changeover[line].get((q, p))
            if v is not None:
                mat[i, j] = v
    fig = go.Figure(data=go.Heatmap(
        z=mat, x=skus, y=skus, colorscale="RdYlGn_r", zmin=0, zmax=6,
        colorbar=dict(title="horas"),
        hovertemplate="prev: %{y}<br>next: %{x}<br>%{z:.2f}h<extra></extra>",
    ))
    # Mark the actual path taken (diagonal-shifted blue squares)
    for i in range(n - 1):
        fig.add_shape(type="rect",
                      x0=i + 0.5, x1=i + 1.5, y0=i - 0.5, y1=i + 0.5,
                      line=dict(color="blue", width=2.5), fillcolor="rgba(0,0,0,0)")
    fig.update_layout(
        title=f"L{line} · changeovers reales 2025 · cuadros azules = secuencia elegida",
        height=520, xaxis_title="SKU siguiente", yaxis_title="SKU previo",
        margin=dict(l=80, r=20, t=60, b=120),
    )
    fig.update_xaxes(tickangle=-90)
    return fig


def migration_figure(ctx: OptimizerContext, base_ind, opt_ind) -> go.Figure:
    """Sankey: which line each SKU was on (base) → which line (optimised)."""
    base_of = {s: l for l in LINES for s in base_ind[l]}
    opt_of = {s: l for l in LINES for s in opt_ind[l]}
    labels = [f"Plan · L{l}" for l in LINES] + [f"GA · L{l}" for l in LINES]
    src, tgt, vals, link_color, link_lbl = [], [], [], [], []
    for sku in ctx.skus:
        b = base_of[sku]
        o = opt_of[sku]
        s_idx = LINES.index(b)
        t_idx = 3 + LINES.index(o)
        src.append(s_idx)
        tgt.append(t_idx)
        vals.append(1)
        # Migrations get full opacity in destination colour, stayers get pale grey.
        migrated = b != o
        c = LINE_COLOR_HEX[o] if migrated else "#cccccc"
        # Convert hex to rgba.
        h = c.lstrip("#")
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        link_color.append(f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{0.75 if migrated else 0.25})")
        link_lbl.append(f"{sku} ({'migra' if migrated else 'stay'})")

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20, thickness=22,
            line=dict(color="black", width=0.4),
            label=labels,
            color=[LINE_COLOR_HEX[l] for l in LINES] * 2,
        ),
        link=dict(source=src, target=tgt, value=vals,
                  color=link_color, label=link_lbl),
    )])
    n_mig = sum(1 for s in ctx.skus if base_of[s] != opt_of[s])
    fig.update_layout(
        title=f"Migraciones de SKU · {n_mig} de {len(ctx.skus)} SKUs cambian de línea",
        height=520, margin=dict(l=10, r=10, t=60, b=20),
    )
    return fig


def roi_table(ctx: OptimizerContext, base_bd, opt_bd) -> pd.DataFrame:
    rows = []
    for line in LINES:
        cap = HOURS_PER_WEEK[line]
        b, o = base_bd[line], opt_bd[line]
        rows.append({
            "Línea": f"L{line}",
            "Capacidad (h)": cap,
            "Plan horas": round(b["total"], 1),
            "Plan changeover": round(b["changeover"], 1),
            "Plan holgura": round(cap - b["total"], 1),
            "GA horas": round(o["total"], 1),
            "GA changeover": round(o["changeover"], 1),
            "GA holgura": round(cap - o["total"], 1),
            "Ahorrado (h)": round(b["total"] - o["total"], 1),
        })
    total_b = sum(base_bd[l]["total"] for l in LINES)
    total_o = sum(opt_bd[l]["total"] for l in LINES)
    rows.append({
        "Línea": "TOTAL",
        "Capacidad (h)": sum(HOURS_PER_WEEK.values()),
        "Plan horas": round(total_b, 1),
        "Plan changeover": round(sum(base_bd[l]["changeover"] for l in LINES), 1),
        "Plan holgura": round(sum(HOURS_PER_WEEK.values()) - total_b, 1),
        "GA horas": round(total_o, 1),
        "GA changeover": round(sum(opt_bd[l]["changeover"] for l in LINES), 1),
        "GA holgura": round(sum(HOURS_PER_WEEK.values()) - total_o, 1),
        "Ahorrado (h)": round(total_b - total_o, 1),
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🍺 LineWise · Genetic Scheduler · Semana 18-22 May 2026")
st.caption(
    "Optimiza la asignación SKU→línea y la secuencia intra-línea para L14/L17/L19 "
    "del Prat, usando la matriz de changeovers real del histórico 2025 y la "
    "demanda semanal del **Diario Hl_Planif**."
)

# Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.header("Configuración")
    data_dir = st.text_input("Directorio de datos OPERACIONS",
                             value=str(DEFAULT_DATA_DIR))
    st.divider()
    st.subheader("Hiperparámetros GA")
    pop_size = st.slider("Tamaño de población", 20, 200, 60, step=10)
    n_gen = st.slider("Generaciones", 30, 400, 150, step=10)
    seed = st.number_input("Random seed", value=42, step=1)
    run_btn = st.button("▶ Ejecutar GA", type="primary", use_container_width=True)
    st.divider()
    st.caption(
        "El GA usa lookup histórico real (no modelos): throughput mediano por "
        "(SKU, línea) y matriz de changeovers construida por "
        "`PostMortemAnalyzer.build_transition_matrices()`."
    )

# Load data once
try:
    ctx = get_context(data_dir)
except Exception as exc:
    st.error(f"No pude cargar los datos desde `{data_dir}`: {exc}")
    st.stop()

base_ind = baseline_individual(ctx)
base_bd = breakdown(ctx, base_ind)

# Decide whether to re-run.
if "result" not in st.session_state or run_btn:
    progress = st.progress(0.0, text="Corriendo evolución…")
    best_holder = {"fit": None}

    def _cb(gen, best_fit, median):
        progress.progress((gen + 1) / n_gen,
                          text=f"Generación {gen + 1}/{n_gen} · "
                               f"mejor = {best_fit:.1f}h · mediana = {median:.1f}h")
        best_holder["fit"] = best_fit

    t0 = time.time()
    best_ind, history = evolve(ctx, pop_size=pop_size, n_gen=n_gen, seed=seed,
                               on_generation=_cb)
    elapsed = time.time() - t0
    progress.empty()
    st.session_state["result"] = {
        "best_ind": best_ind,
        "history": history,
        "fitness": evaluate_schedule(ctx, best_ind)[0],
        "elapsed": elapsed,
    }

result = st.session_state["result"]
opt_ind = result["best_ind"]
opt_bd = breakdown(ctx, opt_ind)

# KPI strip ----------------------------------------------------------------
total_b = sum(base_bd[l]["total"] for l in LINES)
total_o = sum(opt_bd[l]["total"] for l in LINES)
saved = total_b - total_o
saved_pct = saved / total_b * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Plan del planner",  f"{total_b:.1f} h")
c2.metric("Optimizado GA",     f"{total_o:.1f} h", delta=f"{-saved:.1f} h",
          delta_color="inverse")
c3.metric("Ahorrado",           f"{saved:+.1f} h",  delta=f"{saved_pct:+.1f}%")
c4.metric("L19 (cuello)",       f"{opt_bd['19']['total']:.1f} / 115 h",
          delta=f"{base_bd['19']['total'] - opt_bd['19']['total']:+.1f} h vs plan",
          delta_color="normal")

# Priority compliance banner
prio_msgs = []
for sku, line in PRIORITY_ORDERS:
    if sku not in ctx.skus:
        prio_msgs.append(f"⚠️ `{sku}` no está en la demanda esta semana — regla N/A.")
    elif sku in opt_ind[line]:
        pos = opt_ind[line].index(sku) + 1
        n = len(opt_ind[line])
        ok = pos <= max(1, int(0.25 * n))
        prio_msgs.append(
            f"{'✅' if ok else '❌'} `{sku}` en L{line}: slot {pos}/{n}"
        )
    else:
        prio_msgs.append(f"❌ `{sku}` no asignado a L{line}.")
st.info(" · ".join(prio_msgs))

st.divider()

# Tabs ---------------------------------------------------------------------
tab_kpi, tab_gantt, tab_mig, tab_co, tab_conv, tab_data = st.tabs([
    "📊 ROI", "📅 Gantt baseline vs GA", "🔄 Migraciones",
    "🔥 Changeovers", "📈 Convergencia", "📋 Datos crudos"
])

with tab_kpi:
    st.subheader("Reparto de horas por línea")
    st.plotly_chart(stacked_hours_figure(base_bd, opt_bd), use_container_width=True)
    st.subheader("Tabla ROI semana")
    df_roi = roi_table(ctx, base_bd, opt_bd)
    # Highlight savings: green positive, red negative
    def _color_saved(v):
        if isinstance(v, (int, float)):
            if v > 0:
                return "color: #2ca02c; font-weight: bold;"
            if v < 0:
                return "color: #d62728; font-weight: bold;"
        return ""
    st.dataframe(
        df_roi.style.applymap(_color_saved, subset=["Ahorrado (h)"]),
        use_container_width=True, hide_index=True,
    )

with tab_gantt:
    st.subheader("Gantt — Plan del planner")
    st.plotly_chart(gantt_figure(ctx, base_ind, "Baseline (diario)"),
                    use_container_width=True)
    st.subheader("Gantt — Schedule optimizado por GA")
    st.plotly_chart(gantt_figure(ctx, opt_ind, "GA optimizado"),
                    use_container_width=True)
    st.caption(
        "Hover sobre cada bloque para ver HL, throughput, formato. "
        "Bloques rojos = changeover. Gris = arranque. Línea roja vertical = capacidad."
    )

with tab_mig:
    st.subheader("¿Qué SKUs movió el GA entre líneas?")
    st.plotly_chart(migration_figure(ctx, base_ind, opt_ind),
                    use_container_width=True)
    base_of = {s: l for l in LINES for s in base_ind[l]}
    opt_of = {s: l for l in LINES for s in opt_ind[l]}
    migr = pd.DataFrame([
        {"SKU": s, "Formato": ctx.sku_format[s],
         "HL semana": ctx.volumes[s],
         "Plan línea": f"L{base_of[s]}", "GA línea": f"L{opt_of[s]}",
         "Motivo": "Migración" if base_of[s] != opt_of[s] else "—"}
        for s in ctx.skus
    ]).sort_values(["Motivo", "Formato", "SKU"], ascending=[False, True, True])
    st.dataframe(migr, use_container_width=True, hide_index=True)

with tab_co:
    st.subheader("Matriz de changeovers reales (2025)")
    st.caption(
        "Cada celda = horas medias de cambio de formato observadas en 2025 "
        "para el par `(SKU previo → SKU siguiente)` en esa línea. "
        "Los cuadros azules son las transiciones que la mejor solución del GA acabó eligiendo."
    )
    line_pick = st.radio("Línea", LINES, horizontal=True, key="co_line")
    st.plotly_chart(changeover_heatmap(ctx, opt_ind, line_pick),
                    use_container_width=True)

with tab_conv:
    st.subheader(f"Convergencia GA · pop={pop_size}, ngen={n_gen}, seed={seed}")
    st.plotly_chart(convergence_figure(result["history"], total_b, total_o),
                    use_container_width=True)
    st.caption(
        f"GA ejecutado en {result.get('elapsed', 0):.1f}s. "
        "Cuando la línea verde queda muy por debajo de la roja, el GA encontró "
        "una planificación claramente mejor que la del planner. Si las dos "
        "líneas se tocan, la planificación original ya era casi óptima."
    )

with tab_data:
    st.subheader("Demanda semanal (Diario Hl_Planif)")
    st.dataframe(
        ctx.weekly[["sku", "original_line", "original_tren", "hl_total",
                    "first_fecha"]],
        use_container_width=True, hide_index=True,
    )
    st.subheader("Schedule optimizado — formato Gantt")
    st.dataframe(schedule_to_gantt(ctx, opt_ind), use_container_width=True,
                 hide_index=True)
    st.download_button(
        "⬇ Descargar Gantt CSV",
        schedule_to_gantt(ctx, opt_ind).to_csv(index=False).encode(),
        file_name="schedule_optimizado.csv", mime="text/csv",
    )

st.divider()
st.caption(
    "GA: chromosome `{línea: [skus_ordenados]}` · smart-init feasible · "
    "PMX-flavoured crossover · swap + migrate mutation · simulator = lookup "
    "histórico 2025. Reglas mentor: L14 ⊃ {1/2, 1/3} · L17 ⊃ {1/3} · "
    "L19 ⊃ {1/2, 1/3, 2/5} · caps 110/115/115h · `VI1324MY` urgente en L17."
)
