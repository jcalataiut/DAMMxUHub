"""LineWise · Optuna Streamlit dashboard.

Two main flows:
1. **Histórico semanal 2025** — explore every week of last year, line by line,
   compare theoretical (what our simulator would have predicted) vs real
   (h_tot really used). Real must be ≥ theoretical because reality includes
   unplanned change-related events the simulator can't see in advance.
2. **Propuesta Optuna 18-22 May 2026** — the Bayesian optimized schedule for
   the test week, contrasted with the planner's own plan. Same total HL on
   both sides → same theoretical at the plant level; the only thing that
   differs is the distribution, which is what reduces the real execution time.

Run with:
    streamlit run app_optuna_streamlit.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    schedule_to_gantt,
    throughput_rate,
)
from optuna_optimizer import (  # noqa: E402
    load_historical_executed,
    run_study,
    theoretical_hours_for_week,
    weekly_sequence,
)

# ---------------------------------------------------------------------------
# Page config + constants
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LineWise · Optuna Scheduler",
    page_icon="🧬", layout="wide", initial_sidebar_state="expanded",
)

DEFAULT_DATA_DIR = Path("/Users/josecalatayud/DAMMxEHub/DAMMxUHub/OPERACIONS")
LINE_COLOR = {"14": "#1f77b4", "17": "#ff7f0e", "19": "#2ca02c"}
FMT_COLOR = {"1/2": "#4c72b0", "1/3": "#dd8452", "2/5": "#55a868"}


# ---------------------------------------------------------------------------
# Cached pipelines
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando datos OPERACIONS y matrices históricas…")
def get_context(data_dir: str) -> OptimizerContext:
    return build_context(Path(data_dir))


@st.cache_resource(show_spinner="Cargando histórico ejecutado 2025…")
def get_historical(data_dir: str) -> pd.DataFrame:
    return load_historical_executed(Path(data_dir))


# ---------------------------------------------------------------------------
# Plotly builders
# ---------------------------------------------------------------------------

def historical_gantt(week_seqs: Dict[str, pd.DataFrame], week_label: str) -> go.Figure:
    """Each OF as a horizontal bar of width = h_tot, coloured by SKU."""
    rows = []
    for line, df in week_seqs.items():
        if df.empty:
            continue
        for _, r in df.iterrows():
            rows.append({
                "line": f"L{line}", "sku": r["sku"], "of": r["of"],
                "fecha": r["fecha"], "start_h": r["start_h"],
                "dur_h": r["dur_h"], "hl": r["hl"], "oee": r["oee"],
            })
    if not rows:
        fig = go.Figure()
        fig.update_layout(title=f"Semana {week_label} — sin datos", height=300)
        return fig

    df_all = pd.DataFrame(rows)
    skus_unique = df_all["sku"].dropna().unique().tolist()
    palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Pastel
    sku_color = {sku: palette[i % len(palette)] for i, sku in enumerate(skus_unique)}

    fig = go.Figure()
    for _, r in df_all.iterrows():
        hover = (
            f"<b>{r['sku']}</b><br>OF {r['of']}<br>"
            f"Línea: {r['line']}<br>"
            f"Fecha: {pd.to_datetime(r['fecha']).date()}<br>"
            f"Duración real: {r['dur_h']:.2f}h<br>"
            f"HL producidos: {r['hl']:,.0f}<br>"
            f"OEE: {r['oee']:.2%}" if pd.notna(r["oee"]) else
            f"<b>{r['sku']}</b><br>OF {r['of']}<br>Duración real: {r['dur_h']:.2f}h"
        )
        fig.add_trace(go.Bar(
            x=[r["dur_h"]], y=[r["line"]], base=r["start_h"], orientation="h",
            marker=dict(color=sku_color.get(r["sku"], "#999"),
                        line=dict(color="black", width=0.4)),
            text=r["sku"] if r["dur_h"] >= 3 else "",
            textposition="inside", insidetextanchor="middle",
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        ))
    for line in LINES:
        fig.add_vline(x=HOURS_PER_WEEK[line], line_dash="dash",
                      line_color="red", opacity=0.4)
    fig.update_layout(
        barmode="stack", title=f"Secuencia real ejecutada · semana {week_label}",
        xaxis_title="Horas acumuladas desde el inicio de la semana",
        yaxis=dict(categoryorder="array", categoryarray=[f"L{l}" for l in LINES]),
        height=380, plot_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.update_xaxes(gridcolor="#eee")
    return fig


def theo_vs_real_bars(theo_real: Dict[str, Dict[str, float]],
                       week_label: str) -> go.Figure:
    x = [f"L{l}" for l in LINES]
    theo = [theo_real[l]["theoretical"] for l in LINES]
    sim = [theo_real[l]["simulator"] for l in LINES]
    real = [theo_real[l]["real"] for l in LINES]
    diff = [r - t for r, t in zip(real, theo)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=theo, name="Teórico ideal (planificación)",
                          marker_color="#4c72b0",
                          text=[f"{v:.0f}h" for v in theo], textposition="outside"))
    fig.add_trace(go.Bar(x=x, y=sim, name="Simulador (mediana histórica)",
                          marker_color="#8c8c8c",
                          text=[f"{v:.0f}h" for v in sim], textposition="outside"))
    fig.add_trace(go.Bar(x=x, y=real, name="Real ejecutado (h_tot)",
                          marker_color="#dd8452",
                          text=[f"{v:.0f}h" for v in real], textposition="outside"))
    for i, l in enumerate(LINES):
        fig.add_annotation(
            x=x[i], y=max(theo[i], real[i], sim[i]) + 10,
            text=f"<b>Real − Teórico: {diff[i]:+.0f}h</b><br>"
                 f"({(diff[i]/theo[i]*100 if theo[i] else 0):+.0f}%)",
            showarrow=False,
            font=dict(size=10, color="#d62728" if diff[i] > 0 else "#2ca02c"),
        )
    fig.update_layout(
        barmode="group",
        title=f"Teórico ideal · simulador · real ejecutado — semana {week_label}<br>"
              "<sub>Teórico ideal = HL ÷ ritmo p90 histórico (lo que el planner espera) · "
              "Simulador = HL ÷ ritmo mediano (lo que nuestra GA/Optuna usan) · "
              "Real = h_tot medidas en planta</sub>",
        yaxis_title="Horas semana", height=440, plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(gridcolor="#eee")
    return fig


def optuna_gantt(ctx: OptimizerContext, individual, title: str) -> go.Figure:
    gantt = schedule_to_gantt(ctx, individual)

    def _color(row):
        if row["type"] == "changeover":
            return "#d62728"
        if row["type"] == "startup":
            return "#bdbdbd"
        return FMT_COLOR.get(row["format"], "#999")

    gantt["color"] = gantt.apply(_color, axis=1)
    fig = go.Figure()
    for _, r in gantt.iterrows():
        hover = (
            f"<b>{r['task']}</b><br>Línea: {r['line']}<br>"
            f"Inicio: t+{r['start_h']:.1f}h · duración: {r['duration_h']:.2f}h"
        )
        if r["type"] == "production":
            hover += (f"<br>HL: {r['hl']:,.0f}<br>"
                      f"Throughput: {r['rate_hl_per_h']:.0f} HL/h<br>"
                      f"Formato: {r['format']}")
        fig.add_trace(go.Bar(
            x=[r["duration_h"]], y=[r["line"]], base=r["start_h"], orientation="h",
            marker=dict(color=r["color"], line=dict(color="black", width=0.4)),
            text=r["task"] if r["type"] == "production" and r["duration_h"] >= 2 else "",
            textposition="inside", insidetextanchor="middle",
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        ))
    for line in LINES:
        fig.add_vline(x=HOURS_PER_WEEK[line], line_dash="dash",
                      line_color="red", opacity=0.4)
    fig.update_layout(
        barmode="stack", title=title,
        xaxis_title="Horas desde inicio de semana",
        yaxis=dict(categoryorder="array", categoryarray=[f"L{l}" for l in LINES]),
        height=360, plot_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.update_xaxes(gridcolor="#eee")
    return fig


def stacked_hours(base_bd, opt_bd) -> go.Figure:
    categories, parts = [], {"Producción": [], "Changeover": [], "Arranque": [], "Holgura": []}
    colors = {"Producción": "#4c72b0", "Changeover": "#dd8452",
              "Arranque": "#8c8c8c", "Holgura": "#cfe2f3"}
    for line in LINES:
        for scenario, bd in (("Plan", base_bd), ("Optuna", opt_bd)):
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
            text=[f"{v:.0f}h" if v > 4 else "" for v in values],
            textposition="inside",
        ))
    for i, line in enumerate(LINES):
        fig.add_shape(type="line", x0=i*2-0.45, x1=i*2+1.45,
                       y0=HOURS_PER_WEEK[line], y1=HOURS_PER_WEEK[line],
                       line=dict(color="red", dash="dash", width=1.6))
        fig.add_annotation(x=i*2+1.45, y=HOURS_PER_WEEK[line],
                            text=f"cap {HOURS_PER_WEEK[line]:.0f}h",
                            showarrow=False, font=dict(color="red", size=10),
                            xanchor="left", yanchor="middle")
    for cat_i, label in enumerate(categories):
        total = sum(parts[p][cat_i] for p in ("Producción", "Changeover", "Arranque"))
        fig.add_annotation(x=label, y=total+3, text=f"<b>{total:.1f}h</b>",
                            showarrow=False, font=dict(size=11))
    fig.update_layout(
        barmode="stack", title="Reparto de horas — Plan vs Optuna",
        xaxis_title="", yaxis_title="Horas",
        height=460, plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=120, t=60, b=60),
    )
    fig.update_yaxes(gridcolor="#eee",
                      range=[0, max(HOURS_PER_WEEK.values()) + 25])
    return fig


def convergence(study, baseline_total: float, optimised_total: float) -> go.Figure:
    rows = []
    running_best = float("inf")
    for t in study.trials:
        if t.value is None:
            continue
        feasible = t.user_attrs.get("feasible", False)
        if feasible and t.value < running_best:
            running_best = t.value
        rows.append({"trial": t.number, "value": t.value,
                     "feasible": feasible, "running_best": running_best})
    df = pd.DataFrame(rows)
    ok = df[df["feasible"]]
    ko = df[~df["feasible"]]
    fig = go.Figure()
    if not ko.empty:
        fig.add_trace(go.Scatter(
            x=ko["trial"], y=ko["value"].clip(upper=baseline_total*1.5),
            mode="markers", name=f"infeasible ({len(ko)})",
            marker=dict(color="#d62728", size=5, opacity=0.4),
        ))
    if not ok.empty:
        fig.add_trace(go.Scatter(
            x=ok["trial"], y=ok["value"], mode="markers",
            name=f"feasible ({len(ok)})",
            marker=dict(color="#1f77b4", size=6, opacity=0.55),
        ))
        fig.add_trace(go.Scatter(
            x=ok["trial"], y=ok["running_best"], mode="lines",
            name="mejor factible (running)",
            line=dict(color="#2ca02c", width=3),
        ))
    fig.add_hline(y=baseline_total, line_dash="dash", line_color="#d62728",
                  annotation_text=f"baseline {baseline_total:.1f}h",
                  annotation_position="top right")
    fig.add_hline(y=optimised_total, line_dash="dot", line_color="#2ca02c",
                  annotation_text=f"Optuna {optimised_total:.1f}h",
                  annotation_position="bottom right")
    fig.update_layout(
        title=f"Convergencia Optuna · ahorro {baseline_total-optimised_total:+.1f}h "
              f"({(baseline_total-optimised_total)/baseline_total*100:+.1f}%)",
        xaxis_title="trial", yaxis_title="fitness (horas)",
        height=400, plot_bgcolor="white",
        yaxis_range=[min(ok["value"].min() if not ok.empty else baseline_total,
                          optimised_total) * 0.95,
                      baseline_total * 1.15],
    )
    fig.update_yaxes(gridcolor="#eee")
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🧬 LineWise · Optuna Bayesian Scheduler")
st.caption(
    "Explora el histórico ejecutado de 2025 semana a semana y compara la "
    "planificación propuesta por Optuna para la semana 18-22 de mayo 2026 "
    "contra el plan del planner."
)

with st.sidebar:
    st.header("Configuración")
    data_dir = st.text_input("Directorio OPERACIONS", value=str(DEFAULT_DATA_DIR))
    st.divider()
    st.subheader("Hiperparámetros Optuna")
    n_trials = st.slider("Trials", 50, 1000, 300, step=50)
    seed = st.number_input("Random seed", value=42, step=1)
    run_btn = st.button("▶ Ejecutar Optuna", type="primary", use_container_width=True)
    st.divider()
    st.caption(
        "El **teórico** se calcula con el simulador (throughput mediano "
        "histórico + matriz de changeovers reales 2025).  \n"
        "El **real** son las horas `h_tot` realmente medidas en planta para "
        "cada OF. Real ≥ teórico siempre (incluye paradas no planificadas)."
    )

# Load context + historical
try:
    ctx = get_context(data_dir)
    hist = get_historical(data_dir)
except Exception as exc:
    st.error(f"No pude cargar datos desde `{data_dir}`: {exc}")
    st.stop()

base_ind = baseline_individual(ctx)
base_bd = breakdown(ctx, base_ind)
baseline_total = sum(base_bd[l]["total"] for l in LINES)

# Run Optuna lazily (only when button pressed or first time)
if "optuna_result" not in st.session_state or run_btn:
    progress = st.progress(0.0, text="Inicializando Optuna…")
    holder = {"best": float("inf"), "feasible_count": 0}

    def _cb(n, val, feasible):
        if feasible and val < holder["best"]:
            holder["best"] = val
        if feasible:
            holder["feasible_count"] += 1
        progress.progress(
            n / n_trials,
            text=f"Trial {n}/{n_trials} · mejor feasible: "
                 f"{holder['best']:.1f}h · feasibles: {holder['feasible_count']}",
        )

    res = run_study(ctx, n_trials=n_trials, seed=seed, on_trial=_cb)
    progress.empty()
    st.session_state["optuna_result"] = res

opt = st.session_state["optuna_result"]
opt_ind = opt["schedule"]
opt_bd = opt["breakdown"]
opt_total = sum(opt_bd[l]["total"] for l in LINES)

# Tabs ---------------------------------------------------------------------
tab_hist, tab_optuna, tab_compare = st.tabs([
    "📜 Histórico semanal 2025",
    "🎯 Propuesta Optuna · 18-22 May 2026",
    "⚖ Teórico vs Real · marco conceptual",
])


# ===========================================================================
# Tab 1 — Historical weekly explorer
# ===========================================================================
with tab_hist:
    st.subheader("Explorador semana a semana · año 2025")
    weeks_available = sorted(hist["week"].dropna().unique().tolist())
    if not weeks_available:
        st.warning("No encontré semanas en el histórico.")
        st.stop()
    col_pick, col_meta = st.columns([3, 2])
    with col_pick:
        week_pick = st.select_slider(
            "Semana", options=weeks_available,
            value=weeks_available[len(weeks_available) // 2],
        )
    week_seqs = weekly_sequence(hist, week_pick)
    with col_meta:
        n_of_total = sum(len(df) for df in week_seqs.values())
        total_hl = sum(df["hl"].fillna(0).sum() for df in week_seqs.values())
        total_real_h = sum(df["h_tot"].fillna(0).sum() for df in week_seqs.values())
        c1, c2, c3 = st.columns(3)
        c1.metric("OFs ejecutadas", n_of_total)
        c2.metric("HL producidos", f"{total_hl:,.0f}")
        c3.metric("Horas reales", f"{total_real_h:.1f} h")

    st.markdown("#### Secuencia real ejecutada en planta")
    st.plotly_chart(historical_gantt(week_seqs, week_pick),
                    use_container_width=True)

    st.markdown("#### Teórico ideal · Simulador · Real ejecutado · esta semana")
    theo_real = theoretical_hours_for_week(ctx, week_seqs, historical=hist)
    c_chart, c_table = st.columns([3, 2])
    with c_chart:
        st.plotly_chart(theo_vs_real_bars(theo_real, week_pick),
                        use_container_width=True)
    with c_table:
        rows = []
        for line in LINES:
            r = theo_real[line]
            gap = r["real"] - r["theoretical"]
            rows.append({
                "Línea": f"L{line}", "OFs": r["n_of"],
                "Teórico ideal (h)": round(r["theoretical"], 1),
                "Simulador (h)": round(r["simulator"], 1),
                "Real h_tot (h)": round(r["real"], 1),
                "Gap real−teórico": round(gap, 1),
                "Gap (%)": f"{(gap/r['theoretical']*100 if r['theoretical'] else 0):+.0f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
    st.caption(
        "El **teórico ideal** usa el ritmo p90 histórico de cada (SKU, línea) — "
        "lo que el planner asumiría sobre el papel cuando todo va bien. "
        "El **simulador** usa la mediana, lo que GA/Optuna usan para optimizar. "
        "El **real** es el `h_tot` medido en planta. "
        "Real − Teórico es el sobrecoste de ejecución (paradas, micro-paros, calidad, "
        "cambios extra) que una mejor secuenciación puede reducir."
    )

    with st.expander("📋 Listado de OFs ejecutadas esta semana"):
        for line in LINES:
            st.markdown(f"**Línea L{line}**")
            df_line = week_seqs[line]
            if df_line.empty:
                st.caption("Sin actividad esta semana.")
            else:
                show = df_line.copy()
                show["fecha"] = pd.to_datetime(show["fecha"]).dt.date
                show["oee"] = show["oee"].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "—")
                st.dataframe(
                    show[["of", "fecha", "sku", "hl", "h_tot", "oee"]]
                        .rename(columns={"h_tot": "horas reales"}),
                    use_container_width=True, hide_index=True,
                )


# ===========================================================================
# Tab 2 — Optuna proposal for the test week
# ===========================================================================
with tab_optuna:
    st.subheader("Propuesta Optuna para 18-22 May 2026")

    saved = baseline_total - opt_total
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Plan del planner",  f"{baseline_total:.1f} h")
    c2.metric("Optuna (predicho)", f"{opt_total:.1f} h", delta=f"{-saved:.1f} h",
              delta_color="inverse")
    c3.metric("Ahorrado teórico", f"{saved:+.1f} h", delta=f"{saved/baseline_total*100:+.1f}%")
    c4.metric("Todas las caps OK",
              "✓ feasible" if opt["feasible"] else "✗ infeasible",
              delta=f"L17 {opt_bd['17']['total']:.1f}/115h",
              delta_color="off")

    # Priority message
    msgs = []
    for sku, line in PRIORITY_ORDERS:
        if sku not in ctx.skus:
            msgs.append(f"⚠️ `{sku}` no está en esta semana — regla N/A.")
        elif sku in opt_ind[line]:
            pos = opt_ind[line].index(sku) + 1
            n = len(opt_ind[line])
            ok = pos <= max(1, int(0.25 * n))
            msgs.append(f"{'✅' if ok else '❌'} `{sku}` en L{line}: slot {pos}/{n}")
    if msgs:
        st.info(" · ".join(msgs))

    st.markdown("#### Reparto de horas — Plan vs Optuna")
    st.plotly_chart(stacked_hours(base_bd, opt_bd), use_container_width=True)

    st.markdown("#### Gantt — Plan del planner")
    st.plotly_chart(optuna_gantt(ctx, base_ind, "Baseline (plan diario)"),
                    use_container_width=True)
    st.markdown("#### Gantt — Schedule optimizado por Optuna")
    st.plotly_chart(optuna_gantt(ctx, opt_ind, "Optuna optimizado"),
                    use_container_width=True)

    st.markdown("#### Convergencia del estudio Optuna")
    st.plotly_chart(convergence(opt["study"], baseline_total, opt_total),
                    use_container_width=True)
    st.caption(
        f"{n_trials} trials ejecutados en {opt['elapsed_s']:.1f}s. "
        "El sampler TPE usa `constraints_func` para aprender qué combinaciones "
        "violaban capacidad, y los evita en los siguientes trials."
    )


# ===========================================================================
# Tab 3 — Conceptual frame: theoretical vs real
# ===========================================================================
with tab_compare:
    st.subheader("Marco conceptual · teórico vs real, plan vs propuesta")
    st.markdown(
        """
**El argumento de valor.**

- El **tiempo teórico de una planificación** es la suma de
  producción (HL ÷ throughput mediano) + changeovers (matriz histórica) +
  arranque. Es el ideal: lo que el simulador predice si todo va perfecto.
- El **tiempo real** que la empresa registra en planta siempre es **mayor** que
  el teórico porque incluye microparos, paradas de calidad, cambios extra,
  problemas de proveedor, etc. Lo vemos en cada semana de 2025 (tab anterior).
- Para la semana **18-22 mayo 2026**, los HL totales son fijos (28 SKUs,
  36.933 HL). Tanto el plan del planner como la propuesta Optuna producen la
  misma demanda → **a nivel de producción teórica son comparables**.
- La diferencia que captura Optuna está en (1) **mejor distribución entre
  líneas** (descarga L19) y (2) **menos changeovers** al agrupar SKUs con
  matriz de cambio barata. Ambas cosas son robustez de cara a la ejecución.

Por eso esperamos que **si nuestra propuesta se ejecutase**, el real
ejecutado también baje: menos cambios → menos sorpresas → menor gap
real-teórico.
        """
    )

    st.markdown("#### Tiempo teórico predicho · Plan vs Optuna (semana 18-22 May 2026)")
    df_compare = pd.DataFrame([
        {"Plan": "Plan del planner", "Línea": f"L{l}", "Horas": base_bd[l]["total"]}
        for l in LINES
    ] + [
        {"Plan": "Plan del planner", "Línea": "TOTAL", "Horas": baseline_total},
        {"Plan": "Optuna", "Línea": "TOTAL", "Horas": opt_total},
    ] + [
        {"Plan": "Optuna", "Línea": f"L{l}", "Horas": opt_bd[l]["total"]}
        for l in LINES
    ])
    fig = px.bar(df_compare, x="Línea", y="Horas", color="Plan", barmode="group",
                 color_discrete_map={"Plan del planner": "#d62728", "Optuna": "#2ca02c"},
                 text=df_compare["Horas"].round(1).astype(str) + "h",
                 height=420)
    fig.update_traces(textposition="outside")
    fig.update_layout(plot_bgcolor="white",
                       title="Mismo plant total ≈ misma demanda, pero el reparto cambia → menos changeovers")
    fig.update_yaxes(gridcolor="#eee")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Histórico 2025 · gap real-teórico medio por línea")
    # Average gap across all historical weeks
    avg_rows = []
    for line in LINES:
        per_week = []
        for wk in sorted(hist["week"].dropna().unique()):
            seqs = weekly_sequence(hist, wk)
            tr = theoretical_hours_for_week(ctx, seqs, historical=hist)
            if tr[line]["theoretical"] > 0:
                per_week.append({
                    "week": wk,
                    "gap": tr[line]["real"] - tr[line]["theoretical"],
                    "gap_pct": (tr[line]["real"] - tr[line]["theoretical"]) /
                                tr[line]["theoretical"] * 100,
                    "real": tr[line]["real"],
                    "theo": tr[line]["theoretical"],
                })
        if per_week:
            d = pd.DataFrame(per_week)
            avg_rows.append({
                "Línea": f"L{line}",
                "Semanas analizadas": len(d),
                "Gap medio (h)": round(d["gap"].mean(), 1),
                "Gap medio (%)": f"{d['gap_pct'].mean():+.0f}%",
                "Real medio (h)": round(d["real"].mean(), 1),
                "Teórico medio (h)": round(d["theo"].mean(), 1),
            })
    if avg_rows:
        st.dataframe(pd.DataFrame(avg_rows), use_container_width=True,
                     hide_index=True)
        st.caption(
            "Este gap medio es la prima de ejecución que paga la fábrica cada "
            "semana frente al ideal. Reducir changeovers (lo que hace Optuna) "
            "ataca directamente parte de esta prima."
        )

st.divider()
st.caption(
    "Datos: OPERACIONS/ · Simulador: throughput mediano por (SKU,línea) + "
    "matriz de changeovers reales 2025. Optimización: Optuna TPE con "
    "constraints_func + repair pass. Caps: L14=110h, L17=L19=115h. "
    "Urgente: `VI1324MY` → L17."
)
