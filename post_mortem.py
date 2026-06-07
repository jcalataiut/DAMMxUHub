"""
PostMortemAnalyzer — Núcleo analítico para el reto LineWise (DAMM x Engineering HUB).

Detecta ineficiencias históricas en las líneas 14, 17 y 19 de El Prat mediante
modelado matricial y teoría de grafos dirigidos.

Granularidad de análisis: semanal (ISO week).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from data_loaders import add_graph_node_columns, classify_graph_edge

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

LINES = ["14", "17", "19"]

# Umbral: si el tiempo de mantenimiento supera este % del tiempo total de la OF,
# se considera que el OEE está contaminado por mantenimiento, no por secuencia.
MAINTENANCE_CONTAMINATION_THRESHOLD = 0.20

# Umbral estadístico para identificar "black spots" (desviaciones significativas).
BLACK_SPOT_STD_THRESHOLD = 1.5

# Nombre de la columna interna que representa la semana ISO
WEEK_COL = "year_week"


def _normalize_tren(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza la columna tren: Excel lee float64 (14.0) → str ('14')."""
    if "tren" not in df.columns:
        return df
    df = df.copy()
    df["tren"] = pd.to_numeric(df["tren"], errors="coerce").apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal
# ─────────────────────────────────────────────────────────────────────────────

class PostMortemAnalyzer:
    """
    Analiza el histórico de producción de latas de El Prat para identificar
    qué secuencias de productos degradan el OEE y cuantificar su impacto.

    Parameters
    ----------
    df_oee : pd.DataFrame
        Datos de OEE por OF. Columnas requeridas:
        ['of', 'fecha', 'sku', 'tren', 'oee', 'disponibilidad', 'rendimiento', 'marca', 'envase']
    df_cambios : pd.DataFrame
        Datos de cambios de formato. Columnas requeridas:
        ['of', 'fecha', 'sku', 'n_cambios', 'freq_total', 'c_principal',
         'c_brand', 'c_envase', 'c_producto']
    df_mantenimiento : pd.DataFrame
        Datos de mantenimiento. Columnas requeridas:
        ['of', 'tren', 'n_llamadas', 't_espera', 't_intervencion', 't_total',
         't_total_paro', 'oee']
    df_tiempo : pd.DataFrame
        Desglose temporal por OF y máquina. Columnas requeridas:
        ['of', 'tren', 'sku', 'h_tot', 'par_tot', 'oee']
    df_volumen : pd.DataFrame
        Volumen producido por OF. Columnas requeridas:
        ['of', 'fecha', 'sku', 'tren', 'uds', 'hl', 'oee']
    """

    def __init__(
        self,
        df_oee: pd.DataFrame,
        df_cambios: pd.DataFrame,
        df_mantenimiento: pd.DataFrame,
        df_tiempo: pd.DataFrame,
        df_volumen: pd.DataFrame,
    ) -> None:
        self.df_oee = _normalize_tren(df_oee)
        self.df_cambios = _normalize_tren(df_cambios)
        self.df_mantenimiento = _normalize_tren(df_mantenimiento)
        self.df_tiempo = _normalize_tren(df_tiempo)
        self.df_volumen = _normalize_tren(df_volumen)

        # Resultados intermedios (se rellenan al llamar a los métodos)
        self._df_clean: Optional[pd.DataFrame] = None
        self._sequences: Optional[pd.DataFrame] = None
        self._transition_matrices: Optional[Dict[str, pd.DataFrame]] = None
        self._graphs: Optional[Dict[str, nx.DiGraph]] = None
        self._black_spots: Optional[pd.DataFrame] = None

        self._prepare_base_dataset()

    # ─────────────────────────────────────────────────────────────────────────
    # PREPARACIÓN INTERNA
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_base_dataset(self) -> None:
        """Construye el dataset maestro uniendo OEE, cambios, volumen y tiempo."""
        df = self.df_oee.copy()

        # Enriquecer con volumen
        vol_cols = ["of", "hl", "uds"]
        available_vol = [c for c in vol_cols if c in self.df_volumen.columns]
        df = df.merge(
            self.df_volumen[available_vol].drop_duplicates("of"),
            on="of", how="left"
        )

        # Enriquecer con tiempo total de OF
        tiempo_agg = (
            self.df_tiempo.groupby("of")["h_tot"]
            .sum()
            .reset_index()
            .rename(columns={"h_tot": "h_tot_of"})
        )
        df = df.merge(tiempo_agg, on="of", how="left")

        # Enriquecer con datos de cambios
        cambios_cols = [
            c for c in ["of", "n_cambios", "freq_total", "c_principal",
                        "c_brand", "c_envase", "c_producto", "material_precio"]
            if c in self.df_cambios.columns
        ]
        df = df.merge(
            self.df_cambios[cambios_cols].drop_duplicates("of"),
            on="of", how="left"
        )
        if "material_precio" not in df.columns and "mat_precio" in df.columns:
            df["material_precio"] = df["mat_precio"]
        df = add_graph_node_columns(df)

        # Enriquecer con mantenimiento
        mant_agg = (
            self.df_mantenimiento.groupby("of")
            .agg(
                n_llamadas=("n_llamadas", "sum"),
                t_mant_total=("t_total", "sum"),
                t_mant_paro=("t_total_paro", "sum"),
            )
            .reset_index()
        )
        df = df.merge(mant_agg, on="of", how="left")

        # Columnas temporales auxiliares
        df["fecha"] = pd.to_datetime(df["fecha"])
        df = df.dropna(subset=["fecha"]).copy()
        df["mes"] = df["fecha"].dt.month
        # isocalendar() devuelve UInt32 nullable — astype("Int64") evita el error con NaT
        cal = df["fecha"].dt.isocalendar()
        df["semana_iso"] = cal.week.astype("Int64")
        df["anio"] = cal.year.astype("Int64")
        df[WEEK_COL] = df["fecha"].dt.to_period("W-SUN")

        # Orden cronológico proxy: número de OF
        df["of_num"] = df["of"].apply(self._extract_of_number)

        self._df_master = df

    @staticmethod
    def _extract_of_number(of_id: str) -> int:
        """Extrae el componente numérico del ID de una OF para ordenación cronológica."""
        digits = "".join(filter(str.isdigit, str(of_id).split("-")[0]))
        return int(digits) if digits else 0

    # ─────────────────────────────────────────────────────────────────────────
    # 1. LIMPIEZA Y AISLAMIENTO DE MANTENIMIENTO
    # ─────────────────────────────────────────────────────────────────────────

    def clean_and_isolate_maintenance(self) -> pd.DataFrame:
        """
        Separa las caídas de OEE causadas por mantenimiento de las causadas
        por decisiones de secuenciación.

        Estrategia:
        - Si t_mant_paro / h_tot_of > umbral → la OF está 'contaminada' por mant.
        - Para OFs contaminadas se imputa el OEE ajustado eliminando el tiempo
          de paro de mantenimiento del denominador.
        - Se añade la columna 'oee_seq' (OEE limpio de efectos de mantenimiento)
          que es la variable objetivo del análisis post-mortem de secuenciación.

        Returns
        -------
        pd.DataFrame
            Dataset maestro enriquecido con columnas:
            ['mant_contaminada', 'mant_ratio', 'oee_seq', 'oee_adj']
        """
        df = self._df_master.copy()

        # Ratio de tiempo de parada por mantenimiento sobre tiempo total
        df["mant_ratio"] = np.where(
            df["h_tot_of"] > 0,
            df["t_mant_paro"].fillna(0) / df["h_tot_of"],
            0.0,
        )

        # Bandera: OF contaminada por mantenimiento
        df["mant_contaminada"] = df["mant_ratio"] > MAINTENANCE_CONTAMINATION_THRESHOLD

        # OEE ajustado: imputamos quitando el tiempo de mantenimiento del total
        # OEE_adj = OEE_real × h_tot / (h_tot - t_mant_paro)
        # — recupera la señal de secuenciación eliminando el ruido de mantenimiento
        df["oee_adj"] = np.where(
            df["mant_contaminada"] & (df["h_tot_of"] > df["t_mant_paro"].fillna(0)),
            df["oee"] * df["h_tot_of"] / (df["h_tot_of"] - df["t_mant_paro"].fillna(0)),
            df["oee"],
        )
        df["oee_adj"] = df["oee_adj"].clip(upper=1.0)

        # OEE de secuenciación: usamos el ajustado solo si la OF no está muy
        # contaminada (> 50% parada por mantenimiento → imputación por mediana de SKU+línea)
        mediana_sku_linea = (
            df[~df["mant_contaminada"]]
            .groupby(["sku", "tren"])["oee"]
            .median()
            .reset_index()
            .rename(columns={"oee": "oee_mediana_sku_linea"})
        )
        df = df.merge(mediana_sku_linea, on=["sku", "tren"], how="left")

        muy_contaminada = df["mant_ratio"] > 0.50
        df["oee_seq"] = np.where(
            muy_contaminada,
            df["oee_mediana_sku_linea"],   # imputación robusta
            df["oee_adj"],                  # ajuste parcial
        )

        # Cobertura de imputación
        n_contaminadas = df["mant_contaminada"].sum()
        n_muy_contaminadas = muy_contaminada.sum()
        print(
            f"[clean_and_isolate_maintenance] "
            f"OFs contaminadas (>{MAINTENANCE_CONTAMINATION_THRESHOLD:.0%}): "
            f"{n_contaminadas}/{len(df)} ({n_contaminadas/len(df):.1%})\n"
            f"  → imputadas por mediana SKU-línea (>50%): {n_muy_contaminadas}"
        )

        self._df_clean = df
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # SECUENCIAS SEMANALES (uso interno)
    # ─────────────────────────────────────────────────────────────────────────

    def _reconstruct_sequences(self) -> pd.DataFrame:
        """
        Reconstruye la secuencia semanal de OFs por línea y extrae transiciones.

        Hipótesis de orden intra-semana: número de OF como proxy cronológico.
        Esta hipótesis es robusta cuando las OFs son emitidas secuencialmente
        por el sistema ERP, lo cual es el comportamiento estándar de SAP/Blue Yonder.

        Returns
        -------
        pd.DataFrame
            Secuencias con columnas adicionales:
            ['sku_prev', 'prev_node', 'edge_type', 'oee_seq_prev',
             'es_primera_of_semana', 'tipo_transicion']
        """
        if self._df_clean is None:
            self.clean_and_isolate_maintenance()

        df = (
            self._df_clean
            .sort_values(["tren", WEEK_COL, "of_num"])
            .copy()
        )

        grp = df.groupby(["tren", WEEK_COL])

        # Producto anterior en la misma línea y semana
        df["sku_prev"] = grp["sku"].shift(1)
        for col in ["node", "node_marca", "node_volumen", "node_pack", "node_envase"]:
            df[f"prev_{col}"] = grp[col].shift(1)
        df["oee_seq_prev"] = grp["oee_seq"].shift(1)
        df["h_tot_prev"] = grp["h_tot_of"].shift(1)

        # ¿Es la primera OF de la semana en esa línea? (no tiene transición real)
        df["es_primera_of_semana"] = df["sku_prev"].isna()

        # Clasificación de la transición: identidad del grafo por atributo de nodo.
        df["edge_type"] = df.apply(classify_graph_edge, axis=1)
        df["tipo_transicion"] = df["edge_type"]

        self._sequences = df
        return df

    @staticmethod
    def _classify_transition(row: pd.Series) -> str:
        """Clasifica el tipo de transición según las columnas de cambio disponibles."""
        if pd.isna(row.get("sku_prev")):
            return "inicio_semana"
        if row.get("n_cambios", 0) == 0:
            return "continuacion"

        c_principal = str(row.get("c_principal", "")).strip()
        mapping = {
            "Contenido": "cambio_cerveza",
            "Pack. Secundario": "cambio_pack_secundario",
            "Palet": "cambio_palet",
            "Pack, Primario": "cambio_pack_primario",
            "Volumen Envase": "cambio_envase",
            "Marca": "cambio_marca",
            "Referencia": "cambio_referencia",
            "Tapa/Tapón": "cambio_tapon",
        }
        return mapping.get(c_principal, "cambio_otro")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. MATRICES DE TRANSICIÓN
    # ─────────────────────────────────────────────────────────────────────────

    def build_transition_matrices(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Genera matrices de transición N×N por línea.

        Cada celda (node_i → node_j) contiene:
        - oee_mean: OEE medio del nodo destino cuando viene del nodo origen
        - oee_std: desviación estándar del OEE
        - oee_degradation: diferencia vs. OEE baseline del nodo destino
        - count: número de veces que ocurrió esa transición
        - changeover_h_mean: duración media del cambio (horas)

        Returns
        -------
        Dict[str, Dict[str, pd.DataFrame]]
            Clave exterior: línea ('14', '17', '19')
            Claves interiores: 'oee_mean', 'oee_degradation', 'count', 'changeover_h'
        """
        if self._sequences is None:
            self._reconstruct_sequences()

        seq = self._sequences.copy()
        transitions = seq[~seq["es_primera_of_semana"]].copy()

        # Baseline OEE por nodo y línea (mediana de OFs sin cambio o con continuación)
        baseline = (
            seq[seq["tipo_transicion"].isin(["C0_self", "inicio_semana"])]
            .groupby(["tren", "node"])["oee_seq"]
            .median()
            .reset_index()
            .rename(columns={"oee_seq": "oee_baseline"})
        )
        # Fallback: mediana global por nodo-línea si no hay continuaciones
        baseline_global = (
            seq.groupby(["tren", "node"])["oee_seq"]
            .median()
            .reset_index()
            .rename(columns={"oee_seq": "oee_baseline_global"})
        )
        baseline = baseline.merge(baseline_global, on=["tren", "node"], how="right")
        baseline["oee_baseline"] = baseline["oee_baseline"].fillna(
            baseline["oee_baseline_global"]
        )

        transitions = transitions.merge(baseline, on=["tren", "node"], how="left")
        transitions["oee_degradation"] = (
            transitions["oee_baseline"] - transitions["oee_seq"]
        )

        result: Dict[str, Dict[str, pd.DataFrame]] = {}

        for line in LINES:
            df_l = transitions[transitions["tren"] == line].copy()
            if df_l.empty:
                continue

            agg = df_l.groupby(["prev_node", "node"]).agg(
                sku_prev=("sku_prev", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]),
                sku=("sku", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]),
                edge_type=("edge_type", lambda x: x.mode().iloc[0] if len(x.mode()) else "desconocido"),
                node_marca=("node_marca", "first"),
                node_volumen=("node_volumen", "first"),
                node_pack=("node_pack", "first"),
                node_envase=("node_envase", "first"),
                oee_mean=("oee_seq", "mean"),
                oee_std=("oee_seq", "std"),
                oee_degradation=("oee_degradation", "mean"),
                count=("oee_seq", "count"),
                changeover_h_mean=("freq_total", "mean"),
            ).reset_index()

            def _pivot(col: str) -> pd.DataFrame:
                p = agg.pivot(index="prev_node", columns="node", values=col)
                p.index.name = "from_node"
                p.columns.name = "to_node"
                return p

            result[line] = {
                "oee_mean": _pivot("oee_mean"),
                "oee_degradation": _pivot("oee_degradation"),
                "count": _pivot("count"),
                "changeover_h": _pivot("changeover_h_mean"),
                "_raw": agg,  # tabla larga para análisis directo
            }

            print(
                f"[build_transition_matrices] Línea {line}: "
                f"{len(agg)} transiciones únicas "
                f"({agg['prev_node'].nunique()} nodos origen, "
                f"{agg['node'].nunique()} nodos destino)"
            )

        self._transition_matrices = result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 3. GRAFO DIRIGIDO
    # ─────────────────────────────────────────────────────────────────────────

    def build_directed_graph(self) -> Dict[str, nx.DiGraph]:
        """
        Modela las transiciones como un Grafo Dirigido (DiGraph) por línea.

        Nodos: marca × volumen × pack × envase
        Aristas (A → B): el nodo A fue seguido del nodo B en la misma línea
        Peso de arista: degradación de OEE (positivo = pérdida)

        Métricas calculadas:
        - betweenness_centrality: nodos que conectan muchas transiciones (cuellos de botella)
        - in_degree_weighted: cuánta degradación "recibe" cada SKU desde sus predecesores
        - out_degree_weighted: cuánta degradación "causa" cada SKU en sus sucesores

        Returns
        -------
        Dict[str, nx.DiGraph]
            Un grafo por línea con atributos en nodos y aristas.
        """
        if self._transition_matrices is None:
            self.build_transition_matrices()

        graphs: Dict[str, nx.DiGraph] = {}

        for line, mats in self._transition_matrices.items():
            raw = mats["_raw"]
            G = nx.DiGraph()

            for _, row in raw.iterrows():
                u, v = row["prev_node"], row["node"]
                G.add_node(u)
                G.add_node(v, marca=row.get("node_marca"), volumen=row.get("node_volumen"),
                           pack=row.get("node_pack"), envase=row.get("node_envase"))
                G.add_edge(
                    u, v,
                    weight=float(row["oee_degradation"]),
                    oee_mean=float(row["oee_mean"]),
                    count=int(row["count"]),
                    edge_type=row.get("edge_type", "desconocido"),
                    sku_prev=row.get("sku_prev"),
                    sku=row.get("sku"),
                    changeover_h=float(row["changeover_h_mean"])
                    if pd.notna(row["changeover_h_mean"]) else np.nan,
                )

            if len(G.nodes) == 0:
                continue

            # ── Métricas topológicas ──────────────────────────────────────
            # Betweenness: nodos que aparecen en muchos caminos → cuellos de botella
            bc = nx.betweenness_centrality(G, weight="weight", normalized=True)

            # In-degree ponderado: cuánta degradación acumula cada SKU como destino
            in_deg_w = {
                node: sum(
                    d.get("weight", 0)
                    for _, _, d in G.in_edges(node, data=True)
                )
                for node in G.nodes
            }

            # Out-degree ponderado: cuánta degradación genera cada SKU como origen
            out_deg_w = {
                node: sum(
                    d.get("weight", 0)
                    for _, _, d in G.out_edges(node, data=True)
                )
                for node in G.nodes
            }

            # PageRank inverso: nodos que atraen muchos flujos con alta degradación
            try:
                pr = nx.pagerank(G, weight="weight")
            except nx.exception.PowerIterationFailedConvergence:
                pr = {n: 1 / len(G.nodes) for n in G.nodes}

            # Asignar métricas a nodos
            for node in G.nodes:
                G.nodes[node]["betweenness"] = bc.get(node, 0.0)
                G.nodes[node]["in_deg_weighted"] = in_deg_w.get(node, 0.0)
                G.nodes[node]["out_deg_weighted"] = out_deg_w.get(node, 0.0)
                G.nodes[node]["pagerank"] = pr.get(node, 0.0)

            graphs[line] = G
            print(
                f"[build_directed_graph] Línea {line}: "
                f"{G.number_of_nodes()} nodos, {G.number_of_edges()} aristas"
            )

        self._graphs = graphs
        return graphs

    # ─────────────────────────────────────────────────────────────────────────
    # 4. DETECCIÓN DE BLACK SPOTS
    # ─────────────────────────────────────────────────────────────────────────

    def detect_black_spots(self) -> pd.DataFrame:
        """
        Identifica sistemáticamente las transiciones que destruyen el OEE.

        Metodología:
        1. Calcula el OEE baseline por SKU-línea (mediana histórica).
        2. Identifica transiciones donde la degradación > μ + k·σ (k = 1.5 por defecto).
        3. Filtra por frecuencia mínima (≥2 ocurrencias) para evitar falsos positivos.
        4. Añade el coste en HL perdidos: HL_perdidos = degradación × HL_medio × n_ocurrencias

        Returns
        -------
        pd.DataFrame
            Black spots ordenados por impacto total descendente. Columnas:
            ['tren', 'prev_node', 'node', 'edge_type', 'sku_prev', 'sku',
             'oee_mean', 'oee_baseline',
             'oee_degradation', 'count', 'changeover_h_mean', 'hl_perdidos_estimados',
             'gravedad']
        """
        if self._transition_matrices is None:
            self.build_transition_matrices()

        all_rows: List[pd.DataFrame] = []

        for line, mats in self._transition_matrices.items():
            raw = mats["_raw"].copy()
            raw["tren"] = line

            # Filtrar transiciones con suficiente muestra
            raw = raw[raw["count"] >= 2].copy()
            if raw.empty:
                continue

            # Umbral de black spot: degradación media + 1.5 σ
            mu = raw["oee_degradation"].mean()
            sigma = raw["oee_degradation"].std()
            umbral = mu + BLACK_SPOT_STD_THRESHOLD * sigma

            raw["es_black_spot"] = raw["oee_degradation"] > umbral

            # Enriquecer con tipo de transición (del dataset de secuencias)
            if self._sequences is not None:
                tipo_map = (
                    self._sequences[self._sequences["tren"] == line]
                    .dropna(subset=["prev_node"])
                    .groupby(["prev_node", "node"])["tipo_transicion"]
                    .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "desconocido")
                    .reset_index()
                )
                raw = raw.merge(tipo_map, on=["prev_node", "node"], how="left")

            # Coste estimado en HL perdidos
            # HL_perdidos = degradación_OEE × HL_producidos_medio × n_ocurrencias
            hl_medio_node = (
                self._df_master[self._df_master["tren"] == line]
                .groupby("node")["hl"]
                .median()
                .fillna(0)
            )
            raw["hl_medio_destino"] = raw["node"].map(hl_medio_node).fillna(0)
            raw["hl_perdidos_estimados"] = (
                raw["oee_degradation"].clip(lower=0)
                * raw["hl_medio_destino"]
                * raw["count"]
            )

            # Nivel de gravedad
            raw["gravedad"] = pd.cut(
                raw["oee_degradation"],
                bins=[-np.inf, 0.05, 0.10, 0.20, np.inf],
                labels=["leve", "moderada", "alta", "crítica"],
            )

            all_rows.append(raw[raw["es_black_spot"]].copy())

        if not all_rows:
            return pd.DataFrame()

        result = (
            pd.concat(all_rows, ignore_index=True)
            .sort_values("hl_perdidos_estimados", ascending=False)
            .reset_index(drop=True)
        )

        self._black_spots = result
        print(
            f"[detect_black_spots] {len(result)} black spots detectados "
            f"(umbral degradación > μ + {BLACK_SPOT_STD_THRESHOLD}σ, mín. 2 ocurrencias)"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 5. EXPORTACIÓN DE MÉTRICAS DE EXPLICABILIDAD
    # ─────────────────────────────────────────────────────────────────────────

    def export_explainability_metrics(self) -> Dict[str, pd.DataFrame]:
        """
        Genera las tablas resumen de explicabilidad listas para presentar
        al jurado o a un usuario de negocio.

        Returns
        -------
        Dict[str, pd.DataFrame] con claves:
        - 'top_black_spots': las 10 peores transiciones con justificación
        - 'critical_nodes': nodos más críticos del grafo por línea
        - 'weekly_oee_loss': HL y OEE perdidos por semana (series temporal)
        - 'line_summary': resumen por línea
        """
        if self._black_spots is None:
            self.detect_black_spots()
        if self._graphs is None:
            self.build_directed_graph()

        # ── Top 10 peores transiciones ────────────────────────────────────
        cols_report = [
            "tren", "prev_node", "node", "edge_type", "sku_prev", "sku",
            "oee_mean", "oee_degradation", "count",
            "changeover_h_mean", "hl_perdidos_estimados", "gravedad",
        ]
        available_cols = [c for c in cols_report if c in self._black_spots.columns]
        top_black_spots = (
            self._black_spots[available_cols]
            .head(10)
            .rename(columns={
                "tren": "Línea",
                "prev_node": "Nodo origen",
                "node": "Nodo destino",
                "edge_type": "Tipo arista",
                "sku_prev": "SKU origen observado",
                "sku": "SKU destino observado",
                "oee_mean": "OEE medio resultante",
                "oee_degradation": "Degradación OEE",
                "count": "Veces ocurrido",
                "changeover_h_mean": "Changeover medio (h)",
                "hl_perdidos_estimados": "HL perdidos estimados",
                "gravedad": "Gravedad",
            })
        )

        # ── Nodos críticos del grafo ──────────────────────────────────────
        node_rows: List[dict] = []
        for line, G in self._graphs.items():
            for node, attrs in G.nodes(data=True):
                node_rows.append({
                    "Línea": line,
                    "Nodo": node,
                    "Betweenness": attrs.get("betweenness", 0),
                    "In-degree ponderado (degradación recibida)":
                        attrs.get("in_deg_weighted", 0),
                    "Out-degree ponderado (degradación causada)":
                        attrs.get("out_deg_weighted", 0),
                    "PageRank": attrs.get("pagerank", 0),
                })

        critical_nodes = (
            pd.DataFrame(node_rows)
            .sort_values("In-degree ponderado (degradación recibida)", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )

        # ── Pérdida de OEE por semana ─────────────────────────────────────
        if self._sequences is not None:
            seq_trans = self._sequences[~self._sequences["es_primera_of_semana"]].copy()
            weekly_loss = (
                seq_trans.groupby([WEEK_COL, "tren"])
                .agg(
                    n_transiciones=("oee_seq", "count"),
                    oee_medio=("oee_seq", "mean"),
                    oee_degradacion_total=("oee_seq", lambda x: (
                        self._sequences[
                            self._sequences["es_primera_of_semana"]
                        ].groupby("tren")["oee_seq"].median().get(
                            seq_trans.loc[x.index, "tren"].iloc[0], x.mean()
                        ) - x.mean()
                    )),
                    hl_total=("hl", "sum"),
                )
                .reset_index()
                .rename(columns={"tren": "Línea", WEEK_COL: "Semana"})
            )
        else:
            weekly_loss = pd.DataFrame()

        # ── Resumen por línea ─────────────────────────────────────────────
        line_summary_rows = []
        for line in LINES:
            bs = (
                self._black_spots[self._black_spots["tren"] == line]
                if self._black_spots is not None else pd.DataFrame()
            )
            seq_line = (
                self._sequences[self._sequences["tren"] == line]
                if self._sequences is not None else pd.DataFrame()
            )
            line_summary_rows.append({
                "Línea": line,
                "OEE medio 2025": seq_line["oee_seq"].mean() if not seq_line.empty else np.nan,
                "Nº black spots": len(bs),
                "HL perdidos estimados (black spots)": bs["hl_perdidos_estimados"].sum()
                    if not bs.empty and "hl_perdidos_estimados" in bs.columns else 0,
                "Peor arista": (
                    f"{bs.iloc[0]['prev_node']} → {bs.iloc[0]['node']}"
                    if not bs.empty else "N/A"
                ),
            })
        line_summary = pd.DataFrame(line_summary_rows)

        metrics = {
            "top_black_spots": top_black_spots,
            "critical_nodes": critical_nodes,
            "weekly_oee_loss": weekly_loss,
            "line_summary": line_summary,
        }

        print("[export_explainability_metrics] Métricas de explicabilidad generadas:")
        for k, v in metrics.items():
            print(f"  · {k}: {len(v)} filas")

        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # UTILIDADES PÚBLICAS
    # ─────────────────────────────────────────────────────────────────────────

    def get_sequences(self) -> pd.DataFrame:
        """Retorna el DataFrame de secuencias reconstruidas."""
        if self._sequences is None:
            self._reconstruct_sequences()
        return self._sequences

    def get_clean_data(self) -> pd.DataFrame:
        """Retorna el dataset maestro limpio con OEE ajustado por mantenimiento."""
        if self._df_clean is None:
            self.clean_and_isolate_maintenance()
        return self._df_clean

    def get_transition_raw(self, line: str) -> pd.DataFrame:
        """Retorna la tabla larga de transiciones para una línea específica."""
        if self._transition_matrices is None:
            self.build_transition_matrices()
        return self._transition_matrices[line]["_raw"]

    def get_graph(self, line: str) -> nx.DiGraph:
        """Retorna el grafo dirigido de una línea específica."""
        if self._graphs is None:
            self.build_directed_graph()
        return self._graphs[line]

    def run_full_pipeline(self) -> Dict[str, pd.DataFrame]:
        """
        Ejecuta el pipeline completo en orden:
        1. clean_and_isolate_maintenance
        2. build_transition_matrices
        3. build_directed_graph
        4. detect_black_spots
        5. export_explainability_metrics

        Returns
        -------
        Dict con todas las métricas de explicabilidad.
        """
        print("=" * 60)
        print("  PostMortemAnalyzer — Pipeline completo")
        print("=" * 60)
        self.clean_and_isolate_maintenance()
        self.build_transition_matrices()
        self.build_directed_graph()
        self.detect_black_spots()
        return self.export_explainability_metrics()
