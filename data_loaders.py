"""Shared data loaders for the LineWise operations notebooks.

The notebooks originally had slightly different parsers for the same Excel
files. Keeping the logic here avoids silent drift, especially for the planning
diary where total rows look similar to SKU rows.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import openpyxl
import pandas as pd


LINES = ["14", "17", "19"]
CAN_HL_BY_FORMAT = {
    "1/2": 0.0050,
    "1/3": 0.0033,
    "2/5": 0.0044,
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace(".", "") for c in df.columns]
    return df


def fix_tren(series: pd.Series) -> pd.Series:
    """Normalize Excel-read line identifiers: 14.0 -> '14'."""
    return pd.to_numeric(series, errors="coerce").apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )


def infer_sku_format(sku: str) -> str:
    """Infer can format from the SKU code used in the Damm exports."""
    sku = str(sku)
    if "12" in sku:
        return "1/2"
    if "13" in sku:
        return "1/3"
    if "25" in sku:
        return "2/5"
    return "unknown"


def infer_units_per_case(sku: str, description: Optional[str] = None) -> int:
    """Infer units per planned case from SKU code and optional product text."""
    sku_text = str(sku or "").upper()
    desc_text = str(description or "").upper()
    combined = f"{sku_text} {desc_text}"

    text_match = re.search(r"(?:PACK|CAJA|B-|B|P)\s*(12|20|24)\s*U?", combined)
    if text_match:
        return int(text_match.group(1))

    sku_match = re.search(r"(?:P|B|L|M)(12|20|24)", sku_text)
    if sku_match:
        return int(sku_match.group(1))

    return 24


def planned_hl_from_cases(
    sku: str,
    cases: float,
    *,
    description: Optional[str] = None,
) -> float:
    """Convert planned cases to HL using SKU format and inferred case size."""
    sku_format = infer_sku_format(sku)
    hl_per_unit = CAN_HL_BY_FORMAT.get(sku_format, 0.0)
    units_per_case = infer_units_per_case(sku, description)
    return float(cases or 0.0) * units_per_case * hl_per_unit


def load_operational_excel(path: Path) -> pd.DataFrame:
    """Load one standard Damm operations export and normalize common columns."""
    df = pd.read_excel(path)
    df = normalize_columns(df)
    df = df.rename(
        columns={
            "fecha_fin": "fecha",
            "woid": "of",
            "nº_de_cambios": "n_cambios",
            "frecuencia_total": "freq_total",
            "c_principal": "c_principal",
            "c_brand": "c_brand",
            "c_envase": "c_envase",
            "c_producto": "c_producto",
            "tiempo_en_espera": "t_espera",
            "tiempo_intervención": "t_intervencion",
            "tiempo_total": "t_total",
            "tiempo_total_en_marcha": "t_total_marcha",
            "tiempo_total_en_paro": "t_total_paro",
            "nº_llamadas": "n_llamadas",
            "h_tot": "h_tot",
            "par_tot": "par_tot",
            "%_parada": "pct_parada",
            "disp": "disponibilidad",
            "rend": "rendimiento",
        }
    )
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    if "sku" in df.columns:
        df = df[df["sku"] != "LIMPIEZA"].copy()
    if "tren" in df.columns:
        df["tren"] = fix_tren(df["tren"])
        df = df[df["tren"].isin(LINES)].copy()
    return df


def load_all_operations(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all historical files used by the EDA, post-mortem and optimizer."""
    return {
        "oee": load_operational_excel(data_dir / "OEE 14_17_19_ 2025.xlsx"),
        "cam": load_operational_excel(data_dir / "Cambios 14_17_19_ 2025.xlsx"),
        "mant": load_operational_excel(data_dir / "Mantenimiento 14_17_19_ 2025.xlsx"),
        "tiem": load_operational_excel(data_dir / "Tiempo 14_17_19_ 2025.xlsx"),
        "vol": load_operational_excel(data_dir / "Volumen 14_17_19_ 2025.xlsx"),
    }


def _is_total_or_header_row(label: str) -> bool:
    normalized = label.strip().lower()
    if not normalized:
        return True
    if normalized in {"total", "grand total", "centro", "artículo", "articulo"}:
        return True
    if normalized.startswith("total "):
        return True
    return False


def load_diario_hl(path: Path) -> pd.DataFrame:
    """Parse Diario Hl_Planif into tidy rows.

    Returns one row per (date, original line, SKU) with planned and agreed HL.
    The parser intentionally excludes all total/header rows, including the
    final ``TOTAL`` row that was previously being counted as an L19 SKU.
    """
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        return pd.DataFrame(columns=["fecha", "tren", "sku", "hl_planificado", "hl_acordado"])

    header = rows[0]
    date_cols = {}
    for i, value in enumerate(header):
        value_str = str(value or "")
        if (
            "Programa Prod." in value_str
            and "Artículo" not in value_str
            and "Articulo" not in value_str
            and "TOTAL" not in value_str.upper()
        ):
            date_str = value_str.split("\n")[-1].strip()
            if "/" in date_str:
                fecha = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
                if pd.notna(fecha):
                    date_cols[i] = fecha

    records = []
    current_line = None
    for row in rows[1:]:
        label = str(row[0] or "").strip()
        if not label:
            continue
        if "Tren" in label:
            current_line = next((line for line in LINES if line in label), current_line)
            continue
        if current_line is None or label.startswith("-") or _is_total_or_header_row(label):
            continue

        sku = label.strip()
        if not any(char.isdigit() for char in sku) or " " in sku:
            continue

        for col_idx, fecha in date_cols.items():
            hl_plan_raw = row[col_idx] if col_idx < len(row) else None
            hl_acord_raw = row[col_idx + 1] if (col_idx + 1) < len(row) else None
            try:
                hl_plan = float(hl_plan_raw) if hl_plan_raw is not None else 0.0
                hl_acord = float(hl_acord_raw) if hl_acord_raw is not None else 0.0
            except (TypeError, ValueError):
                continue
            if hl_plan > 0:
                records.append(
                    {
                        "fecha": fecha,
                        "tren": current_line,
                        "sku": sku,
                        "row_order": len(records),
                        "hl_planificado": hl_plan,
                        "hl_acordado": hl_acord,
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["fecha", "tren", "sku", "hl_planificado", "hl_acordado"])

    df = df.groupby(["fecha", "tren", "sku"], as_index=False).agg(
        hl_planificado=("hl_planificado", "sum"),
        hl_acordado=("hl_acordado", "sum"),
        row_order=("row_order", "min"),
    )
    return df.sort_values(["tren", "fecha", "sku"]).reset_index(drop=True)


def weekly_demand_from_diario(df_diario: pd.DataFrame) -> pd.DataFrame:
    """Aggregate diary rows into unique weekly SKU demand."""
    if df_diario.empty:
        return pd.DataFrame(columns=["tren", "sku", "hl_total", "original_tren"])

    by_original_line = (
        df_diario.groupby(["tren", "sku"], as_index=False)["hl_planificado"]
        .sum()
        .rename(columns={"hl_planificado": "hl_total"})
    )
    original_tren = (
        by_original_line.groupby("sku")["tren"]
        .agg(lambda values: ",".join(sorted(set(values))))
        .rename("original_tren")
    )
    first_order = df_diario.groupby("sku")["row_order"].min().rename("row_order")
    first_date = df_diario.groupby("sku")["fecha"].min().rename("first_fecha")
    weekly = (
        by_original_line.groupby("sku", as_index=False)["hl_total"]
        .sum()
        .merge(original_tren, on="sku", how="left")
        .merge(first_order, on="sku", how="left")
        .merge(first_date, on="sku", how="left")
    )
    weekly["tren"] = weekly["original_tren"].str.split(",").str[0]
    return weekly[
        ["tren", "sku", "hl_total", "original_tren", "row_order", "first_fecha"]
    ].sort_values("row_order").reset_index(drop=True)


def original_sequences_from_diario(
    df_diario: pd.DataFrame,
    *,
    skus_filter: set[str] | None = None,
) -> Dict[str, List[str]]:
    """Return the original weekly SKU order per line, preserving diary order."""
    weekly = weekly_demand_from_diario(df_diario)
    if skus_filter is not None:
        weekly = weekly[weekly["sku"].isin(skus_filter)].copy()
    sequences: Dict[str, List[str]] = {}
    for line in LINES:
        sequences[line] = (
            weekly[weekly["original_tren"].astype(str).str.split(",").apply(lambda xs: line in xs)]
            .sort_values(["first_fecha", "row_order", "sku"])["sku"]
            .tolist()
        )
    return sequences


def _date_mask(
    values: pd.Series,
    start_date: Optional[str | pd.Timestamp],
    end_date: Optional[str | pd.Timestamp],
) -> pd.Series:
    mask = pd.Series(True, index=values.index)
    if start_date is not None:
        mask &= values >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= values <= pd.Timestamp(end_date)
    return mask


def _combine_date_time(date_value: pd.Series, time_value: pd.Series) -> pd.Series:
    dates = pd.to_datetime(date_value, errors="coerce")
    time_as_text = time_value.fillna("00:00:00").astype(str)
    delta = pd.to_timedelta(time_as_text, errors="coerce").fillna(pd.Timedelta(0))
    return dates + delta


def load_planificado_producciones(
    path: Path,
    *,
    start_date: Optional[str | pd.Timestamp] = None,
    end_date: Optional[str | pd.Timestamp] = None,
) -> pd.DataFrame:
    """Load the 2026 planned production export and estimate planned HL.

    The source quantity is in ``CAJ``. Planned HL is estimated as:
    ``Cntd plan * inferred units per case * HL per unit``.
    """
    raw = pd.read_excel(path)
    df = raw.rename(
        columns={
            "Material": "sku",
            "Tren": "tren",
            "Fecha ini.": "fecha_ini",
            "Hora ini.": "hora_ini",
            "Definición de turno": "turno",
            "Cntd JDA": "cntd_jda",
            "Cntd plan": "cntd_plan",
            "Pndt. Env": "pendiente_env",
            "Unidad medida base": "unidad",
            "Fecha fin": "fecha_fin",
            "Secuencia": "secuencia",
        }
    ).copy()
    df["row_order"] = range(len(df))
    df["sku"] = df["sku"].astype(str).str.strip()
    df["tren"] = fix_tren(df["tren"])
    df["fecha_ini"] = pd.to_datetime(df["fecha_ini"], errors="coerce")
    df["fecha_fin"] = pd.to_datetime(df["fecha_fin"], errors="coerce")
    df["start_ts"] = _combine_date_time(df["fecha_ini"], df.get("hora_ini", pd.Series(index=df.index)))
    df["cntd_plan"] = pd.to_numeric(df["cntd_plan"], errors="coerce").fillna(0.0)
    df["cntd_jda"] = pd.to_numeric(df.get("cntd_jda", 0), errors="coerce").fillna(0.0)
    df["pendiente_env"] = pd.to_numeric(df.get("pendiente_env", 0), errors="coerce").fillna(0.0)

    df = df[df["tren"].isin(LINES)].copy()
    df = df[df["sku"].notna() & (df["sku"] != "") & (df["cntd_plan"] > 0)].copy()
    df = df[_date_mask(df["fecha_ini"], start_date, end_date)].copy()

    df["format"] = df["sku"].apply(infer_sku_format)
    df["units_per_case"] = df["sku"].apply(infer_units_per_case)
    df["hl_per_unit"] = df["format"].map(CAN_HL_BY_FORMAT).fillna(0.0)
    df["uds_plan"] = df["cntd_plan"] * df["units_per_case"]
    df["hl_plan"] = df["uds_plan"] * df["hl_per_unit"]

    keep = [
        "start_ts",
        "fecha_ini",
        "fecha_fin",
        "hora_ini",
        "turno",
        "tren",
        "sku",
        "cntd_plan",
        "unidad",
        "units_per_case",
        "uds_plan",
        "hl_plan",
        "format",
        "secuencia",
        "row_order",
    ]
    return df[[col for col in keep if col in df.columns]].sort_values(
        ["tren", "start_ts", "row_order"]
    ).reset_index(drop=True)


def planned_demand_from_planificado(df_plan: pd.DataFrame) -> pd.DataFrame:
    """Aggregate planned production rows into weekly demand by SKU."""
    if df_plan.empty:
        return pd.DataFrame(columns=["tren", "sku", "hl_total", "original_tren", "row_order"])
    by_line = (
        df_plan.groupby(["tren", "sku"], as_index=False)
        .agg(
            hl_total=("hl_plan", "sum"),
            cntd_plan=("cntd_plan", "sum"),
            uds_plan=("uds_plan", "sum"),
            row_order=("row_order", "min"),
        )
    )
    original_tren = (
        by_line.groupby("sku")["tren"]
        .agg(lambda values: ",".join(sorted(set(values))))
        .rename("original_tren")
    )
    weekly = (
        by_line.groupby("sku", as_index=False)
        .agg(
            hl_total=("hl_total", "sum"),
            cntd_plan=("cntd_plan", "sum"),
            uds_plan=("uds_plan", "sum"),
            row_order=("row_order", "min"),
        )
        .merge(original_tren, on="sku", how="left")
    )
    weekly["tren"] = weekly["original_tren"].str.split(",").str[0]
    return weekly[
        ["tren", "sku", "hl_total", "cntd_plan", "uds_plan", "original_tren", "row_order"]
    ].sort_values("row_order").reset_index(drop=True)


def load_real_production_week(
    path: Path,
    *,
    start_date: Optional[str | pd.Timestamp] = None,
    end_date: Optional[str | pd.Timestamp] = None,
) -> pd.DataFrame:
    """Load actual production for the beta comparison week."""
    raw = pd.read_excel(path)
    df = raw.rename(
        columns={
            "OF": "of",
            "Fecha Fin": "fecha_fin",
            "SKU": "sku",
            "TREN": "tren",
            "UDS": "uds_real",
            "HL": "hl_real",
            "OEE": "oee",
            "DISP": "disp",
            "REND.": "rend",
            "CALID.": "calid",
            "INEF.": "inef",
            "Marca": "marca",
            "Material Precio": "material_precio",
            "Envase": "envase",
            "Tipo Envase": "tipo_envase",
        }
    ).copy()
    df["row_order"] = range(len(df))
    df["fecha_fin"] = pd.to_datetime(df["fecha_fin"], errors="coerce")
    df["sku"] = df["sku"].astype(str).str.strip()
    df["tren"] = fix_tren(df["tren"])
    df["uds_real"] = pd.to_numeric(df["uds_real"], errors="coerce").fillna(0.0)
    df["hl_real"] = pd.to_numeric(df["hl_real"], errors="coerce").fillna(0.0)
    for col in ["oee", "disp", "rend", "calid", "inef"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["tren"].isin(LINES)].copy()
    df = df[df["sku"].notna() & (df["sku"] != "") & (df["sku"].str.lower() != "nan")].copy()
    df = df[_date_mask(df["fecha_fin"], start_date, end_date)].copy()
    df["format"] = df["sku"].apply(infer_sku_format)
    return df.sort_values(["tren", "fecha_fin", "row_order"]).reset_index(drop=True)


def _weighted_average(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    valid = df[value_col].notna() & df[weight_col].gt(0)
    if not valid.any():
        return float("nan")
    return float((df.loc[valid, value_col] * df.loc[valid, weight_col]).sum() / df.loc[valid, weight_col].sum())


def plan_vs_actual_comparison(
    df_plan: pd.DataFrame,
    df_real: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Compare planned vs actual production at line/SKU and line levels."""
    plan_sku = (
        df_plan.groupby(["tren", "sku"], as_index=False)
        .agg(
            hl_plan=("hl_plan", "sum"),
            uds_plan=("uds_plan", "sum"),
            cntd_plan=("cntd_plan", "sum"),
            first_plan_ts=("start_ts", "min"),
            plan_order=("row_order", "min"),
        )
    )
    real_sku_base = (
        df_real.groupby(["tren", "sku"], as_index=False)
        .agg(
            hl_real=("hl_real", "sum"),
            uds_real=("uds_real", "sum"),
            first_real_date=("fecha_fin", "min"),
            real_order=("row_order", "min"),
        )
    )
    oee_rows = []
    for keys, group in df_real.groupby(["tren", "sku"], sort=False):
        oee_rows.append(
            {
                "tren": keys[0],
                "sku": keys[1],
                "oee_real": _weighted_average(group, "oee", "hl_real") if "oee" in group else float("nan"),
                "disp_real": _weighted_average(group, "disp", "hl_real") if "disp" in group else float("nan"),
                "rend_real": _weighted_average(group, "rend", "hl_real") if "rend" in group else float("nan"),
            }
        )
    real_sku = real_sku_base.merge(pd.DataFrame(oee_rows), on=["tren", "sku"], how="left")

    by_sku = plan_sku.merge(real_sku, on=["tren", "sku"], how="outer")
    numeric_cols = ["hl_plan", "uds_plan", "cntd_plan", "hl_real", "uds_real"]
    by_sku[numeric_cols] = by_sku[numeric_cols].fillna(0.0)
    by_sku["delta_hl"] = by_sku["hl_real"] - by_sku["hl_plan"]
    by_sku["attainment_pct"] = by_sku.apply(
        lambda row: row["hl_real"] / row["hl_plan"] if row["hl_plan"] > 0 else float("nan"),
        axis=1,
    )
    by_sku["abs_delta_hl"] = by_sku["delta_hl"].abs()
    by_sku["status"] = "OK"
    by_sku.loc[(by_sku["hl_plan"] > 0) & (by_sku["hl_real"] == 0), "status"] = "NO_PRODUCIDO"
    by_sku.loc[(by_sku["hl_plan"] == 0) & (by_sku["hl_real"] > 0), "status"] = "NO_PLANIFICADO"
    by_sku.loc[(by_sku["hl_plan"] > 0) & (by_sku["hl_real"] > 0) & (by_sku["attainment_pct"] < 0.95), "status"] = "BAJO_PLAN"
    by_sku.loc[(by_sku["hl_plan"] > 0) & (by_sku["attainment_pct"] > 1.05), "status"] = "SOBRE_PLAN"

    plan_line = df_plan.groupby("tren", as_index=False).agg(
        hl_plan=("hl_plan", "sum"),
        uds_plan=("uds_plan", "sum"),
        cntd_plan=("cntd_plan", "sum"),
        skus_plan=("sku", "nunique"),
    )
    real_line = df_real.groupby("tren", as_index=False).agg(
        hl_real=("hl_real", "sum"),
        uds_real=("uds_real", "sum"),
        skus_real=("sku", "nunique"),
    )
    real_line_oee = []
    for line, group in df_real.groupby("tren", sort=False):
        real_line_oee.append(
            {
                "tren": line,
                "oee_real": _weighted_average(group, "oee", "hl_real") if "oee" in group else float("nan"),
                "disp_real": _weighted_average(group, "disp", "hl_real") if "disp" in group else float("nan"),
                "rend_real": _weighted_average(group, "rend", "hl_real") if "rend" in group else float("nan"),
            }
        )
    by_line = plan_line.merge(real_line, on="tren", how="outer").merge(
        pd.DataFrame(real_line_oee), on="tren", how="left"
    )
    for col in ["hl_plan", "uds_plan", "cntd_plan", "skus_plan", "hl_real", "uds_real", "skus_real"]:
        by_line[col] = by_line[col].fillna(0.0)
    by_line["delta_hl"] = by_line["hl_real"] - by_line["hl_plan"]
    by_line["attainment_pct"] = by_line.apply(
        lambda row: row["hl_real"] / row["hl_plan"] if row["hl_plan"] > 0 else float("nan"),
        axis=1,
    )

    return {
        "by_sku": by_sku.sort_values(["tren", "status", "abs_delta_hl"], ascending=[True, True, False]).reset_index(drop=True),
        "by_line": by_line.sort_values("tren").reset_index(drop=True),
    }


def planned_sequences_from_planificado(df_plan: pd.DataFrame) -> Dict[str, list[str]]:
    """Return planned SKU sequence per line from the planned production export."""
    sequences: Dict[str, list[str]] = {}
    for line in LINES:
        rows = df_plan[df_plan["tren"] == line].sort_values(["start_ts", "row_order"])
        sequences[line] = rows.drop_duplicates("sku", keep="first")["sku"].tolist()
    return sequences


def actual_sequences_from_production(df_real: pd.DataFrame) -> Dict[str, list[str]]:
    """Return actual SKU sequence per line from the real production export."""
    sequences: Dict[str, list[str]] = {}
    for line in LINES:
        rows = df_real[df_real["tren"] == line].sort_values(["fecha_fin", "row_order"])
        sequences[line] = rows.drop_duplicates("sku", keep="first")["sku"].tolist()
    return sequences
