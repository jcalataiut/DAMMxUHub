from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_loaders import (
    LINES,
    add_graph_node_columns,
    classify_graph_edge,
    load_all_operations,
    load_diario_hl,
    weekly_demand_from_diario,
)
from post_mortem import PostMortemAnalyzer

RAW_DIR = Path(__file__).resolve().parent / "raw_data"
CLEAN_DIR = Path(__file__).resolve().parent / "clean_data"

PHYSICAL_FORMAT_BY_LINE = {
    "14": {"1/2", "1/3"},
    "17": {"1/3"},
    "19": {"1/2", "1/3", "2/5"},
}

_FORMAT_RE = __import__("re").compile(r"(33|50|44)")
_CL_TO_FMT = {33: "1/3", 50: "1/2", 44: "2/5"}


def _parse_format(sku: str) -> str:
    m = _FORMAT_RE.search(str(sku).upper())
    cl = int(m.group(1)) if m else 33
    return _CL_TO_FMT[cl]


def build_2025_frames(hist_weeks: pd.DataFrame):
    """Compute cumulative transitions per week for 2025 animation."""
    df = hist_weeks.sort_values(["line", "fecha", "of"])
    weeks = sorted(df["week"].unique())

    rows = []
    for wk_idx, week in enumerate(weeks, start=1):
        current = df[df["week"] <= week]
        for line in LINES:
            line_df = current[current["line"] == line].sort_values("fecha")
            seq = line_df.dropna(subset=["node"]).copy()
            nodes = seq["node"].tolist()
            skus = seq["sku"].tolist()
            attrs = seq[["node_marca", "node_volumen", "node_pack", "node_envase"]].to_dict("records")
            for i in range(len(nodes) - 1):
                row = {
                    "prev_node": nodes[i], "node": nodes[i + 1],
                    "prev_node_marca": attrs[i]["node_marca"],
                    "prev_node_volumen": attrs[i]["node_volumen"],
                    "prev_node_pack": attrs[i]["node_pack"],
                    "prev_node_envase": attrs[i]["node_envase"],
                    "node_marca": attrs[i + 1]["node_marca"],
                    "node_volumen": attrs[i + 1]["node_volumen"],
                    "node_pack": attrs[i + 1]["node_pack"],
                    "node_envase": attrs[i + 1]["node_envase"],
                }
                edge_type = classify_graph_edge(pd.Series(row))
                rows.append({
                    "week": wk_idx, "line": line,
                    "prev_node": nodes[i], "next_node": nodes[i + 1],
                    "prev_sku": skus[i], "next_sku": skus[i + 1],
                    "edge_type": edge_type,
                })

    frames = pd.DataFrame(rows)
    agg = frames.groupby(["week", "line", "prev_node", "next_node", "edge_type"]).size().reset_index(name="count")
    # Backward-compatible aliases for older app/notebook code. Semantics are nodes.
    agg["prev_sku"] = agg["prev_node"]
    agg["next_sku"] = agg["next_node"]
    agg["line"] = agg["line"].astype(str)
    agg.to_csv(CLEAN_DIR / "frames_2025.csv", index=False)

    node_degree = (
        agg.groupby(["week", "line", "prev_node"])["count"].sum()
        .reset_index().rename(columns={"prev_node": "node"})
    )
    node_in = (
        agg.groupby(["week", "line", "next_node"])["count"].sum()
        .reset_index().rename(columns={"next_node": "node"})
    )
    nodes = pd.merge(node_degree, node_in, on=["week", "line", "node"], how="outer").fillna(0)
    nodes["degree"] = nodes["count_x"] + nodes["count_y"]
    nodes["line"] = nodes["line"].astype(str)
    nodes["sku"] = nodes["node"]  # backward-compatible alias; semantic value is node.
    nodes = nodes[["week", "line", "node", "sku", "degree"]].sort_values(["week", "line", "node"])
    nodes.to_csv(CLEAN_DIR / "nodes_2025.csv", index=False)

    black_spots = _detect_2025_black_spots(agg)
    black_spots.to_csv(CLEAN_DIR / "black_spots_2025.csv", index=False)

    print(f"  2025 frames: {len(weeks)} weeks, {len(agg)} transitions")


def _detect_2025_black_spots(agg: pd.DataFrame) -> pd.DataFrame:
    final = agg[agg["week"] == agg["week"].max()].copy()
    final["line"] = final["line"].astype(str)
    stats = final.groupby(["line", "prev_node", "next_node", "edge_type"])["count"].sum().reset_index()
    means = stats.groupby("line")["count"].mean().to_dict()
    stds = stats.groupby("line")["count"].std().to_dict()
    spots = []
    for _, row in stats.iterrows():
        line = row["line"]
        z = (row["count"] - means.get(line, 0)) / max(stds.get(line, 1), 0.01)
        if z > 1.5:
            spots.append({
                "line": line, "prev_node": row["prev_node"], "next_node": row["next_node"],
                "prev_sku": row["prev_node"], "next_sku": row["next_node"],
                "edge_type": row["edge_type"],
            })
    return pd.DataFrame(spots)


def main():
    CLEAN_DIR.mkdir(exist_ok=True)
    print("Loading raw Excel data...")
    ops = load_all_operations(RAW_DIR)
    df_oee, df_cam, df_mant, df_tiem, df_vol = (
        ops["oee"], ops["cam"], ops["mant"], ops["tiem"], ops["vol"]
    )

    print("Loading weekly demand...")
    weekly = weekly_demand_from_diario(
        load_diario_hl(RAW_DIR / "Diario Hl_Planif.xlsx")
    )
    weekly["original_line"] = weekly["original_tren"].str.split(",").str[0]
    weekly.to_csv(CLEAN_DIR / "demand.csv", index=False)
    skus = weekly["sku"].tolist()
    sku_format = {sku: _parse_format(sku) for sku in skus}

    print("Computing throughput rates...")
    hl_per_h = (
        df_vol.merge(df_tiem[["of", "h_tot"]], on="of", how="left")
        .dropna(subset=["sku", "tren", "hl", "h_tot"])
    )
    hl_per_h = hl_per_h[hl_per_h["h_tot"] > 0].copy()
    hl_per_h["rate"] = hl_per_h["hl"] / hl_per_h["h_tot"]
    hl_per_h = hl_per_h[hl_per_h["rate"].between(20, 800)]
    throughput = (
        hl_per_h.groupby(["sku", "tren"], as_index=False)["rate"]
        .median()
        .rename(columns={"tren": "line"})
    )
    throughput.to_csv(CLEAN_DIR / "throughput_rates.csv", index=False)

    print("Building transition matrices from 2025 data...")
    pm = PostMortemAnalyzer(
        df_oee=df_oee, df_cambios=df_cam,
        df_mantenimiento=df_mant, df_tiempo=df_tiem, df_volumen=df_vol,
    )
    pm.clean_and_isolate_maintenance()
    transitions = pm.build_transition_matrices()

    changeover_rows = []
    for line in LINES:
        raw = transitions.get(line, {}).get("_raw")
        if raw is not None and not raw.empty:
            for _, row in raw.iterrows():
                hours = row["changeover_h_mean"]
                if pd.notna(hours):
                    h = float(hours) / 60.0 if float(hours) > 30 else float(hours)
                    changeover_rows.append({
                        "line": line,
                        "prev_node": row["prev_node"],
                        "next_node": row["node"],
                        "edge_type": row["edge_type"],
                        "prev_sku": row["sku_prev"],
                        "next_sku": row["sku"],
                        "hours": round(h, 4),
                        "count": int(row["count"]),
                    })
    pd.DataFrame(changeover_rows).to_csv(CLEAN_DIR / "changeovers.csv", index=False)

    print("Extracting SKU info...")
    pd.DataFrame([{"sku": s, "format": f} for s, f in sku_format.items()]).to_csv(
        CLEAN_DIR / "sku_info.csv", index=False
    )

    print("Computing historical (sku, line) pairs...")
    hist_pairs: set[tuple[str, str]] = set()
    for df in (df_oee, df_vol, df_tiem):
        if "sku" in df.columns and "tren" in df.columns:
            for sku, tren in df[["sku", "tren"]].dropna().itertuples(index=False):
                hist_pairs.add((str(sku), str(tren)))
    hist_df = pd.DataFrame(sorted(hist_pairs), columns=["sku", "line"])
    hist_df.to_csv(CLEAN_DIR / "historical_pairs.csv", index=False)

    print("Computing SKU eligibility...")
    eligible_rows = []
    for sku in skus:
        fmt = sku_format[sku]
        opts = [
            l for l in LINES
            if fmt in PHYSICAL_FORMAT_BY_LINE[l] and (sku, l) in hist_pairs
        ]
        is_fallback = False
        if not opts:
            is_fallback = True
            opts = [weekly.loc[weekly["sku"] == sku, "original_line"].iloc[0]]
        eligible_rows.append({
            "sku": sku, "eligible_lines": ",".join(opts), "is_fallback": is_fallback,
        })
    pd.DataFrame(eligible_rows).to_csv(CLEAN_DIR / "sku_eligibility.csv", index=False)

    print("Building historical weeks table...")
    hist_weeks = df_oee[[
        "of", "fecha", "tren", "sku", "marca", "envase", "tipo_envase"
    ]].dropna(subset=["of"]).copy()
    if "material_precio" in df_cam.columns:
        hist_weeks = hist_weeks.merge(
            df_cam[["of", "material_precio"]].drop_duplicates("of"),
            on="of", how="left"
        )
    elif "material_precio" in df_vol.columns:
        hist_weeks = hist_weeks.merge(
            df_vol[["of", "material_precio"]].drop_duplicates("of"),
            on="of", how="left"
        )
    hist_weeks["fecha"] = pd.to_datetime(hist_weeks["fecha"], errors="coerce")
    hist_weeks = hist_weeks.dropna(subset=["fecha"])
    hist_weeks = hist_weeks.rename(columns={"tren": "line"})
    hist_weeks = add_graph_node_columns(hist_weeks)
    if "h_tot" in df_tiem.columns:
        h_tot = df_tiem.groupby("of", as_index=False)["h_tot"].sum()
        hist_weeks = hist_weeks.merge(h_tot, on="of", how="left")
    else:
        hist_weeks["h_tot"] = np.nan
    if "hl" in df_vol.columns:
        hist_weeks = hist_weeks.merge(
            df_vol.groupby("of", as_index=False)["hl"].sum(), on="of", how="left"
        )
    else:
        hist_weeks["hl"] = np.nan
    if "oee" in df_oee.columns:
        oee_of = df_oee.groupby("of", as_index=False)["oee"].mean()
        hist_weeks = hist_weeks.merge(oee_of, on="of", how="left")
    else:
        hist_weeks["oee"] = np.nan

    hist_weeks["week"] = hist_weeks["fecha"].dt.to_period("W-SUN").astype(str)
    hist_weeks["week_start"] = hist_weeks["fecha"].dt.to_period("W-SUN").dt.start_time
    hist_weeks = hist_weeks.sort_values(["line", "fecha", "of"]).reset_index(drop=True)
    hist_weeks.to_csv(CLEAN_DIR / "historical_weeks.csv", index=False)

    print("Building 2025 animation frames...")
    build_2025_frames(hist_weeks)

    print("Saving parameters...")
    params = {
        "hours_per_week": {"14": 110.0, "17": 115.0, "19": 115.0},
        "startup_hours": {"14": 1.0, "17": 1.5, "19": 1.5},
        "priority_orders": [["VI1324MY", "17"]],
        "lines": LINES,
    }
    json.dump(params, open(CLEAN_DIR / "params.json", "w"), indent=2)
    print(f"Done. Clean data saved to {CLEAN_DIR}")


if __name__ == "__main__":
    main()
