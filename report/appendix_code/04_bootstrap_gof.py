"""
Parametric-bootstrap goodness-of-fit calibration.

The usual one-sample KS/CvM p-values are not calibrated when the parametric
family is fitted on the same sample being tested. This script uses a parametric
bootstrap: for each fitted family, simulate from the fitted model, re-fit the
same family with floc=0, and recompute KS, CvM and Anderson-Darling statistics.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

EDGE_TYPES = ["C_pack", "C_brand", "C_vol", "C0_self", "C_envase"]
DIST_CATALOG = {
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "LogNormal": stats.lognorm,
    "InvGauss": stats.invgauss,
}


def parse_vol(te):
    for v in ["1/3", "1/2", "2/5"]:
        if v in str(te):
            return v
    return "UK"


def parse_pack(mp):
    mp = str(mp).upper()
    if any(k in mp for k in ["PACK 24", "BANDEJA24", "B24"]):
        return "P24"
    if any(k in mp for k in ["PACK 12", "P12"]):
        return "P12"
    if any(k in mp for k in ["PACK C. 6", "PACK 6"]):
        return "P6"
    if any(k in mp for k in ["RETRACTIL", "RETR"]):
        return "RETR"
    if "PACK" in mp:
        return "PACK"
    return "UNI"


def classify_edge(r):
    if r["Marca"] != r["prev_Marca"]:
        return "C_brand"
    if r["vol"] != r["prev_vol"]:
        return "C_vol"
    if r["pack"] != r["prev_pack"]:
        return "C_pack"
    if r["Envase"] != r["prev_Envase"]:
        return "C_envase"
    return "C0_self"


def load_h1():
    xl = pd.read_excel(BASE / "raw_data/Cambios 14_17_19_ 2025.xlsx")
    hw = pd.read_csv(BASE / "clean_data/historical_weeks.csv")
    xl = xl.merge(hw[["of", "line"]].drop_duplicates(), left_on="OF", right_on="of", how="left")
    xl = xl.dropna(subset=["line"])
    xl["line"] = xl["line"].astype(int)
    xl = xl.sort_values(["line", "Fecha Fin"]).reset_index(drop=True)
    xl["Marca"] = xl["Marca"].str.strip()
    xl["vol"] = xl["Tipo Envase"].apply(parse_vol)
    xl["pack"] = xl["Material Precio"].apply(parse_pack)
    xl["node"] = xl["Marca"] + "|" + xl["vol"] + "|" + xl["pack"] + "|" + xl["Envase"]
    xl["prev_sku"] = xl.groupby("line")["SKU"].shift(1)
    for col in ["Marca", "vol", "pack", "Envase", "node"]:
        xl[f"prev_{col}"] = xl.groupby("line")[col].shift(1)

    ch = xl[
        xl["Frecuencia Total"].notna()
        & xl["C. PRINCIPAL"].notna()
        & xl["prev_sku"].notna()
    ].copy()
    ch = ch.rename(columns={"Frecuencia Total": "hours", "Fecha Fin": "fecha"})
    ch["fecha"] = pd.to_datetime(ch["fecha"])
    ch["chtype"] = ch.apply(classify_edge, axis=1)
    ch["half"] = np.where(ch["fecha"] < pd.Timestamp("2025-07-01"), "H1", "H2")
    return ch[ch["half"] == "H1"].copy()


def ad_statistic(sample, dist, params):
    x = np.sort(np.asarray(sample, dtype=float))
    n = len(x)
    u = np.clip(dist.cdf(x, *params), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    return float(-n - np.sum((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))) / n)


def ks_statistic(sample, dist, params):
    return float(stats.kstest(sample, dist.cdf, args=params).statistic)


def cvm_statistic(sample, dist, params):
    return float(stats.cramervonmises(sample, dist.cdf, args=params).statistic)


def fit_and_stats(sample, dist):
    params = dist.fit(sample, floc=0)
    ll = float(dist.logpdf(sample, *params).sum())
    aic = -2 * ll + 2 * 2
    return {
        "params": params,
        "AIC": aic,
        "KS_D": ks_statistic(sample, dist, params),
        "CvM_W2": cvm_statistic(sample, dist, params),
        "AD_A2": ad_statistic(sample, dist, params),
    }


def bootstrap_pvalues(sample, dist, obs, b, rng):
    n = len(sample)
    boot = np.empty((b, 3), dtype=float)
    ok = 0
    attempts = 0
    max_attempts = b * 4
    while ok < b and attempts < max_attempts:
        attempts += 1
        sim = dist.rvs(*obs["params"], size=n, random_state=rng)
        sim = np.asarray(sim, dtype=float)
        sim = sim[np.isfinite(sim) & (sim > 0)]
        if len(sim) != n:
            continue
        try:
            sim_stats = fit_and_stats(sim, dist)
        except Exception:
            continue
        vals = [sim_stats["KS_D"], sim_stats["CvM_W2"], sim_stats["AD_A2"]]
        if np.all(np.isfinite(vals)):
            boot[ok, :] = vals
            ok += 1

    if ok < b:
        boot = boot[:ok, :]

    p_ks = (np.sum(boot[:, 0] >= obs["KS_D"]) + 1) / (len(boot) + 1)
    p_cvm = (np.sum(boot[:, 1] >= obs["CvM_W2"]) + 1) / (len(boot) + 1)
    p_ad = (np.sum(boot[:, 2] >= obs["AD_A2"]) + 1) / (len(boot) + 1)
    return p_ks, p_cvm, p_ad, len(boot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    h1 = load_h1()
    rows = []
    for edge_type in EDGE_TYPES:
        sample = h1.loc[h1["chtype"] == edge_type, "hours"].to_numpy(dtype=float)
        for dist_name, dist in DIST_CATALOG.items():
            obs = fit_and_stats(sample, dist)
            p_ks, p_cvm, p_ad, b_ok = bootstrap_pvalues(sample, dist, obs, args.B, rng)
            rows.append(
                {
                    "Tipo": edge_type,
                    "Dist": dist_name,
                    "n": len(sample),
                    "AIC": obs["AIC"],
                    "KS_D": obs["KS_D"],
                    "KS_p_boot": p_ks,
                    "CvM_W2": obs["CvM_W2"],
                    "CvM_p_boot": p_cvm,
                    "AD_A2": obs["AD_A2"],
                    "AD_p_boot": p_ad,
                    "B": b_ok,
                }
            )
            print(
                f"{edge_type:9s} {dist_name:9s} "
                f"KS p={p_ks:.4f} CvM p={p_cvm:.4f} AD p={p_ad:.4f} B={b_ok}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gof_bootstrap_corrected_all.csv", index=False)

    gamma = df[df["Dist"] == "Gamma"].copy()
    gamma.to_csv(OUT / "gof_bootstrap_corrected_gamma.csv", index=False)

    selected = []
    for edge_type, gdf in df.groupby("Tipo", sort=False):
        accepted = gdf[
            (gdf["KS_p_boot"] > 0.05)
            & (gdf["CvM_p_boot"] > 0.05)
            & (gdf["AD_p_boot"] > 0.05)
        ]
        pool = accepted if len(accepted) else gdf
        best = pool.sort_values("AIC").iloc[0].copy()
        best["selection_rule"] = "accepted_min_aic" if len(accepted) else "fallback_min_aic"
        selected.append(best)
    pd.DataFrame(selected).to_csv(OUT / "gof_bootstrap_corrected_selected.csv", index=False)


if __name__ == "__main__":
    main()
