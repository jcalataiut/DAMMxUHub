from pathlib import Path
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import cramervonmises_2samp as cvm2


BASE = Path(__file__).resolve().parents[2]
FIG = BASE / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

EDGE_TYPE_ORDER = ["C_pack", "C_brand", "C_vol", "C0_self", "C_envase"]
GROUP_ORDER = ["C_pack", "C_brand", "C0_self", "C_envase"]
LINE_ORDER = [17, 19, 14]

PALETTE = {
    "C_pack": "#2F5D7C",
    "C_brand": "#D98E04",
    "C_vol": "#6E6E8F",
    "C0_self": "#4C8C6A",
    "C_envase": "#B75D69",
    "L17": "#2F5D7C",
    "L19": "#4C8C6A",
    "L14": "#B75D69",
    "gray": "#7A7A7A",
    "light_gray": "#D9D9D9",
    "dark": "#252525",
}

DIST_CATALOG = {
    "Gamma": (stats.gamma, 2),
    "Weibull": (stats.weibull_min, 2),
    "LogNormal": (stats.lognorm, 2),
    "InvGauss": (stats.invgauss, 2),
}


def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 240,
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


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


def load_changeovers():
    xl = pd.read_excel(BASE / "raw_data/Cambios 14_17_19_ 2025.xlsx")
    hw = pd.read_csv(BASE / "clean_data/historical_weeks.csv")
    xl = xl.merge(hw[["of", "line"]].drop_duplicates(), left_on="OF", right_on="of", how="left")
    xl = xl.dropna(subset=["line"])
    xl["line"] = xl["line"].astype(int)
    xl_sorted = xl.sort_values(["line", "Fecha Fin"]).reset_index(drop=True)
    xl_sorted["Marca"] = xl_sorted["Marca"].str.strip()
    xl_sorted["vol"] = xl_sorted["Tipo Envase"].apply(parse_vol)
    xl_sorted["pack"] = xl_sorted["Material Precio"].apply(parse_pack)
    xl_sorted["node"] = (
        xl_sorted["Marca"] + "|" + xl_sorted["vol"] + "|" + xl_sorted["pack"] + "|" + xl_sorted["Envase"]
    )
    xl_sorted["prev_sku"] = xl_sorted.groupby("line")["SKU"].shift(1)
    for col in ["Marca", "vol", "pack", "Envase", "node"]:
        xl_sorted[f"prev_{col}"] = xl_sorted.groupby("line")[col].shift(1)

    ch = xl_sorted[
        xl_sorted["Frecuencia Total"].notna()
        & xl_sorted["C. PRINCIPAL"].notna()
        & xl_sorted["prev_sku"].notna()
    ].copy()
    ch = ch.rename(
        columns={
            "Frecuencia Total": "hours",
            "C. PRINCIPAL": "ctype_label",
            "Fecha Fin": "fecha",
            "SKU": "next_sku",
            "Nº de Cambios": "n_cambios",
        }
    )
    ch["fecha"] = pd.to_datetime(ch["fecha"])
    ch["edge"] = ch["line"].astype(str) + ":" + ch["prev_node"] + "->" + ch["node"]
    ch["chtype"] = ch.apply(classify_edge, axis=1)
    return ch


def fit_gof(h1):
    gof_rows = []
    type_priors = {}
    gamma_priors = {}
    for g in EDGE_TYPE_ORDER:
        s = h1[h1["chtype"] == g]["hours"].values
        rows = []
        for dname, (dist, k_params) in DIST_CATALOG.items():
            params = dist.fit(s, floc=0)
            ll = dist.logpdf(s, *params).sum()
            aic = -2 * ll + 2 * k_params
            ks_d, ks_p = stats.kstest(s, dist.cdf, args=params)
            cvm_r = stats.cramervonmises(s, dist.cdf, args=params)
            row = {
                "Tipo": g,
                "Dist": dname,
                "n": len(s),
                "AIC": aic,
                "KS_D": ks_d,
                "KS_p": ks_p,
                "CvM_W2": cvm_r.statistic,
                "CvM_p": cvm_r.pvalue,
                "params": params,
            }
            rows.append(row)
            gof_rows.append(row)
        best = sorted(rows, key=lambda r: r["AIC"])[0]
        type_priors[g] = {
            "dist_name": best["Dist"],
            "dist": DIST_CATALOG[best["Dist"]][0],
            "params": best["params"],
            "mean": DIST_CATALOG[best["Dist"]][0].mean(*best["params"]),
            "n_h1": len(s),
        }
        gamma_priors[g] = stats.gamma.fit(s, floc=0)

    s = h1["hours"].values
    rows = []
    for dname, (dist, k_params) in DIST_CATALOG.items():
        params = dist.fit(s, floc=0)
        ll = dist.logpdf(s, *params).sum()
        rows.append(((-2 * ll + 2 * k_params), dname, dist, params))
    _, dname, dist, params = sorted(rows, key=lambda x: x[0])[0]
    type_priors["GLOBAL"] = {
        "dist_name": dname,
        "dist": dist,
        "params": params,
        "mean": dist.mean(*params),
        "n_h1": len(s),
    }
    gamma_priors["GLOBAL"] = stats.gamma.fit(s, floc=0)
    return gof_rows, type_priors, gamma_priors


def sequential_bayes(h1_obs, h2_obs, prior="informative"):
    h1_obs = np.asarray(h1_obs, dtype=float)
    h2_obs = np.asarray(h2_obs, dtype=float)
    if prior == "informative":
        a0, b0 = len(h1_obs), h1_obs.sum()
    else:
        a0, b0 = 1.0, 0.0

    a_seq, b_seq = [a0], [b0]
    for x in h2_obs:
        a_seq.append(a_seq[-1] + 1)
        b_seq.append(b_seq[-1] + x)
    a_arr = np.array(a_seq, dtype=float)
    b_arr = np.array(b_seq, dtype=float)
    mean_seq = np.where(a_arr > 1, b_arr / (a_arr - 1), np.nan)
    lo, hi = [], []
    for a, b in zip(a_arr, b_arr):
        if a <= 1 or b <= 0:
            lo.append(np.nan)
            hi.append(np.nan)
            continue
        lam_hi = stats.gamma.ppf(0.975, a, scale=1 / b)
        lam_lo = stats.gamma.ppf(0.025, a, scale=1 / b)
        lo.append(1 / lam_hi)
        hi.append(1 / lam_lo)
    return a_arr, b_arr, mean_seq, np.array(lo), np.array(hi)


def expected_duration_pdf(mu_grid, a, b):
    mu_grid = np.asarray(mu_grid, dtype=float)
    if a <= 0 or b <= 0:
        return np.full_like(mu_grid, np.nan, dtype=float)
    lam = 1 / mu_grid
    return stats.gamma.pdf(lam, a, scale=1 / b) / (mu_grid**2)


def rank_means(groups):
    ranks = stats.rankdata(np.concatenate(groups))
    means, idx = [], 0
    for g in groups:
        means.append(ranks[idx : idx + len(g)].mean())
        idx += len(g)
    return np.array(means)


def ordered_rank_statistic(groups):
    r = rank_means(groups)
    return sum(r[j] - r[i] for i in range(len(r)) for j in range(i + 1, len(r)))


def pair_order_statistic(groups):
    r = rank_means(groups)
    return r[1] - r[0]


def permutation_test(groups, stat_fn, n_perm=10_000, seed=42):
    rng = np.random.default_rng(seed)
    sizes = [len(g) for g in groups]
    pooled = np.concatenate(groups)
    obs = stat_fn(groups)
    null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(pooled)
        pg, idx = [], 0
        for n_i in sizes:
            pg.append(p[idx : idx + n_i])
            idx += n_i
        null[b] = stat_fn(pg)
    p_val = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, null, p_val


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def centered_axes(n=5, figsize=(16, 8.4)):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 6, hspace=0.38, wspace=0.55)
    positions = [(0, 0, 2), (0, 2, 4), (0, 4, 6), (1, 1, 3), (1, 3, 5)]
    axes = [fig.add_subplot(gs[row, c0:c1]) for row, c0, c1 in positions[:n]]
    return fig, axes


def fig01(ch):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    axes[0].hist(ch["hours"], bins=50, color=PALETTE["C_brand"], edgecolor="white", linewidth=0.4)
    axes[0].axvline(ch["hours"].mean(), color=PALETTE["C_envase"], ls="--", lw=1.6, label=f"Mean={ch['hours'].mean():.2f} h")
    axes[0].axvline(ch["hours"].median(), color=PALETTE["C_pack"], ls="--", lw=1.6, label=f"Median={ch['hours'].median():.2f} h")
    axes[0].set_xlabel("Changeover duration (h)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    groups = [ch[ch["chtype"] == g]["hours"].values for g in GROUP_ORDER]
    bp = axes[1].boxplot(groups, patch_artist=True, medianprops={"color": PALETTE["dark"], "lw": 1.7})
    for patch, g in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(PALETTE[g])
        patch.set_alpha(0.82)
    axes[1].set_xticks(range(1, 5))
    axes[1].set_xticklabels(GROUP_ORDER, rotation=20, ha="right")
    axes[1].set_ylabel("Changeover duration (h)")
    for i, arr in enumerate(groups):
        axes[1].text(i + 1, arr.mean() + 0.1, f"{arr.mean():.2f} h", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "nb07_fig01.png")


def fig02(h1, gof_rows):
    n = len(EDGE_TYPE_ORDER)
    fig, axes = plt.subplots(2, n, figsize=(3.7 * n, 6.7))
    dist_colors = {
        "Gamma": PALETTE["C_pack"],
        "Weibull": PALETTE["C_brand"],
        "LogNormal": PALETTE["C0_self"],
        "InvGauss": PALETTE["C_envase"],
    }
    for col, g in enumerate(EDGE_TYPE_ORDER):
        s = h1[h1["chtype"] == g]["hours"].values
        x = np.linspace(0.01, np.percentile(s, 97), 300)
        axes[0, col].hist(s, bins=25, density=True, alpha=0.38, color=PALETTE[g], edgecolor="white", linewidth=0.3)
        for dname, (dist, _) in DIST_CATALOG.items():
            r = next(r for r in gof_rows if r["Tipo"] == g and r["Dist"] == dname)
            axes[0, col].plot(x, dist.pdf(x, *r["params"]), lw=1.8, color=dist_colors[dname], label=dname)
        axes[0, col].set_xlabel("Hours")
        axes[0, col].set_ylabel(f"{g}\nDensity")
        axes[0, col].legend(fontsize=6)

        r_g = next(r for r in gof_rows if r["Tipo"] == g and r["Dist"] == "Gamma")
        probs = (np.arange(1, len(s) + 1) - 0.5) / len(s)
        emp_q = np.sort(s)
        th_q = stats.gamma.ppf(probs, *r_g["params"])
        lim = max(emp_q.max(), th_q.max()) * 1.05
        axes[1, col].scatter(th_q, emp_q, s=8, alpha=0.55, color=PALETTE[g])
        axes[1, col].plot([0, lim], [0, lim], color=PALETTE["dark"], ls="--", lw=1.2)
        axes[1, col].set_xlabel("Gamma theoretical quantiles")
        axes[1, col].set_ylabel(f"{g}\nEmpirical quantiles")
    fig.tight_layout()
    save(fig, "nb07_fig02.png")


def fig03(h1, h2):
    plot_types = [g for g in EDGE_TYPE_ORDER if len(h1[h1["chtype"] == g]) >= 2 and len(h2[h2["chtype"] == g]) >= 2]
    fig, axes = centered_axes(len(plot_types))
    for ax, g in zip(axes, plot_types):
        color = PALETTE[g]
        s_h1 = h1[h1["chtype"] == g]["hours"].values
        s_h2 = h2[h2["chtype"] == g].sort_values("fecha")["hours"].values
        _, _, mean_i, lo_i, hi_i = sequential_bayes(s_h1, s_h2, "informative")
        _, _, mean_u, lo_u, hi_u = sequential_bayes(s_h1, s_h2, "uniform")
        k = np.arange(len(mean_i))
        ax.fill_between(k, lo_i, hi_i, alpha=0.18, color=color, label="95% CI informative")
        ax.fill_between(k, lo_u, hi_u, alpha=0.10, color=PALETTE["gray"], label="95% CI weak")
        ax.plot(k, mean_i, "-", color=color, lw=2.2, label="Informative")
        ax.plot(k, mean_u, "--", color=PALETTE["dark"], lw=1.8, label="Weak")
        ax.axhline(s_h1.mean(), color=color, ls=":", lw=1.2, alpha=0.8, label=f"H1 mean={s_h1.mean():.2f} h")
        ax.axhline(s_h2.mean(), color=PALETTE["dark"], ls=":", lw=1.2, alpha=0.8, label=f"H2 mean={s_h2.mean():.2f} h")
        diff = mean_i[-1] - mean_u[-1]
        ax.text(k[-1], max(mean_i[-1], mean_u[-1]), f"  Δ={diff:+.2f} h", va="bottom", fontsize=8)
        ax.set_xlabel("New H2 observations")
        ax.set_ylabel(f"{g}\nPosterior E[X] (h)")
        ax.legend()
    save(fig, "nb07_fig03.png")


def fig04(h1, h2):
    plot_types = [g for g in EDGE_TYPE_ORDER if len(h1[h1["chtype"] == g]) >= 2 and len(h2[h2["chtype"] == g]) >= 2]
    fig, axes = plt.subplots(len(plot_types), 3, figsize=(13.5, 2.8 * len(plot_types)))
    for row, g in enumerate(plot_types):
        color = PALETTE[g]
        s_h1 = h1[h1["chtype"] == g]["hours"].values
        s_h2 = h2[h2["chtype"] == g].sort_values("fecha")["hours"].values
        a_i, b_i, mean_i, _, _ = sequential_bayes(s_h1, s_h2, "informative")
        a_u, b_u, mean_u, _, _ = sequential_bayes(s_h1, s_h2, "uniform")
        k_end = len(s_h2)
        x_max = max(np.percentile(np.concatenate([s_h1, s_h2]), 98) * 2.4, np.nanmax([mean_i[0], mean_i[k_end], mean_u[k_end]]) * 2.1)
        grid = np.linspace(0.03, x_max, 700)
        axes[row, 0].plot(grid, expected_duration_pdf(grid, a_i[0], b_i[0]), color=color, lw=2.2, label="Informative prior")
        axes[row, 0].axhline(0, color=PALETTE["dark"], lw=1.4, alpha=0.55, label="Weak prior: improper")
        axes[row, 1].plot(grid, expected_duration_pdf(grid, a_i[k_end], b_i[k_end]), color=color, lw=2.2, label=f"Informative ({mean_i[k_end]:.2f} h)")
        axes[row, 1].plot(grid, expected_duration_pdf(grid, a_u[k_end], b_u[k_end]), color=PALETTE["dark"], lw=2.0, ls="--", label=f"Weak ({mean_u[k_end]:.2f} h)")
        axes[row, 1].axvline(s_h2.mean(), color=PALETTE["gray"], ls=":", lw=1.2, label=f"H2 mean={s_h2.mean():.2f} h")
        axes[row, 2].plot([0, 1], [1.0, 1.0], color=color, lw=2.0)
        axes[row, 2].scatter([0, 1], [1.0, 1.0], color=color, s=42, zorder=3)
        axes[row, 2].text(0, 1.08, f"{mean_i[0]:.2f} h", ha="center", fontsize=8)
        axes[row, 2].text(1, 1.08, f"{mean_i[k_end]:.2f} h", ha="center", fontsize=8)
        axes[row, 2].text(-0.08, 1.0, "Informative", ha="right", va="center", fontsize=8, color=color)
        axes[row, 2].plot([0, 1], [0.35, 0.35], color=PALETTE["dark"], lw=2.0, ls="--")
        axes[row, 2].scatter([1], [0.35], color=PALETTE["dark"], s=42, zorder=3)
        axes[row, 2].text(0, 0.43, "improper", ha="center", fontsize=8, color=PALETTE["gray"])
        axes[row, 2].text(1, 0.43, f"{mean_u[k_end]:.2f} h", ha="center", fontsize=8)
        axes[row, 2].text(-0.08, 0.35, "Weak", ha="right", va="center", fontsize=8)
        axes[row, 2].set_xlim(-0.35, 1.15)
        axes[row, 2].set_ylim(0, 1.35)
        axes[row, 2].set_xticks([0, 1])
        axes[row, 2].set_xticklabels(["before H2", "after H2"])
        axes[row, 2].set_yticks([])
        for col in range(3):
            axes[row, col].set_ylabel(f"{g}\nDensity" if col < 2 else g)
            axes[row, col].legend(fontsize=6, loc="upper right")
        axes[row, 0].set_xlabel("Expected duration (h)")
        axes[row, 1].set_xlabel("Expected duration (h)")
    fig.tight_layout()
    save(fig, "nb07_fig04.png")


def fig05(h1, h2, type_priors):
    fig, axes = centered_axes(len(EDGE_TYPE_ORDER))
    for ax, g in zip(axes, EDGE_TYPE_ORDER):
        color = PALETTE[g]
        s1 = h1[h1["chtype"] == g]["hours"].values
        s2 = h2[h2["chtype"] == g]["hours"].values
        prior = type_priors[g]
        dist = prior["dist"]
        params = prior["params"]
        x = np.linspace(0.01, np.percentile(np.concatenate([s1, s2]), 97), 300)
        ax.hist(s1, bins=25, density=True, alpha=0.50, color=color, edgecolor="white", linewidth=0.3, label=f"H1 (n={len(s1)})")
        ax.hist(s2, bins=25, density=True, alpha=0.30, color=PALETTE["gray"], edgecolor="white", linewidth=0.3, label=f"H2 (n={len(s2)})")
        ax.plot(x, dist.pdf(x, *params), "-", color=color, lw=2.4, label=prior["dist_name"])
        ax.set_xlabel("Hours")
        ax.set_ylabel(f"{g}\nDensity")
        ax.legend()
    save(fig, "nb07_fig05.png")


def fig06(h1, gamma_priors):
    rows = []
    global_alpha, _, global_scale = gamma_priors["GLOBAL"]
    edge_counts = h1.groupby(["edge", "line", "chtype"])["hours"].agg(["count", "mean"]).reset_index()
    for _, row in edge_counts.iterrows():
        edge = row["edge"]
        obs = h1[h1["edge"] == edge]["hours"].values
        if len(obs) >= 2:
            alpha, _, scale = stats.gamma.fit(obs, floc=0)
            source = "edge specific"
        else:
            alpha, scale, source = global_alpha, global_scale, "global fallback"
        rows.append({"edge": edge, "line": int(row["line"]), "chtype": row["chtype"], "n": len(obs), "alpha": alpha, "scale": scale, "source": source})
    priors = pd.DataFrame(rows)
    ep = priors[priors["source"] == "edge specific"].copy()
    pct_gt1 = 100 * (ep["alpha"] > 1).mean()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    axes[0].hist(ep["alpha"].clip(upper=20), bins=30, color=PALETTE["C_pack"], edgecolor="white", linewidth=0.4)
    axes[0].axvline(1.0, color=PALETTE["C_envase"], ls="--", lw=1.8, label="alpha = 1")
    axes[0].axvline(ep["alpha"].median(), color=PALETTE["C_brand"], ls="--", lw=1.4, label=f"Median alpha={ep['alpha'].median():.2f}")
    axes[0].set_xlabel("Gamma alpha (capped at 20)")
    axes[0].set_ylabel("Edges")
    axes[0].legend()
    line_colors = {14: PALETTE["L14"], 17: PALETTE["L17"], 19: PALETTE["L19"]}
    for line, grp in ep.groupby("line"):
        axes[1].scatter(grp["alpha"].clip(upper=20), grp["scale"], s=22, alpha=0.65, label=f"L{line}", color=line_colors[line])
    axes[1].axvline(1.0, color=PALETTE["C_envase"], ls="--", lw=1.4, alpha=0.8)
    axes[1].set_xlabel("Gamma alpha (capped at 20)")
    axes[1].set_ylabel("Gamma scale")
    axes[1].legend(title=f"{pct_gt1:.0f}% alpha > 1", fontsize=8)
    fig.tight_layout()
    save(fig, "nb07_fig06.png")


def fig07_08_09(ch):
    groups_type = [ch[ch["chtype"] == g]["hours"].values for g in GROUP_ORDER]
    r_type = rank_means(groups_type)
    t_obs, t_null, p_val = permutation_test(groups_type, ordered_rank_statistic, seed=43)
    alpha_a = 0.05 / 6
    pairs_a = list(combinations(range(4), 2))
    post_rows, post_nulls = [], {}
    for i, j in pairs_a:
        obs, null, pv = permutation_test([groups_type[i], groups_type[j]], pair_order_statistic, seed=100 + 10 * i + j)
        post_nulls[(i, j)] = null
        post_rows.append({"p": pv, "T": obs})
    post = pd.DataFrame(post_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    bp = axes[0].boxplot(groups_type, patch_artist=True, medianprops={"color": PALETTE["dark"], "lw": 1.7})
    for patch, g in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(PALETTE[g])
        patch.set_alpha(0.84)
    axes[0].set_xticks(range(1, 5))
    axes[0].set_xticklabels(GROUP_ORDER, rotation=20, ha="right")
    axes[0].set_ylabel("Changeover duration (h)")
    for i, arr in enumerate(groups_type):
        axes[0].text(i + 1, arr.mean() + 0.1, f"{arr.mean():.2f} h", ha="center", fontsize=8)
    axes[1].bar(range(1, 5), r_type, color=[PALETTE[g] for g in GROUP_ORDER], edgecolor="white", linewidth=0.8)
    axes[1].set_xticks(range(1, 5))
    axes[1].set_xticklabels(GROUP_ORDER, rotation=20, ha="right")
    axes[1].set_ylabel("Mean rank")
    for i, r in enumerate(r_type):
        axes[1].text(i + 1, r + 4, f"{r:.0f}", ha="center", fontsize=8)
    axes[2].hist(t_null, bins=60, density=True, color=PALETTE["light_gray"], edgecolor="white", linewidth=0.4, label="Permuted labels")
    axes[2].axvline(t_obs, color=PALETTE["C_envase"], lw=2.2, label=f"Observed T={t_obs:.1f}\np={p_val:.4f}")
    axes[2].axvline(np.percentile(t_null, 95), color=PALETTE["C_brand"], lw=1.5, ls="--", label="95th percentile")
    axes[2].set_xlabel("Ordered statistic T")
    axes[2].set_ylabel("Density")
    axes[2].legend()
    fig.tight_layout()
    save(fig, "nb07_fig07.png")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0), gridspec_kw={"width_ratios": [1.0, 1.15]})
    p_mat = np.full((4, 4), np.nan)
    for idx, _ in post.iterrows():
        i, j = pairs_a[idx]
        p_mat[i, j] = post.loc[idx, "p"]
    cmap = sns.light_palette(PALETTE["C_pack"], as_cmap=True, reverse=True)
    im = axes[0].imshow(np.nan_to_num(p_mat, nan=1.0), vmin=0, vmax=0.10, cmap=cmap, aspect="equal")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Permutation p-value")
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(4))
    axes[0].set_xticklabels(GROUP_ORDER, rotation=25, ha="right")
    axes[0].set_yticklabels(GROUP_ORDER)
    for i in range(4):
        for j in range(4):
            if i >= j:
                txt = "-"
            else:
                val = p_mat[i, j]
                sig = "***" if val < alpha_a else ("*" if val < 0.05 else "")
                txt = f"{val:.4f}\n{sig}"
            axes[0].text(j, i, txt, ha="center", va="center", fontsize=8, color=PALETTE["dark"])
    for (i, j), color in zip([(0, 1), (1, 2), (2, 3)], [PALETTE["C_pack"], PALETTE["C_brand"], PALETTE["C0_self"]]):
        null = post_nulls[(i, j)]
        row = post.iloc[pairs_a.index((i, j))]
        axes[1].hist(null, bins=45, density=True, histtype="step", lw=2.0, color=color, label=f"{GROUP_ORDER[i]} <= {GROUP_ORDER[j]}")
        axes[1].axvline(row["T"], color=color, lw=2.0, ls="--")
    axes[1].axvline(0, color=PALETTE["dark"], lw=1, alpha=0.55)
    axes[1].set_xlabel("Pairwise statistic T_ij = R_j - R_i")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "nb07_fig08.png")

    groups_line = [ch[ch["line"] == line]["hours"].values for line in LINE_ORDER]
    r_line = rank_means(groups_line)
    t_line, null_line, p_line = permutation_test(groups_line, ordered_rank_statistic, seed=44)
    alpha_b = 0.05 / 3
    pairs_b = list(combinations(range(3), 2))
    line_post = []
    for i, j in pairs_b:
        obs, _, pv = permutation_test([groups_line[i], groups_line[j]], pair_order_statistic, seed=300 + 10 * i + j)
        line_post.append({"p": pv, "T": obs})
    line_post = pd.DataFrame(line_post)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw={"width_ratios": [1.0, 1.2, 1.0]})
    bp = axes[0].boxplot(groups_line, patch_artist=True, medianprops={"color": PALETTE["dark"], "lw": 1.7})
    for patch, line in zip(bp["boxes"], LINE_ORDER):
        patch.set_facecolor(PALETTE[f"L{line}"])
        patch.set_alpha(0.84)
    axes[0].set_xticks(range(1, 4))
    axes[0].set_xticklabels([f"L{line}" for line in LINE_ORDER])
    axes[0].set_ylabel("Changeover duration (h)")
    for i, arr in enumerate(groups_line):
        axes[0].text(i + 1, arr.mean() + 0.08, f"{arr.mean():.2f} h", ha="center", fontsize=8)
    axes[1].hist(null_line, bins=60, density=True, color=PALETTE["light_gray"], edgecolor="white", linewidth=0.4, label="Permuted labels")
    axes[1].axvline(t_line, color=PALETTE["C_envase"], lw=2.2, label=f"Observed T={t_line:.1f}\np={p_line:.4f}")
    axes[1].axvline(np.percentile(null_line, 95), color=PALETTE["C_brand"], lw=1.5, ls="--", label="95th percentile")
    axes[1].set_xlabel("Ordered statistic T")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    p_mat = np.full((3, 3), np.nan)
    for idx, _ in line_post.iterrows():
        i, j = pairs_b[idx]
        p_mat[i, j] = line_post.loc[idx, "p"]
    cmap = sns.light_palette(PALETTE["C_pack"], as_cmap=True, reverse=True)
    im = axes[2].imshow(np.nan_to_num(p_mat, nan=1.0), vmin=0, vmax=0.10, cmap=cmap, aspect="equal")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Permutation p-value")
    axes[2].set_xticks(range(3))
    axes[2].set_yticks(range(3))
    axes[2].set_xticklabels([f"L{line}" for line in LINE_ORDER])
    axes[2].set_yticklabels([f"L{line}" for line in LINE_ORDER])
    for i in range(3):
        for j in range(3):
            if i >= j:
                txt = "-"
            else:
                val = p_mat[i, j]
                sig = "***" if val < alpha_b else ("*" if val < 0.05 else "")
                txt = f"{val:.4f}\n{sig}"
            axes[2].text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.tight_layout()
    save(fig, "nb07_fig09.png")


def main():
    apply_style()
    ch = load_changeovers()
    h1 = ch[ch["fecha"] < pd.Timestamp("2025-07-01")].copy()
    h2 = ch[ch["fecha"] >= pd.Timestamp("2025-07-01")].copy()
    gof_rows, type_priors, gamma_priors = fit_gof(h1)
    fig01(ch)
    fig02(h1, gof_rows)
    fig03(h1, h2)
    fig04(h1, h2)
    fig05(h1, h2, type_priors)
    fig06(h1, gamma_priors)
    fig07_08_09(ch)
    print(f"Regenerated figures in {FIG}")


if __name__ == "__main__":
    main()
