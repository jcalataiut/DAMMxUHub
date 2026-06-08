"""
Appendix A.3 -- Permutation tests for ordered alternatives.
Source: notebooks/07_statistical_analysis.ipynb (Section 3).

Only numpy + scipy.stats.rankdata are needed: the null distribution is built
by shuffling the group labels (10,000 permutations), so no parametric
assumption on the changeover-duration distribution is required.
"""
import numpy as np
from scipy import stats
from itertools import combinations


def rank_means(groups):
    """Mean of the joint ranks within each group."""
    ranks = stats.rankdata(np.concatenate(groups))
    R, idx = [], 0
    for g in groups:
        R.append(ranks[idx:idx + len(g)].mean()); idx += len(g)
    return np.array(R)


def ordered_rank_statistic(groups):
    """T_k = sum_{i<j} (R_j - R_i).

    Large values support the ordered alternative. For k=4 this equals
    3R4 + R3 - R2 - 3R1; for k=3 this equals 2R3 - 2R1.
    """
    R = rank_means(groups)
    return sum(R[j] - R[i] for i in range(len(R)) for j in range(i + 1, len(R)))


def pair_order_statistic(groups):
    """Two-group ordered statistic T_ij = R_j - R_i (large -> group i < group j)."""
    R1, R2 = rank_means(groups)
    return R2 - R1


def permutation_test(groups, stat_fn, n_perm=10_000, seed=42, alternative='greater'):
    """Label-permutation test: returns (observed stat, null sample, p-value)."""
    rng = np.random.default_rng(seed)
    sizes = [len(g) for g in groups]
    pooled = np.concatenate(groups)
    T_obs = stat_fn(groups)
    T_null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(pooled)
        pg, idx = [], 0
        for n_i in sizes:
            pg.append(p[idx:idx + n_i]); idx += n_i
        T_null[b] = stat_fn(pg)
    if alternative == 'greater':
        p_val = (np.sum(T_null >= T_obs) + 1) / (n_perm + 1)
    else:  # two-sided around the null mean
        c = T_null.mean()
        p_val = (np.sum(np.abs(T_null - c) >= abs(T_obs - c)) + 1) / (n_perm + 1)
    return T_obs, T_null, p_val


# ---- Application A (edge types, k=4) and B (lines, k=3) ----------------------
# groups_type = [hours[chtype==g] for g in ['C_pack','C_brand','C0_self','C_envase']]
# T_obs, _, p_ord  = permutation_test(groups_type, ordered_rank_statistic, seed=43)
#
# Bonferroni-corrected ordered post-hoc over the C(4,2)=6 pairs:
# for (i, j) in combinations(range(4), 2):
#     permutation_test([groups_type[i], groups_type[j]], pair_order_statistic)
#
# groups_line = [hours[line==l] for l in [17, 19, 14]]
# T_line, _, p_line = permutation_test(groups_line, ordered_rank_statistic, seed=44)
# post-hoc lines: same pair_order_statistic over the C(3,2)=3 ordered pairs.
