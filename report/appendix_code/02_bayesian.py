"""
Appendix A.2 -- Bayesian conjugate updating and stationarity tests.
Source: notebooks/07_statistical_analysis.ipynb (Sections 2.3-2.6).

Gamma-Exponential conjugate model:
    X_i | lambda ~ Exp(lambda),   lambda ~ Gamma(a0, rate=b0)
    posterior after n obs:  lambda | x ~ Gamma(a0 + n, b0 + sum x_i)
    predictive duration:    E[X] = E[1/lambda] = b / (a - 1)   (a > 1)
Quantiles of lambda -> 1/quantile give the 95% interval for E[X]
(implemented with scipy.stats.gamma.ppf).
"""
import numpy as np
from scipy import stats


def sequential_bayes(h1_obs, h2_obs, prior='informative'):
    """Sequential Gamma-Exponential update; returns posterior mean path + 95% CI.

    informative : a0 = n_H1, b0 = sum_H1 x   (H1 enters as a pseudo-sample)
    uniform     : a0 = 1,    b0 = 0          (Jeffreys-type, learns only from H2)
    """
    h1_obs = np.asarray(h1_obs, float)
    h2_obs = np.asarray(h2_obs, float)
    if prior == 'informative':
        a0, b0 = len(h1_obs), h1_obs.sum()
    elif prior == 'uniform':
        a0, b0 = 1.0, 0.0
    else:
        raise ValueError("prior must be 'informative' or 'uniform'")

    a_seq, b_seq = [a0], [b0]
    for x in h2_obs:                      # one observation at a time
        a_seq.append(a_seq[-1] + 1)
        b_seq.append(b_seq[-1] + x)
    a_arr, b_arr = np.array(a_seq, float), np.array(b_seq, float)
    mean_seq = np.where(a_arr > 1, b_arr / (a_arr - 1), np.nan)

    ci_lo, ci_hi = [], []
    for a, b in zip(a_arr, b_arr):
        if a <= 1 or b <= 0:
            ci_lo.append(np.nan); ci_hi.append(np.nan); continue
        lam_hi = stats.gamma.ppf(0.975, a, scale=1 / b)   # high lambda -> low E[X]
        lam_lo = stats.gamma.ppf(0.025, a, scale=1 / b)
        ci_lo.append(1 / lam_hi); ci_hi.append(1 / lam_lo)
    return a_arr, b_arr, mean_seq, np.array(ci_lo), np.array(ci_hi)


def stationarity_tests(s1, s2):
    """Two-sample H1-vs-H2 distribution-equality tests.

    KS  : stats.ks_2samp            (supremum of |F1 - F2|)
    AD  : stats.anderson_ksamp      (Anderson-Darling, tail-weighted)
    CvM : stats.cramervonmises_2samp(Cramer-von Mises, uniform weight)
    p > 0.05  ->  do not reject H0: F_H1 = F_H2  (process is stationary).
    """
    ks  = stats.ks_2samp(s1, s2)
    ad  = stats.anderson_ksamp([s1, s2])
    cvm = stats.cramervonmises_2samp(s1, s2)
    return {
        'KS_D': ks.statistic,  'KS_p': ks.pvalue,
        'AD_stat': ad.statistic, 'AD_sig_level': ad.significance_level,
        'CvM_stat': cvm.statistic, 'CvM_p': cvm.pvalue,
    }


def empirical_bayes_prior(ch, edge_col='edge', min_n=2):
    """Per-edge Gamma prior when n >= min_n; otherwise fall back to the
    global Gamma prior used as a hierarchical regulariser."""
    a_g, _, sc_g = stats.gamma.fit(ch['hours'].values, floc=0)   # global prior
    priors = {}
    for edge, grp in ch.groupby(edge_col):
        s = grp['hours'].values
        if len(s) >= min_n:
            a, _, sc = stats.gamma.fit(s, floc=0)
            priors[edge] = (a, sc, 'specific')
        else:
            priors[edge] = (a_g, sc_g, 'global')
    return priors, (a_g, sc_g)
