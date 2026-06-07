"""
Appendix A.1 -- Data preparation, graph construction and Goodness-of-Fit.
Source: notebooks/07_statistical_analysis.ipynb (Sections 1 and 2.1-2.2).

Software stack: pandas (data handling), numpy (numerics),
scipy.stats (parametric families + GoF tests).
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

np.random.seed(42)
BASE = Path.cwd() if (Path.cwd() / 'raw_data').exists() else Path.cwd().parent

# --- Load raw changeover log and attach the production line ------------------
xl = pd.read_excel(BASE / 'raw_data/Cambios 14_17_19_ 2025.xlsx')
hw = pd.read_csv(BASE / 'clean_data/historical_weeks.csv')
xl = xl.merge(hw[['of', 'line']].drop_duplicates(),
              left_on='OF', right_on='of', how='left').dropna(subset=['line'])
xl['line'] = xl['line'].astype(int)

# --- Build the graph node = Marca x vol x pack x Envase ----------------------
def parse_vol(te):
    for v in ['1/3', '1/2', '2/5']:
        if v in str(te):
            return v
    return 'UK'

def parse_pack(mp):
    mp = str(mp).upper()
    if any(k in mp for k in ['PACK 24', 'BANDEJA24', 'B24']): return 'P24'
    if any(k in mp for k in ['PACK 12', 'P12']):              return 'P12'
    if any(k in mp for k in ['PACK C. 6', 'PACK 6']):         return 'P6'
    if any(k in mp for k in ['RETRACTIL', 'RETR']):           return 'RETR'
    if 'PACK' in mp:                                          return 'PACK'
    return 'UNI'

xl = xl.sort_values(['line', 'Fecha Fin']).reset_index(drop=True)
xl['Marca'] = xl['Marca'].str.strip()
xl['vol']   = xl['Tipo Envase'].apply(parse_vol)
xl['pack']  = xl['Material Precio'].apply(parse_pack)
xl['node']  = xl['Marca'] + '|' + xl['vol'] + '|' + xl['pack'] + '|' + xl['Envase']
for col in ['Marca', 'vol', 'pack', 'Envase', 'node']:
    xl[f'prev_{col}'] = xl.groupby('line')[col].shift(1)

ch = xl[xl['Frecuencia Total'].notna() & xl['prev_node'].notna()].copy()
ch = ch.rename(columns={'Frecuencia Total': 'hours', 'Fecha Fin': 'fecha'})
ch['fecha'] = pd.to_datetime(ch['fecha'])

# --- Classify the edge type by the attribute that changes --------------------
def classify_edge(r):
    if r['Marca']  != r['prev_Marca']:  return 'C_brand'
    if r['vol']    != r['prev_vol']:    return 'C_vol'
    if r['pack']   != r['prev_pack']:   return 'C_pack'
    if r['Envase'] != r['prev_Envase']: return 'C_envase'
    return 'C0_self'

ch['chtype'] = ch.apply(classify_edge, axis=1)

# --- Temporal split: H1 = Jan-Jun (prior), H2 = Jul-Dec (validation) ---------
SPLIT = pd.Timestamp('2025-07-01')
ch['half'] = np.where(ch['fecha'] < SPLIT, 'H1', 'H2')
h1, h2 = ch[ch['half'] == 'H1'], ch[ch['half'] == 'H2']

# --- Goodness of Fit: fit 4 families by MLE, compare by AIC + KS + CvM --------
# stats.{gamma, weibull_min, lognorm, invgauss}.fit(s, floc=0)  -> MLE
# stats.kstest(s, dist.cdf, args=params)                        -> KS test
# stats.cramervonmises(s, dist.cdf, args=params)                -> CvM test
DIST_CATALOG = {'Gamma': stats.gamma, 'Weibull': stats.weibull_min,
                'LogNormal': stats.lognorm, 'InvGauss': stats.invgauss}
EDGE_TYPES = ['C_pack', 'C_brand', 'C_vol', 'C0_self', 'C_envase']

def gof_table(sample):
    rows = []
    for name, dist in DIST_CATALOG.items():
        params = dist.fit(sample, floc=0)              # MLE, location fixed at 0
        ll  = dist.logpdf(sample, *params).sum()
        aic = -2 * ll + 2 * 2                           # k = 2 free parameters
        ks_d, ks_p = stats.kstest(sample, dist.cdf, args=params)
        cvm = stats.cramervonmises(sample, dist.cdf, args=params)
        rows.append({'Dist': name, 'AIC': aic, 'KS_D': ks_d, 'KS_p': ks_p,
                     'CvM_W2': cvm.statistic, 'CvM_p': cvm.pvalue, 'params': params})
    return sorted(rows, key=lambda r: r['AIC'])          # best = lowest AIC

best_family = {}
for g in EDGE_TYPES:
    s = h1[h1['chtype'] == g]['hours'].values
    if len(s) >= 10:
        best_family[g] = gof_table(s)[0]                 # winner per edge type
