"""Reproduce the published 6-benchmark regression (N=260, gemini-3-flash-preview excluded).

Produces the regression values reported in the article (R2_train = 0.463,
R2_CV = 0.373, Table 2, Supplementary Tables S8-S10 and Extended Data Table 1). It
builds the 260 configurations from ORIGINAL_DATA plus the per-instance
SWE-bench Verified / Terminal-Bench results, excluding the ten
gemini-3-flash-preview rows that Supplementary Table S11 identifies as out of
sample, and prints the fit, 5-fold CV, Holm-Bonferroni, cluster-robust and
LODO results together with the ACI variant.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extended_mixed_effects_6benchmarks import (
    ORIGINAL_DATA, COORDINATION_METRICS, INTELLIGENCE_SCORES,
    TOOL_COUNTS, AGENT_COUNTS, FORMULA, FEATURE_COLS,
    build_dataframe, standardize, normalize_model_name
)
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Load CSV, exclude gemini-3-flash-preview
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'per_instance_results_swe_tb.csv')
df_csv = pd.read_csv(csv_path)
df_csv = df_csv[df_csv['model'] != 'gemini-3-flash-preview']
grp = df_csv.groupby(['dataset', 'agent_type', 'provider', 'model']).agg(
    resolution_rate=('resolved', 'mean'),
).reset_index()
new_rows = [[r['dataset'], r['agent_type'], r['provider'], r['model'], r['resolution_rate']] for _, r in grp.iterrows()]
print(f"New rows (excl gemini-3-flash-preview): {len(new_rows)}")

all_data = ORIGINAL_DATA + new_rows
print(f"Total rows: {len(all_data)}")

df, intel_mean = build_dataframe(all_data)
print(f"N = {len(df)}, Intel centering offset = {intel_mean:.4f}")

df_scaled, scaler = standardize(df, FEATURE_COLS)
model = smf.ols(FORMULA, data=df_scaled).fit()

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = []
for train_idx, test_idx in kf.split(df_scaled):
    cv_m = smf.ols(FORMULA, data=df_scaled.iloc[train_idx]).fit()
    yp = cv_m.predict(df_scaled.iloc[test_idx])
    yt = df_scaled.iloc[test_idx]['performance'].values
    ss_r = np.sum((yt - yp)**2); ss_t = np.sum((yt - yt.mean())**2)
    cv_r2.append(1 - ss_r/ss_t if ss_t > 0 else np.nan)

# How the configuration-level folds above treat the 52 (benchmark, model) choice
# sets: each set's five rows are scattered across folds, so no set is ever held out
# as a unit. This is why this cross-validation estimates configuration-level error
# and cannot evaluate a choice among the five architectures of an unseen set.
_key = (df['dataset'].astype(str) + '|' + df['model_short'].astype(str)).values
_fold = np.empty(len(df), int)
for _k, (_, _te) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(df_scaled)):
    _fold[_te] = _k
_spread = pd.Series(_fold).groupby(_key).nunique()
print(f"Choice sets: {len(_spread)}; test folds per set range {_spread.min()} to {_spread.max()}; "
      f"sets held out intact: {int((_spread == 1).sum())}")
assert (len(_spread), _spread.min(), _spread.max(), int((_spread == 1).sum())) == (52, 2, 5, 0), _spread.describe()

print(f"\n{'='*80}")
print(f"6-BENCHMARK MODEL (N={len(df)}, EXCLUDING gemini-3-flash-preview)")
print(f"{'='*80}")
print(f"R2_train = {model.rsquared:.4f}")
print(f"R2_adj   = {model.rsquared_adj:.4f}")
print(f"R2_CV    = {np.mean(cv_r2):.4f} (+/- {np.std(cv_r2):.4f})")
print(f"AIC      = {model.aic:.2f}")

print(f"\n--- ALL COEFFICIENTS ---")
for var in model.params.index:
    b = model.params[var]; p = model.pvalues[var]; ci = model.conf_int().loc[var]
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '+' if p<0.1 else ''
    print(f"  {var:<30s}  beta={b:>8.4f}  CI=[{ci[0]:>7.4f}, {ci[1]:>7.4f}]  p={p:.4f} {sig}")

# Holm-Bonferroni
pvals_s = model.pvalues.drop('Intercept')
preds = list(pvals_s.index); pvals = pvals_s.values
rej_holm, p_holm, _, _ = multipletests(pvals, method='holm')
print(f"\n--- HOLM-BONFERRONI ({len(pvals)} hypotheses) ---")
for pred, rp, hp, rej in sorted(zip(preds, pvals, p_holm, rej_holm), key=lambda x: x[1]):
    print(f"  {pred:<30s}  raw_p={rp:.4f}  holm_p={hp:.4f}  {'SURVIVES' if rej else ''}")

# Cluster-robust
robust = model.get_robustcov_results(cov_type='cluster', groups=df_scaled['dataset'])
print(f"\n--- CLUSTER-ROBUST (G={df_scaled['dataset'].nunique()}) ---")
# robust.params / robust.pvalues are plain arrays in recent statsmodels; take names from the base fit
robust_p = np.asarray(robust.pvalues)
for i, var in enumerate(model.params.index):
    if var == 'Intercept': continue
    pn = model.pvalues[var]; pr = float(robust_p[i])
    st = 'SURVIVES' if pr<0.05 else ('LOST' if pn<0.05 else '')
    print(f"  {var:<30s}  naive_p={pn:.4f}  robust_p={pr:.4f}  {st}")

# LODO
print(f"\n--- LODO CV ---")
datasets = df['dataset'].unique(); lodo_r2s = []
for held_out in datasets:
    tr = df[df['dataset']!=held_out].copy(); te = df[df['dataset']==held_out].copy()
    sc = StandardScaler()
    tr[FEATURE_COLS] = sc.fit_transform(tr[FEATURE_COLS])
    te[FEATURE_COLS] = sc.transform(te[FEATURE_COLS])
    try:
        m = smf.ols(FORMULA, data=tr).fit()
        yp = m.predict(te); yt = te['performance'].values
        r2 = 1 - np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2)
    except: r2 = np.nan
    lodo_r2s.append(r2)
    print(f"  {held_out:<35s}  R2={r2:.4f}")
print(f"  Mean LODO R2 = {np.mean(lodo_r2s):.4f}")

# ACI model
print(f"\n{'='*80}")
print(f"ACI MODEL (N={len(df)})")
print(f"{'='*80}")
# Compute ACI = mean SA performance across all 6 benchmarks
sa = df[df['architecture']=='single-agent'][['dataset','model_short','performance']].copy()
aci = sa.groupby('model_short')['performance'].mean().to_dict()
df['aci'] = df['model_short'].map(aci)
aci_mean = df['aci'].mean()
df['aci_centered'] = df['aci'] - aci_mean
df['aci_sq_centered'] = df['aci_centered']**2
df['aci_x_efficiency'] = df['aci_centered'] * df['efficiency']
df['msg_density_x_aci'] = df['message_density'] * df['aci_centered']
df['aci_x_tools'] = df['aci_centered'] * np.log1p(df['n_tools'])
ACI_FORMULA = FORMULA.replace('intelligence_centered','aci_centered').replace('intelligence_sq_centered','aci_sq_centered').replace('intel_x_efficiency','aci_x_efficiency').replace('msg_density_x_intel','msg_density_x_aci').replace('intel_x_tools','aci_x_tools')
aci_cols = [c.replace('intelligence_centered','aci_centered').replace('intelligence_sq_centered','aci_sq_centered') for c in FEATURE_COLS]
# Add new cols
for c in ['aci_centered','aci_sq_centered','aci_x_efficiency','msg_density_x_aci','aci_x_tools']:
    if c not in aci_cols: aci_cols.append(c)
df_aci = df.copy()
sc2 = StandardScaler()
all_feat = list(set(FEATURE_COLS + aci_cols))
df_aci[all_feat] = sc2.fit_transform(df_aci[all_feat])
aci_model = smf.ols(ACI_FORMULA, data=df_aci).fit()
# ACI CV
cv_r2_aci = []
for train_idx, test_idx in kf.split(df_aci):
    cv_m = smf.ols(ACI_FORMULA, data=df_aci.iloc[train_idx]).fit()
    yp = cv_m.predict(df_aci.iloc[test_idx])
    yt = df_aci.iloc[test_idx]['performance'].values
    ss_r = np.sum((yt-yp)**2); ss_t = np.sum((yt-yt.mean())**2)
    cv_r2_aci.append(1 - ss_r/ss_t if ss_t > 0 else np.nan)
print(f"ACI R2_train = {aci_model.rsquared:.4f}")
print(f"ACI R2_CV    = {np.mean(cv_r2_aci):.4f} (+/- {np.std(cv_r2_aci):.4f})")
print(f"ACI AIC      = {aci_model.aic:.2f}")

# ACI correlation with Intelligence Index
corr = df[['aci','intelligence']].corr().iloc[0,1]
print(f"ACI-Intelligence correlation r = {corr:.4f}")

# Extended Data Table 1: nested specifications, same standardisation and five-fold split
print(f"\n--- EXTENDED DATA TABLE 1 (N={len(df)}) ---")
_I, _T, _A = "intelligence_centered", "log_tools", "log_agents"
_COORD = ["log_overhead", "message_density", "redundancy", "efficiency", "log_error_amp"]
_f1 = f"performance ~ {_I} + intelligence_sq_centered + {_T} + {_A}"
_f2 = _f1 + " + " + " + ".join(_COORD)
_f3 = _f2 + " + single_agent_baseline"
for _lab, _f in (("Intelligence + Tools + Agents", _f1), ("  + Coordination structure", _f2),
                 ("  + Single-agent baseline", _f3), ("  + Interaction terms (equation 1)", FORMULA)):
    _m = smf.ols(_f, data=df_scaled).fit()
    _cv = []
    for _tr, _te in KFold(n_splits=5, shuffle=True, random_state=42).split(df_scaled):
        _yp = smf.ols(_f, data=df_scaled.iloc[_tr]).fit().predict(df_scaled.iloc[_te])
        _yt = df_scaled.iloc[_te]["performance"].values
        _cv.append(1 - np.sum((_yt - _yp) ** 2) / np.sum((_yt - _yt.mean()) ** 2))
    print(f"  {_lab:<36s} R2_train={_m.rsquared:.3f}  R2_CV={np.mean(_cv):.3f}  AIC={_m.aic:.1f}")
