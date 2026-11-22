"""ReCySE final analyses, step 2: post-hoc descriptive diagnostics from saved calibrated test predictions.
No retraining, recalibration, member reselection or threshold change. Thresholds are the frozen stage-A values."""
import numpy as np, pandas as pd
D = '/home/claude/fin/data/'; O = '/home/claude/fin/out/'
import os; os.makedirs(O, exist_ok=True)
CFG = ['Hybrid-2', 'Hybrid-3', 'Hybrid-4', 'Accuracy Top-2', 'Accuracy Top-3', 'Accuracy Top-4',
       'All-9 equal-weight', 'Best single (composite)']
CLS = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
ALPHAS = [0.01, 0.03, 0.05]
pr = pd.read_csv(D + 'hybridk_ensemble_evaluation/test_predictions_all_configs.csv.gz')
pr = pr[pr.config.isin(CFG)]
sa = pd.read_csv(D + 'risk_targeting_exploratory/stageA_thresholds_calibration.csv')
PC = [c for c in pr.columns if c.startswith('p_calibrated__')]
assert [c.split('__im_')[1] for c in PC] == CLS

# ---------- 1. reliability diagram bins (15 equal-width, top-class confidence, pooled over 15 runs)
nb = 15; b = np.linspace(0, 1, nb + 1); rel = []
for c in CFG:
    g = pr[pr.config == c]; p = g[PC].values; conf = p.max(1); corr = (p.argmax(1) == g.y_true.values).astype(float)
    for i in range(nb):
        m = (conf > b[i]) & (conf <= b[i + 1])
        rel.append({'config': c, 'bin_lo': b[i], 'bin_hi': b[i + 1], 'n': int(m.sum()),
                    'mean_conf': conf[m].mean() if m.any() else np.nan, 'accuracy': corr[m].mean() if m.any() else np.nan})
rel = pd.DataFrame(rel); rel.to_csv(O + 'reliability_bins_pooled.csv', index=False)
pooled_ece = rel.assign(w=lambda d: d.n / d.groupby('config').n.transform('sum'),
                        gap=lambda d: (d.accuracy - d.mean_conf).abs()).fillna(0).assign(x=lambda d: d.w * d.gap) \
    .groupby('config').x.sum()

# ---------- 2. risk-coverage curves (per run, then mean over 15 runs)
grid = np.round(np.arange(0.01, 1.0001, 0.01), 2); rc = []
for (c, k, s), g in pr.groupby(['config', 'fold', 'seed']):
    p = g[PC].values; conf = p.max(1); err = (p.argmax(1) != g.y_true.values).astype(float)
    o = np.argsort(-conf, kind='stable'); n = len(conf); ce = np.cumsum(err[o])
    for cv in grid:
        m = max(1, int(np.ceil(cv * n - 1e-9))); rc.append({'config': c, 'fold': k, 'seed': s, 'coverage': cv, 'risk': ce[m - 1] / m})
rc = pd.DataFrame(rc)
fm = rc.groupby(['config', 'coverage', 'fold']).risk.mean()
rcs = rc.groupby(['config', 'coverage']).risk.mean().to_frame('risk_mean').join(
    fm.groupby(['config', 'coverage']).std(ddof=1).rename('risk_sd_fold')).reset_index()
rcs.to_csv(O + 'risk_coverage_curves_mean.csv', index=False)

# ---------- 3. class-level coverage, referral and selective risk at frozen thresholds
rows = []
for t in sa[sa.config.isin(CFG) & sa.alpha.round(4).isin(ALPHAS)].itertuples():
    g = pr[(pr.config == t.config) & (pr.fold == t.fold) & (pr.seed == t.seed)]
    p = g[PC].values; y = g.y_true.values; conf = p.max(1); corr = p.argmax(1) == y; acc = conf >= t.tau
    for j, cn in enumerate(CLS):
        m = y == j; na = int((acc & m).sum())
        rows.append({'config': t.config, 'fold': t.fold, 'seed': t.seed, 'alpha': round(t.alpha, 2), 'class': cn,
                     'n_class': int(m.sum()), 'n_accepted': na, 'coverage': na / m.sum(), 'referral': 1 - na / m.sum(),
                     'selective_risk': float((~corr[acc & m]).mean()) if na else np.nan})
cl = pd.DataFrame(rows); cl.to_csv(O + 'class_level_per_run.csv', index=False)
K = ['alpha', 'config', 'class']
fmc = cl.groupby(K + ['fold'])[['coverage', 'referral', 'selective_risk']].mean()
sdc = fmc.groupby(K).std(ddof=1).add_suffix('_sd_fold')
agg = cl.groupby(K).agg(n_class_mean=('n_class', 'mean'), coverage=('coverage', 'mean'), referral=('referral', 'mean'),
                         selective_risk=('selective_risk', 'mean'),
                         runs_undefined=('n_accepted', lambda x: int((x == 0).sum())),
                         runs_risk_le_alpha=('selective_risk', lambda x: np.nan)).reset_index()
agg = agg.merge(sdc.reset_index(), on=K)
le = cl.assign(le=lambda d: d.selective_risk <= d.alpha).query('n_accepted>0').groupby(K).le.sum()
agg['runs_risk_le_alpha'] = agg.set_index(K).index.map(le).fillna(0).astype(int)
agg['config'] = pd.Categorical(agg.config, CFG); agg['class'] = pd.Categorical(agg['class'], CLS)
agg = agg.sort_values(K).reset_index(drop=True); agg.to_csv(O + 'class_level_summary.csv', index=False)

# ---------- internal consistency: class-level pooled back to overall equals stage-B
sb = pd.read_csv(D + 'risk_targeting_exploratory/stageB_test_results_per_run.csv'); sb['alpha'] = sb.alpha.round(2)
rb = cl.groupby(['config', 'fold', 'seed', 'alpha']).apply(
    lambda d: pd.Series({'cov': d.n_accepted.sum() / d.n_class.sum(),
                         'risk': (d.selective_risk.fillna(0) * d.n_accepted).sum() / d.n_accepted.sum() if d.n_accepted.sum() else np.nan}),
    include_groups=False).reset_index()
x = rb.merge(sb, on=['config', 'fold', 'seed', 'alpha'])
print('class->overall coverage maxdiff', (x['cov'] - x.test_coverage).abs().max(),
      'risk maxdiff', (x['risk'] - x.test_selective_risk).abs().max(), 'rows', len(x))
pum = pd.read_csv(D + 'hybridk_ensemble_evaluation/per_unit_test_metrics.csv')
au = rcs.groupby('config').apply(lambda d: np.trapezoid(d.risk_mean, d.coverage), include_groups=False)
print('pooled ECE vs mean per-run ECE\n', pd.concat([pooled_ece.rename('pooled'), pum.groupby('config').ece.mean().rename('mean_run'),
                                                      au.rename('AURC_grid'), pum.groupby('config').aurc.mean().rename('AURC_reported')], axis=1).loc[CFG].round(4))
pooled_ece.to_csv(O + 'pooled_ece.csv')
