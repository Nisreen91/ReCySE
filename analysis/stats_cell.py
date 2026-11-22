# ======================================================================================
# Hybrid-K - STATISTICAL, ACCURACY-RELIABILITY AND SELECTIVE-PREDICTION ANALYSIS (CPU only)
#  Reads the saved outputs only. Nothing is retrained, re-selected or re-weighted.
#  * Unit of inference = OUTER FOLD (n = 5). The three seeds of a fold share the same test images,
#    so seed results are averaged within each fold before any test; the 15 fold-seed values are
#    never treated as independent.
#  * Paired tests on the 5 fold-level differences:
#      - primary: Nadeau-Bengio corrected paired t (variance x (1/J + n_test/n_train), df = J-1),
#        which is conservative for overlapping training sets;
#      - also reported: uncorrected paired t, and the exact sign-flip permutation test
#        (with J = 5 its smallest attainable two-sided p is 2/32 = 0.0625, so it can never reach 0.05).
#    Holm correction over the 8 metrics within each comparison, and over all comparisons x metrics.
#  * Labels: "supported" only if the Holm-adjusted corrected-t p < 0.05; "consistent trend" if all
#    5 folds point the same way without support; otherwise "inconclusive". These are not
#    superiority or equivalence claims.
#  * Selective prediction: confidence = max calibrated probability; ordering and AURC exactly as in
#    the existing metric (np.argsort(-conf)). Test risk-coverage results are descriptive.
#    Operational thresholds are chosen on CALIBRATION data only, using the frozen ensemble temperatures.
# ======================================================================================
import os, json, glob, hashlib, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import softmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from google.colab import drive
    if not os.path.isdir('/content/drive/MyDrive'):
        drive.mount('/content/drive')
except Exception:
    pass

OUT = Path(os.environ.get('HYBRIDK_OUT', '/content/drive/MyDrive/HybridK_Phase1_corrected'))
ENS, RUNS = OUT / 'hybridk_ensemble_evaluation', OUT / 'fold_results'
NEW = OUT / 'hybridk_statistical_analysis'
FIG = NEW / 'figures'
MODELS = ['swin_tiny_patch4_window7_224', 'tiny_vit_5m_224', 'vit_tiny_patch16_224', 'densenet121', 'vgg16',
          'mobilevit_xs', 'efficientformerv2_s0', 'resnet50', 'efficientnet_b0']
PRETTY = {'swin_tiny_patch4_window7_224': 'Swin-Tiny', 'tiny_vit_5m_224': 'TinyViT-5M',
          'vit_tiny_patch16_224': 'ViT-Tiny', 'densenet121': 'DenseNet121', 'vgg16': 'VGG16',
          'mobilevit_xs': 'MobileViT-XS', 'efficientformerv2_s0': 'EfficientFormerV2-S0',
          'resnet50': 'ResNet50', 'efficientnet_b0': 'EfficientNet-B0'}
KEY = {v: k for k, v in PRETTY.items()}
SEEDS, FOLDS, KS = [42, 123, 456], [0, 1, 2, 3, 4], [2, 3, 4]
METRIC_KEYS = ['acc', 'f1', 'auroc', 'ece', 'wc_ece', 'aurc', 'brier', 'nll']
MNAME = {'acc': 'Accuracy', 'f1': 'Macro-F1', 'auroc': 'Macro-AUROC', 'ece': 'ECE', 'wc_ece': 'WC-ECE',
         'aurc': 'AURC', 'brier': 'Brier', 'nll': 'NLL (exact)'}
HIGHER = {'acc', 'f1', 'auroc'}
COVERAGES = [0.70, 0.80, 0.90, 0.95, 1.00]
ALPHA = 0.05
EPS = 1e-12
t0 = time.time()

# ---------------------------------------------------------------- 1. read-only protection
def sha(f): return hashlib.sha256(open(f, 'rb').read()).hexdigest()
protected = []
for sub in ('fold_results', 'calibration_corrected', 'crossfit_composite', 'hybridk_ensemble_evaluation',
            'hybridk_robustness_analysis'):
    protected += [f for f in sorted(glob.glob(str(OUT / sub / '**' / '*'), recursive=True)) if os.path.isfile(f)]
hash_before = {f: sha(f) for f in protected}
print(f'Protected files: {len(protected)} ({time.time() - t0:.0f}s)')

# ---------------------------------------------------------------- 2. load saved test results
units = pd.read_csv(ENS / 'per_unit_test_metrics.csv')
frozen = pd.read_csv(ENS / 'frozen_configurations_calibration_only.csv')
selm = pd.read_csv(ENS / 'selected_members_per_fold.csv')
preds = pd.read_csv(ENS / 'test_predictions_all_configs.csv.gz')
SINGLES = [f'Single: {PRETTY[m]}' for m in MODELS]
HYB = [f'Hybrid-{K}' for K in KS]; ACC = [f'Accuracy Top-{K}' for K in KS]
CONFIGS = HYB + ACC + ['All-9 equal-weight', 'Best single (composite)', 'Best single (accuracy)'] + SINGLES
assert set(CONFIGS) <= set(units.config), 'missing configurations in per_unit_test_metrics.csv'
assert (units.groupby('config').size().reindex(CONFIGS) == 15).all()
PCOLS = [c for c in preds.columns if c.startswith('p_calibrated__')]
CLASS_NAMES = [c.replace('p_calibrated__', '') for c in PCOLS]

# split sizes for the Nadeau-Bengio correction (from the run files' split records)
ratios = []
for k in FOLDS:
    d = json.load(open(RUNS / f'multiclass__{MODELS[0]}__rep{SEEDS[0]}__fold{k}.json'))
    ratios.append(d['split']['n_test'] / d['split']['n_train'])
RHO = float(np.mean(ratios))
print(f'n_test / n_train (mean over folds) = {RHO:.4f} -> NB variance factor 1/5 + {RHO:.4f}')

# fold-level means (seeds averaged within fold)
fold_means = units.groupby(['config', 'fold'])[METRIC_KEYS].mean()

# ---------------------------------------------------------------- 3. paired statistics
def holm(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = ~np.isnan(p)
    q = p[ok]; o = np.argsort(q); m = len(q); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * q[i])); adj[i] = run
    out[ok] = adj; return out

def signflip_p(d):
    d = np.asarray(d, float)
    if np.allclose(d, 0): return np.nan
    obs = abs(d.mean()); n = len(d)
    vals = [abs((np.array(s) * d).mean()) for s in itertools.product([1, -1], repeat=n)]
    return float(np.mean([v >= obs - 1e-15 for v in vals]))

def paired(a_cfg, b_cfg, key):
    A = fold_means.loc[a_cfg][key].reindex(FOLDS).values; B = fold_means.loc[b_cfg][key].reindex(FOLDS).values
    d = A - B; J = len(d); m = d.mean(); s = d.std(ddof=1)
    ua = units[units.config == a_cfg].set_index(['fold', 'seed'])[key]
    ub = units[units.config == b_cfg].set_index(['fold', 'seed'])[key].reindex(ua.index)
    du = (ua - ub).values
    sign = 1 if key in HIGHER else -1                       # improvement = sign * difference
    tq = stats.t.ppf(1 - ALPHA / 2, J - 1)
    if s < 1e-15:
        p_c = p_u = np.nan; ci_c = ci_u = (m, m); dz = np.nan
    else:
        se_u = s / np.sqrt(J); se_c = s * np.sqrt(1 / J + RHO)
        p_u = 2 * stats.t.sf(abs(m / se_u), J - 1); p_c = 2 * stats.t.sf(abs(m / se_c), J - 1)
        ci_u = (m - tq * se_u, m + tq * se_u); ci_c = (m - tq * se_c, m + tq * se_c); dz = m / s
    return {'metric': key, 'direction': 'higher' if key in HIGHER else 'lower', 'mean_A': A.mean(), 'mean_B': B.mean(),
            'mean_diff': m, 'sd_fold_diff': s, 'ci95_corrected_low': ci_c[0], 'ci95_corrected_high': ci_c[1],
            'ci95_uncorrected_low': ci_u[0], 'ci95_uncorrected_high': ci_u[1],
            'rel_diff_pct_of_B': 100 * m / B.mean() if abs(B.mean()) > 1e-15 else np.nan,
            'improvement': sign * m, 'cohens_dz': dz, 'p_corrected_t': p_c, 'p_uncorrected_t': p_u,
            'p_signflip_exact': signflip_p(d),
            'folds_A_better': int((sign * d > 1e-12).sum()), 'folds_tied': int((np.abs(d) <= 1e-12).sum()),
            'units_A_better_of_15': int((sign * du > 1e-12).sum()), 'units_tied_of_15': int((np.abs(du) <= 1e-12).sum()),
            **{f'fold{k}_diff': d[i] for i, k in enumerate(FOLDS)}}

PAIRS = []
for K in KS:
    PAIRS += [(f'Hybrid-{K}', f'Accuracy Top-{K}', 'Accuracy Top-K'), (f'Hybrid-{K}', 'All-9 equal-weight', 'All-9'),
              (f'Hybrid-{K}', 'Best single (composite)', 'Best single')]
ident = {}
for a, b, _ in PAIRS:
    ma = selm[selm.config == a].set_index('fold').members; mb = selm[selm.config == b].set_index('fold').members
    ident[(a, b)] = int((ma.reindex(FOLDS) == mb.reindex(FOLDS)).sum())
rows = []
for a, b, fam in PAIRS:
    for key in METRIC_KEYS:
        rows.append({'comparison': f'{a} vs {b}', 'hybrid': a, 'baseline': b, 'baseline_type': fam,
                     'K': int(a.split('-')[1]), 'folds_identical_members': ident[(a, b)], **paired(a, b, key)})
cmp = pd.DataFrame(rows)
cmp['p_holm_within_comparison'] = cmp.groupby('comparison').p_corrected_t.transform(lambda p: pd.Series(holm(p.values), index=p.index))
cmp['p_holm_global'] = holm(cmp.p_corrected_t.values)

def label(r):
    if r.folds_tied == 5 and abs(r.mean_diff) < 1e-15: return 'identical (no difference)'
    dirn = 'Hybrid-K better' if r.improvement > 0 else 'Hybrid-K worse'
    if r.p_holm_within_comparison < ALPHA: return f'supported: {dirn}'
    if r.folds_A_better == 5 or (5 - r.folds_A_better - r.folds_tied) == 5: return f'consistent trend: {dirn} (5/5 folds)'
    return 'inconclusive'
cmp['finding'] = cmp.apply(label, axis=1)
cmp['supported_after_global_holm'] = cmp.p_holm_global < ALPHA

# ---------------------------------------------------------------- 4. trade-off and Pareto (descriptive)
summ_rows = []
for c in CONFIGS:
    u = units[units.config == c]; fm = fold_means.loc[c]
    summ_rows.append({'config': c, **{f'{k}_mean': u[k].mean() for k in METRIC_KEYS},
                      **{f'{k}_sd_fold': fm[k].std(ddof=1) for k in METRIC_KEYS},
                      **{f'{k}_sd_15units': u[k].std(ddof=1) for k in METRIC_KEYS}})
summ = pd.DataFrame(summ_rows).set_index('config')
PARETO_SET = HYB + ACC + ['All-9 equal-weight'] + SINGLES          # aliases of single models excluded
def nondominated(V, higher):
    S = V * np.where(higher, 1, -1)                     # maximize all
    nd = np.ones(len(S), bool)
    for i in range(len(S)):
        dom = (S >= S[i] - 1e-15).all(1) & (S > S[i] + 1e-15).any(1)
        nd[i] = not dom.any()
    return nd
hv = np.array([k in HIGHER for k in METRIC_KEYS])
par_rows = []
for k in FOLDS:
    V = np.array([fold_means.loc[(c, k), METRIC_KEYS].values for c in PARETO_SET], float)
    nd = nondominated(V, hv)
    for c, f in zip(PARETO_SET, nd): par_rows.append({'fold': k, 'config': c, 'nondominated_8_metrics': bool(f)})
par = pd.DataFrame(par_rows)
pareto = par.groupby('config', sort=False).nondominated_8_metrics.sum().rename('folds_nondominated_8_metrics_of_5').to_frame()
Vall = summ.loc[PARETO_SET, [f'{k}_mean' for k in METRIC_KEYS]].values
pareto['nondominated_8_metrics_on_means'] = nondominated(Vall, hv)
for rk in ['ece', 'wc_ece', 'aurc', 'brier', 'nll']:
    V2 = summ.loc[PARETO_SET, ['acc_mean', f'{rk}_mean']].values
    pareto[f'front_acc_vs_{rk}'] = nondominated(V2, np.array([True, False]))
pareto = pareto.reindex(PARETO_SET)

# ---------------------------------------------------------------- 5. selective prediction on test (descriptive)
def aurc_existing(y, p):                                 # identical to the existing AURC metric
    conf = p.max(1); corr = (p.argmax(1) == y).astype(float); o = np.argsort(-conf)
    risk = 1 - np.cumsum(corr[o]) / np.arange(1, len(y) + 1); cov = np.arange(1, len(y) + 1) / len(y)
    return float((np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(risk, cov)), risk, cov
GRID = np.linspace(0.05, 1.0, 96)
SEL_CFG = HYB + ACC + ['All-9 equal-weight', 'Best single (composite)']
sel_rows, curve_rows, check_rows = [], [], []
for (c, k, s), g in preds[preds.config.isin(SEL_CFG)].groupby(['config', 'fold', 'seed']):
    y = g.y_true.values; p = g[PCOLS].values
    a, risk, cov = aurc_existing(y, p); n = len(y)
    saved = units[(units.config == c) & (units.fold == k) & (units.seed == s)].iloc[0]
    conf = p.max(1)
    check_rows.append({'config': c, 'fold': k, 'seed': s, 'aurc_diff': abs(a - saved.aurc),
                       'n_tied_confidence': int(len(conf) - len(np.unique(conf))),
                       'acc_diff': abs((p.argmax(1) == y).mean() - saved.acc), 'ypred_match': bool((p.argmax(1) == g.y_pred.values).all())})
    for cv in COVERAGES:
        nc = int(np.ceil(cv * n - 1e-9))
        sel_rows.append({'config': c, 'fold': k, 'seed': s, 'coverage': cv, 'n_accepted': nc, 'selective_risk': risk[nc - 1]})
    idx = np.ceil(GRID * n - 1e-9).astype(int) - 1
    curve_rows.append(pd.DataFrame({'config': c, 'fold': k, 'seed': s, 'coverage': GRID, 'risk': risk[idx]}))
chk = pd.DataFrame(check_rows)
assert chk.acc_diff.max() < 1e-12 and chk.ypred_match.all(), 'saved predictions inconsistent'
assert chk.aurc_diff.max() < 1e-3, 'AURC recomputed from saved probabilities differs by more than tie effects'
if chk.aurc_diff.max() > 1e-9:
    print(f'  NOTE: AURC recomputed from saved probabilities differs by up to {chk.aurc_diff.max():.1e} in '
          f'{int((chk.aurc_diff > 1e-9).sum())} runs. The existing AURC orders tied confidences arbitrarily '
          f'(saturated probabilities; {int((chk.n_tied_confidence > 0).sum())} runs have ties), and the CSV '
          f'round-trip changes the tie order. Saved AURC values are used in all statistical tests.')
selr = pd.DataFrame(sel_rows); curves = pd.concat(curve_rows, ignore_index=True)
sel_fold = selr.groupby(['config', 'coverage', 'fold']).selective_risk.mean()
sel_summary = selr.groupby(['config', 'coverage']).selective_risk.agg(['mean']).join(
    sel_fold.groupby(['config', 'coverage']).std(ddof=1).rename('sd_fold')).reset_index()
srows = []
for a, b, fam in [p for p in PAIRS if p[2] in ('Accuracy Top-K', 'All-9')] + [(f'Hybrid-{K}', 'Best single (composite)', 'Best single') for K in KS]:
    for cv in COVERAGES:
        A = sel_fold.loc[(a, cv)].reindex(FOLDS).values; B = sel_fold.loc[(b, cv)].reindex(FOLDS).values
        d = A - B; m = d.mean(); sd = d.std(ddof=1); tq = stats.t.ppf(1 - ALPHA / 2, 4)
        se = sd * np.sqrt(1 / 5 + RHO)
        p_c = 2 * stats.t.sf(abs(m / se), 4) if sd > 1e-15 else np.nan
        srows.append({'comparison': f'{a} vs {b}', 'baseline_type': fam, 'coverage': cv, 'risk_A': A.mean(), 'risk_B': B.mean(),
                      'mean_diff': m, 'ci95_corrected_low': m - tq * se, 'ci95_corrected_high': m + tq * se,
                      'p_corrected_t': p_c, 'p_signflip_exact': signflip_p(d), 'folds_A_lower_risk': int((d < -1e-12).sum()),
                      'folds_tied': int((np.abs(d) <= 1e-12).sum())})
selcmp = pd.DataFrame(srows)
selcmp['p_holm_within_comparison'] = selcmp.groupby('comparison').p_corrected_t.transform(lambda p: pd.Series(holm(p.values), index=p.index))

# ---------------------------------------------------------------- 6. operational thresholds from CALIBRATION data only
print(f'Rebuilding ensemble calibration confidences from the run files ({time.time() - t0:.0f}s) ...')
run = {}
for m in MODELS:
    for s in SEEDS:
        for k in FOLDS:
            d = json.load(open(RUNS / f'multiclass__{m}__rep{s}__fold{k}.json'))
            ca = d['calib']; ck = [f'{c}/{f}' for c, f in zip(ca['classes'], ca['filenames'])]
            o = np.argsort(ck)
            tk = [f'{c}/{f}' for c, f in zip(d['classes'], d['filenames'])]; ot = np.argsort(tk)
            run[(m, s, k)] = {'ck': [ck[i] for i in o], 'cy': np.asarray(ca['y_true'])[o],
                              'cp': softmax(np.asarray(ca['logits'], float)[o], axis=1),
                              'tk': [tk[i] for i in ot], 'tp': np.asarray(d['y_prob'], float)[ot], 'ty': np.asarray(d['y_true'])[ot]}
            del d
def clean(p):
    p = np.clip(np.nan_to_num(p), 0, None); s = p.sum(1, keepdims=True); return p / np.where(s < 1e-12, 1, s)
def ens_prob(members, part, s, k, beta):
    lg = np.log(np.clip(np.mean([clean(run[(m, s, k)][part]) for m in members], axis=0), EPS, 1.0))
    return softmax(beta * lg, axis=1)
thr_rows, recon_diff = [], 0.0
TARGETS = [0.70, 0.80, 0.90, 0.95]
for _, fz in frozen[frozen.config.isin(SEL_CFG)].iterrows():
    c, k, s, beta = fz.config, int(fz.fold), int(fz.seed), 1 / float(fz['T'])
    mem = [KEY[x] for x in fz.members.split(' + ')]
    for m in mem:
        assert run[(m, s, k)]['ck'] == run[(mem[0], s, k)]['ck'] and run[(m, s, k)]['tk'] == run[(mem[0], s, k)]['tk']
    pc = ens_prob(mem, 'cp', s, k, beta); yc = run[(mem[0], s, k)]['cy']
    pt = ens_prob(mem, 'tp', s, k, beta); yt = run[(mem[0], s, k)]['ty']
    g = preds[(preds.config == c) & (preds.fold == k) & (preds.seed == s)]
    gk = [f'{a}/{b}' for a, b in zip(g['class'], g.filename)]; go = np.argsort(gk)
    assert [gk[i] for i in go] == run[(mem[0], s, k)]['tk']
    recon_diff = max(recon_diff, float(np.abs(g[PCOLS].values[go] - pt).max()))
    cc = np.sort(pc.max(1))[::-1]; ct = pt.max(1); corr_t = (pt.argmax(1) == yt)
    for tg in TARGETS:
        tau = cc[int(np.ceil(tg * len(cc) - 1e-9)) - 1]           # threshold chosen on calibration data
        acc_t = ct >= tau
        thr_rows.append({'config': c, 'fold': k, 'seed': s, 'target_coverage': tg, 'threshold_from_calibration': tau,
                         'calib_coverage': float((pc.max(1) >= tau).mean()),
                         'calib_selective_risk': float(1 - (pc.argmax(1) == yc)[pc.max(1) >= tau].mean()),
                         'test_coverage': float(acc_t.mean()),
                         'test_selective_risk': float(1 - corr_t[acc_t].mean()) if acc_t.any() else np.nan})
assert recon_diff < 1e-6, f'ensemble reconstruction differs from saved test predictions ({recon_diff:.2e})'
thr = pd.DataFrame(thr_rows)
thr_fold = thr.groupby(['config', 'target_coverage', 'fold'])[['test_coverage', 'test_selective_risk', 'calib_selective_risk']].mean()
thr_summary = thr_fold.groupby(['config', 'target_coverage']).agg(['mean', 'std']).reset_index()
thr_summary.columns = ['_'.join([x for x in c if x]) if isinstance(c, tuple) else c for c in thr_summary.columns]
print(f'  ensemble probabilities rebuilt; max difference from saved test predictions = {recon_diff:.1e}')

# ---------------------------------------------------------------- 7. tables
NEW.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
fmt = lambda v, d=4: '' if pd.isna(v) else f'{v:.{d}f}'
def pfmt(p): return 'n/a' if pd.isna(p) else ('<0.001' if p < 0.001 else f'{p:.3f}')
T1 = pd.DataFrame({MNAME[k]: [f"{summ.loc[c, k + '_mean']:.4f} ± {summ.loc[c, k + '_sd_fold']:.4f}" for c in CONFIGS]
                   for k in METRIC_KEYS}, index=CONFIGS)
T1.index.name = 'Configuration (mean over 15 runs ± SD of 5 fold means)'
T2 = cmp.assign(cell=lambda d: [f'{r.mean_diff:+.4f} [{r.ci95_corrected_low:+.4f}, {r.ci95_corrected_high:+.4f}] '
                                f'p_Holm={pfmt(r.p_holm_within_comparison)}; {r.folds_A_better}/5'
                                for r in d.itertuples()]).pivot(index='comparison', columns='metric', values='cell')[METRIC_KEYS]
T2.columns = [MNAME[k] for k in METRIC_KEYS]
T2 = T2.reindex([f'{a} vs {b}' for a, b, _ in PAIRS])
T3 = cmp.pivot(index='comparison', columns='metric', values='finding')[METRIC_KEYS].reindex(T2.index)
T3.columns = T2.columns
TO = []
for a, b, fam in PAIRS:
    r = cmp[(cmp.hybrid == a) & (cmp.baseline == b)].set_index('metric')
    TO.append({'comparison': f'{a} vs {b}', 'folds_identical_members': ident[(a, b)],
               **{f'{MNAME[k]} diff [95% CI]': f"{r.loc[k, 'mean_diff']:+.4f} [{r.loc[k, 'ci95_corrected_low']:+.4f}, "
                  f"{r.loc[k, 'ci95_corrected_high']:+.4f}]" for k in ['acc', 'f1', 'ece', 'wc_ece', 'brier', 'nll', 'aurc']},
               'accuracy CI includes 0': bool(r.loc['acc', 'ci95_corrected_low'] <= 0 <= r.loc['acc', 'ci95_corrected_high']),
               'calibration metrics improved in >=4/5 folds': ', '.join(MNAME[k] for k in ['ece', 'wc_ece', 'brier', 'nll']
                                                                         if r.loc[k, 'folds_A_better'] >= 4) or 'none',
               'accuracy finding': r.loc['acc', 'finding'],
               'ECE finding': r.loc['ece', 'finding'], 'NLL finding': r.loc['nll', 'finding']})
T4 = pd.DataFrame(TO)
T5 = sel_summary.assign(cell=lambda d: [f'{m:.4f} ± {s:.4f}' for m, s in zip(d['mean'], d.sd_fold)]).pivot(
    index='config', columns='coverage', values='cell').reindex(SEL_CFG)
T5.columns = [f'risk @ {int(c * 100)}% coverage' for c in T5.columns]
T5['AURC'] = [f"{summ.loc[c, 'aurc_mean']:.4f} ± {summ.loc[c, 'aurc_sd_fold']:.4f}" for c in SEL_CFG]
T6 = selcmp.assign(cell=lambda d: [f'{r.mean_diff:+.4f} [{r.ci95_corrected_low:+.4f}, {r.ci95_corrected_high:+.4f}] '
                                   f'{r.folds_A_lower_risk}/5' for r in d.itertuples()]).pivot(
    index='comparison', columns='coverage', values='cell')
T6.columns = [f'risk diff @ {int(c * 100)}%' for c in T6.columns]
T7 = thr_summary.copy()

files_csv = {'paired_comparisons_full.csv': cmp, 'trade_off_summary.csv': T4, 'pareto_analysis.csv': pareto.reset_index(),
             'pareto_per_fold.csv': par, 'metrics_summary_numeric.csv': summ.reset_index(),
             'selective_risk_per_unit.csv': selr, 'selective_risk_summary.csv': sel_summary,
             'selective_risk_paired_comparisons.csv': selcmp, 'risk_coverage_curves.csv': curves,
             'calibration_threshold_per_unit.csv': thr, 'calibration_threshold_summary.csv': T7,
             'consistency_checks.csv': chk}
for n, t in files_csv.items(): t.to_csv(NEW / n, index=False)
for n, t in {'Table1_metrics_mean_sd.csv': T1, 'Table2_paired_differences_CI.csv': T2, 'Table3_findings.csv': T3,
             'Table5_selective_risk.csv': T5, 'Table6_selective_risk_differences.csv': T6}.items():
    t.to_csv(NEW / n)
try:
    with pd.ExcelWriter(NEW / 'statistical_tables.xlsx') as xw:
        for n, t in {'T1_metrics': T1, 'T2_paired_diff': T2, 'T3_findings': T3, 'T4_tradeoff': T4.set_index('comparison'),
                     'T5_selective_risk': T5, 'T6_selective_diff': T6, 'T7_calib_thresholds': T7.set_index('config'),
                     'Pareto': pareto}.items():
            t.to_excel(xw, sheet_name=n[:31])
except Exception as e:
    print('  Excel export skipped:', e)

# ---------------------------------------------------------------- 8. figures (colour-blind-safe; shape + label as secondary encoding)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})
C_H, C_A, C_9, C_S, C_G = '#2a78d6', '#eb6834', '#1baf7a', '#52514e', '#b8b7b0'
MK = {2: 'o', 3: 's', 4: '^'}
def style(c):
    if c.startswith('Hybrid'): return C_H, MK[int(c[-1])], c.replace('Hybrid-', 'H')
    if c.startswith('Accuracy'): return C_A, MK[int(c[-1])], c.replace('Accuracy Top-', 'A')
    if c.startswith('All-9'): return C_9, 'D', 'All-9'
    if c.startswith('Best single'): return C_S, 'v', 'Best single'
    return C_G, '.', c.replace('Single: ', '')
def save(fig, name):
    fig.savefig(FIG / f'{name}.png'); fig.savefig(FIG / f'{name}.pdf'); plt.close(fig)

# F1: forest plot of fold-level differences (improvement scale, metric units, corrected 95% CI)
fig, axs = plt.subplots(2, 4, figsize=(15, 7.5), sharey=True)
order = [f'{a} vs {b}' for a, b, _ in PAIRS]
for ax, key in zip(axs.flat, METRIC_KEYS):
    r = cmp[cmp.metric == key].set_index('comparison').reindex(order)
    sgn = 1 if key in HIGHER else -1
    for i, (cname, row) in enumerate(r.iterrows()):
        col = {'Accuracy Top-K': C_A, 'All-9': C_9, 'Best single': C_S}[row.baseline_type]
        lo, hi = sorted([sgn * row.ci95_corrected_low, sgn * row.ci95_corrected_high])
        ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle='round')
        ax.plot(row.improvement, i, MK[row.K], color=col, ms=7, mec=col, mew=1.5, zorder=3,
                mfc=col if row.p_holm_within_comparison < ALPHA else 'white')
    ax.axvline(0, color='#0b0b0b', lw=0.8); ax.set_title(MNAME[key] + (' (higher better)' if key in HIGHER else ' (lower better)'))
    ax.set_xlabel('improvement of Hybrid-K\n(positive = Hybrid-K better)')
axs[0, 0].set_yticks(range(len(order))); axs[0, 0].set_yticklabels(order); axs[0, 0].invert_yaxis()
axs[1, 0].set_yticks(range(len(order))); axs[1, 0].set_yticklabels(order)
fig.suptitle('Paired fold-level differences, mean and 95% CI (Nadeau-Bengio corrected, 5 outer folds). '
             'Filled marker = Holm-adjusted p < 0.05; open = not supported.', fontsize=10)
fig.tight_layout(); save(fig, 'Fig1_forest_paired_differences')

# F2: accuracy vs reliability with Pareto fronts
fig, axs = plt.subplots(2, 2, figsize=(11, 9))
for ax, rk in zip(axs.flat, ['ece', 'nll', 'brier', 'aurc']):
    for c in PARETO_SET:
        col, mk, lab = style(c)
        x, y = summ.loc[c, 'acc_mean'], summ.loc[c, f'{rk}_mean']
        ax.errorbar(x, y, xerr=summ.loc[c, 'acc_sd_fold'], yerr=summ.loc[c, f'{rk}_sd_fold'], fmt=mk, color=col,
                    ms=7 if col != C_G else 9, mec='white', mew=1.2, ecolor=col, elinewidth=1, capsize=0, alpha=0.95, zorder=3)
        off, ha = ((-6, 6), 'right') if col == C_H else (((6, -11), 'left') if col == C_A else ((4, 4), 'left'))
        ax.annotate(lab, (x, y), xytext=off, textcoords='offset points', fontsize=7, ha=ha,
                    color='#0b0b0b' if col != C_G else '#8a897f')
    fr = pareto.index[pareto[f'front_acc_vs_{rk}']]
    P = summ.loc[fr, ['acc_mean', f'{rk}_mean']].sort_values('acc_mean')
    ax.plot(P.acc_mean, P[f'{rk}_mean'], '--', color='#52514e', lw=1, zorder=2, label='Pareto front (test means, descriptive)')
    ax.set_xlabel('Accuracy (higher better)'); ax.set_ylabel(f'{MNAME[rk]} (lower better)')
handles = [plt.Line2D([], [], marker='o', color=C_H, ls='', label='Hybrid-K (H2, H3, H4)'),
           plt.Line2D([], [], marker='o', color=C_A, ls='', label='Accuracy Top-K (A2, A3, A4)'),
           plt.Line2D([], [], marker='D', color=C_9, ls='', label='All-9 equal-weight'),
           plt.Line2D([], [], marker='.', color=C_G, ls='', ms=10, label='Individual models'),
           plt.Line2D([], [], ls='--', color='#52514e', label='2-objective Pareto front (descriptive)')]
fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Accuracy versus reliability on the test folds (mean over 15 runs; bars = SD of 5 fold means). '
             'Marker shape: circle K=2, square K=3, triangle K=4.', fontsize=10)
fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, 'Fig2_accuracy_vs_reliability_pareto')

# F3: trade-off plane, fold-level differences
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, rk in zip(axs, ['ece', 'nll', 'brier']):
    for a, b, fam in PAIRS:
        col = {'Accuracy Top-K': C_A, 'All-9': C_9, 'Best single': C_S}[fam]; K = int(a[-1])
        ra = cmp[(cmp.hybrid == a) & (cmp.baseline == b) & (cmp.metric == 'acc')].iloc[0]
        rr = cmp[(cmp.hybrid == a) & (cmp.baseline == b) & (cmp.metric == rk)].iloc[0]
        dx = [ra[f'fold{k}_diff'] for k in FOLDS]; dy = [-rr[f'fold{k}_diff'] for k in FOLDS]
        ax.scatter(dx, dy, s=14, color=col, alpha=0.35, marker=MK[K], linewidths=0)
        ax.errorbar(ra.mean_diff, -rr.mean_diff, xerr=[[ra.mean_diff - ra.ci95_corrected_low], [ra.ci95_corrected_high - ra.mean_diff]],
                    yerr=[[rr.ci95_corrected_high - rr.mean_diff], [rr.mean_diff - rr.ci95_corrected_low]],
                    fmt=MK[K], color=col, ms=8, mec='white', mew=1.2, elinewidth=1.2)
    ax.axhline(0, color='#0b0b0b', lw=0.8); ax.axvline(0, color='#0b0b0b', lw=0.8)
    ax.set_xlabel('Accuracy difference (Hybrid-K minus baseline)'); ax.set_ylabel(f'{MNAME[rk]} improvement (baseline minus Hybrid-K)')
    ax.set_title(f'Accuracy vs {MNAME[rk]}: upper right = better on both')
handles = [plt.Line2D([], [], marker='o', color=C_A, ls='', label='vs Accuracy Top-K'),
           plt.Line2D([], [], marker='o', color=C_9, ls='', label='vs All-9'),
           plt.Line2D([], [], marker='o', color=C_S, ls='', label='vs best single'),
           plt.Line2D([], [], marker='o', color='#8a897f', ls='', label='K=2'),
           plt.Line2D([], [], marker='s', color='#8a897f', ls='', label='K=3'),
           plt.Line2D([], [], marker='^', color='#8a897f', ls='', label='K=4')]
fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.suptitle('Trade-off plane: mean with 95% corrected CI; faint points are the 5 fold-level differences', fontsize=10)
fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, 'Fig3_tradeoff_plane')

# F4: risk-coverage curves
cm = curves.groupby(['config', 'fold', 'coverage']).risk.mean().reset_index()
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, K in zip(axs, KS):
    for c in [f'Hybrid-{K}', f'Accuracy Top-{K}', 'All-9 equal-weight', 'Best single (composite)']:
        col, mk, lab = style(c)
        g = cm[cm.config == c].groupby('coverage').risk; mu, sd = g.mean(), g.std(ddof=1)
        ax.plot(mu.index, mu.values, color=col, lw=2, label=c)
        ax.fill_between(mu.index, (mu - sd).values, (mu + sd).values, color=col, alpha=0.12, linewidth=0)
        ax.plot([cv for cv in COVERAGES], [mu.reindex([cv], method='nearest').iloc[0] for cv in COVERAGES], mk, color=col,
                ms=6, mec='white', mew=1.2)
    for cv in COVERAGES: ax.axvline(cv, color='#e6e5e1', lw=0.8, zorder=0)
    ax.set_title(f'K = {K}'); ax.set_xlabel('Coverage'); ax.legend(frameon=False, fontsize=8, loc='upper left')
axs[0].set_ylabel('Selective risk (1 - accuracy on accepted cells)')
fig.suptitle('Risk-coverage curves on the test folds (confidence-based rejection; mean of 5 fold means, band = SD). Descriptive.', fontsize=10)
fig.tight_layout(); save(fig, 'Fig4_risk_coverage_curves')

# F5: selective risk differences at prespecified coverage
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, K in zip(axs, KS):
    for b, col in [(f'Accuracy Top-{K}', C_A), ('All-9 equal-weight', C_9), ('Best single (composite)', C_S)]:
        r = selcmp[selcmp.comparison == f'Hybrid-{K} vs {b}'].sort_values('coverage')
        x = np.arange(len(COVERAGES)) + {C_A: -0.2, C_9: 0, C_S: 0.2}[col]
        ax.errorbar(x, -r.mean_diff, yerr=[(r.ci95_corrected_high - r.mean_diff).values, (r.mean_diff - r.ci95_corrected_low).values],
                    fmt=MK[K], color=col, ms=7, mec='white', mew=1.2, elinewidth=1.2, label=f'vs {b}')
    ax.axhline(0, color='#0b0b0b', lw=0.8); ax.set_xticks(range(len(COVERAGES)))
    ax.set_xticklabels([f'{int(c * 100)}%' for c in COVERAGES]); ax.set_xlabel('Coverage'); ax.set_title(f'Hybrid-{K}')
    ax.legend(frameon=False, fontsize=8)
axs[0].set_ylabel('Risk reduction by Hybrid-K\n(positive = lower selective risk)')
fig.suptitle('Selective-risk differences at prespecified coverages, mean and 95% corrected CI over 5 outer folds', fontsize=10)
fig.tight_layout(); save(fig, 'Fig5_selective_risk_differences')

# ---------------------------------------------------------------- 9. limitations text
LIMITS = f"""LIMITATIONS AND HOW TO REPORT THESE RESULTS
1. Five outer folds are the unit of inference (n = 5, df = 4). Power is low and confidence intervals are wide.
   Non-significant differences are not evidence of equivalence.
2. The three seeds re-use the same test images within a fold; they were averaged, not counted as replicates.
   Seed-level win counts (out of 15) are descriptive only.
3. The Nadeau-Bengio correction (variance factor 1/5 + n_test/n_train = {1 / 5 + RHO:.3f}) is a heuristic for
   overlapping training sets; it is conservative relative to the uncorrected paired t, which is also reported.
4. The exact sign-flip test with 5 folds cannot give p < 0.0625; it is reported as a sanity check, not for decisions.
5. Holm correction is applied over the 8 metrics within each comparison (primary) and over all
   {len(cmp)} tests (stricter). The 8 metrics are correlated; Holm remains valid but is conservative.
6. Where Hybrid-K and a baseline have identical members in a fold, the difference is exactly 0 in that fold.
7. Pareto fronts and risk-coverage curves are computed on test folds and are descriptive; nothing is selected from them.
8. Operational thresholds are chosen on calibration data whose temperature was fitted on the same subset;
   the resulting calibration coverage may therefore be slightly optimistic. Test coverage is reported as achieved.
9. Parent images are not patient identifiers; all results are on one dataset (SIPaKMeD) and one taxonomy.
"""
(NEW / 'limitations.txt').write_text(LIMITS)
json.dump({'unit_of_inference': 'outer fold (seed results averaged within fold)', 'n_folds': 5,
           'primary_test': 'Nadeau-Bengio corrected paired t, df=4', 'rho_n_test_over_n_train': RHO,
           'also_reported': ['uncorrected paired t', 'exact sign-flip permutation (min p 0.0625)', 'Cohen dz',
                             'seed-level win counts (descriptive)'],
           'multiplicity': 'Holm within comparison (8 metrics) [primary], Holm global [secondary]',
           'finding_labels': {'supported': 'Holm-adjusted corrected p < 0.05',
                              'consistent trend': 'same direction in 5/5 folds, not supported',
                              'inconclusive': 'otherwise', 'identical': 'zero difference in all folds'},
           'selective_prediction': {'confidence': 'max calibrated probability', 'ordering': 'np.argsort(-conf) as in AURC',
                                    'coverages': COVERAGES, 'n_accepted': 'ceil(coverage x n)',
                                    'operational_thresholds': 'chosen on calibration data (frozen ensemble T) for targets '
                                                              + str(TARGETS)},
           'pareto': 'descriptive, test means and per-fold; excludes best-single aliases'},
          open(NEW / 'run_config.json', 'w'), indent=2)

# ---------------------------------------------------------------- 10. verification and summary
hash_after = {f: sha(f) for f in protected}
changed = [f for f in protected if hash_before[f] != hash_after[f]]
print('\nVERIFICATION')
print(f'  configurations: {len(CONFIGS)} x 15 runs loaded; paired tests: {len(cmp)}; selective comparisons: {len(selcmp)}')
print(f'  saved AURC / accuracy reproduced from saved probabilities: max diff {chk.aurc_diff.max():.1e} / {chk.acc_diff.max():.1e}')
print(f'  ensemble test probabilities rebuilt from run files: max diff {recon_diff:.1e}')
print(f'  protected files modified: {len(changed)} (must be 0)')
assert not changed

pd.set_option('display.width', 260); pd.set_option('display.max_colwidth', 70)
print('\nTABLE 1. Test metrics, mean over 15 runs ± SD of the 5 fold means'); print(T1.loc[HYB + ACC + ['All-9 equal-weight', 'Best single (composite)']].to_string())
print('\nTABLE 2. Paired differences (Hybrid-K minus baseline), fold-level mean [95% corrected CI], Holm p, folds favouring Hybrid-K')
for key in ['acc', 'ece', 'nll', 'brier', 'aurc']:
    print(f'  {MNAME[key]}'); print(T2[[MNAME[key]]].to_string(header=False))
print('\nFINDINGS (counts over all comparisons x metrics)'); print(cmp.finding.value_counts().to_string())
sup = cmp[cmp.finding.str.startswith('supported')]
if len(sup):
    print('  supported findings:'); print(sup[['comparison', 'metric', 'mean_diff', 'ci95_corrected_low', 'ci95_corrected_high',
                                               'p_holm_within_comparison', 'supported_after_global_holm']].round(4).to_string(index=False))
else:
    print('  No difference is statistically supported after Holm correction; report differences as descriptive trends.')
print('\nPARETO (descriptive): folds in which each configuration is non-dominated on all 8 metrics')
print(pareto[['folds_nondominated_8_metrics_of_5', 'nondominated_8_metrics_on_means']].to_string())
print('\nSELECTIVE RISK on test (mean ± SD of fold means), descriptive'); print(T5.to_string())
print('\nSELECTIVE RISK DIFFERENCES (Hybrid-K minus baseline) [95% corrected CI], folds with lower Hybrid-K risk')
print(T6.to_string())
print('\nTHRESHOLDS CHOSEN ON CALIBRATION DATA, applied to test (mean over folds)')
print(thr_summary[thr_summary.config.isin(HYB + ['All-9 equal-weight'])][['config', 'target_coverage', 'test_coverage_mean',
      'test_selective_risk_mean', 'calib_selective_risk_mean']].round(4).to_string(index=False))
print('\n' + LIMITS)
print(f'Saved to {NEW} (tables, CSVs, statistical_tables.xlsx) and {FIG} (PNG + PDF). Done in {time.time() - t0:.0f}s.')
