# ======================================================================================
# Hybrid-K - EMPIRICAL RISK-TARGETED SELECTIVE PREDICTION (EXPLORATORY; CPU only; no retraining)
#  Framework stages, all fitted on development (calibration) data, then applied once to the test folds:
#    member selection -> ensemble temperature (both frozen, from hybridk_ensemble_evaluation/)
#    -> confidence threshold tau chosen for an application-defined target selective risk alpha.
#  Target risk alpha (requirement) is distinct from the threshold tau (estimated from calibration data).
#  Threshold rule (fixed in advance): candidate tau = every distinct calibration confidence (full resolution);
#    choose the lowest tau (largest calibration coverage) whose one-sided 90% Clopper-Pearson upper bound on the
#    calibration selective risk is <= alpha; if none qualifies, refer every cell.
#  NO FORMAL RISK GUARANTEE is claimed: the calibration subset also selected the members and fitted the temperature,
#  cells are clustered within parent images, and the 90% bound is applied without multiplicity correction.
#  EXPLORATORY: the test folds were examined in earlier analyses, and this design was chosen after a previous
#  test-half analysis.
# ======================================================================================
import os, json, glob, hashlib, time, datetime
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
RUNS, ENS = OUT / 'fold_results', OUT / 'hybridk_ensemble_evaluation'
NEW = OUT / 'risk_targeting_exploratory'
assert not (NEW / 'stageA_thresholds_calibration.csv').exists(), f'{NEW} already holds results; rename it deliberately to re-run.'
NEW.mkdir(parents=True, exist_ok=True)
MODELS = ['swin_tiny_patch4_window7_224', 'tiny_vit_5m_224', 'vit_tiny_patch16_224', 'densenet121', 'vgg16',
          'mobilevit_xs', 'efficientformerv2_s0', 'resnet50', 'efficientnet_b0']
PRETTY = {'swin_tiny_patch4_window7_224': 'Swin-Tiny', 'tiny_vit_5m_224': 'TinyViT-5M', 'vit_tiny_patch16_224': 'ViT-Tiny',
          'densenet121': 'DenseNet121', 'vgg16': 'VGG16', 'mobilevit_xs': 'MobileViT-XS',
          'efficientformerv2_s0': 'EfficientFormerV2-S0', 'resnet50': 'ResNet50', 'efficientnet_b0': 'EfficientNet-B0'}
KEY = {v: k for k, v in PRETTY.items()}
SEEDS, FOLDS, KS = [42, 123, 456], [0, 1, 2, 3, 4], [2, 3, 4]
CONFIGS = ['Hybrid-2', 'Hybrid-3', 'Hybrid-4', 'Accuracy Top-2', 'Accuracy Top-3', 'Accuracy Top-4',
           'All-9 equal-weight', 'Best single (composite)']
ALPHAS = [round(0.01 * i, 2) for i in range(1, 11)]
BOUND_LEVEL, EPS = 0.90, 1e-12
TEST_ALPHAS = [0.03, 0.05]
COMPARISONS = ([(c, 'Best single (composite)') for c in CONFIGS[:7]] + [(f'Hybrid-{K}', f'Accuracy Top-{K}') for K in KS])
t0 = time.time()
json.dump({'status': 'EXPLORATORY; no formal risk guarantee; test folds examined earlier; design chosen after a previous test-half analysis',
           'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
           'target_risk_alpha': ALPHAS, 'definition': 'alpha = maximum acceptable error rate among automatically accepted cells',
           'threshold_rule': 'candidate tau = every distinct calibration confidence; lowest tau with one-sided 90% Clopper-Pearson '
                             'upper bound on calibration selective risk <= alpha; refer all if none qualifies',
           'score': 'max probability of the frozen ensemble with its frozen full-calibration temperature',
           'configurations': CONFIGS, 'test_use': 'thresholds frozen and saved before test predictions are read; test applied once',
           'outcomes': ['test coverage', 'referral rate', 'achieved test selective risk', 'target met (achieved <= alpha)', 'tau'],
           'tests': {'alphas': TEST_ALPHAS, 'outcomes': ['coverage', 'selective_risk'], 'comparisons': [f'{a} vs {b}' for a, b in COMPARISONS],
                     'method': 'seeds averaged within fold; Nadeau-Bengio corrected paired t (df=4); Holm within alpha x outcome'},
           'not_claimed': ['formal risk guarantee', 'superiority of Hybrid-K']}, open(NEW / 'protocol.json', 'w'), indent=2)

def sha(f): return hashlib.sha256(open(f, 'rb').read()).hexdigest()
protected = [f for sub in ('fold_results', 'calibration_corrected', 'crossfit_composite', 'hybridk_ensemble_evaluation',
                           'hybridk_robustness_analysis', 'hybridk_statistical_analysis', 'risk_control_crossfit', 'risk_control_experiment')
             for f in sorted(glob.glob(str(OUT / sub / '**' / '*'), recursive=True)) if os.path.isfile(f)]
hash_before = {f: sha(f) for f in protected}

def clean(p):
    p = np.clip(np.nan_to_num(p), 0, None); s = p.sum(1, keepdims=True); return p / np.where(s < 1e-12, 1, s)
def ens_prob(prob_list, T):
    lg = np.log(np.clip(np.mean([clean(p) for p in prob_list], axis=0), EPS, 1.0)); return softmax(lg / T, axis=1)
def choose_tau(conf, corr, alpha):
    o = np.argsort(-conf, kind='stable'); c = conf[o]; err = np.cumsum(~corr[o]); n = np.arange(1, len(c) + 1)
    last = np.r_[c[1:] != c[:-1], True]                  # positions where all cells with this confidence are included
    c, err, n = c[last], err[last], n[last]
    ucb = np.where(err >= n, 1.0, stats.beta.ppf(BOUND_LEVEL, err + 1, np.maximum(n - err, 1)))
    ok = np.where(ucb <= alpha)[0]
    if not len(ok): return np.inf, 0.0, np.nan, np.nan
    i = ok.max()                                          # largest calibration coverage that qualifies
    return float(c[i]), n[i] / len(conf), err[i] / n[i], float(ucb[i])

frozen = pd.read_csv(ENS / 'frozen_configurations_calibration_only.csv'); frozen = frozen[frozen.config.isin(CONFIGS)]
assert len(frozen) == len(CONFIGS) * 15, 'frozen configurations incomplete'
key = lambda cls, fns: [f'{c}/{f}' for c, f in zip(cls, fns)]

# ---------------- Stage A: calibration data only
CAL = {}
for m in MODELS:
    for s in SEEDS:
        for k in FOLDS:
            d = json.load(open(RUNS / f'multiclass__{m}__rep{s}__fold{k}.json'))
            ca = d['calib']; ck = key(ca['classes'], ca['filenames']); o = np.argsort(ck)
            CAL[(m, s, k)] = ([ck[i] for i in o], np.asarray(ca['y_true'], int)[o], softmax(np.asarray(ca['logits'], float)[o], axis=1))
            del d
thr_rows = []
for r in frozen.itertuples():
    k, s = int(r.fold), int(r.seed); mem = [KEY[x] for x in r.members.split(' + ')]
    ck0, y, _ = CAL[(mem[0], s, k)]
    for m in mem: assert CAL[(m, s, k)][0] == ck0 and np.array_equal(CAL[(m, s, k)][1], y)
    p = ens_prob([CAL[(m, s, k)][2] for m in mem], float(r.T)); conf, corr = p.max(1), p.argmax(1) == y
    for a in ALPHAS:
        tau, cov, risk, ucb = choose_tau(conf, corr, a)
        thr_rows.append({'config': r.config, 'fold': k, 'seed': s, 'alpha': a, 'tau': tau, 'calib_coverage': cov,
                         'calib_selective_risk': risk, 'calib_ucb90': ucb, 'n_calib': len(y)})
thr = pd.DataFrame(thr_rows); thr.to_csv(NEW / 'stageA_thresholds_calibration.csv', index=False)
FSHA = sha(NEW / 'stageA_thresholds_calibration.csv')
print(f'[{time.time() - t0:.0f}s] Stage A: {len(thr)} thresholds chosen on calibration data and saved (sha256 {FSHA[:12]}...)')

# ---------------- Stage B: apply frozen thresholds to the test folds once
TE = {}
for m in MODELS:
    for s in SEEDS:
        for k in FOLDS:
            d = json.load(open(RUNS / f'multiclass__{m}__rep{s}__fold{k}.json'))
            tk = key(d['classes'], d['filenames']); o = np.argsort(tk)
            TE[(m, s, k)] = ([tk[i] for i in o], np.asarray(d['y_true'], int)[o], np.asarray(d['y_prob'], float)[o]); del d
preds = pd.read_csv(ENS / 'test_predictions_all_configs.csv.gz'); PC = [c for c in preds.columns if c.startswith('p_calibrated__')]
rows, maxdiff = [], 0.0
for r in frozen.itertuples():
    k, s = int(r.fold), int(r.seed); mem = [KEY[x] for x in r.members.split(' + ')]
    tk0, y, _ = TE[(mem[0], s, k)]
    for m in mem: assert TE[(m, s, k)][0] == tk0 and np.array_equal(TE[(m, s, k)][1], y)
    p = ens_prob([TE[(m, s, k)][2] for m in mem], float(r.T))
    g = preds[(preds.config == r.config) & (preds.fold == k) & (preds.seed == s)]
    gk = key(g['class'], g.filename); go = np.argsort(gk); assert [gk[i] for i in go] == tk0
    maxdiff = max(maxdiff, float(np.abs(g[PC].values[go] - p).max()))
    conf, corr = p.max(1), p.argmax(1) == y
    for t in thr[(thr.config == r.config) & (thr.fold == k) & (thr.seed == s)].itertuples():
        acc = conf >= t.tau; na = int(acc.sum())
        rows.append({'config': r.config, 'fold': k, 'seed': s, 'alpha': t.alpha, 'tau': t.tau, 'calib_coverage': t.calib_coverage,
                     'test_coverage': acc.mean(), 'referral_rate': 1 - acc.mean(), 'n_accepted': na,
                     'test_selective_risk': float((~corr[acc]).mean()) if na else np.nan,
                     'target_met': bool(na == 0 or (~corr[acc]).mean() <= t.alpha)})
assert maxdiff < 1e-6, f'rebuilt probabilities differ from saved predictions ({maxdiff:.1e})'
assert sha(NEW / 'stageA_thresholds_calibration.csv') == FSHA
res = pd.DataFrame(rows); res.to_csv(NEW / 'stageB_test_results_per_run.csv', index=False)

# ---------------- summaries
fm = res.groupby(['alpha', 'config', 'fold'])[['test_coverage', 'test_selective_risk', 'referral_rate']].mean()
summ = res.groupby(['alpha', 'config']).agg(coverage=('test_coverage', 'mean'), referral=('referral_rate', 'mean'),
                                            selective_risk=('test_selective_risk', 'mean'), runs_target_met=('target_met', 'sum'),
                                            runs_refer_all=('n_accepted', lambda x: int((x == 0).sum())),
                                            tau_median=('tau', lambda x: float(np.median(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan),
                                            tau_min=('tau', lambda x: float(x[np.isfinite(x)].min()) if np.isfinite(x).any() else np.nan),
                                            tau_max=('tau', lambda x: float(x[np.isfinite(x)].max()) if np.isfinite(x).any() else np.nan)).reset_index()
sd = fm.groupby(['alpha', 'config']).std(ddof=1).rename(columns={'test_coverage': 'coverage_sd_fold', 'test_selective_risk': 'selective_risk_sd_fold',
                                                                 'referral_rate': 'referral_sd_fold'}).reset_index()
summ = summ.merge(sd, on=['alpha', 'config']); summ.to_csv(NEW / 'summary_all_targets.csv', index=False)
compact = summ[summ.alpha.isin([0.01, 0.03, 0.05, 0.10])].copy()
compact['Coverage'] = [f'{c:.3f} ± {s:.3f}' for c, s in zip(compact.coverage, compact.coverage_sd_fold)]
compact['Selective risk'] = ['n/a' if not np.isfinite(r) else f'{r:.4f} ± {s:.4f}' for r, s in zip(compact.selective_risk, compact.selective_risk_sd_fold)]
compact['Target met (runs/15)'] = compact.runs_target_met.astype(int)
compact['tau median [range]'] = ['refer all' if not np.isfinite(m) else f'{m:.3f} [{lo:.3f}, {hi:.3f}]' for m, lo, hi in
                                 zip(compact.tau_median, compact.tau_min, compact.tau_max)]
compact = compact[['alpha', 'config', 'Coverage', 'Selective risk', 'Target met (runs/15)', 'tau median [range]']]
compact.to_csv(NEW / 'Table_compact_alpha_1_3_5_10.csv', index=False)

RHO = float(np.mean([(lambda sp: sp['n_test'] / sp['n_train'])(json.load(open(RUNS / f'multiclass__{MODELS[0]}__rep42__fold{k}.json'))['split']) for k in FOLDS]))
def holm(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = ~np.isnan(p); q = p[ok]; o = np.argsort(q); run = 0; adj = np.empty(len(q))
    for i_, j in enumerate(o): run = max(run, min(1, (len(q) - i_) * q[j])); adj[j] = run
    out[ok] = adj; return out
crow = []
for a in TEST_ALPHAS:
    for A_, B_ in COMPARISONS:
        for outc, col in [('coverage', 'test_coverage'), ('selective_risk', 'test_selective_risk')]:
            A = fm.loc[(a, A_)][col].reindex(FOLDS).values; B = fm.loc[(a, B_)][col].reindex(FOLDS).values
            d = (A - B)[~np.isnan(A - B)]; n = len(d)
            mu = d.mean() if n else np.nan; s_ = d.std(ddof=1) if n > 1 else 0.0; se = s_ * np.sqrt(1 / max(n, 1) + RHO)
            tq = stats.t.ppf(0.975, max(n - 1, 1))
            crow.append({'alpha': a, 'outcome': outc, 'comparison': f'{A_} vs {B_}', 'n_folds': n, 'mean_A': np.nanmean(A), 'mean_B': np.nanmean(B),
                         'mean_diff': mu, 'ci95_low': mu - tq * se if s_ > 0 else mu, 'ci95_high': mu + tq * se if s_ > 0 else mu,
                         'p_corrected_t': 2 * stats.t.sf(abs(mu / se), n - 1) if s_ > 0 and n > 1 else np.nan,
                         'folds_A_higher': int((d > 1e-12).sum())})
cmp = pd.DataFrame(crow)
cmp['p_holm'] = cmp.groupby(['alpha', 'outcome']).p_corrected_t.transform(lambda p: pd.Series(holm(p.values), index=p.index))
cmp['finding'] = np.where(cmp.p_holm < 0.05, 'statistically supported', np.where((cmp.folds_A_higher == cmp.n_folds) | (cmp.folds_A_higher == 0),
                                                                                  'consistent direction in all folds (descriptive)', 'inconclusive'))
cmp.to_csv(NEW / 'fold_level_comparisons_alpha_3_5.csv', index=False)

# ---------------- two-panel sensitivity figure
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
def style(c):
    if c.startswith('Hybrid'): return '#2a78d6', {'2': ':', '3': '--', '4': '-'}[c[-1]], {'2': 'o', '3': 's', '4': '^'}[c[-1]]
    if c.startswith('Accuracy'): return '#eb6834', {'2': ':', '3': '--', '4': '-'}[c[-1]], {'2': 'o', '3': 's', '4': '^'}[c[-1]]
    if c.startswith('All'): return '#1baf7a', '-', 'D'
    return '#52514e', '-', 'v'
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
xs = np.array(ALPHAS) * 100
for c in CONFIGS:
    col, ls, mk = style(c); t = summ[summ.config == c].set_index('alpha').reindex(ALPHAS)
    axs[0].plot(xs, t.coverage.values * 100, ls=ls, marker=mk, color=col, lw=2, ms=5, mec='white', label=c)
    axs[1].plot(xs, t.selective_risk.values * 100, ls=ls, marker=mk, color=col, lw=2, ms=5, mec='white', label=c)
axs[1].plot(xs, xs, color='#0b0b0b', lw=0.9, ls='-', label='achieved = target')
axs[0].set_xlabel('target selective risk α (%)'); axs[0].set_ylabel('test coverage (%)  [referral = 100 − coverage]')
axs[1].set_xlabel('target selective risk α (%)'); axs[1].set_ylabel('achieved test selective risk (%)')
axs[0].set_title('Coverage at each target'); axs[1].set_title('Achieved selective risk versus target')
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=8)
fig.suptitle('Empirical risk targeting (exploratory): thresholds chosen on calibration data, applied once to test folds; '
             'mean over 15 runs. No formal risk guarantee.', fontsize=10)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(NEW / 'Fig_risk_targeting_sensitivity.png'); fig.savefig(NEW / 'Fig_risk_targeting_sensitivity.pdf'); plt.close(fig)

# ---------------- report
changed = [f for f in protected if sha(f) != hash_before[f]]; assert not changed
pd.set_option('display.width', 240); pd.set_option('display.max_colwidth', 40)
print(f'\nVERIFICATION: {len(res)} test evaluations (8 configurations x 15 runs x 10 targets); rebuilt probabilities max diff {maxdiff:.1e}; '
      f'thresholds unchanged after test (sha256 match); protected files modified: {len(changed)}')
print('\nCOVERAGE (mean over 15 runs) by target'); print(summ.pivot(index='config', columns='alpha', values='coverage').reindex(CONFIGS).round(3).to_string())
print('\nACHIEVED SELECTIVE RISK (mean over runs with accepted cells) by target'); print(summ.pivot(index='config', columns='alpha', values='selective_risk').reindex(CONFIGS).round(4).to_string())
print('\nRUNS MEETING THE TARGET (of 15) by target'); print(summ.pivot(index='config', columns='alpha', values='runs_target_met').reindex(CONFIGS).astype(int).to_string())
print('\nCOMPACT TABLE (alpha = 1, 3, 5, 10%)'); print(compact.to_string(index=False))
print('\nFOLD-LEVEL COMPARISONS AT 3% AND 5% (A minus B; Holm within target and outcome)')
print(cmp.round(4)[['alpha', 'outcome', 'comparison', 'n_folds', 'mean_diff', 'ci95_low', 'ci95_high', 'p_holm', 'finding']].to_string(index=False))
print('\nEXPLORATORY: thresholds are estimated from calibration data that also selected members and fitted temperatures; cells are '
      'clustered within parent images; no formal risk guarantee is claimed.')
print(f'Saved to {NEW}. Done in {time.time() - t0:.0f}s.')
