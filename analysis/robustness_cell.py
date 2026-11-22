# ======================================================================================
# Hybrid-K - ROBUSTNESS ANALYSIS: LEAVE-ONE-METRIC-OUT (LOMO) AND DIRICHLET WEIGHT PERTURBATION
#  * CPU only. Calibration data only: reads the saved per-fold normalized metrics, composite scores and
#    Top-K selections from crossfit_composite/ (cross-fitted calibration metrics, seed-averaged).
#  * Normalization is NOT redone: each normalized metric is a per-metric min-max over the nine models,
#    so it does not depend on the weights. Only the weights change.
#  * Reference = the saved equal-weight (1/8) composite and its Top-2/3/4 per outer fold.
#  * LOMO (primary, as specified): drop one metric, equal weights 1/7 on the remaining seven.
#    LOMO (original-script variant, reported for continuity): drop one metric, then 0.5 x mean of the
#    remaining classification metrics + 0.5 x mean of the remaining reliability metrics.
#  * Dirichlet: 5,000 draws from Dirichlet(1,...,1) over the 8 metrics (mean = equal weights), fixed seed
#    20240517, as in the original script. The same 5,000 weight vectors are used for every outer fold.
#  * Ranking: descending score, ties broken by the original model order (same rule as the saved selection).
#  * No test data are read. Nothing is retrained. Outputs go to a NEW folder: hybridk_robustness_analysis/.
# ======================================================================================
import os, json, glob, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau, spearmanr
try:
    from google.colab import drive
    if not os.path.isdir('/content/drive/MyDrive'):
        drive.mount('/content/drive')
except Exception:
    pass

OUT = Path(os.environ.get('HYBRIDK_OUT', '/content/drive/MyDrive/HybridK_Phase1_corrected'))
CFC = OUT / 'crossfit_composite'
NEW = OUT / 'hybridk_robustness_analysis'
MODEL_ORDER = ['Swin-Tiny', 'TinyViT-5M', 'ViT-Tiny', 'DenseNet121', 'VGG16', 'MobileViT-XS',
               'EfficientFormerV2-S0', 'ResNet50', 'EfficientNet-B0']           # original order (tie-break)
METRIC_KEYS = ['acc', 'f1', 'auroc', 'ece', 'wc_ece', 'aurc', 'brier', 'nll']
METRIC_NAME = {'acc': 'Accuracy', 'f1': 'Macro-F1', 'auroc': 'Macro-AUROC', 'ece': 'ECE', 'wc_ece': 'WC-ECE',
               'aurc': 'AURC', 'brier': 'Brier', 'nll': 'NLL'}
CLASS_KEYS, REL_KEYS = ['acc', 'f1', 'auroc'], ['ece', 'wc_ece', 'aurc', 'brier', 'nll']
FOLDS, KS = [0, 1, 2, 3, 4], [2, 3, 4]
N_DRAWS, RNG_SEED = 5000, 20240517
t0 = time.time()

# ---------------------------------------------------------------- 1. read-only protection
def sha(f): return hashlib.sha256(open(f, 'rb').read()).hexdigest()
protected = []
for sub in ('fold_results', 'calibration_corrected', 'crossfit_composite', 'hybridk_ensemble_evaluation'):
    protected += [f for f in sorted(glob.glob(str(OUT / sub / '**' / '*'), recursive=True)) if os.path.isfile(f)]
hash_before = {f: sha(f) for f in protected}
print(f'Protected files: {len(protected)} ({time.time() - t0:.0f}s)')

# ---------------------------------------------------------------- 2. load saved calibration-based results
normm = pd.read_csv(CFC / 'normalized_metrics_per_fold.csv')
scores = pd.read_csv(CFC / 'composite_scores_and_rankings_per_fold.csv')
topk = pd.read_csv(CFC / 'topk_selection_per_fold.csv')
cfg = json.load(open(CFC / 'run_config.json'))
assert cfg.get('data_used', '').startswith('calibration subset only'), 'unexpected source of the composite'
NORM, REF_SCORE, REF_ORDER, REF_TOPK = {}, {}, {}, {}

def order_of(s):                                     # descending, ties -> original model order
    return np.argsort(-np.asarray(s), kind='stable', axis=-1)

for k in FOLDS:
    N = normm[normm.fold == k].set_index('model').reindex(MODEL_ORDER)[[f'{m}_norm' for m in METRIC_KEYS]].values
    assert N.shape == (9, 8) and np.isfinite(N).all() and (N >= -1e-12).all() and (N <= 1 + 1e-12).all()
    sc = scores[scores.fold == k].set_index('model').reindex(MODEL_ORDER)
    s_eq = N @ np.full(8, 1 / 8)
    assert np.allclose(s_eq, sc.composite_score.values, atol=1e-10), f'fold {k}: composite not reproduced'
    o = order_of(s_eq)
    rank = np.empty(9, int); rank[o] = np.arange(1, 10)
    assert np.array_equal(rank, sc['rank'].values), f'fold {k}: ranking not reproduced'
    for K in KS:
        saved = topk[(topk.fold == k) & (topk.K == K)].selected_models.iloc[0].split(' + ')
        assert [MODEL_ORDER[i] for i in o[:K]] == saved, f'fold {k} K {K}: saved selection not reproduced'
    NORM[k], REF_SCORE[k], REF_ORDER[k] = N, s_eq, o
    REF_TOPK[k] = {K: set(o[:K].tolist()) for K in [1] + KS}
print('Saved equal-weight composite, rankings and Top-2/3/4 reproduced exactly from the saved normalized metrics.')

def ranks_from_order(o):                             # 1-D order -> ranks 1..9 (model order)
    r = np.empty(9, int); r[o] = np.arange(1, 10)
    return r
def jacc(a, b): return len(a & b) / len(a | b)
def names(idx): return ' + '.join(MODEL_ORDER[i] for i in idx)

# ---------------------------------------------------------------- 3. LOMO
lomo_rows = []
VARIANTS = {'equal_1/7 (primary)': None, 'original_group_0.5/0.5': 'group'}
for variant, kind in VARIANTS.items():
    for drop in METRIC_KEYS:
        keep = [m for m in METRIC_KEYS if m != drop]
        w = np.zeros(8)
        if kind is None:
            for m in keep: w[METRIC_KEYS.index(m)] = 1 / 7
        else:
            ck, rk = [m for m in CLASS_KEYS if m != drop], [m for m in REL_KEYS if m != drop]
            for m in ck: w[METRIC_KEYS.index(m)] = 0.5 / len(ck)
            for m in rk: w[METRIC_KEYS.index(m)] = 0.5 / len(rk)
        assert abs(w.sum() - 1) < 1e-12 and w[METRIC_KEYS.index(drop)] == 0
        for k in FOLDS:
            s = NORM[k] @ w; o = order_of(s); r = ranks_from_order(o); r0 = ranks_from_order(REF_ORDER[k])
            row = {'variant': variant, 'dropped_metric': METRIC_NAME[drop], 'fold': k,
                   **{f'w_{m}': w[i] for i, m in enumerate(METRIC_KEYS)},
                   'kendall_tau_vs_reference': kendalltau(r, r0)[0], 'spearman_vs_reference': spearmanr(r, r0)[0],
                   'top1': MODEL_ORDER[o[0]], 'top1_same': o[0] == REF_ORDER[k][0]}
            for K in KS:
                sset = set(o[:K].tolist())
                row[f'top{K}'] = names(o[:K]); row[f'top{K}_same_set'] = sset == REF_TOPK[k][K]
                row[f'top{K}_jaccard'] = jacc(sset, REF_TOPK[k][K])
                row[f'top{K}_removed'] = ', '.join(MODEL_ORDER[i] for i in sorted(REF_TOPK[k][K] - sset))
                row[f'top{K}_added'] = ', '.join(MODEL_ORDER[i] for i in sorted(sset - REF_TOPK[k][K]))
            row.update({f'rank_{m}': int(r[i]) for i, m in enumerate(MODEL_ORDER)})
            lomo_rows.append(row)
lomo = pd.DataFrame(lomo_rows)

# ---------------------------------------------------------------- 4. Dirichlet weight perturbation
rng = np.random.default_rng(RNG_SEED)
W = rng.dirichlet(np.ones(8), size=N_DRAWS)
assert np.allclose(W.sum(1), 1) and W.shape == (N_DRAWS, 8)
draw_rows, freq_rows, set_rows, rank_rows = [], [], [], []
for k in FOLDS:
    S = W @ NORM[k].T                                # (draws, 9)
    O = order_of(S)                                  # stable per row -> original-order tie-break
    R = np.empty_like(O); np.put_along_axis(R, O, np.tile(np.arange(1, 10), (N_DRAWS, 1)), axis=1)
    r0 = ranks_from_order(REF_ORDER[k])
    taus = np.array([kendalltau(R[d], r0)[0] for d in range(N_DRAWS)])
    rhos = 1 - 6 * ((R - r0) ** 2).sum(1) / (9 * (81 - 1))       # Spearman (no ties in ranks)
    rec = {'fold': k, 'draw': np.arange(N_DRAWS), 'kendall_tau': taus, 'spearman': rhos}
    for i, m in enumerate(METRIC_KEYS): rec[f'w_{m}'] = W[:, i]
    for K in [1] + KS:
        sets = [frozenset(O[d, :K].tolist()) for d in range(N_DRAWS)]
        same = np.array([s == REF_TOPK[k][K] for s in sets])
        jac = np.array([jacc(set(s), REF_TOPK[k][K]) for s in sets])
        rec[f'top{K}_same_as_reference'] = same; rec[f'top{K}_jaccard'] = jac
        rec[f'top{K}_set'] = [names(sorted(s, key=lambda i: R[d, i])) for d, s in enumerate(sets)]
        cnt = pd.Series(sets).value_counts()
        modal = cnt.index[0]
        set_rows.append({'fold': k, 'K': K, 'reference_set': names(REF_ORDER[k][:K]),
                         'reference_retained_pct': 100 * same.mean(), 'mean_jaccard': jac.mean(),
                         'distinct_sets': len(cnt), 'modal_set': names(sorted(modal, key=lambda i: r0[i])),
                         'modal_pct': 100 * cnt.iloc[0] / N_DRAWS, 'modal_equals_reference': modal == REF_TOPK[k][K]})
        sel = np.zeros(9)
        for s in sets:
            for i in s: sel[i] += 1
        for i, m in enumerate(MODEL_ORDER):
            freq_rows.append({'fold': k, 'K': K, 'model': m, 'selected_pct': 100 * sel[i] / N_DRAWS,
                              'in_reference': i in REF_TOPK[k][K]})
    for i, m in enumerate(MODEL_ORDER):
        rank_rows.append({'fold': k, 'model': m, 'reference_rank': int(r0[i]), 'mean_rank': R[:, i].mean(),
                          'sd_rank': R[:, i].std(ddof=1), 'min_rank': int(R[:, i].min()), 'max_rank': int(R[:, i].max()),
                          'pct_rank_1': 100 * (R[:, i] == 1).mean()})
    draw_rows.append(pd.DataFrame(rec))
    print(f'  fold {k}: Dirichlet draws done ({time.time() - t0:.0f}s)')
draws = pd.concat(draw_rows, ignore_index=True)
dsets, dfreq, drank = pd.DataFrame(set_rows), pd.DataFrame(freq_rows), pd.DataFrame(rank_rows)

# ---------------------------------------------------------------- 5. publication-ready summary tables
lp = lomo[lomo.variant == 'equal_1/7 (primary)']
T1 = pd.DataFrame([{'Dropped metric': METRIC_NAME[d], **{f'Top-{K} unchanged (folds of 5)':
                    int(lp[lp.dropped_metric == METRIC_NAME[d]][f'top{K}_same_set'].sum()) for K in KS},
                    **{f'Top-{K} mean Jaccard': round(lp[lp.dropped_metric == METRIC_NAME[d]][f'top{K}_jaccard'].mean(), 3)
                       for K in KS},
                    'Top-1 unchanged (folds of 5)': int(lp[lp.dropped_metric == METRIC_NAME[d]].top1_same.sum()),
                    "Kendall's tau vs reference, mean (min)":
                        f"{lp[lp.dropped_metric == METRIC_NAME[d]].kendall_tau_vs_reference.mean():.3f} "
                        f"({lp[lp.dropped_metric == METRIC_NAME[d]].kendall_tau_vs_reference.min():.3f})"}
                   for d in METRIC_KEYS])
lo = lomo[lomo.variant == 'original_group_0.5/0.5']
T1b = pd.DataFrame([{'Dropped metric': METRIC_NAME[d], **{f'Top-{K} unchanged (folds of 5)':
                     int(lo[lo.dropped_metric == METRIC_NAME[d]][f'top{K}_same_set'].sum()) for K in KS}}
                    for d in METRIC_KEYS])
T2 = dsets[dsets.K.isin(KS)].assign(**{
    'Reference retained (%)': lambda d: d.reference_retained_pct.round(1), 'Mean Jaccard': lambda d: d.mean_jaccard.round(3),
    'Modal (%)': lambda d: d.modal_pct.round(1)})[['fold', 'K', 'reference_set', 'Reference retained (%)', 'Mean Jaccard',
                                                  'distinct_sets', 'modal_set', 'Modal (%)', 'modal_equals_reference']]
T2.columns = ['Fold', 'K', 'Reference (equal-weight) set', 'Reference retained (%)', 'Mean Jaccard',
              'Distinct sets', 'Modal set', 'Modal (%)', 'Modal = reference']
ref_count = {(m, K): sum(MODEL_ORDER.index(m) in REF_TOPK[k][K] for k in FOLDS) for m in MODEL_ORDER for K in [1] + KS}
T3 = (dfreq.groupby(['model', 'K']).selected_pct.mean().unstack('K').reindex(MODEL_ORDER))
T3.columns = [f'Top-{K} selected, Dirichlet (% of draws, mean over folds)' if K > 1
              else 'Rank 1, Dirichlet (% of draws, mean over folds)' for K in T3.columns]
for K in [1] + KS:
    T3[f'{"Rank 1" if K == 1 else f"Top-{K}"} in reference (folds of 5)'] = [ref_count[(m, K)] for m in T3.index]
T3['Mean rank, Dirichlet (mean over folds)'] = drank.groupby('model').mean_rank.mean().reindex(T3.index)
T3 = T3.sort_values('Mean rank, Dirichlet (mean over folds)').round(2)
T4 = draws.groupby('fold').agg(tau_mean=('kendall_tau', 'mean'), tau_p05=('kendall_tau', lambda x: np.percentile(x, 5)),
                               tau_p95=('kendall_tau', lambda x: np.percentile(x, 95)), rho_mean=('spearman', 'mean'),
                               rho_p05=('spearman', lambda x: np.percentile(x, 5)))
T4.loc['all folds'] = [draws.kendall_tau.mean(), np.percentile(draws.kendall_tau, 5), np.percentile(draws.kendall_tau, 95),
                       draws.spearman.mean(), np.percentile(draws.spearman, 5)]
T4 = T4.round(3); T4.columns = ["Kendall's tau, mean", "Kendall's tau, P5", "Kendall's tau, P95",
                                "Spearman's rho, mean", "Spearman's rho, P5"]
T0 = pd.DataFrame([{'Fold': k, **{f'Top-{K}': names(REF_ORDER[k][:K]) for K in KS}} for k in FOLDS])

# ---------------------------------------------------------------- 6. save
NEW.mkdir(parents=True, exist_ok=True)
lomo.to_csv(NEW / 'lomo_detailed.csv', index=False)
draws.to_csv(NEW / 'dirichlet_draws_detailed.csv.gz', index=False, compression='gzip')
dsets.to_csv(NEW / 'dirichlet_set_stability_per_fold.csv', index=False)
dfreq.to_csv(NEW / 'dirichlet_model_selection_frequency_per_fold.csv', index=False)
drank.to_csv(NEW / 'dirichlet_rank_distribution_per_fold.csv', index=False)
pd.DataFrame(W, columns=[f'w_{m}' for m in METRIC_KEYS]).to_csv(NEW / 'dirichlet_weights.csv', index_label='draw')
tables = {'T0_reference_selection': T0, 'T1_LOMO_equal_weights': T1, 'T1b_LOMO_original_group_variant': T1b,
          'T2_Dirichlet_set_stability': T2, 'T3_model_selection_frequency': T3.reset_index().rename(columns={'model': 'Model'}),
          'T4_ranking_agreement': T4.reset_index().rename(columns={'fold': 'Fold'})}
for n, t in tables.items(): t.to_csv(NEW / f'{n}.csv', index=False)
try:
    with pd.ExcelWriter(NEW / 'robustness_summary_tables.xlsx') as xw:
        for n, t in tables.items(): t.to_excel(xw, sheet_name=n[:31], index=False)
except Exception as e:
    print('  Excel export skipped:', e)
json.dump({'source': str(CFC), 'data': 'calibration-based normalized metrics (cross-fitted, seed-averaged); no test data',
           'reference': 'saved equal-weight composite (1/8 per metric) and its Top-K per fold',
           'normalization': 'unchanged (saved per-metric min-max over the nine models)',
           'lomo_primary': 'drop one metric, 1/7 on each remaining metric',
           'lomo_original_variant': 'drop one metric, 0.5 x mean(remaining classification) + 0.5 x mean(remaining reliability), '
                                    'as in the original script',
           'dirichlet': {'alpha': [1.0] * 8, 'draws': N_DRAWS, 'seed': RNG_SEED, 'generator': 'numpy default_rng',
                         'same_weights_for_all_folds': True},
           'tie_break': 'original model order (stable sort)', 'model_order': MODEL_ORDER,
           'agreement': "Kendall's tau-b and Spearman's rho between full 9-model rankings and the reference ranking; "
                        'Jaccard index between selected sets and the reference set'},
          open(NEW / 'run_config.json', 'w'), indent=2)

# ---------------------------------------------------------------- 7. verification and report
hash_after = {f: sha(f) for f in protected}
changed = [f for f in protected if hash_before[f] != hash_after[f]]
print('\nVERIFICATION')
print(f'  LOMO evaluations: {len(lomo)} (expected 2 variants x 8 metrics x 5 folds = 80)')
print(f'  Dirichlet evaluations: {len(draws)} (expected {N_DRAWS} x 5 folds = {N_DRAWS * 5})')
print(f'  protected files modified: {len(changed)} (must be 0); test data read: none')
assert len(lomo) == 80 and len(draws) == N_DRAWS * 5 and not changed

pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
print('\nREFERENCE (equal-weight) SELECTIONS'); print(T0.to_string(index=False))
print('\nLOMO, equal weights on the remaining seven metrics'); print(T1.to_string(index=False))
print('\nLOMO, original-script group-weighted variant'); print(T1b.to_string(index=False))
print('\nDIRICHLET SET STABILITY (5,000 draws per fold)'); print(T2.to_string(index=False))
print('\nMODEL SELECTION FREQUENCY'); print(T3.to_string())
print('\nRANKING AGREEMENT WITH THE REFERENCE (Dirichlet draws)'); print(T4.to_string())
print(f'\nSaved to {NEW}:')
for f in sorted(NEW.iterdir()): print('  ', f.name)
print(f'Done in {time.time() - t0:.0f}s. Calibration data only. Test results untouched.')
