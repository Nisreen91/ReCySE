# ======================================================================================
# Hybrid-K - GROUPED CROSS-FITTING + RELIABILITY-AWARE COMPOSITE SCORE + TOP-K SELECTION
#  * CPU only. No retraining. Reads only the saved run files (read-only, SHA-256 checked).
#  * Uses ONLY the calibration subset of each run (logits, labels, class names, parent groups).
#    Test-fold predictions, labels and metrics are never read into the analysis.
#  * Temperature scaling inside each calibration subset is cross-fitted: 5 parent-image-grouped,
#    class-stratified inner folds (fixed seed per outer fold, identical for all models and seeds).
#    T is fitted on 4 inner folds (exact log-sum-exp NLL, beta = 1/T in [0.2, 4.0], same corrected
#    procedure as the calibration-correction notebook) and applied to the held-out inner fold.
#  * Eight metrics on the out-of-fold calibrated probabilities; NLL is the exact (unclipped) NLL.
#  * Original Hybrid-K composite: norm_higher / norm_lower min-max over the nine models,
#    equal weights 1/8 = 0.125, rank by descending score, Top-2/3/4. Per outer fold, after
#    averaging each metric over the three seeds.
#  * No ensembles are built or evaluated. Outputs go to a NEW folder: crossfit_composite/.
# ======================================================================================
import os, json, glob, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
try:
    from google.colab import drive
    if not os.path.isdir('/content/drive/MyDrive'):
        drive.mount('/content/drive')
except Exception:
    pass

OUT = Path(os.environ.get('HYBRIDK_OUT', '/content/drive/MyDrive/HybridK_Phase1_corrected'))
RUNS = OUT / 'fold_results'
NEW = OUT / 'crossfit_composite'
# original Hybrid-K model order (ALL_NINE); also used as the deterministic tie-break order
MODELS = ['swin_tiny_patch4_window7_224', 'tiny_vit_5m_224', 'vit_tiny_patch16_224', 'densenet121', 'vgg16',
          'mobilevit_xs', 'efficientformerv2_s0', 'resnet50', 'efficientnet_b0']
PRETTY = {'swin_tiny_patch4_window7_224': 'Swin-Tiny', 'tiny_vit_5m_224': 'TinyViT-5M',
          'vit_tiny_patch16_224': 'ViT-Tiny', 'densenet121': 'DenseNet121', 'vgg16': 'VGG16',
          'mobilevit_xs': 'MobileViT-XS', 'efficientformerv2_s0': 'EfficientFormerV2-S0',
          'resnet50': 'ResNet50', 'efficientnet_b0': 'EfficientNet-B0'}
SEEDS, FOLDS, KS = [42, 123, 456], [0, 1, 2, 3, 4], [2, 3, 4]
N_INNER = 5
CF_SEED = lambda fold: 1000 + fold          # fixed, reproducible inner split per outer fold
T_LO, T_HI = 0.25, 5.0
GRID = np.geomspace(1 / T_HI, 1 / T_LO, 4001)
METRIC_KEYS = ['acc', 'f1', 'auroc', 'ece', 'wc_ece', 'aurc', 'brier', 'nll']
DIRECTION = {'acc': 'higher', 'f1': 'higher', 'auroc': 'higher', 'ece': 'lower', 'wc_ece': 'lower',
             'aurc': 'lower', 'brier': 'lower', 'nll': 'lower'}
WEIGHTS = {k: 0.125 for k in METRIC_KEYS}
assert abs(sum(WEIGHTS.values()) - 1) < 1e-12
t0 = time.time()

# ---------------------------------------------------------------- original eight metrics (unchanged)
def _clean(p):
    p = np.nan_to_num(np.asarray(p, float), nan=0.0, posinf=0.0, neginf=0.0); p = np.clip(p, 0, None)
    s = p.sum(1, keepdims=True); p = p / np.where(s < 1e-12, 1, s); z = p.sum(1) < 1e-12
    if z.any(): p[z] = 1 / p.shape[1]
    return p
def _ece_on(conf, corr, nb=15):
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (conf > b[i]) & (conf <= b[i + 1])
        if m.sum(): e += m.sum() / len(conf) * abs(corr[m].mean() - conf[m].mean())
    return e
def m_acc(y, p): return float((_clean(p).argmax(1) == y).mean())
def m_f1(y, p):
    p = _clean(p); q = p.argmax(1); f = []
    for c in range(p.shape[1]):
        tp = ((q == c) & (y == c)).sum(); fp = ((q == c) & (y != c)).sum(); fn = ((q != c) & (y == c)).sum()
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f))
def m_ece(y, p): p = _clean(p); return float(_ece_on(p.max(1), (p.argmax(1) == y).astype(float)))
def m_wc_ece(y, p):
    p = _clean(p)
    return float(np.nanmax([_ece_on(p[y == c].max(1), (p[y == c].argmax(1) == c).astype(float)) if (y == c).any()
                            else np.nan for c in range(p.shape[1])]))
def m_brier(y, p): p = _clean(p); return float(np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, 1)))
def m_aurc(y, p):
    p = _clean(p); conf = p.max(1); corr = (p.argmax(1) == y).astype(float); o = np.argsort(-conf)
    risk = 1 - np.cumsum(corr[o]) / np.arange(1, len(y) + 1); cov = np.arange(1, len(y) + 1) / len(y)
    return float((np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(risk, cov))
def m_auroc(y, p):
    try: return float(roc_auc_score(y, _clean(p), multi_class='ovr', average='macro'))
    except Exception: return float('nan')

# ---------------------------------------------------------------- corrected temperature fit (same as correction notebook)
def exact_nll_vec(lg, y, beta):
    z = beta * lg
    return logsumexp(z, axis=1) - z[np.arange(len(y)), y]
def fit_beta(lg, y):
    vals = np.empty(len(GRID)); yi = np.arange(len(y))
    for a in range(0, len(GRID), 500):
        z = GRID[a:a + 500, None, None] * lg[None]
        vals[a:a + 500] = (logsumexp(z, axis=2) - z[:, yi, y]).mean(axis=1)
    i = int(np.argmin(vals)); lo, hi = GRID[max(i - 1, 0)], GRID[min(i + 1, len(GRID) - 1)]
    b = GRID[i]
    if hi > lo:
        r = minimize_scalar(lambda t: exact_nll_vec(lg, y, t).mean(), bounds=(lo, hi), method='bounded',
                            options={'xatol': 1e-12})
        if r.fun <= vals[i]: b = r.x
    guard = exact_nll_vec(lg, y, b).mean() > exact_nll_vec(lg, y, 1.0).mean()
    return (1.0, True) if guard else (float(b), False)

# ---------------------------------------------------------------- 1. read-only protection
def sha(f): return hashlib.sha256(open(f, 'rb').read()).hexdigest()
files = {(m, s, k): RUNS / f'multiclass__{m}__rep{s}__fold{k}.json' for m in MODELS for s in SEEDS for k in FOLDS}
missing = [str(f) for f in files.values() if not f.exists()]
assert not missing, f'{len(missing)} run files missing, e.g. {missing[:3]}'
protected = [str(f) for f in files.values()]
cc_dir = OUT / 'calibration_corrected'
cc_files = sorted(glob.glob(str(cc_dir / '**' / '*'), recursive=True))
cc_files = [f for f in cc_files if os.path.isfile(f)]
protected += cc_files
hash_before = {f: sha(f) for f in protected}
print(f'Run files found: {len(files)}/135 | corrected-calibration files protected: {len(cc_files)}')
if not cc_files:
    print('  NOTE: calibration_corrected/ not found or empty. Run the calibration-correction notebook first '
          'if it has not been verified.')

# ---------------------------------------------------------------- 2. load CALIBRATION data only
calib = {}
for (m, s, k), f in files.items():
    d = json.load(open(f))
    c = d['calib']
    cn = list(d['class_names'])
    assert d['model_name'] == m and int(d['replication_seed']) == s and int(d['fold_idx']) == k, f'header mismatch {f}'
    test_groups = set(d['group_ids'])                              # identifiers only, for the disjointness check
    calib[(m, s, k)] = {'filenames': list(c['filenames']), 'classes': list(c['classes']),
                        'group_ids': list(c['group_ids']), 'y': np.asarray(c['y_true'], int),
                        'logits': np.asarray(c['logits'], float), 'class_names': cn,
                        'disjoint_from_test': len(test_groups & set(c['group_ids'])) == 0}
    del d, test_groups                                             # nothing from the test fold is kept

# ---------------------------------------------------------------- 3. alignment checks
checks = []
CLASS_NAMES = calib[(MODELS[0], SEEDS[0], FOLDS[0])]['class_names']
for k in FOLDS:
    ref = calib[(MODELS[0], SEEDS[0], k)]
    key_of = lambda r: [f'{c}/{fn}' for c, fn in zip(r['classes'], r['filenames'])]   # filenames repeat across classes
    ref_order = {u: i for i, u in enumerate(key_of(ref))}
    assert len(ref_order) == len(ref['filenames']), f'fold {k}: duplicate class/filename keys in calibration subset'
    for m in MODELS:
        for s in SEEDS:
            r = calib[(m, s, k)]
            assert r['class_names'] == CLASS_NAMES, f'{m} s{s} f{k}: class order differs'
            keys = key_of(r)
            assert len(set(keys)) == len(keys) and set(keys) == set(ref_order), f'{m} s{s} f{k}: calibration cells differ'
            idx = np.argsort([ref_order[u] for u in keys])                 # align to the reference order
            reordered = not np.array_equal(idx, np.arange(len(idx)))
            for key in ('filenames', 'classes', 'group_ids'):
                r[key] = [r[key][i] for i in idx]
            r['y'], r['logits'] = r['y'][idx], r['logits'][idx]
            n, C = r['logits'].shape
            ok = {'filenames': r['filenames'] == ref['filenames'],
                  'labels': np.array_equal(r['y'], ref['y']),
                  'group_ids': r['group_ids'] == ref['group_ids'],
                  'classes': r['classes'] == ref['classes'],
                  'label_matches_class_name': all(CLASS_NAMES[y] == c for y, c in zip(r['y'], r['classes'])),
                  'parent_prefix_in_group': all(g.split('|')[0] == c for g, c in zip(r['group_ids'], r['classes'])),
                  'logits_finite': bool(np.isfinite(r['logits']).all()),
                  'n_classes': C == len(CLASS_NAMES),
                  'calib_parents_disjoint_from_test': r['disjoint_from_test']}
            checks.append({'model': PRETTY[m], 'seed': s, 'fold': k, 'n_calib': n, 'reordered': reordered, **ok})
            assert all(ok.values()), f'{m} s{s} f{k}: alignment failed {[a for a, v in ok.items() if not v]}'
checks = pd.DataFrame(checks)
print(f'Alignment verified for {len(checks)} runs: filenames, labels, parent groups, class order, '
      f'class names, finite logits, calibration/test parent disjointness '
      f'({int(checks.reordered.sum())} runs needed re-ordering by filename)')

# ---------------------------------------------------------------- 4. fixed grouped inner split per outer fold
NEW.mkdir(parents=True, exist_ok=True)
inner, split_rows = {}, []
for k in FOLDS:
    ref = calib[(MODELS[0], SEEDS[0], k)]
    sgk = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=CF_SEED(k))
    lab = np.full(len(ref['y']), -1)
    for j, (_, h) in enumerate(sgk.split(np.zeros(len(ref['y'])), ref['y'], groups=ref['group_ids'])):
        lab[h] = j
    assert (lab >= 0).all()
    g = pd.DataFrame({'group_id': ref['group_ids'], 'inner': lab})
    assert (g.groupby('group_id').inner.nunique() == 1).all(), f'fold {k}: a parent image spans inner folds'
    inner[k] = lab
    split_rows += [{'outer_fold': k, 'filename': fn, 'group_id': gid, 'class': c, 'y_true': int(y), 'inner_fold': int(j)}
                   for fn, gid, c, y, j in zip(ref['filenames'], ref['group_ids'], ref['classes'], ref['y'], lab)]
splits = pd.DataFrame(split_rows)
splits.to_csv(NEW / 'crossfit_inner_splits.csv', index=False)
print('Inner split sizes (cells per inner fold):')
print(splits.groupby(['outer_fold', 'inner_fold']).size().unstack().to_string())

# ---------------------------------------------------------------- 5. cross-fitted temperature scaling and metrics
run_rows = []
for k in FOLDS:
    lab = inner[k]
    for m in MODELS:
        for s in SEEDS:
            r = calib[(m, s, k)]; lg, y = r['logits'], r['y']
            p_oof = np.zeros_like(lg); nll_cell = np.zeros(len(y)); Ts, guards = [], 0
            for j in range(N_INNER):
                fit, held = lab != j, lab == j
                b, gflag = fit_beta(lg[fit], y[fit])                  # fitted without the held-out parents
                p_oof[held] = softmax(b * lg[held], axis=1)
                nll_cell[held] = exact_nll_vec(lg[held], y[held], b)
                Ts.append(1 / b); guards += int(gflag)
            assert np.allclose(p_oof.sum(1), 1)
            met = {'acc': m_acc(y, p_oof), 'f1': m_f1(y, p_oof), 'auroc': m_auroc(y, p_oof), 'ece': m_ece(y, p_oof),
                   'wc_ece': m_wc_ece(y, p_oof), 'aurc': m_aurc(y, p_oof), 'brier': m_brier(y, p_oof),
                   'nll': float(nll_cell.mean())}
            assert all(np.isfinite(v) for v in met.values()), f'{m} s{s} f{k}: non-finite metric {met}'
            run_rows.append({'fold': k, 'model_key': m, 'model': PRETTY[m], 'seed': s, 'n_calib': len(y),
                             **{f'T_inner{j}': Ts[j] for j in range(N_INNER)},
                             'T_inner_min': min(Ts), 'T_inner_max': max(Ts), 'inner_fits_fallback_T1': guards,
                             'inner_fits_at_T_bound': int(sum(min(abs(t - T_LO), abs(t - T_HI)) < 1e-3 for t in Ts)),
                             **met})
    print(f'  outer fold {k}: cross-fitting done ({time.time() - t0:.0f}s)')
runs = pd.DataFrame(run_rows)
runs.to_csv(NEW / 'per_run_crossfit_calibration_metrics.csv', index=False)

# ---------------------------------------------------------------- 6. original Hybrid-K normalization and composite
def norm_higher(v):                                   # original Hybrid-K function (unchanged)
    v = np.array(v, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    return np.ones_like(v) if hi - lo < 1e-12 else (v - lo) / (hi - lo)
def norm_lower(v):                                    # original Hybrid-K function (unchanged)
    return 1.0 - norm_higher(v)

seedavg = runs.groupby(['fold', 'model_key'], sort=False)[METRIC_KEYS].mean()
assert (runs.groupby(['fold', 'model_key']).size() == len(SEEDS)).all()
norm_rows, score_rows, topk_rows, zr_rows = [], [], [], []
for k in FOLDS:
    M = seedavg.loc[k].reindex(MODELS)                # rows in the original model order
    assert M.notna().all().all() and len(M) == 9
    N = pd.DataFrame(index=MODELS)
    for key in METRIC_KEYS:
        v = M[key].values
        N[key] = norm_higher(v) if DIRECTION[key] == 'higher' else norm_lower(v)
        rng = float(np.nanmax(v) - np.nanmin(v))
        zr_rows.append({'fold': k, 'metric': key, 'range': rng, 'zero_range': rng < 1e-12})
    assert N.notna().all().all() and ((N.values >= -1e-12) & (N.values <= 1 + 1e-12)).all()
    score = sum(WEIGHTS[key] * N[key].values for key in METRIC_KEYS)
    assert np.allclose(score, N[METRIC_KEYS].values.mean(axis=1))          # identical to the original 1/8 mean
    order = np.argsort(-score, kind='stable')         # ties broken by original model order (stable)
    rank = np.empty(9, int); rank[order] = np.arange(1, 10)
    srt = np.sort(score)[::-1]
    ties = [(PRETTY[MODELS[order[i]]], PRETTY[MODELS[order[i + 1]]]) for i in range(8) if abs(srt[i] - srt[i + 1]) < 1e-12]
    for i, m in enumerate(MODELS):
        norm_rows.append({'fold': k, 'model': PRETTY[m], **{f'{key}_raw_seedavg': M.loc[m, key] for key in METRIC_KEYS},
                          **{f'{key}_norm': N.loc[m, key] for key in METRIC_KEYS}})
        score_rows.append({'fold': k, 'model': PRETTY[m], 'composite_score': score[i], 'rank': int(rank[i])})
    for K in KS:
        sel = [PRETTY[MODELS[i]] for i in order[:K]]
        boundary_tie = abs(score[order[K - 1]] - score[order[K]]) < 1e-12
        topk_rows.append({'fold': k, 'K': K, 'selected_models': ' + '.join(sel),
                          **{f'member_{i + 1}': sel[i] for i in range(K)},
                          'score_of_last_selected': score[order[K - 1]], 'score_of_first_excluded': score[order[K]],
                          'margin_at_cutoff': score[order[K - 1]] - score[order[K]], 'tie_at_cutoff': boundary_tie})
    if ties:
        print(f'  fold {k}: exact composite ties (broken by original model order): {ties}')
normdf = pd.DataFrame(norm_rows); scoredf = pd.DataFrame(score_rows).sort_values(['fold', 'rank'])
topk = pd.DataFrame(topk_rows); zr = pd.DataFrame(zr_rows)
freq = pd.DataFrame([{'model': PRETTY[m], **{f'Top-{K}_folds_selected': int(sum(
    PRETTY[m] in r.selected_models.split(' + ') for _, r in topk[topk.K == K].iterrows())) for K in KS},
    'mean_rank': scoredf[scoredf.model == PRETTY[m]]['rank'].mean()} for m in MODELS]).sort_values('mean_rank')

normdf.to_csv(NEW / 'normalized_metrics_per_fold.csv', index=False)
scoredf.to_csv(NEW / 'composite_scores_and_rankings_per_fold.csv', index=False)
topk.to_csv(NEW / 'topk_selection_per_fold.csv', index=False)
freq.to_csv(NEW / 'selection_frequency_across_folds.csv', index=False)
zr.to_csv(NEW / 'zero_range_check.csv', index=False)
checks.to_csv(NEW / 'alignment_checks.csv', index=False)
json.dump({'source': str(RUNS), 'data_used': 'calibration subset only (logits, y_true, class names, group_ids)',
           'test_data_used': 'none (test group IDs read only for the disjointness check)',
           'inner_cv': {'method': 'StratifiedGroupKFold', 'n_splits': N_INNER, 'groups': 'group_id (class|parent image)',
                        'random_state': {str(k): CF_SEED(k) for k in FOLDS},
                        'shared_across': 'all 9 models and 3 seeds within an outer fold'},
           'temperature': {'objective': 'exact NLL via log-sum-exp', 'parameter': 'beta = 1/T',
                           'T_bounds': [T_LO, T_HI], 'grid_points': len(GRID), 'refinement': 'bounded Brent, xatol 1e-12',
                           'guard': 'T=1 if fitted T does not lower fitting-set NLL'},
           'metrics': METRIC_KEYS, 'nll_definition': 'exact (unclipped) NLL from logits', 'ece_bins': 15,
           'seed_aggregation': 'mean of each metric over seeds 42, 123, 456 before normalization',
           'normalization': 'original Hybrid-K norm_higher / norm_lower (min-max over 9 models; zero range -> 1 / 0)',
           'directions': DIRECTION, 'weights': WEIGHTS, 'tie_break': 'original model order (stable sort)',
           'model_order': [PRETTY[m] for m in MODELS], 'K': KS,
           'ensembles_built_or_evaluated': False}, open(NEW / 'run_config.json', 'w'), indent=2)

# ---------------------------------------------------------------- 7. verification and report
hash_after = {f: sha(f) for f in protected}
changed = [f for f in protected if hash_before[f] != hash_after[f]]
print('\nVERIFICATION')
print(f'  runs used: {len(runs)} (expected 135) | protected files modified: {len(changed)} (must be 0)')
print(f'  inner fits: {len(runs) * N_INNER} | fallback to T=1: {int(runs.inner_fits_fallback_T1.sum())} | '
      f'at a T bound: {int(runs.inner_fits_at_T_bound.sum())}')
print(f'  zero-range metrics: {int(zr.zero_range.sum())} of {len(zr)} (fold x metric)')
print(f'  ties at a Top-K cutoff: {int(topk.tie_at_cutoff.sum())}')
assert not changed and len(runs) == 135

pd.set_option('display.width', 200)
print('\nCOMPOSITE SCORE AND RANK PER OUTER FOLD (calibration data, cross-fitted, seed-averaged)')
piv = scoredf.pivot(index='model', columns='fold', values='composite_score').reindex(freq.model.tolist())
rk = scoredf.pivot(index='model', columns='fold', values='rank').reindex(piv.index)
print(pd.DataFrame({f'fold {k}': [f'{piv.loc[i, k]:.3f} (#{int(rk.loc[i, k])})' for i in piv.index] for k in FOLDS},
                   index=piv.index).to_string())
print('\nTOP-K SELECTION PER OUTER FOLD')
print(topk[['fold', 'K', 'selected_models', 'margin_at_cutoff']].round(4).to_string(index=False))
print('\nSELECTION FREQUENCY ACROSS THE 5 OUTER FOLDS')
print(freq.round(2).to_string(index=False))
print(f'\nSaved to {NEW}:')
for f in sorted(NEW.iterdir()): print('  ', f.name)
print(f'Done in {time.time() - t0:.0f}s. Test folds untouched. No ensembles built or evaluated.')
