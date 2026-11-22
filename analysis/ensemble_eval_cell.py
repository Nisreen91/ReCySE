# ======================================================================================
# Hybrid-K - ENSEMBLE CONSTRUCTION AND FINAL TEST EVALUATION (CPU only, no retraining)
#  Original Hybrid-K ensemble (sipakmed_full_pipeline_all_in_one.py, config_folds / pooled_calibrated):
#    1. members of the same fold and seed: unweighted mean of RAW softmax probabilities (soft voting);
#    2. ensemble pseudo-logits = log(clip(mean_prob, 1e-12, 1));
#    3. one temperature T for the ensemble, fitted by NLL with T in [0.25, 5.0];
#    4. calibrated ensemble probabilities = softmax(pseudo-logits / T).
#  Kept as is, with two leakage-related changes only:
#    a. T is fitted on the SAME fold and seed's calibration subset (not on the other test folds);
#    b. the original `calibrate_best` step (keep T only if it lowers ECE on the evaluated fold) is
#       NOT used, because it chose between raw and calibrated outputs using test labels. T is always applied.
#  The T fit uses the corrected exact log-sum-exp objective (beta = 1/T in [0.2, 4.0]) with a T = 1 guard.
#  Top-K memberships are read from crossfit_composite/ and are not recomputed.
#  Stage A (calibration data only) fixes every configuration and T and saves them BEFORE test labels are read.
# ======================================================================================
import os, json, glob, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score
try:
    from google.colab import drive
    if not os.path.isdir('/content/drive/MyDrive'):
        drive.mount('/content/drive')
except Exception:
    pass

OUT = Path(os.environ.get('HYBRIDK_OUT', '/content/drive/MyDrive/HybridK_Phase1_corrected'))
RUNS, CFC = OUT / 'fold_results', OUT / 'crossfit_composite'
NEW = OUT / 'hybridk_ensemble_evaluation'
MODELS = ['swin_tiny_patch4_window7_224', 'tiny_vit_5m_224', 'vit_tiny_patch16_224', 'densenet121', 'vgg16',
          'mobilevit_xs', 'efficientformerv2_s0', 'resnet50', 'efficientnet_b0']     # original order (tie-break)
PRETTY = {'swin_tiny_patch4_window7_224': 'Swin-Tiny', 'tiny_vit_5m_224': 'TinyViT-5M',
          'vit_tiny_patch16_224': 'ViT-Tiny', 'densenet121': 'DenseNet121', 'vgg16': 'VGG16',
          'mobilevit_xs': 'MobileViT-XS', 'efficientformerv2_s0': 'EfficientFormerV2-S0',
          'resnet50': 'ResNet50', 'efficientnet_b0': 'EfficientNet-B0'}
KEY = {v: k for k, v in PRETTY.items()}
SEEDS, FOLDS, KS = [42, 123, 456], [0, 1, 2, 3, 4], [2, 3, 4]
T_LO, T_HI, EPS = 0.25, 5.0, 1e-12
GRID = np.geomspace(1 / T_HI, 1 / T_LO, 4001)
METRIC_KEYS = ['acc', 'f1', 'auroc', 'ece', 'wc_ece', 'aurc', 'brier', 'nll']
HIGHER = {'acc', 'f1', 'auroc'}
t0 = time.time()

# ---------------------------------------------------------------- original eight metrics (NLL exact)
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
def exact_nll_vec(lg, y, beta):
    z = beta * lg
    return logsumexp(z, axis=1) - z[np.arange(len(y)), y]
def all_metrics(y, lg, beta):
    p = softmax(beta * lg, axis=1)
    return {'acc': m_acc(y, p), 'f1': m_f1(y, p), 'auroc': m_auroc(y, p), 'ece': m_ece(y, p),
            'wc_ece': m_wc_ece(y, p), 'aurc': m_aurc(y, p), 'brier': m_brier(y, p),
            'nll': float(exact_nll_vec(lg, y, beta).mean())}

# ---------------------------------------------------------------- corrected temperature fit (calibration data only)
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

# ---------------------------------------------------------------- original soft voting
def soft_vote(prob_list):
    """Original Hybrid-K: unweighted mean of members' raw softmax probabilities -> log pseudo-logits."""
    p = np.mean(np.stack([_clean(q) for q in prob_list]), axis=0)
    return np.log(np.clip(p, EPS, 1.0))

# ---------------------------------------------------------------- 1. read-only protection
def sha(f): return hashlib.sha256(open(f, 'rb').read()).hexdigest()
files = {(m, s, k): RUNS / f'multiclass__{m}__rep{s}__fold{k}.json' for m in MODELS for s in SEEDS for k in FOLDS}
missing = [str(f) for f in files.values() if not f.exists()]
assert not missing, f'{len(missing)} run files missing, e.g. {missing[:3]}'
protected = [str(f) for f in files.values()]
for sub in ('calibration_corrected', 'crossfit_composite'):
    protected += [f for f in sorted(glob.glob(str(OUT / sub / '**' / '*'), recursive=True)) if os.path.isfile(f)]
hash_before = {f: sha(f) for f in protected}
print(f'Protected files (run files + calibration_corrected + crossfit_composite): {len(protected)}')

# ---------------------------------------------------------------- 2. saved selections (not recomputed)
topk = pd.read_csv(CFC / 'topk_selection_per_fold.csv')
ranks = pd.read_csv(CFC / 'composite_scores_and_rankings_per_fold.csv')
normm = pd.read_csv(CFC / 'normalized_metrics_per_fold.csv')
assert len(topk) == 15 and set(zip(topk.fold, topk.K)) == {(k, K) for k in FOLDS for K in KS}
CONFIGS = {}                                           # (config, fold) -> list of model keys
for _, r in topk.iterrows():
    mem = [KEY[x.strip()] for x in r.selected_models.split(' + ')]
    assert len(mem) == r.K == len(set(mem)), f'fold {r.fold} K {r.K}: bad member list'
    rk = ranks[ranks.fold == r.fold].sort_values('rank').model.tolist()
    assert [PRETTY[m] for m in mem] == rk[:r.K], f'fold {r.fold} K {r.K}: selection inconsistent with saved ranking'
    CONFIGS[(f'Hybrid-{r.K}', int(r.fold))] = mem
for k in FOLDS:
    assert set(CONFIGS[('Hybrid-2', k)]) <= set(CONFIGS[('Hybrid-3', k)]) <= set(CONFIGS[('Hybrid-4', k)])
    rk = ranks[ranks.fold == k].sort_values('rank').model.tolist()
    CONFIGS[('Best single (composite)', k)] = [KEY[rk[0]]]
    # Accuracy-based Top-K on the same calibration data (seed-averaged; accuracy does not depend on T)
    acc = normm[normm.fold == k].set_index('model')['acc_raw_seedavg'].reindex([PRETTY[m] for m in MODELS])
    assert acc.notna().all()
    order = np.argsort(-acc.values, kind='stable')
    CONFIGS[('Best single (accuracy)', k)] = [MODELS[order[0]]]
    for K in KS:
        CONFIGS[(f'Accuracy Top-{K}', k)] = [MODELS[i] for i in order[:K]]
    CONFIGS[('All-9 equal-weight', k)] = list(MODELS)
    for m in MODELS:
        CONFIGS[(f'Single: {PRETTY[m]}', k)] = [m]
CONFIG_NAMES = (['Hybrid-2', 'Hybrid-3', 'Hybrid-4', 'Best single (composite)', 'Best single (accuracy)']
                + [f'Accuracy Top-{K}' for K in KS] + ['All-9 equal-weight'] + [f'Single: {PRETTY[m]}' for m in MODELS])
sel = pd.DataFrame([{'config': c, 'fold': k, 'n_members': len(CONFIGS[(c, k)]),
                     'members': ' + '.join(PRETTY[m] for m in CONFIGS[(c, k)])} for c in CONFIG_NAMES for k in FOLDS])
print('Saved Hybrid-K selections loaded and checked against the saved composite ranking (15 fold x K rows).')

# ---------------------------------------------------------------- 3. load runs; alignment checks
data = {}
for (m, s, k), f in files.items():
    d = json.load(open(f))
    assert d['model_name'] == m and int(d['replication_seed']) == s and int(d['fold_idx']) == k
    data[(m, s, k)] = {'class_names': list(d['class_names']),
                       'te': {'filenames': list(d['filenames']), 'classes': list(d['classes']),
                              'group_ids': list(d['group_ids']), 'y': np.asarray(d['y_true'], int),
                              'p_raw': np.asarray(d['y_prob'], float), 'logits': np.asarray(d['logits'], float)},
                       'ca': {'filenames': list(d['calib']['filenames']), 'classes': list(d['calib']['classes']),
                              'group_ids': list(d['calib']['group_ids']), 'y': np.asarray(d['calib']['y_true'], int),
                              'logits': np.asarray(d['calib']['logits'], float)}}
    del d
CLASS_NAMES = data[(MODELS[0], SEEDS[0], FOLDS[0])]['class_names']
keyof = lambda part: [f'{c}/{fn}' for c, fn in zip(part['classes'], part['filenames'])]
align_rows = []
for k in FOLDS:
    for role in ('ca', 'te'):
        ref = data[(MODELS[0], SEEDS[0], k)][role]; ref_keys = keyof(ref)
        pos = {u: i for i, u in enumerate(ref_keys)}; assert len(pos) == len(ref_keys)
        for m in MODELS:
            for s in SEEDS:
                d = data[(m, s, k)]; part = d[role]; kk = keyof(part)
                assert len(set(kk)) == len(kk) and set(kk) == set(pos), f'{m} s{s} f{k} {role}: cells differ'
                idx = np.argsort([pos[u] for u in kk])
                for a in ('filenames', 'classes', 'group_ids'): part[a] = [part[a][i] for i in idx]
                for a in ('y', 'p_raw', 'logits'):
                    if a in part: part[a] = part[a][idx]
                ok = {'class_order': d['class_names'] == CLASS_NAMES, 'filenames': part['filenames'] == ref['filenames'],
                      'labels': np.array_equal(part['y'], ref['y']), 'group_ids': part['group_ids'] == ref['group_ids'],
                      'label_matches_class': all(CLASS_NAMES[y] == c for y, c in zip(part['y'], part['classes'])),
                      'finite': bool(np.isfinite(part['logits']).all())}
                if role == 'te':
                    ok['saved_prob_equals_softmax_logits'] = bool(np.allclose(part['p_raw'], softmax(part['logits'], 1), atol=1e-5))
                align_rows.append({'fold': k, 'role': 'calibration' if role == 'ca' else 'test', 'model': PRETTY[m],
                                   'seed': s, 'n': len(kk), 'reordered': bool((idx != np.arange(len(idx))).any()), **ok})
                assert all(ok.values()), f'{m} s{s} f{k} {role}: {[a for a, v in ok.items() if not v]}'
    assert not set(data[(MODELS[0], SEEDS[0], k)]['ca']['group_ids']) & set(data[(MODELS[0], SEEDS[0], k)]['te']['group_ids'])
align = pd.DataFrame(align_rows)
print(f'Alignment verified: {len(align)} run x role checks (filenames, labels, class order, parent groups); '
      f'{int(align.reordered.sum())} re-ordered by class/filename.')

# ---------------------------------------------------------------- 4. STAGE A: calibration data only -> freeze configurations
NEW.mkdir(parents=True, exist_ok=True)
frozen, frozen_rows = {}, []
for c in CONFIG_NAMES:
    for k in FOLDS:
        mem = CONFIGS[(c, k)]
        for s in SEEDS:
            ca = [data[(m, s, k)]['ca'] for m in mem]
            lg_ca = soft_vote([softmax(x['logits'], axis=1) for x in ca])
            y_ca = ca[0]['y']
            b, guard = fit_beta(lg_ca, y_ca)
            frozen[(c, k, s)] = {'members': mem, 'beta': b}
            frozen_rows.append({'config': c, 'fold': k, 'seed': s, 'members': ' + '.join(PRETTY[m] for m in mem),
                                'n_members': len(mem), 'T': 1 / b, 'fallback_T1': guard,
                                'T_at_bound': min(abs(1 / b - T_LO), abs(1 / b - T_HI)) < 1e-3,
                                'calib_nll_T1': float(exact_nll_vec(lg_ca, y_ca, 1.0).mean()),
                                'calib_nll_fitted': float(exact_nll_vec(lg_ca, y_ca, b).mean())})
frozen_df = pd.DataFrame(frozen_rows)
assert (frozen_df.calib_nll_fitted <= frozen_df.calib_nll_T1 + 1e-12).all()
frozen_df.to_csv(NEW / 'frozen_configurations_calibration_only.csv', index=False)
sel.to_csv(NEW / 'selected_members_per_fold.csv', index=False)
cons = frozen_df.groupby(['config', 'fold']).members.nunique()
assert (cons == 1).all(), 'members differ across seeds within a fold'
print(f'Stage A done: {len(frozen_df)} configurations x fold x seed frozen from calibration data '
      f'({time.time() - t0:.0f}s). Test labels used only for the alignment check so far.')

# ---------------------------------------------------------------- 5. STAGE B: apply frozen configurations to the test fold
unit_rows, pred_frames = [], []
for c in CONFIG_NAMES:
    for k in FOLDS:
        for s in SEEDS:
            fz = frozen[(c, k, s)]; te = [data[(m, s, k)]['te'] for m in fz['members']]
            lg_te = soft_vote([x['p_raw'] for x in te]); y = te[0]['y']
            p_raw = np.exp(lg_te); p_raw = p_raw / p_raw.sum(1, keepdims=True)
            p_cal = softmax(fz['beta'] * lg_te, axis=1)
            met = all_metrics(y, lg_te, fz['beta'])
            unit_rows.append({'config': c, 'fold': k, 'seed': s, 'n_test': len(y), 'T': 1 / fz['beta'],
                              'members': ' + '.join(PRETTY[m] for m in fz['members']), **met})
            fr = pd.DataFrame({'config': c, 'fold': k, 'seed': s, 'filename': te[0]['filenames'],
                               'class': te[0]['classes'], 'group_id': te[0]['group_ids'], 'y_true': y,
                               'y_pred': p_cal.argmax(1)})
            for j, cn in enumerate(CLASS_NAMES):
                fr[f'p_vote_raw__{cn}'] = p_raw[:, j]
            for j, cn in enumerate(CLASS_NAMES):
                fr[f'p_calibrated__{cn}'] = p_cal[:, j]
            pred_frames.append(fr)
units = pd.DataFrame(unit_rows)
assert units[METRIC_KEYS].notna().all().all()
units.to_csv(NEW / 'per_unit_test_metrics.csv', index=False)
pd.concat(pred_frames, ignore_index=True).to_csv(NEW / 'test_predictions_all_configs.csv.gz', index=False,
                                                 compression='gzip')

# ---------------------------------------------------------------- 6. summaries and paired comparisons
summ = units.groupby('config', sort=False)[METRIC_KEYS].agg(['mean', 'std']).reindex(CONFIG_NAMES)
summ.columns = [f'{a}_{b}' for a, b in summ.columns]
summ.insert(0, 'n_units', units.groupby('config').size().reindex(CONFIG_NAMES).values)
summ.to_csv(NEW / 'summary_mean_sd.csv')
pretty_tbl = pd.DataFrame({k: [f"{summ.loc[c, k + '_mean']:.4f} ± {summ.loc[c, k + '_std']:.4f}" for c in CONFIG_NAMES]
                           for k in METRIC_KEYS}, index=CONFIG_NAMES)
pretty_tbl.to_csv(NEW / 'summary_mean_sd_formatted.csv')

PAIRS = ([(f'Hybrid-{K}', 'Best single (composite)') for K in KS] + [(f'Hybrid-{K}', f'Accuracy Top-{K}') for K in KS]
         + [(f'Hybrid-{K}', 'All-9 equal-weight') for K in KS])
idx_cols = ['fold', 'seed']
comp_rows = []
for a, b in PAIRS:
    A = units[units.config == a].set_index(idx_cols)[METRIC_KEYS]
    B = units[units.config == b].set_index(idx_cols)[METRIC_KEYS].reindex(A.index)
    D = A - B
    same = (sel[sel.config == a].set_index('fold').members == sel[sel.config == b].set_index('fold').members).sum()
    for key in METRIC_KEYS:
        better = (D[key] > 1e-12) if key in HIGHER else (D[key] < -1e-12)
        comp_rows.append({'comparison': f'{a} vs {b}', 'metric': key, 'direction': 'higher' if key in HIGHER else 'lower',
                          'mean_diff': D[key].mean(), 'sd_diff': D[key].std(), 'units_A_better': int(better.sum()),
                          'units_tied': int((D[key].abs() <= 1e-12).sum()), 'n_units': len(D),
                          'folds_with_identical_members': int(same)})
comp = pd.DataFrame(comp_rows)
comp.to_csv(NEW / 'paired_comparisons.csv', index=False)

json.dump({'ensemble_method': 'original Hybrid-K soft voting: unweighted mean of raw member softmax probabilities, '
                              'log(clip(p, 1e-12)) pseudo-logits, one ensemble temperature',
           'temperature_fit': 'same fold and seed calibration subset only; exact log-sum-exp NLL; beta=1/T in [0.2, 4.0]; '
                              'T=1 guard', 'original_calibrate_best_step': 'not used (it selected raw vs calibrated '
                              'using ECE on the evaluated fold, i.e. test labels); T always applied',
           'selections': 'read from crossfit_composite/topk_selection_per_fold.csv, not recomputed',
           'baselines': {'Best single (composite)': 'rank 1 of the saved composite ranking per fold',
                         'Accuracy Top-K / Best single (accuracy)': 'seed-averaged calibration accuracy per fold '
                         '(crossfit_composite/normalized_metrics_per_fold.csv), stable tie-break in original model order',
                         'All-9 equal-weight': 'original soft voting over all nine models',
                         'Single models': 'same pipeline with one member'},
           'equal_weight_note': 'the original Hybrid-K voting is already equal-weight; no separate weighted variant exists',
           'nll': 'exact NLL of softmax(pseudo_logits / T) via log-sum-exp', 'units': '5 outer folds x 3 seeds = 15',
           'sd': 'sample SD (ddof=1) over the 15 units; units share test images across seeds, so they are not independent'},
          open(NEW / 'run_config.json', 'w'), indent=2)
align.to_csv(NEW / 'alignment_checks.csv', index=False)

# ---------------------------------------------------------------- 7. verification and report
hash_after = {f: sha(f) for f in protected}
changed = [f for f in protected if hash_before[f] != hash_after[f]]
n_hyb = int(units.config.str.startswith('Hybrid-').sum())
print('\nVERIFICATION')
print(f'  Hybrid-K ensemble evaluations: {n_hyb} (expected 45 = 3 configurations x 5 folds x 3 seeds)')
print(f'  all configuration evaluations: {len(units)} (expected {len(CONFIG_NAMES) * 15})')
print(f'  same members across the 3 seeds of every fold: {bool((cons == 1).all())}')
print(f'  ensemble temperatures: fallback to T=1 in {int(frozen_df.fallback_T1.sum())}, at a bound in '
      f'{int(frozen_df.T_at_bound.sum())} of {len(frozen_df)}')
print(f'  protected files modified: {len(changed)} (must be 0)')
assert n_hyb == 45 and len(units) == len(CONFIG_NAMES) * 15 and not changed

pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
print('\nSELECTED MEMBERS PER FOLD (from the saved calibration-based selection)')
print(sel[sel.config.isin(['Hybrid-2', 'Hybrid-3', 'Hybrid-4', 'Accuracy Top-2', 'Accuracy Top-3', 'Accuracy Top-4'])]
      .pivot(index='fold', columns='config', values='members').to_string())
print('\nTEST METRICS, mean ± SD over 15 fold-and-seed units (NLL = exact)')
print(pretty_tbl.to_string())
print('\nPAIRED DIFFERENCES (A - B), mean ± SD over 15 units; units_A_better counts the better direction')
cp = comp.copy(); cp['diff'] = [f'{a:+.4f} ± {b:.4f}' for a, b in zip(cp.mean_diff, cp.sd_diff)]
print(cp.pivot(index='comparison', columns='metric', values='diff')[METRIC_KEYS].to_string())
print(cp.pivot(index='comparison', columns='metric', values='units_A_better')[METRIC_KEYS].to_string())
print(f'\nSaved to {NEW}:')
for f in sorted(NEW.iterdir()): print('  ', f.name)
print(f'Done in {time.time() - t0:.0f}s. Original files unchanged. No retraining.')
