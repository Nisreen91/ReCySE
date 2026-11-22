# ======================================================================================
# Hybrid-K - CORRECTED TEMPERATURE SCALING for all nine models (CPU only, no retraining)
#  * Uses ONLY the saved calibration-subset logits/labels for fitting. Test data are never used to fit.
#  * Objective: exact, numerically stable NLL via log-sum-exp, parameterized by inverse temperature
#    beta = 1/T in [1/5.0, 1/0.25] = [0.2, 4.0] (same bounds as the original T in [0.25, 5.0]).
#    NLL(beta) is convex in beta -> global minimum found by a dense grid + bounded refinement.
#  * Safeguard: if the fitted beta does not lower calibration NLL relative to T=1, T=1 is used.
#  * Original run files are read-only (SHA-256 checked before/after). New outputs -> calibration_corrected/.
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
NEW = OUT / 'calibration_corrected'
MODELS = {'swin_tiny_patch4_window7_224': 'Swin-Tiny', 'tiny_vit_5m_224': 'TinyViT-5M',
          'vit_tiny_patch16_224': 'ViT-Tiny', 'densenet121': 'DenseNet121', 'vgg16': 'VGG16',
          'resnet50': 'ResNet50', 'efficientnet_b0': 'EfficientNet-B0', 'mobilevit_xs': 'MobileViT-XS',
          'efficientformerv2_s0': 'EfficientFormerV2-S0'}
T_LO, T_HI = 0.25, 5.0
B_LO, B_HI = 1 / T_HI, 1 / T_LO                     # beta bounds 0.2 .. 4.0
GRID = np.geomspace(B_LO, B_HI, 4001)

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
def m_nll(y, p): p = _clean(p); return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))
def m_aurc(y, p):
    p = _clean(p); conf = p.max(1); corr = (p.argmax(1) == y).astype(float); o = np.argsort(-conf)
    risk = 1 - np.cumsum(corr[o]) / np.arange(1, len(y) + 1); cov = np.arange(1, len(y) + 1) / len(y)
    return float((np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(risk, cov))
def m_auroc(y, p):
    try: return float(roc_auc_score(y, _clean(p), multi_class='ovr', average='macro'))
    except Exception: return float('nan')
METRICS = {'acc': m_acc, 'f1': m_f1, 'auroc': m_auroc, 'ece': m_ece, 'wc_ece': m_wc_ece,
           'aurc': m_aurc, 'brier': m_brier, 'nll': m_nll}

# ---------------------------------------------------------------- stable objective and fit
def exact_nll(lg, y, beta):
    """Exact mean NLL of softmax(beta * logits), via log-sum-exp (no clipping, no underflow)."""
    z = beta * lg
    return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(y)), y]))

def fit_beta(lg, y):
    vals = np.empty(len(GRID)); yi = np.arange(len(y))
    for a in range(0, len(GRID), 500):                       # vectorized over the grid, in chunks
        z = GRID[a:a + 500, None, None] * lg[None]
        vals[a:a + 500] = (logsumexp(z, axis=2) - z[:, yi, y]).mean(axis=1)
    i = int(np.argmin(vals))
    lo, hi = GRID[max(i - 1, 0)], GRID[min(i + 1, len(GRID) - 1)]
    if hi > lo:
        r = minimize_scalar(lambda b: exact_nll(lg, y, b), bounds=(lo, hi), method='bounded', options={'xatol': 1e-12})
        b_star = r.x if r.fun <= vals[i] else GRID[i]
    else:
        b_star = GRID[i]
    # local-minimum count on the grid (convexity sanity check; expected 1)
    d = np.sign(np.diff(vals)); d = d[d != 0]
    n_local_min = int(np.sum((d[:-1] < 0) & (d[1:] > 0))) + int(i in (0, len(GRID) - 1))
    nll1, nllb = exact_nll(lg, y, 1.0), exact_nll(lg, y, b_star)
    guard = nllb > nll1                                      # never worse than T = 1
    if guard:
        b_star, nllb = 1.0, nll1
    return float(b_star), nllb, nll1, n_local_min, bool(guard)

def probs(lg, beta): return softmax(beta * lg, axis=1)

# ---------------------------------------------------------------- read-only protection of originals
def sha(f): return hashlib.sha256(open(f, 'rb').read()).hexdigest()
orig_files = {m: sorted(glob.glob(str(OUT / 'fold_results' / f'multiclass__{m}__rep*__fold*.json'))) for m in MODELS}
for m, fs in orig_files.items():
    print(f'{MODELS[m]:22s}: {len(fs)} run files')
    assert len(fs) == 15, f'{MODELS[m]}: expected 15 run files'
hash_before = {f: sha(f) for fs in orig_files.values() for f in fs}

# ---------------------------------------------------------------- main loop
t0 = time.time(); rows = []
for m, fs in orig_files.items():
    (NEW / m).mkdir(parents=True, exist_ok=True)
    for f in fs:
        d = json.load(open(f))
        ca_lg = np.asarray(d['calib']['logits'], float); ca_y = np.asarray(d['calib']['y_true'], int)
        te_lg = np.asarray(d['logits'], float); te_y = np.asarray(d['y_true'], int)
        assert np.isfinite(ca_lg).all() and np.isfinite(te_lg).all()
        T_orig = float(d['temperature'])
        # ---- fit on CALIBRATION data only ----
        beta, nll_new_c, nll_t1_c, nmin, guard = fit_beta(ca_lg, ca_y)
        T_new = 1.0 / beta
        nll_orig_c = exact_nll(ca_lg, ca_y, 1.0 / T_orig)
        # clipped (original metric) NLL on calibration data, for the explanation of the earlier flag
        clip_t1_c = m_nll(ca_y, probs(ca_lg, 1.0)); clip_orig_c = m_nll(ca_y, probs(ca_lg, 1 / T_orig))
        clip_new_c = m_nll(ca_y, probs(ca_lg, beta))
        # ---- evaluate on TEST (no fitting here) ----
        p_raw, p_old, p_new = probs(te_lg, 1.0), probs(te_lg, 1 / T_orig), probs(te_lg, beta)
        met_raw = {k: fn(te_y, p_raw) for k, fn in METRICS.items()}
        met_old = {k: fn(te_y, p_old) for k, fn in METRICS.items()}
        met_new = {k: fn(te_y, p_new) for k, fn in METRICS.items()}
        for dct, b in ((met_raw, 1.0), (met_old, 1 / T_orig), (met_new, beta)):
            dct['nll_exact'] = exact_nll(te_lg, te_y, b)       # unclipped NLL, reported alongside
        raw_diff = max(abs(met_raw[k] - d['metrics_test_raw'][k]) for k in METRICS)
        tag = Path(f).stem.replace('multiclass__', '')
        rec = {'source_file': Path(f).name, 'model_name': m, 'replication_seed': d['replication_seed'],
               'fold_idx': d['fold_idx'], 'temperature_original': T_orig, 'temperature_corrected': T_new,
               'beta_corrected': beta, 'fit_on': 'calibration subset only', 'objective': 'exact NLL, log-sum-exp, beta=1/T in [0.2, 4.0]',
               'guard_fallback_to_T1': guard, 'grid_local_minima': nmin,
               'calib_nll_exact': {'T1': nll_t1_c, 'T_original': nll_orig_c, 'T_corrected': nll_new_c},
               'calib_nll_clipped_metric': {'T1': clip_t1_c, 'T_original': clip_orig_c, 'T_corrected': clip_new_c},
               'y_true': te_y.tolist(), 'filenames': d['filenames'], 'parent_ids': d['parent_ids'],
               'y_prob_raw': p_raw.tolist(), 'y_prob_calibrated_corrected': p_new.tolist(),
               'metrics_test_raw': met_raw, 'metrics_test_calibrated_original': met_old,
               'metrics_test_calibrated_corrected': met_new}
        json.dump(rec, open(NEW / m / f'calibcorr__{tag}.json', 'w'))
        rows.append({'model': MODELS[m], 'seed': d['replication_seed'], 'fold': d['fold_idx'],
                     'T_original': T_orig, 'T_corrected': T_new, 'abs_dT': abs(T_new - T_orig),
                     'T_at_bound': min(abs(T_new - T_LO), abs(T_new - T_HI)) < 1e-3, 'guard_T1': guard,
                     'grid_local_minima': nmin, 'max_abs_logit_calib': float(np.abs(ca_lg).max()),
                     'calib_exact_T1': nll_t1_c, 'calib_exact_orig': nll_orig_c, 'calib_exact_new': nll_new_c,
                     'calib_clip_T1': clip_t1_c, 'calib_clip_orig': clip_orig_c, 'calib_clip_new': clip_new_c,
                     'raw_metric_recompute_maxdiff': raw_diff,
                     **{f'test_{k}_raw': met_raw[k] for k in ['ece', 'wc_ece', 'brier', 'nll', 'nll_exact', 'aurc']},
                     **{f'test_{k}_old': met_old[k] for k in ['ece', 'wc_ece', 'brier', 'nll', 'nll_exact', 'aurc']},
                     **{f'test_{k}_new': met_new[k] for k in ['ece', 'wc_ece', 'brier', 'nll', 'nll_exact', 'aurc']}})
df = pd.DataFrame(rows)
df.to_csv(NEW / 'per_run_calibration_check.csv', index=False)

# ---------------------------------------------------------------- verification
hash_after = {f: sha(f) for f in hash_before}
changed = [f for f in hash_before if hash_before[f] != hash_after[f]]
viol_new = df[df.calib_exact_new > df.calib_exact_T1 + 1e-12]
print('\nVERIFICATION')
print(f'  runs processed: {len(df)} (expected 135) | time {time.time() - t0:.0f}s')
print(f'  original run files modified: {len(changed)} (must be 0)')
print(f'  corrected T: calibration exact NLL above T=1 in {len(viol_new)} runs (must be 0); '
      f'fallback to T=1 used in {int(df.guard_T1.sum())} runs')
print(f'  objective convex on grid (1 local minimum) in {int((df.grid_local_minima <= 1).sum())}/{len(df)} runs')
print(f'  raw test metrics recomputed vs saved: max diff {df.raw_metric_recompute_maxdiff.max():.2e}')
assert not changed and viol_new.empty and len(df) == 135

# ---------------------------------------------------------------- before / after report
print('\nWAS THE ORIGINAL FIT SUBOPTIMAL? (calibration data, exact NLL)')
df['orig_exact_worse_than_T1'] = df.calib_exact_orig > df.calib_exact_T1 + 1e-12
df['orig_clip_worse_than_T1'] = df.calib_clip_orig > df.calib_clip_T1 + 1e-12
df['new_clip_worse_than_T1'] = df.calib_clip_new > df.calib_clip_T1 + 1e-12
df['orig_gap_to_optimum'] = df.calib_exact_orig - df.calib_exact_new
summ = df.groupby('model', sort=False).agg(
    T_orig_mean=('T_original', 'mean'), T_new_mean=('T_corrected', 'mean'), max_abs_dT=('abs_dT', 'max'),
    runs_T_at_bound=('T_at_bound', 'sum'), max_gap_orig_vs_optimum=('orig_gap_to_optimum', 'max'),
    orig_exactNLL_worse_T1=('orig_exact_worse_than_T1', 'sum'),
    orig_clippedNLL_worse_T1=('orig_clip_worse_than_T1', 'sum'),
    new_clippedNLL_worse_T1=('new_clip_worse_than_T1', 'sum'),
    max_abs_logit=('max_abs_logit_calib', 'max'))
print(summ.round(4).to_string())

ba = df.groupby('model', sort=False)[[f'test_{k}_{v}' for k in ['ece', 'brier', 'nll', 'nll_exact', 'wc_ece', 'aurc']
                                     for v in ['raw', 'old', 'new']]].mean()
print('\nTEST-FOLD METRICS, mean over 15 runs (raw | original calibration | corrected calibration)')
out = pd.DataFrame({f'{k}': [f"{r[f'test_{k}_raw']:.4f} | {r[f'test_{k}_old']:.4f} | {r[f'test_{k}_new']:.4f}"
                             for _, r in ba.iterrows()] for k in ['ece', 'brier', 'nll', 'nll_exact']}, index=ba.index)
print(out.to_string())
summ.to_csv(NEW / 'per_model_fit_summary.csv'); ba.to_csv(NEW / 'per_model_test_metrics_before_after.csv')

ef = df[df.model == 'EfficientFormerV2-S0'][['seed', 'fold', 'T_original', 'T_corrected', 'calib_exact_T1',
     'calib_exact_orig', 'calib_exact_new', 'calib_clip_T1', 'calib_clip_orig', 'calib_clip_new', 'max_abs_logit_calib']]
print('\nEfficientFormerV2-S0, per run (calibration subset):')
print(ef.round(4).to_string(index=False))
print(f'\nSaved to {NEW}: per-run JSONs (corrected T, corrected calibrated probabilities, all metrics), '
      'per_run_calibration_check.csv, per_model_fit_summary.csv, per_model_test_metrics_before_after.csv')
print('Original results untouched. Composite Score / Hybrid-K not run.')
