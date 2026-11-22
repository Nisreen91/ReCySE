"""ReCySE final analyses, step 1: integrity and provenance checks on saved predictions (read-only)."""
import numpy as np, pandas as pd, json, sys
sys.path.insert(0, '/home/claude/p1')
D = '/home/claude/fin/data/'
CFG = ['Hybrid-2', 'Hybrid-3', 'Hybrid-4', 'Accuracy Top-2', 'Accuracy Top-3', 'Accuracy Top-4',
       'All-9 equal-weight', 'Best single (composite)']
pr = pd.read_csv(D + 'hybridk_ensemble_evaluation/test_predictions_all_configs.csv.gz')
pum = pd.read_csv(D + 'hybridk_ensemble_evaluation/per_unit_test_metrics.csv')
frz = pd.read_csv(D + 'hybridk_ensemble_evaluation/frozen_configurations_calibration_only.csv')
sa = pd.read_csv(D + 'risk_targeting_exploratory/stageA_thresholds_calibration.csv')
sb = pd.read_csv(D + 'risk_targeting_exploratory/stageB_test_results_per_run.csv')
PC = [c for c in pr.columns if c.startswith('p_calibrated__')]


def ece_on(conf, corr, nb=15):
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (conf > b[i]) & (conf <= b[i + 1])
        if m.sum(): e += m.sum() / len(conf) * abs(corr[m].mean() - conf[m].mean())
    return e


def aurc(conf, corr):
    o = np.argsort(-conf); risk = 1 - np.cumsum(corr[o]) / np.arange(1, len(conf) + 1)
    return float(np.trapezoid(risk, np.arange(1, len(conf) + 1) / len(conf)))


R = {}
pr = pr[pr.config.isin(CFG)]
R['n_rows_8cfg'] = len(pr)
R['runs_per_cfg'] = pr.groupby('config')[['fold', 'seed']].apply(lambda g: len(g.drop_duplicates())).to_dict()
R['unique_images'] = int(pr.drop_duplicates(['class', 'filename']).shape[0])
# each image appears exactly once per seed per config (one test fold)
R['each_image_once_per_seed'] = bool((pr.groupby(['config', 'seed', 'class', 'filename']).size() == 1).all())
# group (parent image) disjointness: no group_id appears in two folds within a seed
gf = pr.groupby(['config', 'seed', 'group_id']).fold.nunique()
R['groups_in_single_fold'] = bool((gf == 1).all())
P = pr[PC].values
R['prob_rowsum_maxdev'] = float(np.abs(P.sum(1) - 1).max())
R['y_pred_equals_argmax'] = bool((P.argmax(1) == pr.y_pred.values).all())
R['nan_probs'] = int(np.isnan(P).sum())

# metric reproduction against per_unit_test_metrics
dd = {'acc': 0, 'ece': 0, 'wc_ece': 0, 'aurc': 0, 'n': 0}
for (c, k, s), g in pr.groupby(['config', 'fold', 'seed']):
    p = g[PC].values; y = g.y_true.values; conf = p.max(1); corr = (p.argmax(1) == y).astype(float)
    ref = pum[(pum.config == c) & (pum.fold == k) & (pum.seed == s)].iloc[0]
    assert ref.n_test == len(g)
    wc = max(ece_on(p[y == j].max(1), (p[y == j].argmax(1) == j).astype(float)) for j in range(5))
    for key, v in [('acc', corr.mean()), ('ece', ece_on(conf, corr)), ('wc_ece', wc), ('aurc', aurc(conf, corr))]:
        dd[key] = max(dd[key], abs(v - ref[key]))
    dd['n'] += 1
R['metric_maxabs_diff_vs_per_unit'] = dd

# stageB reproduction with frozen stage-A thresholds
mx = {'cov': 0, 'risk': 0, 'nacc': 0, 'met_mismatch': 0, 'n': 0}
for t in sa[sa.config.isin(CFG)].itertuples():
    g = pr[(pr.config == t.config) & (pr.fold == t.fold) & (pr.seed == t.seed)]
    p = g[PC].values; conf = p.max(1); corr = p.argmax(1) == g.y_true.values
    acc = conf >= t.tau; na = int(acc.sum())
    risk = float((~corr[acc]).mean()) if na else np.nan
    ref = sb[(sb.config == t.config) & (sb.fold == t.fold) & (sb.seed == t.seed) & np.isclose(sb.alpha, t.alpha)].iloc[0]
    mx['cov'] = max(mx['cov'], abs(acc.mean() - ref.test_coverage))
    mx['nacc'] = max(mx['nacc'], abs(na - ref.n_accepted))
    if na: mx['risk'] = max(mx['risk'], abs(risk - ref.test_selective_risk))
    mx['met_mismatch'] += int(bool(na == 0 or risk <= t.alpha) != bool(ref.target_met))
    mx['n'] += 1
R['stageB_reproduction'] = mx
# frozen config consistency
R['T_match_frozen_vs_per_unit'] = float(np.abs(frz.set_index(['config', 'fold', 'seed'])['T'] -
                                         pum.set_index(['config', 'fold', 'seed'])['T']).max())
R['members_match'] = bool((frz.set_index(['config', 'fold', 'seed']).members ==
                           pum.set_index(['config', 'fold', 'seed']).members).all())
json.dump(R, open('/home/claude/fin/verification.json', 'w'), indent=2, default=str)
print(json.dumps(R, indent=2, default=str))
