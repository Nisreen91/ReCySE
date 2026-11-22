# -*- coding: utf-8 -*-
"""
make_split_manifest.py  --  CPU only. Builds and audits the corrected Hybrid-K split.

Input : a cell list with columns class, filename (one row per cropped SIPaKMeD cell).
        Default: original_fold_assignment.csv produced by sipakmed_parent_image_audit.py
        (it also holds the ORIGINAL cell-level fold for comparison).
Output: split_manifest.csv, split_audit.txt, split_audit.json

Design (fixed once, reused for both models and all three training seeds,
exactly like the original fixed partition):
  * Parent-image ID  = "<class folder>|<NNN>" from crop name NNN_MM.bmp.
    This is a PARENT (cluster) IMAGE ID, not a patient ID.
  * Outer split      = StratifiedGroupKFold(5, shuffle=True, random_state=42)
                       on the 5-class label, grouped by parent image.
  * Calibration set  = inside each outer training portion,
                       StratifiedGroupKFold(8, shuffle=True, random_state=42+k), first fold
                       (~1/8 of the outer-train parent images, ~10% of all cells).
  * Roles per outer fold k: 'test' (held-out fold), 'calib' (temperature fitting and
    epoch monitoring only), 'train' (model fitting).
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

CROP_RE = re.compile(r'^(?P<parent>\d+)_(?P<cell>\d+)\.[A-Za-z]+$')
N_OUTER, N_INNER, SEED = 5, 8, 42


def build(cells: pd.DataFrame) -> pd.DataFrame:
    cells = cells[['class', 'filename']].copy()
    # same index order as phase_1.py: sorted class folders, sorted filenames
    cells = cells.sort_values(['class', 'filename'], kind='mergesort').reset_index(drop=True)
    classes = sorted(cells['class'].unique())
    cells['label'] = cells['class'].map({c: i for i, c in enumerate(classes)})
    m = cells['filename'].str.extract(CROP_RE)
    if m['parent'].isna().any():
        bad = cells.loc[m['parent'].isna(), 'filename'].tolist()[:5]
        raise SystemExit(f'Filenames not matching NNN_MM.ext: {bad}')
    cells['parent_id'] = m['parent']
    cells['group_id'] = cells['class'] + '|' + cells['parent_id']
    y = cells['label'].to_numpy()
    g = cells['group_id'].to_numpy()
    outer = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    cells['test_fold'] = -1
    for k, (_, te) in enumerate(outer.split(np.zeros(len(y)), y, g)):
        cells.loc[te, 'test_fold'] = k
    for k in range(N_OUTER):
        role = np.where(cells['test_fold'] == k, 'test', 'train').astype(object)
        tr = np.where(role == 'train')[0]
        inner = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=SEED + k)
        _, cal_local = next(inner.split(np.zeros(len(tr)), y[tr], g[tr]))
        role[tr[cal_local]] = 'calib'
        cells[f'role_fold{k}'] = role
    return cells


def audit(df: pd.DataFrame, orig: pd.DataFrame | None):
    lines, checks = [], []
    say = lines.append
    say('=' * 80); say('CORRECTED HYBRID-K SPLIT AUDIT (parent-image grouped)'); say('=' * 80)
    say(f'scikit-learn {sklearn.__version__}; outer StratifiedGroupKFold(5, seed 42); '
        f'calibration = 1/{N_INNER} of outer-train parent images (StratifiedGroupKFold, seed 42+k)')
    say(f'Cells: {len(df)} | parent images: {df.group_id.nunique()} | classes: {sorted(df["class"].unique())}')
    say('Grouping unit: PARENT IMAGE (class folder + NNN prefix). Not a patient or slide ID.')
    # every group single-fold
    g_folds = df.groupby('group_id')['test_fold'].nunique()
    checks.append(('Every parent image assigned to exactly one test fold', int((g_folds > 1).sum()) == 0,
                   f'{int((g_folds > 1).sum())} parents split across test folds'))
    rows = []
    for k in range(N_OUTER):
        r = df[f'role_fold{k}']
        G = {s: set(df.loc[r == s, 'group_id']) for s in ('train', 'calib', 'test')}
        ov_tt = len(G['train'] & G['test']); ov_ct = len(G['calib'] & G['test']); ov_tc = len(G['train'] & G['calib'])
        cells_sib = int(df.loc[r == 'test', 'group_id'].isin(G['train'] | G['calib']).sum())
        rows.append({'fold': k, 'train_cells': int((r == 'train').sum()), 'calib_cells': int((r == 'calib').sum()),
                     'test_cells': int((r == 'test').sum()), 'train_parents': len(G['train']),
                     'calib_parents': len(G['calib']), 'test_parents': len(G['test']),
                     'overlap_train_test': ov_tt, 'overlap_calib_test': ov_ct, 'overlap_train_calib': ov_tc,
                     'test_cells_with_sibling_outside_test': cells_sib})
        checks.append((f'Fold {k}: zero parent overlap train/test, calib/test, train/calib',
                       ov_tt == ov_ct == ov_tc == 0, f'{ov_tt}/{ov_ct}/{ov_tc}'))
    t = pd.DataFrame(rows)
    say('\nPer-fold sizes and parent-image overlap (must all be 0):'); say(t.to_string(index=False))
    # class balance
    say('\nClass distribution of test folds (cells):')
    say(pd.crosstab(df['class'], df['test_fold']).to_string())
    say('\nClass distribution of calibration subsets (cells):')
    cal = pd.concat([df.loc[df[f'role_fold{k}'] == 'calib', ['class']].assign(fold=k) for k in range(N_OUTER)], ignore_index=True)
    say(pd.crosstab(cal['class'], cal['fold']).to_string())
    every = all((df[f'role_fold{k}'] == 'calib').groupby(df['class']).sum().min() > 0 for k in range(N_OUTER))
    checks.append(('Every class present in every calibration subset', bool(every), ''))
    every_t = pd.crosstab(df['class'], df['test_fold']).min().min() > 0
    checks.append(('Every class present in every test fold', bool(every_t), ''))
    checks.append(('Each cell is test exactly once', bool((df.test_fold >= 0).all()), ''))
    if orig is not None and 'orig_fold' in orig:
        o = orig.merge(df[['class', 'filename', 'group_id']], on=['class', 'filename'])
        sib = 0
        for f in sorted(o.orig_fold.unique()):
            tr_groups = set(o.loc[o.orig_fold != f, 'group_id'])
            sib += int(o.loc[o.orig_fold == f, 'group_id'].isin(tr_groups).sum())
        say(f'\nFor comparison, ORIGINAL cell-level split: {sib}/{len(o)} test cells '
            f'({100 * sib / len(o):.1f}%) had a sibling from the same parent image in training.')
        say(f'CORRECTED split: {int(t.test_cells_with_sibling_outside_test.sum())}/{len(df)} test cells.')
    say('\nCHECKS'); ok = True
    for name, passed, note in checks:
        ok &= bool(passed); say(f'  [{"PASS" if passed else "FAIL"}] {name}' + (f'  ({note})' if note else ''))
    say('\nOVERALL: ' + ('ALL CHECKS PASSED - zero parent-image overlap.' if ok else 'FAILURES - do not train.'))
    return '\n'.join(lines), t, checks, ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='original_fold_assignment.csv')
    ap.add_argument('--out', default='splits')
    a, _ = ap.parse_known_args(argv)
    src = pd.read_csv(a.cells, dtype=str)
    df = build(src)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cols = ['label', 'class', 'filename', 'parent_id', 'group_id', 'test_fold'] + [f'role_fold{k}' for k in range(N_OUTER)]
    df[cols].to_csv(out / 'split_manifest.csv', index=False)
    orig = src.copy()
    if 'orig_fold' in orig:
        orig['orig_fold'] = orig['orig_fold'].astype(int)
    text, t, checks, ok = audit(df, orig if 'orig_fold' in orig else None)
    (out / 'split_audit.txt').write_text(text, encoding='utf-8')
    json.dump({'all_passed': ok, 'checks': [{'check': c, 'passed': bool(p), 'note': n} for c, p, n in checks],
               'per_fold': t.to_dict(orient='records'), 'sklearn': sklearn.__version__,
               'design': {'outer': 'StratifiedGroupKFold(5, shuffle=True, random_state=42)',
                          'calib': f'StratifiedGroupKFold({N_INNER}, shuffle=True, random_state=42+k), first fold',
                          'group': 'class folder + parent prefix NNN'}},
              open(out / 'split_audit.json', 'w'), indent=2)
    print(text)
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
