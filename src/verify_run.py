"""Read-only sanity checks on a finished run: official test count, no patient overlap, label columns,
prediction rows and ranges, matching configs, early stopping, and that reported numbers reproduce.

python verify_run.py --project <cxr_experiment> --cache /content/cache --report verify_report.json"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
          'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
OFFICIAL_TEST = 25596
CFG_KEYS = ['img_size', 'batch_size', 'max_epochs', 'patience', 'lr', 'weight_decay', 'warmup_epochs', 'gamma',
            'augment', 'mixup_alpha', 'seed', 'select_metric', 'pretrained', 'limit']


class Report:
    def __init__(self):
        self.items = []

    def add(self, level, area, msg):
        self.items.append({'level': level, 'area': area, 'msg': msg})
        print(f'[{level:4}] {area:8} {msg}')

    def ok(self, a, m):
        self.add('PASS', a, m)

    def warn(self, a, m):
        self.add('WARN', a, m)

    def fail(self, a, m):
        self.add('FAIL', a, m)


def _auroc(y, p):
    from scipy.stats import rankdata
    y = y.astype(bool)
    out = []
    for j in range(y.shape[1]):
        n1 = y[:, j].sum()
        n0 = len(y) - n1
        if n1 and n0:
            r = rankdata(p[:, j])
            out.append((r[y[:, j]].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(out))


def check_data(cache, R):
    meta = pd.read_csv(os.path.join(cache, 'meta.csv'))
    done = json.load(open(os.path.join(cache, 'prep_done.json')))
    n_test = int((meta.split == 'test').sum())
    if 'official' in done.get('split_source', ''):
        (R.ok if n_test == OFFICIAL_TEST else R.fail)('data', f'official test split: {n_test} images (expected {OFFICIAL_TEST})')
    else:
        R.warn('data', f"split source is not the official list: {done.get('split_source')}")
    pats = {s: set(meta.loc[meta.split == s, 'Patient ID']) for s in ('train', 'val', 'test')}
    ov = {f'{a}&{b}': len(pats[a] & pats[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    (R.ok if sum(ov.values()) == 0 else R.fail)('data', f'patient overlap between splits: {ov}')
    fl = meta['Finding Labels'].str.split('|')
    bad = sum(int((fl.apply(lambda s, l=l: l in s).astype(int) != meta[l]).sum()) for l in LABELS)
    (R.ok if bad == 0 else R.fail)('data', f'label columns consistent with Finding Labels (mismatches: {bad})')
    R.ok('data', f"split sizes {meta.split.value_counts().to_dict()}")
    return meta


def check_runs(run_root, meta, R, variant='main'):
    vdir = os.path.join(run_root, variant)
    if not os.path.isdir(vdir):
        R.fail('runs', f'missing {vdir}')
        return {}
    Y = meta[LABELS].values
    split_rows = {s: np.where(meta.split.values == s)[0] for s in ('val', 'test')}
    rows0, cfg0, out = None, None, {}
    for k in sorted(os.listdir(vdir)):
        d = os.path.join(vdir, k)
        has = {f: os.path.exists(os.path.join(d, f)) for f in ('preds.npz', 'history.json', 'config.json', 'best.pt', 'last.pt')}
        if not has['preds.npz']:
            R.warn('runs', f'{variant}/{k}: not finished (last.pt present: {has["last.pt"]})')
            continue
        p = np.load(os.path.join(d, 'preds.npz'))
        cfg = json.load(open(os.path.join(d, 'config.json')))
        hist = json.load(open(os.path.join(d, 'history.json')))
        rows = (tuple(p['val_rows']), tuple(p['test_rows']))
        if cfg.get('limit'):
            R.fail('runs', f'{variant}/{k}: trained with limit={cfg["limit"]} (subset run, not the full experiment)')
        if not np.array_equal(p['val_rows'], split_rows['val']) or not np.array_equal(p['test_rows'], split_rows['test']):
            R.fail('runs', f'{variant}/{k}: prediction rows do not match the full val/test split')
        if rows0 is None:
            rows0 = rows
        elif rows != rows0:
            R.fail('runs', f'{variant}/{k}: rows differ from other models')
        for s in ('val', 'test'):
            a = p[s]
            if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
                R.fail('runs', f'{variant}/{k}: {s} probabilities not finite or outside [0,1]')
        c = {x: cfg.get(x) for x in CFG_KEYS}
        if cfg0 is None:
            cfg0 = c
        elif c != cfg0:
            diff = {x: (cfg0[x], c[x]) for x in CFG_KEYS if cfg0[x] != c[x]}
            R.fail('runs', f'{variant}/{k}: config differs from other models: {diff}')
        # early stopping and best-epoch consistency
        aucs = [h['val_mean_auc'] for h in hist]
        best_ep = int(np.argmax(aucs)) + 1
        n = len(hist)
        stopped_ok = n == cfg['max_epochs'] or (n - best_ep) == cfg['patience']
        (R.ok if stopped_ok else R.warn)('runs', f'{variant}/{k}: {n} epochs, best epoch {best_ep}, '
                                         f'early-stopping rule {"consistent" if stopped_ok else "INCONSISTENT"}')
        recomputed = _auroc(Y[p['val_rows']], p['val'])
        gap = abs(recomputed - max(aucs))
        (R.ok if gap < 2e-3 else R.warn)('runs', f'{variant}/{k}: val AUROC from saved preds {recomputed:.4f} vs '
                                         f'history best {max(aucs):.4f} (|diff| {gap:.4f})')
        if variant == 'main' and 'runs_v3' not in run_root and not has['best.pt']:
            R.warn('runs', f'{variant}/{k}: best.pt missing (needed for benchmark / Grad-CAM)')
        out[k] = {'epochs': n, 'best_epoch': best_ep, 'val_auroc_recomputed': recomputed,
                  'test_auroc': _auroc(Y[p['test_rows']], p['test'])}
    return out


def check_results(project, meta, R):
    s = os.path.join(project, 'results', 'summary.json')
    if not os.path.exists(s):
        R.warn('results', 'results/summary.json not found (run the Evaluate cell)')
        return
    S = json.load(open(s))
    Y = meta[LABELS].values
    for k, v in S.get('models', {}).items():
        p = os.path.join(project, 'runs', 'main', k, 'preds.npz')
        if not os.path.exists(p):
            continue
        d = np.load(p)
        a = _auroc(Y[d['test_rows']], d['test'])
        rep = v['test']['mean_auc']
        (R.ok if abs(a - rep) < 1e-6 else R.fail)('results', f'{k}: reported test AUROC {rep:.4f} reproduces ({a:.4f})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--report', default=None)
    a = ap.parse_args()
    R = Report()
    meta = check_data(a.cache, R)
    runs = {v: check_runs(os.path.join(a.project, 'runs'), meta, R, v)
            for v in sorted(os.listdir(os.path.join(a.project, 'runs'))) if os.path.isdir(os.path.join(a.project, 'runs', v))}
    v3 = os.path.join(a.project, 'runs_v3')
    if os.path.isdir(v3):
        for v in sorted(os.listdir(v3)):
            if os.path.isdir(os.path.join(v3, v)):
                runs[f'runs_v3/{v}'] = check_runs(v3, meta, R, v)
    check_results(a.project, meta, R)
    n_fail = sum(i['level'] == 'FAIL' for i in R.items)
    n_warn = sum(i['level'] == 'WARN' for i in R.items)
    print(f'\n{n_fail} FAIL, {n_warn} WARN')
    if a.report:
        json.dump({'items': R.items, 'runs': runs}, open(a.report, 'w'), indent=2, default=float)
    return n_fail


if __name__ == '__main__':
    raise SystemExit(main())
