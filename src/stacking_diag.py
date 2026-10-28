"""Checks the full-stacking results on the same grouped 5-fold OOF protocol as analysis_v3:
reruns stacking on raw probabilities, refits it on standardized logits, and checks feature alignment
(test AUROC should collapse if each model's finding columns are shifted by one).

python stacking_diag.py --cache /content/cache --runs <project>/runs_v3/evaxrecipe --out diag.json"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_v3 import LABELS, _group_folds, auroc  # noqa: E402


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def feats(P, j, mode, space):
    X = [logit(p) if space == 'logit' else p for p in P]
    return np.hstack(X) if mode == 'all' else np.column_stack([x[:, j] for x in X])


def fit_predict(Pfit, yfit, Ppred, mode, space, C=1.0):
    out = np.zeros((len(Ppred[0]), yfit.shape[1]))
    coefs = []
    for j in range(yfit.shape[1]):
        Xf, Xp = feats(Pfit, j, mode, space), feats(Ppred, j, mode, space)
        if space == 'logit':
            mu, sd = Xf.mean(0), Xf.std(0) + 1e-8
            Xf, Xp = (Xf - mu) / sd, (Xp - mu) / sd
        if yfit[:, j].min() == yfit[:, j].max():
            out[:, j] = yfit[:, j].mean()
            continue
        m = LogisticRegression(C=C, max_iter=5000).fit(Xf, yfit[:, j])
        out[:, j] = m.predict_proba(Xp)[:, 1]
        coefs.append(float(np.abs(m.coef_).mean()))
    return out, float(np.mean(coefs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--runs', required=True, help='folder holding <model>/preds.npz')
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    meta = pd.read_csv(os.path.join(a.cache, 'meta.csv'))
    Y = meta[LABELS].values.astype(float)
    keys = a.models or sorted(k for k in os.listdir(a.runs) if os.path.exists(os.path.join(a.runs, k, 'preds.npz')))
    D = {k: np.load(os.path.join(a.runs, k, 'preds.npz')) for k in keys}
    vr, tr = D[keys[0]]['val_rows'], D[keys[0]]['test_rows']
    assert all(np.array_equal(D[k]['val_rows'], vr) and np.array_equal(D[k]['test_rows'], tr) for k in keys)
    Pv = [D[k]['val'].astype(float) for k in keys]
    Pt = [D[k]['test'].astype(float) for k in keys]
    yv, yt = Y[vr], Y[tr]
    folds = _group_folds(meta['Patient ID'].values[vr], 5)
    mac = lambda y, p: float(np.nanmean(auroc(y, p)))
    res = {'models': keys, 'prob_median': {k: float(np.median(p)) for k, p in zip(keys, Pv)},
           'calibrated_avg_reference': 'see analysis_v3.json'}
    for mode in ('all', 'label'):
        for space in ('prob', 'logit'):
            oof = np.zeros_like(yv)
            for f_tr, f_te in folds:
                oof[f_te], _ = fit_predict([p[f_tr] for p in Pv], yv[f_tr], [p[f_te] for p in Pv], mode, space)
            test, coef = fit_predict(Pv, yv, Pt, mode, space)
            insample, _ = fit_predict(Pv, yv, Pv, mode, space)
            r = {'val_oof_auroc': mac(yv, oof), 'val_insample_auroc': mac(yv, insample), 'test_auroc': mac(yt, test),
                 'mean_abs_coef': coef}
            if mode == 'all':  # alignment check: shift every model's finding columns by one at prediction time
                Pt_shuf = [np.roll(p, 1, axis=1) for p in Pt]
                r['test_auroc_labels_shifted'] = mac(yt, fit_predict(Pv, yv, Pt_shuf, mode, space)[0])
            res[f'{mode}_{space}'] = r
            print(f'{mode:5} {space:5} OOF {r["val_oof_auroc"]:.4f}  in-sample {r["val_insample_auroc"]:.4f}  '
                  f'test {r["test_auroc"]:.4f}  |coef| {coef:.2f}'
                  + (f'  labels shifted {r["test_auroc_labels_shifted"]:.4f}' if mode == 'all' else ''))
    res['probability_average'] = {'val': mac(yv, np.mean(Pv, 0)), 'test': mac(yt, np.mean(Pt, 0))}
    print('probability average  val', f"{res['probability_average']['val']:.4f}", 'test', f"{res['probability_average']['test']:.4f}")
    json.dump(res, open(a.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
