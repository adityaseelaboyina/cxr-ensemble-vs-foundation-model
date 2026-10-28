"""Analysis of saved predictions (CPU only): fusion rules with patient-grouped out-of-fold validation,
subset search, diversity, paired patient-level bootstrap with Holm correction, per-finding and subgroup results.
Reads <run_root>/<variant>/<model>/preds.npz and writes everything to out_dir."""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
          'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DISPLAY = {'resnet18': 'ResNet18', 'densenet121': 'DenseNet121', 'coatnet0': 'CoAtNet-0', 'vit_b16': 'ViT-B/16',
           'swin_b': 'Swin-B', 'evax_ti': 'EVA-X-Ti', 'evax_s': 'EVA-X-S', 'evax_b': 'EVA-X-B'}
FAMILY = {'resnet18': 'CNN', 'densenet121': 'CNN', 'coatnet0': 'CNN', 'vit_b16': 'ViT (ImageNet)',
          'swin_b': 'ViT (ImageNet)', 'evax_ti': 'ViT (CXR foundation)', 'evax_s': 'ViT (CXR foundation)',
          'evax_b': 'ViT (CXR foundation)'}
BASE5 = ['resnet18', 'densenet121', 'coatnet0', 'vit_b16', 'swin_b']


def arch(k):
    return k.split('@')[0]


def disp(k):
    a, _, tag = k.partition('@')
    return DISPLAY.get(a, a) + (f' ({tag})' if tag else '')
EPS = 1e-6


# ----------------------------------------------------------------------------- loading

def load_preds(sources):
    """sources: list of (model_key, path_to_preds.npz). Returns dict and the shared row indices."""
    P, rows = {}, None
    for key, path in sources:
        d = np.load(path)
        r = (tuple(d['val_rows']), tuple(d['test_rows']))
        if rows is None:
            rows = r
        elif r != rows:
            raise ValueError(f'{key}: validation/test rows differ from the other models; cannot combine')
        P[key] = (d['val'].astype(np.float64), d['test'].astype(np.float64))
    return P, np.array(rows[0]), np.array(rows[1])


def discover(run_root, variant='main', extra_roots=()):
    src = []
    for root in [run_root, *extra_roots]:
        vdir = os.path.join(root, variant)
        if not os.path.isdir(vdir):
            continue
        for k in sorted(os.listdir(vdir)):
            p = os.path.join(vdir, k, 'preds.npz')
            if os.path.exists(p) and k not in dict(src):
                src.append((k, p))
    order = {k: i for i, k in enumerate(list(DISPLAY))}
    return sorted(src, key=lambda kv: order.get(kv[0], 99))


# ----------------------------------------------------------------------------- metrics

def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def auroc(y, p):
    from scipy.stats import rankdata
    y = y.astype(bool)
    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        n1 = y[:, j].sum()
        n0 = len(y) - n1
        if n1 and n0:
            r = rankdata(p[:, j])
            out[j] = (r[y[:, j]].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def auprc(y, p):
    """Average precision (step-wise, as sklearn.average_precision_score), vectorized per label."""
    y = y.astype(bool)
    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        n1 = y[:, j].sum()
        if n1 == 0:
            continue
        o = np.argsort(-p[:, j], kind='mergesort')
        yj, pj = y[o, j], p[o, j]
        tp = np.cumsum(yj)
        k = np.arange(1, len(yj) + 1)
        # keep the last index of each tied score block
        last = np.r_[pj[1:] != pj[:-1], True]
        prec = tp[last] / k[last]
        rec = tp[last] / n1
        out[j] = np.sum(np.diff(np.r_[0, rec]) * prec)
    return out


def ece(y, p, bins=15):
    """Per-label expected calibration error with equal-width bins."""
    out = np.zeros(y.shape[1])
    edges = np.linspace(0, 1, bins + 1)
    for j in range(y.shape[1]):
        idx = np.clip(np.digitize(p[:, j], edges[1:-1]), 0, bins - 1)
        tot = 0.0
        for b in range(bins):
            m = idx == b
            if m.any():
                tot += m.mean() * abs(p[m, j].mean() - y[m, j].mean())
        out[j] = tot
    return out


def best_thresholds(y, p):
    from sklearn.metrics import precision_recall_curve
    thr = np.full(y.shape[1], 0.5)
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0:
            continue
        pr, rc, t = precision_recall_curve(y[:, j], p[:, j])
        f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
        thr[j] = t[int(np.argmax(f1))]
    return thr


def f1s(y, p, thr):
    y = y.astype(bool)
    pred = p >= thr
    tp = (pred & y).sum(0)
    fp = (pred & ~y).sum(0)
    fn = (~pred & y).sum(0)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    return f1, micro


def evaluate(y, p, thr):
    a, ap, e = auroc(y, p), auprc(y, p), ece(y, p)
    f1, micro = f1s(y, p, thr)
    brier = ((p - y) ** 2).mean(0)
    return {'macro_auroc': float(np.nanmean(a)), 'macro_auprc': float(np.nanmean(ap)),
            'macro_f1': float(f1.mean()), 'micro_f1': float(micro), 'mean_ece': float(e.mean()),
            'mean_brier': float(brier.mean()),
            'per_label': {lab: {'auroc': a[j], 'auprc': ap[j], 'f1': f1[j], 'ece': e[j], 'brier': brier[j],
                                'threshold': float(thr[j]), 'prevalence': float(y[:, j].mean())}
                          for j, lab in enumerate(LABELS)}}


# ----------------------------------------------------------------------------- fusion methods

def _group_folds(groups, k=5):
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=groups))


def _platt_fit(z, y):
    from sklearn.linear_model import LogisticRegression
    if y.min() == y.max():
        return (0.0, 1.0, float(_logit(np.array([y.mean()]))[0]))
    m = LogisticRegression(C=1e4, max_iter=1000).fit(z[:, None], y)
    return (1.0, float(m.coef_[0, 0]), float(m.intercept_[0]))


def _platt_apply(c, z):
    flag, a, b = c
    return _sigmoid(a * z + b) if flag else np.full_like(z, _sigmoid(b))


class Fuser:
    """Fit on validation, apply to any split. `oof` gives patient-grouped out-of-fold validation predictions."""

    name = 'base'

    def fit(self, Pv, yv):
        raise NotImplementedError

    def predict(self, P):
        raise NotImplementedError

    def oof(self, Pv, yv, folds):
        out = np.zeros_like(yv, dtype=np.float64)
        for tr, te in folds:
            f = self.__class__(**self.kw).fit([p[tr] for p in Pv], yv[tr])
            out[te] = f.predict([p[te] for p in Pv])
        return out

    def __init__(self, **kw):
        self.kw = kw


class MeanProb(Fuser):
    name = 'Average (probabilities)'

    def fit(self, Pv, yv):
        return self

    def predict(self, P):
        return np.mean(P, 0)

    def oof(self, Pv, yv, folds):
        return self.predict(Pv)  # nothing is fitted


class CalibratedMean(Fuser):
    """Per-model, per-label Platt scaling on validation, then average of calibrated logits."""
    name = 'Average (calibrated)'

    def fit(self, Pv, yv):
        self.c = [[_platt_fit(_logit(p[:, j]), yv[:, j]) for j in range(yv.shape[1])] for p in Pv]
        return self

    def predict(self, P):
        Z = [np.column_stack([_logit(_platt_apply(c[j], _logit(p[:, j]))) for j in range(p.shape[1])])
             for c, p in zip(self.c, P)]
        return _sigmoid(np.mean(Z, 0))


class WeightedMean(Fuser):
    """Global non-negative model weights (sum to 1) on calibrated logits, chosen to minimize validation log-loss,
    followed by per-label Platt scaling of the fused logit."""
    name = 'Weighted average (val-optimized)'

    def fit(self, Pv, yv):
        from scipy.optimize import minimize
        self.cal = CalibratedMean().fit(Pv, yv)
        Z = np.stack([_logit(self.cal.predict([p])) for p in Pv])  # per-model calibrated logits
        self._Zshape = Z.shape

        def loss(theta):
            w = np.exp(theta) / np.exp(theta).sum()
            q = np.clip(_sigmoid(np.tensordot(w, Z, 1)), EPS, 1 - EPS)
            return -np.mean(yv * np.log(q) + (1 - yv) * np.log(1 - q))
        r = minimize(loss, np.zeros(len(Pv)), method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-4, 'fatol': 1e-7})
        self.w = np.exp(r.x) / np.exp(r.x).sum()
        fused = np.tensordot(self.w, Z, 1)
        self.post = [_platt_fit(fused[:, j], yv[:, j]) for j in range(yv.shape[1])]
        return self

    def predict(self, P):
        Z = np.stack([_logit(self.cal.predict([p])) for p in P])
        fused = np.tensordot(self.w, Z, 1)
        return np.column_stack([_platt_apply(self.post[j], fused[:, j]) for j in range(fused.shape[1])])


class Stack(Fuser):
    """Per-label L2 logistic regression. features='all': all models' probabilities for all labels (5x14=70,
    the configuration used in the main experiment); features='label': only the M probabilities of that label."""
    name = 'Stacking (LR)'

    def __init__(self, features='all', C=1.0):
        super().__init__(features=features, C=C)
        self.features, self.C = features, C
        self.name = 'Stacking, all labels (LR)' if features == 'all' else 'Stacking, label-wise (LR)'

    def _X(self, P, j):
        return np.hstack(P) if self.features == 'all' else np.column_stack([p[:, j] for p in P])

    def fit(self, Pv, yv):
        from sklearn.linear_model import LogisticRegression
        self.m = []
        for j in range(yv.shape[1]):
            if yv[:, j].min() == yv[:, j].max():
                self.m.append(float(yv[:, j].mean()))
            else:
                self.m.append(LogisticRegression(C=self.C, max_iter=3000).fit(self._X(Pv, j), yv[:, j]))
        return self

    def predict(self, P):
        cols = []
        for j, m in enumerate(self.m):
            cols.append(np.full(len(P[0]), m) if isinstance(m, float) else m.predict_proba(self._X(P, j))[:, 1])
        return np.column_stack(cols)


def fusers():
    return [MeanProb(), CalibratedMean(), WeightedMean(), Stack('label'), Stack('all')]


def run_fuser(f, Pv, Pt, yv, folds):
    oof = f.oof(Pv, yv, folds)
    f.fit(Pv, yv)
    test = f.predict(Pt)
    return oof, test, best_thresholds(yv, oof)


# ----------------------------------------------------------------------------- bootstrap

class Bootstrap:
    """Patient-level bootstrap indices shared by every comparison (paired)."""

    def __init__(self, groups, n=1000, seed=42):
        rng = np.random.default_rng(seed)
        uniq, inv = np.unique(groups, return_inverse=True)
        order = np.argsort(inv, kind='stable')
        bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
        members = [order[bounds[g]:bounds[g + 1]] for g in range(len(uniq))]
        self.idx = [np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n)]

    def stat(self, fn):
        return np.array([fn(i) for i in self.idx])


def _ci(a):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return [float('nan'), float('nan')]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def _p_two_sided(d):
    # 2(k+1)/(B+1), k = resamples on the smaller side of zero. Never 0; smallest value with B=1000 is ~0.002.
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float('nan')
    k = min(int((d <= 0).sum()), int((d >= 0).sum()))
    return float(min(1.0, 2 * (k + 1) / (d.size + 1)))


def holm(pvals):
    p = np.nan_to_num(np.asarray(pvals, float), nan=1.0)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ----------------------------------------------------------------------------- main analysis

def analyze(run_root, cache_dir, out_dir, variant='main', extra_roots=(), n_boot=1000, seed=42,
            bench_json=None, max_subset_models=8, extra_sources=(), replace_sources=None, extra_comparisons=()):
    """extra_sources: iterable of (key, preds.npz path) added after the discovered models, e.g.
    ('evax_s', '<project>/runs_v3/evaxrecipe/evax_s/preds.npz')."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    meta = pd.read_csv(os.path.join(cache_dir, 'meta.csv'))
    Y = meta[LABELS].values.astype(np.float64)

    src = discover(run_root, variant, extra_roots)
    src += [(k, p) for k, p in extra_sources if k not in dict(src)]
    if replace_sources:  # sensitivity analyses, e.g. {'vit_b16': '<runs_v3>/vit_lr3e-5/vit_b16/preds.npz'}
        src = [(k, replace_sources.get(k, p)) for k, p in src]
    R_src = {k: p for k, p in src}
    if len(src) < 2:
        raise RuntimeError(f'Need at least two finished models in {run_root}/{variant}; found {[k for k, _ in src]}')
    P, vr, tr = load_preds(src)
    keys = list(P)
    yv, yt = Y[vr], Y[tr]
    gv, gt = meta['Patient ID'].values[vr], meta['Patient ID'].values[tr]
    folds = _group_folds(gv, 5)
    R = {'sources': R_src, 'models': keys, 'n_val': len(vr), 'n_test': len(tr), 'n_test_patients': int(len(np.unique(gt))),
         'variant': variant, 'protocol': 'thresholds/fusers fit on validation; validation metrics are out-of-fold',
         'p_method': 'add_one'}
    print(f'Models: {keys} | val {len(vr)} | test {len(tr)}')

    # ---- single models (thresholds on validation; validation predictions are in-sample for thresholds only)
    systems = {}
    single = {}
    for k in keys:
        pv, pt = P[k]
        thr = best_thresholds(yv, pv)
        single[k] = {'val': evaluate(yv, pv, thr), 'test': evaluate(yt, pt, thr)}
        systems[k] = (pt, thr)
    R['single'] = single

    # ---- fusion methods over the full set and the ImageNet-only five
    sets = {'all': keys}
    if all(k in keys for k in BASE5) and len(keys) > 5:
        sets['base5'] = BASE5
    fusion = {}
    for set_name, ks in sets.items():
        fusion[set_name] = {}
        for f in fusers():
            oof, test, thr = run_fuser(f, [P[k][0] for k in ks], [P[k][1] for k in ks], yv, folds)
            fusion[set_name][f.name] = {'val_oof': evaluate(yv, oof, thr), 'test': evaluate(yt, test, thr),
                                        'weights': getattr(f, 'w', None)}
            systems[f'{set_name}:{f.name}'] = (test, thr)
            print(f'  [{set_name}] {f.name:<34} val-OOF AUROC {fusion[set_name][f.name]["val_oof"]["macro_auroc"]:.4f}')
    R['fusion'] = fusion

    # ---- subset search (calibrated average, no fitting beyond Platt; selection on validation OOF only)
    subset_rows = []
    ks_for_subsets = keys[:max_subset_models]
    for r in range(1, len(ks_for_subsets) + 1):
        for comb in itertools.combinations(ks_for_subsets, r):
            f = CalibratedMean()
            oof, test, thr = run_fuser(f, [P[k][0] for k in comb], [P[k][1] for k in comb], yv, folds)
            subset_rows.append({'models': '+'.join(comb), 'size': r,
                                'val_oof_auroc': float(np.nanmean(auroc(yv, oof))),
                                'test_auroc': float(np.nanmean(auroc(yt, test))),
                                'test_auprc': float(np.nanmean(auprc(yt, test))),
                                'test_macro_f1': float(f1s(yt, test, thr)[0].mean())})
    sub = pd.DataFrame(subset_rows)
    sub.to_csv(os.path.join(out_dir, 'subsets_calibrated_average.csv'), index=False)
    chosen = sub.loc[sub.groupby('size')['val_oof_auroc'].idxmax()].reset_index(drop=True)
    chosen.to_csv(os.path.join(out_dir, 'best_subset_per_size_selected_on_val.csv'), index=False)
    R['best_subset_per_size'] = chosen.to_dict(orient='records')

    # ---- inference cost (params from model definitions; throughput if a per-model benchmark exists)
    cost = _model_costs(keys, bench_json)
    R['cost'] = cost
    if cost:
        sub['cost_ms_per_image'] = sub['models'].apply(lambda s: sum(cost[k].get('ms_per_image', np.nan) for k in s.split('+')))
        sub['params_m'] = sub['models'].apply(lambda s: sum(cost[k]['params_m'] for k in s.split('+')))
        sub.to_csv(os.path.join(out_dir, 'subsets_calibrated_average.csv'), index=False)

    # ---- diversity: pairwise correlation of errors vs pairwise fusion gain
    div = []
    for a, b in itertools.combinations(keys, 2):
        ea, eb = P[a][1] - yt, P[b][1] - yt
        corr = float(np.nanmean([np.corrcoef(ea[:, j], eb[:, j])[0, 1] for j in range(len(LABELS))]))
        rank_corr = float(np.nanmean([pd.Series(P[a][1][:, j]).corr(pd.Series(P[b][1][:, j]), method='spearman')
                                      for j in range(len(LABELS))]))
        pair = sub[sub['models'].isin(['+'.join([a, b]), '+'.join([b, a])])].iloc[0]
        best_member = max(single[a]['test']['macro_auroc'], single[b]['test']['macro_auroc'])
        div.append({'a': a, 'b': b, 'family_a': FAMILY.get(arch(a)), 'family_b': FAMILY.get(arch(b)),
                    'same_family': FAMILY.get(arch(a)) == FAMILY.get(arch(b)), 'error_corr': corr, 'score_spearman': rank_corr,
                    'pair_test_auroc': float(pair['test_auroc']), 'gain_over_better_member': float(pair['test_auroc'] - best_member)})
    div = pd.DataFrame(div)
    div.to_csv(os.path.join(out_dir, 'pairwise_diversity.csv'), index=False)
    from scipy.stats import spearmanr
    rho = spearmanr(div['error_corr'], div['gain_over_better_member'])
    R['diversity'] = {'spearman_errorcorr_vs_gain': float(rho.correlation), 'p': float(rho.pvalue), 'n_pairs': len(div),
                      'note': 'descriptive; pairs share models, so p-values are not independent'}

    # ---- primary comparisons (pre-specified), paired patient-level bootstrap, Holm-adjusted
    best_single = max(keys, key=lambda k: single[k]['val']['macro_auroc'])
    # The ensemble used for headline claims is the fuser with the best patient-grouped OOF validation AUROC.
    sel = {s: max(b, key=lambda n: b[n]['val_oof']['macro_auroc']) for s, b in fusion.items()}
    R['selected_fuser'] = sel
    ens_all = f"all:{sel['all']}"
    full = 'all:Stacking, all labels (LR)'
    comps = [('EnsVsBest', 'Ensemble (val-selected fuser) vs best single model', ens_all, best_single),
             ('StackVsCalAvg', 'Full stacking vs calibrated averaging', full, 'all:Average (calibrated)'),
             ('StackVsAvg', 'Full stacking vs probability averaging', full, 'all:Average (probabilities)'),
             ('CalAvgVsBest', 'Calibrated averaging vs best single model', 'all:Average (calibrated)', best_single)]
    if 'base5' in sets:
        evax = [k for k in keys if arch(k).startswith('evax')]
        if evax:
            fm = max(evax, key=lambda k: single[k]['val']['macro_auroc'])
            ens5 = f"base5:{sel['base5']}"
            comps += [('FiveVsFM', 'ImageNet five-model ensemble vs CXR foundation model', ens5, fm),
                      ('EnsVsFM', 'Ensemble including foundation model vs foundation model alone', ens_all, fm),
                      ('WithVsWithoutFM', 'Ensemble with vs without foundation model', ens_all, ens5)]
    comps += [c for c in extra_comparisons if c[2] in systems and c[3] in systems]
    print(f'Bootstrapping {n_boot} patient-level resamples...')
    B = Bootstrap(gt, n_boot, seed)
    names = sorted({c[2] for c in comps} | {c[3] for c in comps})
    stats = {}
    for nm in names:
        p, thr = systems[nm]
        stats[nm] = {'auroc': B.stat(lambda i: np.nanmean(auroc(yt[i], p[i]))),
                     'auprc': B.stat(lambda i: np.nanmean(auprc(yt[i], p[i]))),
                     'macro_f1': B.stat(lambda i: f1s(yt[i], p[i], thr)[0].mean())}
    prim = []
    for cid, title, a, b in comps:
        row = {'id': cid, 'comparison': title, 'a': a, 'b': b}
        for met in ('auroc', 'auprc', 'macro_f1'):
            pa, pb = systems[a][0], systems[b][0]
            point = {'auroc': lambda p: np.nanmean(auroc(yt, p)), 'auprc': lambda p: np.nanmean(auprc(yt, p)),
                     'macro_f1': None}[met]
            if met == 'macro_f1':
                da = f1s(yt, systems[a][0], systems[a][1])[0].mean() - f1s(yt, systems[b][0], systems[b][1])[0].mean()
            else:
                da = point(pa) - point(pb)
            d = stats[a][met] - stats[b][met]
            row[f'{met}_delta'] = float(da)
            row[f'{met}_ci'] = _ci(d)
            row[f'{met}_p'] = _p_two_sided(d)
        prim.append(row)
    # Holm over distinct (a, b) pairs; an identical pair (e.g. when the foundation model is also the best single
    # model) is one hypothesis and shares its adjusted p-value.
    pairs = list(dict.fromkeys((r['a'], r['b']) for r in prim))
    adj = dict(zip(pairs, holm([next(r['auroc_p'] for r in prim if (r['a'], r['b']) == pr) for pr in pairs])))
    for r in prim:
        r['auroc_p_holm'] = float(adj[(r['a'], r['b'])])
        r['duplicate_of'] = next(q['id'] for q in prim if (q['a'], q['b']) == (r['a'], r['b']))
    R['n_hypotheses'] = len(pairs)
    R['primary_comparisons'] = prim
    R['best_single_by_val'] = best_single
    R['system_cis'] = {nm: {m: _ci(v) for m, v in s.items()} for nm, s in stats.items()}

    # ---- per-label gains (stacking vs best single), paired bootstrap, Holm across 14 labels
    pa, pb = systems[ens_all][0], systems[best_single][0]
    da = B.stat(lambda i: auroc(yt[i], pa[i]) - auroc(yt[i], pb[i]))
    point = auroc(yt, pa) - auroc(yt, pb)
    pl = []
    for j, lab in enumerate(LABELS):
        pl.append({'label': lab, 'prevalence_test': float(yt[:, j].mean()), 'n_pos_test': int(yt[:, j].sum()),
                   'delta_auroc': float(point[j]), 'ci': _ci(da[:, j]), 'p': _p_two_sided(da[:, j])})
    for r, a in zip(pl, holm([r['p'] for r in pl])):
        r['p_holm'] = float(a)
    R['per_label_stack_vs_best'] = pl
    pd.DataFrame(pl).to_csv(os.path.join(out_dir, 'per_label_gain.csv'), index=False)

    # ---- subgroups with CIs
    tm = meta.iloc[tr].reset_index(drop=True)
    sg = {}
    groups = []
    if 'Patient Gender' in tm:
        groups += [('Male', (tm['Patient Gender'] == 'M').values), ('Female', (tm['Patient Gender'] == 'F').values)]
    if 'Patient Age' in tm:
        age = tm['Patient Age'].values
        groups += [('Age <40', age < 40), ('Age 40-59', (age >= 40) & (age < 60)), ('Age >=60', (age >= 60) & (age <= 100))]
    p_full = systems[ens_all][0]
    for name, m in groups:
        if m.sum() < 50:
            continue
        sub_b = Bootstrap(gt[m], max(200, n_boot // 5), seed)
        ym, pm = yt[m], p_full[m]
        sg[name] = {'images': int(m.sum()), 'patients': int(len(np.unique(gt[m]))),
                    'macro_auroc': float(np.nanmean(auroc(ym, pm))),
                    'macro_auroc_ci': _ci(sub_b.stat(lambda i: np.nanmean(auroc(ym[i], pm[i]))))}
    R['subgroups'] = sg

    _write_tables(R, out_dir)
    _write_macros(R, out_dir)
    _figures(R, sub, div, pl, systems, yt, ens_all, best_single, fig_dir, cost)
    with open(os.path.join(out_dir, 'analysis_v3.json'), 'w') as f:
        json.dump(_py(R), f, indent=2)
    print(f'Done in {time.time() - t0:.0f}s -> {out_dir}')
    return R


def _model_costs(keys, bench_json):
    """Parameter counts from the model definitions; latency from bench_v3 output if available."""
    out = {}
    try:
        sys.path.insert(0, HERE)
        import timm
        import evax
        names = {'resnet18': 'resnet18', 'densenet121': 'densenet121', 'coatnet0': 'coatnet_0_rw_224',
                 'vit_b16': 'vit_base_patch16_224', 'swin_b': 'swin_base_patch4_window7_224'}
        for k in keys:
            m = evax.build(arch(k)) if arch(k).startswith('evax') else timm.create_model(names[arch(k)], pretrained=False, num_classes=14)
            out[k] = {'params_m': sum(p.numel() for p in m.parameters()) / 1e6}
    except Exception as e:  # noqa: BLE001
        print('cost: could not build models for parameter counts:', e)
        return {}
    if bench_json and os.path.exists(bench_json):
        b = json.load(open(bench_json))
        for k in keys:
            if arch(k) in b.get('per_model', {}):
                out[k].update(b['per_model'][arch(k)])
    return out


# ----------------------------------------------------------------------------- outputs

def _py(o):
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _f(x, d=3):
    return '---' if x is None or (isinstance(x, float) and not math.isfinite(x)) else f'{x:.{d}f}'


def _table(caption, label, header, rows, align, note=None, small=False, wide=False):
    env = 'table*' if wide else 'table'
    L = [rf'\begin{{{env}}}[t]', rf'\caption{{{caption}}}', rf'\label{{{label}}}', r'\centering']
    if small:
        L += [r'\footnotesize', r'\setlength{\tabcolsep}{3.5pt}']
    L += [rf'\begin{{tabular}}{{{align}}}', r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
    for r in rows:
        L.append(r'\midrule' if r == 'MID' else ' & '.join(r) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    if note:
        L.append(rf'\\[2pt]{{\footnotesize {note}}}')
    L += [rf'\end{{{env}}}', '']
    return '\n'.join(L)


_FUSER_TAG = {'Average (probabilities)': 'Avg', 'Average (calibrated)': 'CalAvg',
              'Weighted average (val-optimized)': 'WAvg', 'Stacking, label-wise (LR)': 'StackLW',
              'Stacking, all labels (LR)': 'Stack'}
_MODEL_TAG = {'resnet18': 'ResNet', 'densenet121': 'Dense', 'coatnet0': 'CoAt', 'vit_b16': 'ViTB', 'swin_b': 'Swin',
              'evax_ti': 'EvaxTi', 'evax_s': 'EvaxS', 'evax_b': 'EvaxB'}


def _write_macros(R, out_dir):
    """\v<Set><System><Metric> macros, e.g. \vAllStackAUROC, \vBaseFiveCalAvgECE, \vEvaxSAUROC,
    and \vComp<A..G><Delta|Lo|Hi|P> for the pre-specified comparisons (in the order of R['primary_comparisons'])."""
    L = ['% Auto-generated by analysis_v3.py from saved predictions. Do not edit.']
    m = lambda n, v: L.append(rf'\newcommand{{\v{n}}}{{{v}}}')
    m('NTest', f"{R['n_test']:,}")
    m('NTestPat', f"{R['n_test_patients']:,}")
    m('NVal', f"{R['n_val']:,}")
    m('BestSingle', disp(R['best_single_by_val']))
    m('SelectedFuser', R['selected_fuser']['all'])
    met = [('macro_auroc', 'AUROC', 3), ('macro_auprc', 'AUPRC', 3), ('macro_f1', 'MacroF', 3), ('mean_ece', 'ECE', 3),
           ('mean_brier', 'Brier', 4)]
    for k, v in R['single'].items():
        for key, tag, d in met:
            m(f"{_MODEL_TAG.get(arch(k), arch(k))}{'Evaxrec' if '@' in k else ''}{tag}", _f(v['test'][key], d))
    for set_name, block in R['fusion'].items():
        st = 'All' if set_name == 'all' else 'BaseFive'
        for name, v in block.items():
            for key, tag, d in met:
                m(f'{st}{_FUSER_TAG[name]}{tag}', _f(v['test'][key], d))
    for c in R['primary_comparisons']:
        L.append(f"% {c['id']}: {c['comparison']}")
        t = c['id']
        m(f'{t}Delta', f"{c['auroc_delta']:+.4f}")
        m(f'{t}Lo', f"{c['auroc_ci'][0]:+.4f}")
        m(f'{t}Hi', f"{c['auroc_ci'][1]:+.4f}")
        m(f'{t}P', _f(c['auroc_p_holm'], 3))
        m(f'{t}FDelta', f"{c['macro_f1_delta']:+.3f}")
    for i, r in enumerate(R['best_subset_per_size']):
        m(f"Subset{'ABCDEFGH'[i]}AUROC", _f(r['test_auroc']))
    with open(os.path.join(out_dir, 'macros_v3.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def _write_tables(R, out_dir):
    T = {}
    # fusion comparison
    rows = []
    for set_name, block in R['fusion'].items():
        title = 'All models' if set_name == 'all' else 'ImageNet-pretrained five'
        if len(R['fusion']) > 1:
            rows.append([rf'\multicolumn{{6}}{{l}}{{\textit{{{title}}}}}'])
        for name, v in block.items():
            t = v['test']
            rows.append([name, _f(t['macro_auroc']), _f(t['macro_auprc']), _f(t['macro_f1']), _f(t['mean_ece']),
                         _f(t['mean_brier'], 4)])
    T['fusion'] = _table('Fusion Methods on the Test Set', 'tab:fusion',
                         ['Method', 'AUROC', 'AUPRC', 'Macro-F1', 'ECE', 'Brier'], rows, 'lccccc',
                         note='All fusers, calibrators, and thresholds are fit on the validation set.', small=True)
    # singles
    rows = [[disp(k), _f(v['test']['macro_auroc']), _f(v['test']['macro_auprc']), _f(v['test']['macro_f1']),
             _f(v['test']['mean_ece'])] for k, v in R['single'].items()]
    T['single'] = _table('Individual Backbones on the Test Set', 'tab:single',
                         ['Model', 'AUROC', 'AUPRC', 'Macro-F1', 'ECE'], rows, 'lcccc')
    # primary comparisons
    rows = []
    for c in R['primary_comparisons']:
        if c['duplicate_of'] != c['id']:
            continue
        rows.append([c['comparison'], f"{c['auroc_delta']:+.4f}",
                     f"[{c['auroc_ci'][0]:+.4f}, {c['auroc_ci'][1]:+.4f}]", _f(c['auroc_p_holm'], 3),
                     f"{c['macro_f1_delta']:+.3f}"])
    T['primary'] = _table('Pre-Specified Paired Comparisons (Test, Patient-Level Bootstrap)', 'tab:primary',
                          ['Comparison', r'$\Delta$AUROC', '95\\% CI', '$p_{\\text{Holm}}$', r'$\Delta$F1'], rows,
                          'lcccc', small=False, wide=True)
    # subsets
    rows = [[str(int(r['size'])), r['models'].replace('+', ' + ').replace('_', r'\_'), _f(r['val_oof_auroc']),
             _f(r['test_auroc'])] for r in R['best_subset_per_size']]
    T['subsets'] = _table('Best Ensemble of Each Size (Selected on Validation)', 'tab:subsets',
                          ['$k$', 'Members', 'Val AUROC', 'Test AUROC'], rows, 'clcc', small=True)
    # per-label
    rows = [[r['label'].replace('_', ' '), _f(r['prevalence_test'] * 100, 1), f"{r['delta_auroc']:+.3f}",
             f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]", _f(r['p_holm'], 3)] for r in R['per_label_stack_vs_best']]
    T['perlabel'] = _table(f"Per-Finding AUROC Change, Ensemble vs {disp(R['best_single_by_val'])}",
                           'tab:perlabel', ['Finding', 'Prev.\\ (\\%)', r'$\Delta$AUROC', '95\\% CI', '$p_{\\text{Holm}}$'],
                           rows, 'lcccc', small=True)
    for k, t in T.items():
        with open(os.path.join(out_dir, f'tab_v3_{k}.tex'), 'w') as f:
            f.write(t)


def _figures(R, sub, div, pl, systems, yt, full, best_single, fig_dir, cost):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})

    # reliability diagram: pooled over labels
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], ls='--', c='0.6', lw=0.8)
    for nm, lab in [(best_single, disp(best_single)), ('all:Average (probabilities)', 'Average'),
                    ('all:Average (calibrated)', 'Calibrated avg.'), (full, 'Selected ensemble')]:
        p = systems[nm][0].ravel()
        y = yt.ravel()
        q = np.quantile(p, np.linspace(0, 1, 16))
        idx = np.clip(np.searchsorted(q[1:-1], p), 0, 14)
        xs = [p[idx == b].mean() for b in range(15) if (idx == b).any()]
        ys = [y[idx == b].mean() for b in range(15) if (idx == b).any()]
        ax.plot(xs, ys, marker='o', ms=2.5, lw=1, label=lab)
    ax.set_xlabel('Predicted probability (equal-mass bins)')
    ax.set_ylabel('Observed frequency')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'reliability.pdf'))
    plt.close(fig)

    # ensemble size curve and Pareto frontier
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    xcol = 'cost_ms_per_image' if 'cost_ms_per_image' in sub and sub['cost_ms_per_image'].notna().all() else (
        'params_m' if 'params_m' in sub else 'size')
    xl = {'cost_ms_per_image': 'Inference cost (ms / image, summed over members)', 'params_m': 'Parameters (M)',
          'size': 'Number of models'}[xcol]
    ax.scatter(sub[xcol], sub['test_auroc'], s=8, c=sub['size'], cmap='viridis', alpha=0.8)
    s = sub.sort_values(xcol)
    front, best = [], -1
    for _, r in s.iterrows():
        if r['val_oof_auroc'] > best:
            best = r['val_oof_auroc']
            front.append(r)
    fr = pd.DataFrame(front)
    ax.plot(fr[xcol], fr['test_auroc'], c='k', lw=0.8, label='Frontier (chosen on val)')
    ax.set_xlabel(xl)
    ax.set_ylabel('Test macro-AUROC')
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'pareto.pdf'))
    plt.close(fig)

    # diversity vs gain
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for same, mk in [(True, 'o'), (False, '^')]:
        d = div[div['same_family'] == same]
        ax.scatter(d['error_corr'], d['gain_over_better_member'] * 100, marker=mk, s=14,
                   label='Same family' if same else 'Different family')
    ax.axhline(0, c='0.6', lw=0.8)
    ax.set_xlabel('Mean error correlation (test)')
    ax.set_ylabel('AUROC gain over better member (pts)')
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'diversity_gain.pdf'))
    plt.close(fig)

    # per-label gain vs prevalence
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    d = pd.DataFrame(pl)
    lo = d['delta_auroc'] - d['ci'].apply(lambda c: c[0])
    hi = d['ci'].apply(lambda c: c[1]) - d['delta_auroc']
    ax.errorbar(d['prevalence_test'] * 100, d['delta_auroc'] * 100, yerr=[lo * 100, hi * 100], fmt='o', ms=3, lw=0.8)
    for _, r in d.iterrows():
        ax.annotate(r['label'].replace('_', ' ')[:10], (r['prevalence_test'] * 100, r['delta_auroc'] * 100), fontsize=5.5,
                    xytext=(2, 2), textcoords='offset points')
    ax.axhline(0, c='0.6', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Test prevalence (%)')
    ax.set_ylabel(r'$\Delta$AUROC, ensemble vs best single (pts)')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'perlabel_gain.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_root', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--extra_roots', nargs='*', default=[])
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--bench_json', default=None)
    ap.add_argument('--extra_preds', nargs='*', default=[], help='key=path/to/preds.npz')
    a = ap.parse_args()
    extra = [tuple(s.split('=', 1)) for s in a.extra_preds]
    analyze(a.run_root, a.cache_dir, a.out_dir, extra_roots=a.extra_roots, n_boot=a.n_boot, bench_json=a.bench_json,
            extra_sources=extra)
