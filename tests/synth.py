"""Synthetic stand-in for a finished run (NOT real results). Mimics the on-disk layout of cxr_pipeline."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from analysis_v3 import LABELS
root = sys.argv[1]; rng = np.random.default_rng(0)
n_pat = 3000; imgs = rng.integers(1, 6, n_pat); pid = np.repeat(np.arange(n_pat), imgs); N = len(pid)
prev = np.array([.10,.025,.12,.18,.05,.06,.012,.045,.04,.02,.022,.015,.03,.002])
pat_risk = rng.normal(0, 1, n_pat)[pid]
Y = (rng.random((N, 14)) < np.clip(prev * np.exp(0.6 * pat_risk[:, None]), 0, .9)).astype(np.uint8)
Y[:, 13] |= (np.arange(N) % 400 == 0)  # make sure rare label has positives
split = np.where(pid < 2000, 'train', np.where(pid < 2300, 'val', 'test'))
meta = pd.DataFrame({'Image Index': [f'{i:08d}.png' for i in range(N)], 'Patient ID': pid,
                     'Patient Age': rng.integers(5, 95, n_pat)[pid], 'Patient Gender': np.where(rng.random(n_pat) < .55, 'M', 'F')[pid],
                     'Finding Labels': ['|'.join(l for l, v in zip(LABELS, r) if v) or 'No Finding' for r in Y], 'split': split})
for j, l in enumerate(LABELS): meta[l] = Y[:, j]
cache = os.path.join(root, 'cache'); os.makedirs(cache, exist_ok=True)
meta.to_csv(f'{cache}/meta.csv', index=False)
json.dump({'split_source': 'synthetic', 'patient_overlap': 0}, open(f'{cache}/prep_done.json', 'w'))
shared = rng.normal(0, 1, (N, 14)); fam = {f: rng.normal(0, 1, (N, 14)) for f in ['cnn', 'vit', 'fm']}
models = {'resnet18': ('cnn', 1.0), 'densenet121': ('cnn', 1.25), 'coatnet0': ('cnn', 1.3), 'vit_b16': ('vit', 1.05),
          'swin_b': ('vit', 1.3), 'evax_s': ('fm', 1.55)}
vr, tr = np.where(split == 'val')[0], np.where(split == 'test')[0]
for v in ['main']:
    for k, (f, s) in models.items():
        z = 2.2 * s * (Y - .5) + 0.9 * shared + 0.8 * fam[f] + 0.7 * rng.normal(0, 1, (N, 14))
        logit = z / 1.5 + np.log(prev / (1 - prev)) + 2.0   # miscalibrated upward like pos_weight training
        p = 1 / (1 + np.exp(-logit))
        d = os.path.join(root, 'runs' if not k.startswith('evax') else 'runs_v3', v, k); os.makedirs(d, exist_ok=True)
        np.savez_compressed(f'{d}/preds.npz', val=p[vr].astype(np.float32), test=p[tr].astype(np.float32), val_rows=vr, test_rows=tr)
        h = [{'epoch': e + 1, 'val_mean_auc': .7 + .01 * min(e, 6), 'epoch_seconds': 60} for e in range(12)]
        from analysis_v3 import auroc
        h[6]['val_mean_auc'] = float(np.nanmean(auroc(Y[vr], p[vr].astype(np.float32).astype(np.float64))))
        json.dump(h, open(f'{d}/history.json', 'w'))
        json.dump({'img_size': 224, 'batch_size': 32, 'max_epochs': 40, 'patience': 5, 'lr': 1e-4, 'weight_decay': 1e-4,
                   'warmup_epochs': 1, 'gamma': 2.0, 'augment': True, 'mixup_alpha': .4, 'seed': 42,
                   'select_metric': 'val_mean_auc', 'pretrained': True, 'limit': None}, open(f'{d}/config.json', 'w'))
print('synthetic run written', N, 'images')
