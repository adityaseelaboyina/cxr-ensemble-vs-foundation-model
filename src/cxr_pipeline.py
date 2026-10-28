"""Main experiment on NIH ChestX-ray14: data prep with the official split, training the five ImageNet
backbones (shared recipe), evaluation, latency benchmark and Grad-CAM. Every step resumes where it left off."""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
          'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
          'Pleural_Thickening', 'Hernia']

# key: (timm model name, display name). Same pretrained weights as the original notebook.
MODELS = {
    'resnet18': ('resnet18.tv_in1k', 'ResNet18'),
    'densenet121': ('densenet121.tv_in1k', 'DenseNet121'),
    'coatnet0': ('coatnet_0_rw_224.sw_in1k', 'CoAtNet-0'),
    'vit_b16': ('vit_base_patch16_224', 'ViT-B/16'),
    'swin_b': ('swin_base_patch4_window7_224', 'Swin-B'),
}
MODEL_KEYS = list(MODELS)
CNN_KEYS = ['resnet18', 'densenet121', 'coatnet0']
TRANSFORMER_KEYS = ['vit_b16', 'swin_b']
CAM_LAYERS = {'densenet121': 'features', 'resnet18': 'layer4'}

DEFAULTS = dict(
    img_size=224, batch_size=32, max_epochs=40, patience=5, lr=1e-4, weight_decay=1e-4,
    warmup_epochs=1, gamma=2.0, augment=True, mixup_alpha=0.4, seed=42, num_workers=None,
    select_metric='val_mean_auc',
)

# Every experiment in the paper is one of these variants.
VARIANTS = {
    'main': {},
    'gamma0': {'gamma': 0.0},          # no focal loss (weighted BCE); also gamma = 0 in the sweep
    'gamma1': {'gamma': 1.0},
    'gamma3': {'gamma': 3.0},
    'no_aug': {'augment': False, 'mixup_alpha': 0.0},
    'lr1e-3': {'lr': 1e-3},
    'lr5e-5': {'lr': 5e-5},
}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- utils

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _py(o):
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(_py(obj), f, indent=2)


def _atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _amp_dtype(device):
    if device.type != 'cuda':
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _workers(cfg):
    if cfg.get('num_workers') is not None:
        return cfg['num_workers']
    return max(1, min(12, (os.cpu_count() or 2)))


# ----------------------------------------------------------------------------- data

def _find_root(path):
    for dirpath, _, files in os.walk(path):
        if 'Data_Entry_2017.csv' in files:
            return dirpath
    raise FileNotFoundError(f'Data_Entry_2017.csv not found under {path}')


def _index_images(root):
    idx = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith('.png'):
                idx[f] = os.path.join(dirpath, f)
    return idx


def _load_resize(path, size):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert('L').resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def prepare_data(raw_dir, cache_dir, img_size=224, val_frac=0.1, seed=42, workers=None):
    """Patient-wise split + 224x224 uint8 image cache (about 5.6 GB)."""
    from sklearn.model_selection import GroupShuffleSplit
    os.makedirs(cache_dir, exist_ok=True)
    done_p = os.path.join(cache_dir, 'prep_done.json')
    if os.path.exists(done_p):
        print('Cache already prepared:', cache_dir)
        return pd.read_csv(os.path.join(cache_dir, 'meta.csv'))

    root = _find_root(raw_dir)
    df = pd.read_csv(os.path.join(root, 'Data_Entry_2017.csv'))
    paths = _index_images(root)
    n0 = len(df)
    df = df[df['Image Index'].isin(paths)].copy()
    print(f'Images found: {len(df)} of {n0} listed')
    findings = df['Finding Labels'].str.split('|')
    for lab in LABELS:
        df[lab] = findings.apply(lambda s, lab=lab: lab in s).astype(np.uint8)

    tv_list = os.path.join(root, 'train_val_list.txt')
    te_list = os.path.join(root, 'test_list.txt')
    if os.path.exists(tv_list) and os.path.exists(te_list):
        test_names = set(open(te_list).read().split())
        df['split'] = np.where(df['Image Index'].isin(test_names), 'test', 'trainval')
        split_source = 'official NIH train_val_list / test_list'
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        _, te = next(gss.split(df, groups=df['Patient ID']))
        df['split'] = 'trainval'
        df.iloc[te, df.columns.get_loc('split')] = 'test'
        split_source = 'patient-wise random 10% test (official lists not found)'

    tv = np.where(df['split'] == 'trainval')[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_i, va_i = next(gss.split(tv, groups=df['Patient ID'].values[tv]))
    df.iloc[tv[tr_i], df.columns.get_loc('split')] = 'train'
    df.iloc[tv[va_i], df.columns.get_loc('split')] = 'val'
    df = df.reset_index(drop=True)

    pats = {s: set(df.loc[df.split == s, 'Patient ID']) for s in ('train', 'val', 'test')}
    overlap = len(pats['train'] & pats['test']) + len(pats['val'] & pats['test']) + len(pats['train'] & pats['val'])
    print('Split source:', split_source, '| patient overlap between splits:', overlap)

    img_p = os.path.join(cache_dir, f'images_{img_size}.npy')
    arr = np.lib.format.open_memmap(img_p, mode='w+', dtype=np.uint8, shape=(len(df), img_size, img_size))
    files = [paths[n] for n in df['Image Index']]
    workers = workers or max(1, os.cpu_count() or 1)
    t0 = time.time()
    with Pool(workers) as pool:
        for i, im in enumerate(pool.imap(partial(_load_resize, size=img_size), files, chunksize=64)):
            arr[i] = im
            if (i + 1) % 10000 == 0:
                print(f'  resized {i + 1}/{len(files)} ({time.time() - t0:.0f}s)')
    arr.flush()
    del arr
    keep = ['Image Index', 'Patient ID', 'Patient Age', 'Patient Gender', 'View Position', 'Finding Labels', 'split'] + LABELS
    df[[c for c in keep if c in df.columns]].to_csv(os.path.join(cache_dir, 'meta.csv'), index=False)
    _write_json({'split_source': split_source, 'patient_overlap': overlap,
                 'counts': df['split'].value_counts().to_dict(), 'img_size': img_size}, done_p)
    print(f'Cache ready in {time.time() - t0:.0f}s:', df['split'].value_counts().to_dict())
    return df


def get_data(local_cache, drive_cache=None, raw_dir=None, img_size=224, workers=None):
    """Use the Drive cache if it exists, otherwise download from Kaggle and build it."""
    names = ['meta.csv', f'images_{img_size}.npy', 'prep_done.json']

    def complete(d):
        return d and all(os.path.exists(os.path.join(d, n)) for n in names)

    if complete(local_cache):
        print('Local cache ready:', local_cache)
        return local_cache
    if complete(drive_cache):
        os.makedirs(local_cache, exist_ok=True)
        t0 = time.time()
        for n in names:
            shutil.copy(os.path.join(drive_cache, n), os.path.join(local_cache, n))
        print(f'Copied cache from Drive in {time.time() - t0:.0f}s')
        return local_cache
    if raw_dir is None:
        import kagglehub
        raw_dir = kagglehub.dataset_download('nih-chest-xrays/data')
    prepare_data(raw_dir, local_cache, img_size=img_size, workers=workers)
    if drive_cache:
        os.makedirs(drive_cache, exist_ok=True)
        for n in names:
            shutil.copy(os.path.join(local_cache, n), os.path.join(drive_cache, n))
        print('Saved cache to Drive:', drive_cache)
    return local_cache


def _build_augment():
    import albumentations as A
    major = int(A.__version__.split('.')[0])
    noise = (A.GaussNoise(std_range=(0.0, 0.02), p=0.3) if major >= 2
             else A.GaussNoise(var_limit=(0.0, (0.02 * 255) ** 2), p=0.3))
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), scale=(0.9, 1.1), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        noise,
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


class CXRDataset(Dataset):
    """Reads cached uint8 images. Normalization happens on the GPU."""

    def __init__(self, img_path, rows, labels, augment=False):
        self.img_path, self.rows, self.labels, self.augment = img_path, np.asarray(rows), labels, augment
        self.arr = None
        self.tf = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.arr is None:
            self.arr = np.load(self.img_path, mmap_mode='r')
            if self.augment:
                self.tf = _build_augment()
        img = np.array(self.arr[self.rows[i]])
        if self.tf is not None:
            img = self.tf(image=img)['image']
        return torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(self.labels[i])


def _to_input(x, device, channels_last=False):
    x = x.to(device, non_blocking=True).float().div_(255).unsqueeze(1).expand(-1, 3, -1, -1)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.contiguous(memory_format=torch.channels_last) if channels_last else x.contiguous()


def _load_split(cache_dir, img_size=224):
    meta = pd.read_csv(os.path.join(cache_dir, 'meta.csv'))
    Y = meta[LABELS].values.astype(np.float32)
    idx = {s: np.where(meta['split'].values == s)[0] for s in ('train', 'val', 'test')}
    return meta, Y, idx, os.path.join(cache_dir, f'images_{img_size}.npy')


# ----------------------------------------------------------------------------- model and loss

def create_model(key, pretrained=True):
    import timm
    return timm.create_model(MODELS[key][0], pretrained=pretrained, num_classes=len(LABELS))


class FocalLoss(nn.Module):
    """Class-balanced focal loss. Positives are weighted by pos_weight = neg/pos (about 1/f_j).
    gamma = 0 gives weighted BCE."""

    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('pos_weight', pos_weight if pos_weight is not None else torch.tensor(1.0))

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = bce if self.gamma == 0 else (1 - torch.exp(-bce)) ** self.gamma * bce
        weight = 1 + (self.pos_weight - 1) * targets
        return (weight * loss).mean()


@torch.no_grad()
def predict(model, loader, device, dtype=None, channels_last=False):
    model.eval()
    out = []
    for x, _ in loader:
        x = _to_input(x, device, channels_last)
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            out.append(torch.sigmoid(model(x).float()).cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- metrics

def per_label_auc(y, p):
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


def best_thresholds(y, p):
    """Per-label threshold that maximizes F1 on the given (validation) data."""
    from sklearn.metrics import precision_recall_curve
    thr = np.full(y.shape[1], 0.5)
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0:
            continue
        prec, rec, t = precision_recall_curve(y[:, j], p[:, j])
        f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
        thr[j] = t[int(np.argmax(f1))]
    return thr


def _counts(y, pred):
    y = y.astype(bool)
    tp = (pred & y).sum(0)
    fp = (pred & ~y).sum(0)
    fn = (~pred & y).sum(0)
    return tp, fp, fn


def summarize(y, p, thr):
    pred = p >= np.asarray(thr)
    tp, fp, fn = _counts(y, pred)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    auc = per_label_auc(y, p)
    return {
        'macro_f1': float(f1.mean()),
        'micro_f1': float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)),
        'mean_auc': float(np.nanmean(auc)),
        'per_class': {lab: {'precision': precision[j], 'recall': recall[j], 'f1': f1[j], 'auc': auc[j],
                            'threshold': float(np.asarray(thr).reshape(-1)[j] if np.ndim(thr) else thr),
                            'support': int(y[:, j].sum())} for j, lab in enumerate(LABELS)},
    }


def bootstrap(y, systems, groups, n=1000, seed=42):
    """Patient-level bootstrap. systems: name -> (probs, thresholds). Same resamples for every system."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    members = [order[bounds[g]:bounds[g + 1]] for g in range(len(uniq))]
    yb = y.astype(bool)
    out = {k: {'macro_f1': [], 'micro_f1': [], 'mean_auc': []} for k in systems}
    preds = {k: p >= np.asarray(t) for k, (p, t) in systems.items()}
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        for k, (p, _) in systems.items():
            tp, fp, fn = _counts(yb[idx], preds[k][idx])
            f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
            out[k]['macro_f1'].append(f1.mean())
            out[k]['micro_f1'].append(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1))
            out[k]['mean_auc'].append(np.nanmean(per_label_auc(yb[idx], p[idx])))
    return {k: {m: np.array(v) for m, v in d.items()} for k, d in out.items()}


def _ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ----------------------------------------------------------------------------- training

def _loaders(img_path, Y, idx, cfg, train_rows=None, eval_bs=None, device=None):
    nw = _workers(cfg)
    pin = device is not None and device.type == 'cuda'
    rows = idx['train'] if train_rows is None else train_rows
    dl_tr = DataLoader(CXRDataset(img_path, rows, Y[rows], augment=cfg['augment']),
                       batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                       num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
    ebs = eval_bs or cfg['batch_size'] * 2
    ekw = dict(num_workers=min(nw, 4), pin_memory=pin)
    dl_va = DataLoader(CXRDataset(img_path, idx['val'], Y[idx['val']]), batch_size=ebs, **ekw)
    dl_te = DataLoader(CXRDataset(img_path, idx['test'], Y[idx['test']]), batch_size=ebs, **ekw)
    return dl_tr, dl_va, dl_te


def train_model(key, variant, run_root, cache_dir, overrides=None, pretrained=True, limit=None):
    """Fine-tune one backbone. Resumes from the last finished epoch.
    limit: use only this many images per split (for quick tests)."""
    cfg = {**DEFAULTS, **VARIANTS[variant], **(overrides or {})}
    out = os.path.join(run_root, variant, key)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, 'preds.npz')):
        print(f'[{variant}/{key}] already finished, skipping')
        return
    seed_everything(cfg['seed'])
    device = _device()
    meta, Y, idx, img_path = _load_split(cache_dir, cfg['img_size'])
    if limit:
        rng = np.random.default_rng(cfg['seed'])
        idx = {s: np.sort(rng.choice(v, min(limit, len(v)), replace=False)) for s, v in idx.items()}
    tr = idx['train']
    pos = Y[tr].sum(0)
    pos_weight = torch.tensor((len(tr) - pos) / np.maximum(pos, 1), dtype=torch.float32)

    dl_tr, dl_va, dl_te = _loaders(img_path, Y, idx, cfg, device=device)
    channels_last = key in CNN_KEYS and device.type == 'cuda'
    model = create_model(key, pretrained).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    spe = len(dl_tr)
    total, warm = spe * cfg['max_epochs'], max(1, spe * cfg['warmup_epochs'])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    loss_fn = FocalLoss(cfg['gamma'], pos_weight).to(device)
    dtype = _amp_dtype(device)
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)

    start, best, bad, history = 0, -np.inf, 0, []
    last_p, best_p = os.path.join(out, 'last.pt'), os.path.join(out, 'best.pt')
    if os.path.exists(last_p):
        ck = torch.load(last_p, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        start, best, bad, history = ck['epoch'], ck['best'], ck['bad'], ck['history']
        print(f'[{variant}/{key}] resuming after epoch {start} (best {cfg["select_metric"]} {best:.4f})')
    _write_json({'key': key, 'variant': variant, 'pretrained': pretrained, 'limit': limit,
                 'n_train': len(idx['train']), 'n_val': len(idx['val']), 'n_test': len(idx['test']),
                 'pos_weight': pos_weight.numpy(), **cfg}, os.path.join(out, 'config.json'))

    yv = Y[idx['val']]
    for epoch in range(start, cfg['max_epochs']):
        if bad >= cfg['patience']:
            break
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, y in dl_tr:
            x = _to_input(x, device, channels_last)
            y = y.to(device, non_blocking=True)
            lam, perm = 1.0, None
            if cfg['mixup_alpha'] > 0:
                lam = float(np.random.beta(cfg['mixup_alpha'], cfg['mixup_alpha']))
                perm = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[perm]
            with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
                logits = model(x)
            logits = logits.float()
            loss = loss_fn(logits, y) if perm is None else lam * loss_fn(logits, y) + (1 - lam) * loss_fn(logits, y[perm])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        train_s = time.time() - t0
        pv = predict(model, dl_va, device, dtype, channels_last)
        rec = {
            'epoch': epoch + 1, 'train_loss': tot / max(n, 1),
            'val_mean_auc': float(np.nanmean(per_label_auc(yv, pv))),
            'val_macro_f1@0.5': summarize(yv, pv, 0.5)['macro_f1'],
            'val_macro_f1_tuned': summarize(yv, pv, best_thresholds(yv, pv))['macro_f1'],
            'lr': opt.param_groups[0]['lr'], 'train_seconds': train_s, 'epoch_seconds': time.time() - t0,
        }
        history.append(rec)
        score = rec[cfg['select_metric']]
        if score > best:
            best, bad = score, 0
            _atomic_save(model.state_dict(), best_p)
        else:
            bad += 1
        _atomic_save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                      'scaler': scaler.state_dict(), 'epoch': epoch + 1, 'best': best, 'bad': bad,
                      'history': history}, last_p)
        note = '  *best' if bad == 0 else f'  (no improvement {bad}/{cfg["patience"]})'
        print(f"[{variant}/{key}] ep {epoch + 1:2d} loss {rec['train_loss']:.4f} | val AUC {rec['val_mean_auc']:.4f} "
              f"F1@0.5 {rec['val_macro_f1@0.5']:.4f} F1tuned {rec['val_macro_f1_tuned']:.4f} | "
              f"{rec['epoch_seconds'] / 60:.1f} min{note}")

    model.load_state_dict(torch.load(best_p, map_location=device))
    pv = predict(model, dl_va, device, dtype, channels_last)
    pt = predict(model, dl_te, device, dtype, channels_last)
    np.savez_compressed(os.path.join(out, 'preds.npz'), val=pv, test=pt, val_rows=idx['val'], test_rows=idx['test'])
    _write_json(history, os.path.join(out, 'history.json'))
    if os.path.exists(last_p):
        os.remove(last_p)
    if variant != 'main' and os.path.exists(best_p):
        os.remove(best_p)          # only the main checkpoints are needed later
    print(f'[{variant}/{key}] done: best {cfg["select_metric"]} {best:.4f}')


def probe(keys, cache_dir, steps=25, epochs_guess=12):
    """Times real training steps per model to estimate the GPU budget for one run."""
    cfg = dict(DEFAULTS)
    device = _device()
    meta, Y, idx, img_path = _load_split(cache_dir)
    dl_tr, _, _ = _loaders(img_path, Y, idx, cfg, device=device)
    dtype = _amp_dtype(device)
    rows = []
    for key in keys:
        cl = key in CNN_KEYS and device.type == 'cuda'
        model = create_model(key, pretrained=False).to(device)
        if cl:
            model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = FocalLoss(2.0).to(device)
        it = iter(dl_tr)
        t0, seen = None, 0
        for s in range(steps):
            x, y = next(it)
            if s == 5:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0, seen = time.time(), 0
            x, y = _to_input(x, device, cl), y.to(device)
            with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            seen += x.size(0)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        ips = seen / (time.time() - t0)
        ep_min = (len(idx['train']) / ips + (len(idx['val']) + len(idx['test'])) / (3 * ips)) / 60
        rows.append({'model': MODELS[key][1], 'train_img_per_s': round(ips), 'min_per_epoch': round(ep_min, 1),
                     f'hours_for_{epochs_guess}_epochs': round(ep_min * epochs_guess / 60, 1)})
        del model, opt
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"Total for one full run of these models at ~{epochs_guess} epochs each: "
          f"{df[f'hours_for_{epochs_guess}_epochs'].sum():.1f} GPU hours (early stopping usually ends sooner or later)")
    return df


# ----------------------------------------------------------------------------- evaluation

def _load_preds(run_root, variant, keys):
    got = {}
    for k in keys:
        p = os.path.join(run_root, variant, k, 'preds.npz')
        if os.path.exists(p):
            d = np.load(p)
            got[k] = (d['val'], d['test'], d['val_rows'], d['test_rows'])
    return got


def _fit_lr(X, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    if y.min() == y.max():
        return ('const', float(y.mean()))
    return LogisticRegression(C=C, max_iter=2000).fit(X, y)


def _pred_lr(m, X):
    if isinstance(m, tuple):
        return np.full(len(X), m[1])
    return m.predict_proba(X)[:, 1]


def stack(Xv, yv, gv, Xt, folds=5, C=1.0):
    """Per-label logistic regression on concatenated base-model probabilities.
    Thresholds come from out-of-fold validation predictions, so nothing is tuned on its own fit."""
    from sklearn.model_selection import GroupKFold
    oof = np.zeros_like(yv, dtype=np.float64)
    for tr, te in GroupKFold(n_splits=folds).split(Xv, groups=gv):
        for j in range(yv.shape[1]):
            oof[te, j] = _pred_lr(_fit_lr(Xv[tr], yv[tr, j], C), Xv[te])
    models = [_fit_lr(Xv, yv[:, j], C) for j in range(yv.shape[1])]
    test = np.column_stack([_pred_lr(m, Xt) for m in models])
    return oof, test, best_thresholds(yv, oof), models


def _ensemble(preds, keys, yv, gv, how):
    Pv = [preds[k][0] for k in keys]
    Pt = [preds[k][1] for k in keys]
    if how == 'average':
        v, t = np.mean(Pv, 0), np.mean(Pt, 0)
        return v, t, best_thresholds(yv, v), None
    return stack(np.hstack(Pv), yv, gv, np.hstack(Pt))


def _fmt(x):
    return '---' if x is None or (isinstance(x, float) and not math.isfinite(x)) else f'{x:.3f}'


def _latex_table(caption, label, header, rows, align, small=False):
    lines = [r'\begin{table}[t]', rf'\caption{{{caption}}}', rf'\label{{{label}}}', r'\centering']
    if small:
        lines += [r'\footnotesize', r'\setlength{\tabcolsep}{4pt}']
    lines += [
             rf'\begin{{tabular}}{{{align}}}', r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
    for r in rows:
        lines.append(r'\midrule' if r == 'MID' else ' & '.join(r) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(lines)


def evaluate_all(run_root, cache_dir, out_dir, sweep_model='densenet121', n_boot=1000, seed=42, folds=5):
    import joblib
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    meta, Y, idx, _ = _load_split(cache_dir)
    S = {}
    tex = {}

    # ---- split statistics
    stats = {}
    for s in ('train', 'val', 'test'):
        m = meta[meta.split == s]
        nlab = m[LABELS].sum(1)
        stats[s] = {'images': len(m), 'patients': m['Patient ID'].nunique(),
                    'avg_labels_all': float(nlab.mean()), 'avg_labels_abnormal': float(nlab[nlab > 0].mean()),
                    'pct_abnormal': float((nlab > 0).mean() * 100)}
    S['splits'] = stats
    S['split_source'] = json.load(open(os.path.join(cache_dir, 'prep_done.json')))['split_source']
    tex['splits'] = (_latex_table('Dataset Splits (Avg.\\ Findings: Mean Number of Findings per Abnormal Image)', 'tab:splits',
                            ['Split', 'Images', 'Patients', 'Avg.\\ Findings'],
                            [[s.capitalize() if s != 'val' else 'Validation', f"{v['images']:,}", f"{v['patients']:,}",
                              f"{v['avg_labels_abnormal']:.2f}"] for s, v in stats.items()], 'lrrc')
)

    # ---- main models
    main = _load_preds(run_root, 'main', MODEL_KEYS)
    if not main:
        raise RuntimeError('No finished main runs found in ' + run_root)
    any_k = next(iter(main))
    vr, trow = main[any_k][2], main[any_k][3]
    yv, yt = Y[vr], Y[trow]
    gv, gt = meta['Patient ID'].values[vr], meta['Patient ID'].values[trow]
    S['n_val'], S['n_test'] = len(vr), len(trow)
    missing = [k for k in MODEL_KEYS if k not in main]
    if missing:
        print('WARNING: main runs missing for', missing, '- ensemble uses the rest')
    S['missing_main_models'] = missing

    systems, rows = {}, {}
    for k in MODEL_KEYS:
        if k not in main:
            continue
        pv, pt = main[k][0], main[k][1]
        thr = best_thresholds(yv, pv)
        systems[k] = (pt, thr)
        rows[k] = {'name': MODELS[k][1], 'test': summarize(yt, pt, thr), 'test@0.5': summarize(yt, pt, 0.5),
                   'val': summarize(yv, pv, thr)}
    keys = [k for k in MODEL_KEYS if k in main]
    ens = {}
    for name, ks, how in [('stacked', keys, 'stack'), ('average', keys, 'average'),
                          ('stacked_cnn_only', [k for k in CNN_KEYS if k in main], 'stack'),
                          ('stacked_transformer_only', [k for k in TRANSFORMER_KEYS if k in main], 'stack')]:
        if len(ks) < 2:
            continue
        v, t, thr, models = _ensemble(main, ks, yv, gv, how)
        systems[name] = (t, thr)
        pretty = {'stacked': 'Stacked ensemble', 'average': 'Average fusion', 'stacked_cnn_only': 'Stacked, CNNs only',
                  'stacked_transformer_only': 'Stacked, Transformers only'}[name]
        rows[name] = {'name': pretty, 'models': ks, 'test': summarize(yt, t, thr), 'test@0.5': summarize(yt, t, 0.5),
                      'val_oof' if how == 'stack' else 'val': summarize(yv, v, thr)}
        ens[name] = (v, t, thr, models)
        if name == 'stacked':
            joblib.dump({'models': models, 'thresholds': thr, 'keys': ks}, os.path.join(out_dir, 'stacker.joblib'))
            np.savez_compressed(os.path.join(out_dir, 'stacked_preds.npz'), val_oof=v, test=t, thr=thr)

    # best single model is chosen on validation, not test
    singles = [k for k in keys]
    best_single = max(singles, key=lambda k: rows[k]['val']['macro_f1'])
    S['best_single_by_val'] = best_single
    boot_sys = {k: systems[k] for k in ['stacked', 'average', best_single] if k in systems}
    print(f'Bootstrapping {n_boot} patient-level resamples of the test set...')
    B = bootstrap(yt, boot_sys, gt, n=n_boot, seed=seed)
    for k in boot_sys:
        rows[k]['test_ci95'] = {m: _ci(B[k][m]) for m in B[k]}
    if 'stacked' in B:
        S['stacked_minus_best_single'] = {
            m: {'delta': rows['stacked']['test'][m] - rows[best_single]['test'][m],
                'ci95': _ci(B['stacked'][m] - B[best_single][m]),
                'p_le_0': float((B['stacked'][m] - B[best_single][m] <= 0).mean())} for m in B['stacked']}
        S['stacked_minus_average'] = {
            m: {'delta': rows['stacked']['test'][m] - rows['average']['test'][m],
                'ci95': _ci(B['stacked'][m] - B['average'][m]),
                'p_le_0': float((B['stacked'][m] - B['average'][m] <= 0).mean())} for m in B['stacked']}
    S['models'] = rows

    trows = [[MODELS[k][1]] + [_fmt(rows[k]['test'][m]) for m in ('macro_f1', 'micro_f1', 'mean_auc')] for k in keys]
    trows += ['MID', ['Average fusion'] + [_fmt(rows['average']['test'][m]) for m in ('macro_f1', 'micro_f1', 'mean_auc')]]
    trows += [[r'\textbf{Stacked Ensemble}'] + [r'\textbf{' + _fmt(rows['stacked']['test'][m]) + '}'
                                                for m in ('macro_f1', 'micro_f1', 'mean_auc')]]
    tex['test'] = (_latex_table('Test Set Performance', 'tab:test', ['Model', 'Macro-F1', 'Micro-F1', 'Mean AUC'], trows, 'lccc')
               + '% thresholds: per label, tuned on validation (stacked: on out-of-fold validation predictions)\n'
               + '% stacked 95% CI: ' + json.dumps({m: [round(a, 3) for a in v] for m, v in rows['stacked']['test_ci95'].items()}) + '\n')

    # ---- per-class table for the ensemble
    pc = rows['stacked']['test']['per_class']
    pd.DataFrame(pc).T.to_csv(os.path.join(out_dir, 'per_class_stacked.csv'))
    tex['perclass'] = (_latex_table('Per-Class Test Performance of the Stacked Ensemble', 'tab:perclass',
                            ['Finding', 'Precision', 'Recall', 'F1', 'AUC', 'Support'],
                            [[lab.replace('_', ' '), _fmt(v['precision']), _fmt(v['recall']), _fmt(v['f1']),
                              _fmt(v['auc']), str(v['support'])] for lab, v in pc.items()], 'lccccr', small=True))

    # ---- subgroups (sex, age) for the stacked ensemble
    st_p, st_thr = systems['stacked']
    tm = meta.iloc[trow].reset_index(drop=True)
    groups_def = []
    if 'Patient Gender' in tm:
        groups_def += [('Male', (tm['Patient Gender'] == 'M').values), ('Female', (tm['Patient Gender'] == 'F').values)]
    if 'Patient Age' in tm:
        age = tm['Patient Age'].values
        groups_def += [('Age $<$40', age < 40), ('Age 40--59', (age >= 40) & (age < 60)), ('Age $\\geq$60', (age >= 60) & (age <= 100))]
    sub = {}
    for name, mask in groups_def:
        if mask.sum() > 0:
            r = summarize(yt[mask], st_p[mask], st_thr)
            sub[name] = {'images': int(mask.sum()), 'macro_f1': r['macro_f1'], 'mean_auc': r['mean_auc']}
    S['subgroups'] = sub
    if sub:
        tex['subgroups'] = _latex_table('Stacked Ensemble Test Performance by Subgroup', 'tab:subgroups',
                                        ['Subgroup', 'Images', 'Macro-F1', 'Mean AUC'],
                                        [[n, f"{v['images']:,}", _fmt(v['macro_f1']), _fmt(v['mean_auc'])] for n, v in sub.items()],
                                        'lrcc')

    # ---- ablations (test set)
    abl = [('Full stacked ensemble', rows['stacked']['test']['macro_f1'])]
    abl.append(('Average fusion instead of stacking', rows['average']['test']['macro_f1']))
    if 'stacked_cnn_only' in rows:
        abl.append(('CNNs only (no Transformers)', rows['stacked_cnn_only']['test']['macro_f1']))
    if 'stacked_transformer_only' in rows:
        abl.append(('Transformers only (no CNNs)', rows['stacked_transformer_only']['test']['macro_f1']))
    S['variants'] = {}
    for v, label in [('gamma0', 'No focal loss'), ('no_aug', 'No augmentation')]:
        vp = _load_preds(run_root, v, MODEL_KEYS)
        if not vp:
            continue
        res = {}
        if all(k in vp for k in keys):
            _, t, thr, _ = _ensemble(vp, keys, yv, gv, 'stack')
            res['stacked'] = summarize(yt, t, thr)
            abl.append((label, res['stacked']['macro_f1']))
        for k, (pv, pt, _, _) in vp.items():
            res[k] = summarize(yt, pt, best_thresholds(yv, pv))
        S['variants'][v] = res
    base = abl[0][1]
    tex['ablation'] = (_latex_table('Ablation Results (Test Macro-F1)', 'tab:ablation', ['Configuration', 'Macro-F1', r'$\Delta$'],
                            [[n if i == 0 else r'\quad ' + n, _fmt(f), '---' if i == 0 else f'{f - base:+.3f}']
                             for i, (n, f) in enumerate(abl)], 'lcc'))

    # single-backbone version of the training ablations (if they were only run on the sweep model)
    sm = sweep_model
    if sm in main:
        sab = [('Full training recipe', rows[sm]['test']['macro_f1'])]
        for v, label in [('gamma0', 'No focal loss'), ('no_aug', 'No augmentation')]:
            if sm in S['variants'].get(v, {}):
                sab.append((label, S['variants'][v][sm]['macro_f1']))
        if len(sab) > 1:
            tex['ablation_single'] = (_latex_table(f'Training Ablations on {MODELS[sm][1]} (Test Macro-F1)', 'tab:ablation_single',
                                    ['Configuration', 'Macro-F1', r'$\Delta$'],
                                    [[n if i == 0 else r'\quad ' + n, _fmt(f), '---' if i == 0 else f'{f - sab[0][1]:+.3f}']
                                     for i, (n, f) in enumerate(sab)], 'lcc'))

    # ---- sweeps on the sweep model
    def sweep(pairs):
        out = []
        for v, label in pairs:
            p = _load_preds(run_root, v, [sm]).get(sm)
            if p is None:
                continue
            thr = best_thresholds(yv, p[0])
            out.append((label, summarize(yt, p[1], thr)['macro_f1'], summarize(yv, p[0], thr)['macro_f1']))
        return out
    g = sweep([('gamma0', '0 (BCE)'), ('gamma1', '1'), ('main', '2'), ('gamma3', '3')])
    lr = sweep([('lr1e-3', '$10^{-3}$'), ('main', '$10^{-4}$'), ('lr5e-5', r'$5\times10^{-5}$')])
    S['sweep_model'] = sm
    S['gamma_sweep'] = [{'gamma': a, 'test_macro_f1': b, 'val_macro_f1': c} for a, b, c in g]
    S['lr_sweep'] = [{'lr': a, 'test_macro_f1': b, 'val_macro_f1': c} for a, b, c in lr]
    if len(g) > 1:
        ref = g[0][1]
        tex['gamma'] = (_latex_table(rf'Focal Loss $\gamma$ Sensitivity ({MODELS[sm][1]}, Test Macro-F1)', 'tab:gamma',
                                [r'$\gamma$', 'Macro-F1', r'$\Delta$'],
                                [[a, _fmt(b), '---' if i == 0 else f'{b - ref:+.3f}'] for i, (a, b, _) in enumerate(g)], 'ccc'))
    if len(lr) > 1:
        tex['lr'] = (_latex_table(f'Initial Learning Rate Sensitivity ({MODELS[sm][1]}, Test Macro-F1)', 'tab:lr',
                                ['Learning Rate', 'Macro-F1'], [[a, _fmt(b)] for a, b, _ in lr], 'cc'))

    # ---- training histories
    hist = {}
    for k in keys:
        hp = os.path.join(run_root, 'main', k, 'history.json')
        if os.path.exists(hp):
            hist[k] = json.load(open(hp))
    S['histories'] = {k: {'epochs_run': len(h), 'best_epoch': int(np.argmax([r['val_mean_auc'] for r in h]) + 1),
                          'avg_epoch_minutes': float(np.mean([r['epoch_seconds'] for r in h]) / 60)} for k, h in hist.items()}

    for k, t in tex.items():
        with open(os.path.join(out_dir, f'tab_{k}.tex'), 'w') as f:
            f.write(t)
    with open(os.path.join(out_dir, 'tables.tex'), 'w') as f:
        f.write('\n'.join(tex.values()))
    _write_macros(S, rows, os.path.join(out_dir, 'results_macros.tex'))
    _write_json(S, os.path.join(out_dir, 'summary.json'))
    _figures(meta, hist, fig_dir)
    _print_summary(S, rows, keys)
    return S


_MACRO_NAMES = {'resnet18': 'ResNet', 'densenet121': 'Dense', 'coatnet0': 'CoAt', 'vit_b16': 'ViT', 'swin_b': 'Swin',
                'average': 'Avg', 'stacked': 'Stack', 'stacked_cnn_only': 'StackCNN', 'stacked_transformer_only': 'StackTrans'}


def _write_macros(S, rows, path):
    """LaTeX macros so the paper text picks up every number automatically."""
    L = ['% Auto-generated by cxr_pipeline.evaluate_all. Do not edit by hand.']

    def m(name, val):
        L.append(rf'\newcommand{{\{name}}}{{{val}}}')

    for s, short in [('train', 'Train'), ('val', 'Val'), ('test', 'Test')]:
        m(f'N{short}', f"{S['splits'][s]['images']:,}")
        m(f'N{short}Pat', f"{S['splits'][s]['patients']:,}")
    for k, nm in _MACRO_NAMES.items():
        if k in rows:
            t = rows[k]['test']
            m(f'{nm}MacroF', f"{t['macro_f1']:.3f}")
            m(f'{nm}MicroF', f"{t['micro_f1']:.3f}")
            m(f'{nm}AUC', f"{t['mean_auc']:.3f}")
            m(f'{nm}MacroFHalf', f"{rows[k]['test@0.5']['macro_f1']:.3f}")
            if 'test_ci95' in rows[k]:
                for met, tag in [('macro_f1', 'MacroF'), ('micro_f1', 'MicroF'), ('mean_auc', 'AUC')]:
                    lo, hi = rows[k]['test_ci95'][met]
                    m(f'{nm}{tag}Lo', f'{lo:.3f}')
                    m(f'{nm}{tag}Hi', f'{hi:.3f}')
    b = S.get('best_single_by_val')
    if b:
        m('BestSingleName', MODELS[b][1])
        for key, tag in [('stacked_minus_best_single', 'VsBest'), ('stacked_minus_average', 'VsAvg')]:
            for met, mt in [('macro_f1', 'MacroF'), ('mean_auc', 'AUC')]:
                d = S[key][met]
                m(f'Delta{tag}{mt}', f"{d['delta']:+.3f}")
                m(f'Delta{tag}{mt}Lo', f"{d['ci95'][0]:+.3f}")
                m(f'Delta{tag}{mt}Hi', f"{d['ci95'][1]:+.3f}")
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


def _figures(meta, hist, fig_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.ticker
    import matplotlib.pyplot as plt
    tr = meta[meta.split == 'train'][LABELS].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar([l.replace('_', ' ') for l in tr.index], tr.values, color='#3b6ea5')
    ax.set_ylabel('Training images')
    ax.tick_params(axis='x', rotation=60)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'label_dist.png'), dpi=300)
    plt.close(fig)
    for metric, ylabel, fname in [('train_loss', 'Training loss', 'train_loss.png'),
                                  ('val_macro_f1_tuned', 'Validation macro-F1', 'val_f1.png'),
                                  ('val_mean_auc', 'Validation mean AUC', 'val_auc.png')]:
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for k, h in hist.items():
            ax.plot([r['epoch'] for r in h], [r[metric] for r in h], marker='o', ms=3, label=MODELS[k][1])
        ax.set_xlabel('Epoch')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, fname), dpi=300)
        plt.close(fig)


def _print_summary(S, rows, keys):
    print('\nTest set (thresholds tuned on validation)')
    print(f"{'Model':<28}{'Macro-F1':>9}{'Micro-F1':>9}{'AUC':>8}")
    for k in keys + ['average', 'stacked_cnn_only', 'stacked_transformer_only', 'stacked']:
        if k in rows:
            t = rows[k]['test']
            print(f"{rows[k]['name']:<28}{t['macro_f1']:>9.3f}{t['micro_f1']:>9.3f}{t['mean_auc']:>8.3f}")
    if 'stacked_minus_best_single' in S:
        d = S['stacked_minus_best_single']['macro_f1']
        print(f"\nStacked vs best single ({S['best_single_by_val']}): macro-F1 {d['delta']:+.3f} "
              f"(95% CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f})")
        d = S['stacked_minus_average']['macro_f1']
        print(f"Stacked vs average fusion: macro-F1 {d['delta']:+.3f} (95% CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f})")


# ----------------------------------------------------------------------------- deployment benchmark

def _load_main_models(run_root, keys, device):
    ms = {}
    for k in keys:
        p = os.path.join(run_root, 'main', k, 'best.pt')
        m = create_model(k, pretrained=False)
        m.load_state_dict(torch.load(p, map_location='cpu'))
        ms[k] = m.to(device).eval()
    return ms


def benchmark(run_root, cache_dir, out_dir, batch_size=64, n_batches=20, check_f1=True):
    """Ensemble inference throughput, FP32 vs mixed precision, and the test macro-F1 change."""
    import joblib
    device = _device()
    meta, Y, idx, img_path = _load_split(cache_dir)
    st = joblib.load(os.path.join(out_dir, 'stacker.joblib'))
    keys = st['keys']
    ms = _load_main_models(run_root, keys, device)
    arr = np.load(img_path, mmap_mode='r')
    rows = idx['test'][:batch_size * n_batches]
    xb = [torch.from_numpy(np.array(arr[rows[i:i + batch_size]])) for i in range(0, len(rows), batch_size)]
    dtype = _amp_dtype(device)
    modes = {'fp32': None}
    if dtype is not None:
        modes['bf16' if dtype == torch.bfloat16 else 'fp16'] = dtype
    res = {'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu', 'batch_size': batch_size}
    for name, dt in modes.items():
        with torch.no_grad():
            for x in xb[:3]:
                for k, m in ms.items():
                    with torch.autocast(device.type, dtype=dt or torch.float32, enabled=dt is not None):
                        m(_to_input(x, device))
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            for x in xb:
                xi = _to_input(x, device)
                for k, m in ms.items():
                    with torch.autocast(device.type, dtype=dt or torch.float32, enabled=dt is not None):
                        m(xi)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        res[name] = {'images_per_s': len(rows) / (time.time() - t0)}
        res[name]['exams_per_hour'] = res[name]['images_per_s'] * 3600
    if len(modes) > 1:
        lo = [m for m in modes if m != 'fp32'][0]
        res['speedup'] = res[lo]['images_per_s'] / res['fp32']['images_per_s']
    if check_f1 and len(modes) > 1:
        cfg = dict(DEFAULTS)
        dl_te = DataLoader(CXRDataset(img_path, idx['test'], Y[idx['test']]), batch_size=128, num_workers=_workers(cfg))
        yt = Y[idx['test']]
        for name, dt in modes.items():
            X = np.hstack([predict(ms[k], dl_te, device, dt) for k in keys])
            p = np.column_stack([_pred_lr(m, X) for m in st['models']])
            res[name]['test_macro_f1'] = summarize(yt, p, st['thresholds'])['macro_f1']
        lo = [m for m in modes if m != 'fp32'][0]
        res['macro_f1_change_vs_fp32'] = res[lo]['test_macro_f1'] - res['fp32']['test_macro_f1']
    _write_json(res, os.path.join(out_dir, 'benchmark.json'))
    L = ['% Auto-generated by cxr_pipeline.benchmark.',
         rf"\newcommand{{\BenchGPU}}{{{res['gpu']}}}",
         rf"\newcommand{{\BenchBatch}}{{{batch_size}}}",
         rf"\newcommand{{\BenchFPThirtyTwoIPS}}{{{res['fp32']['images_per_s']:,.0f}}}"]
    lo = [k for k in modes if k != 'fp32']
    if lo:
        L += [rf"\newcommand{{\BenchLowPrec}}{{{lo[0].upper()}}}",
              rf"\newcommand{{\BenchLowIPS}}{{{res[lo[0]]['images_per_s']:,.0f}}}",
              rf"\newcommand{{\BenchLowPerHour}}{{{res[lo[0]]['exams_per_hour']:,.0f}}}",
              rf"\newcommand{{\BenchSpeedup}}{{{res['speedup']:.1f}}}"]
        if 'macro_f1_change_vs_fp32' in res:
            L.append(rf"\newcommand{{\BenchFOneChange}}{{{res['macro_f1_change_vs_fp32']:+.4f}}}")
    with open(os.path.join(out_dir, 'bench_macros.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(json.dumps(_py(res), indent=2))
    return res


# ----------------------------------------------------------------------------- Grad-CAM

def gradcam(run_root, cache_dir, out_dir, key='densenet121',
            findings=('Cardiomegaly', 'Effusion', 'Pneumothorax', 'Mass')):
    """For each finding, Grad-CAM on the test image with the highest predicted probability among true positives."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    device = _device()
    meta, Y, idx, img_path = _load_split(cache_dir)
    model = _load_main_models(run_root, [key], device)[key]
    d = np.load(os.path.join(run_root, 'main', key, 'preds.npz'))
    pt, trow = d['test'], d['test_rows']
    layer = dict(model.named_modules())[CAM_LAYERS[key]]
    store = {}
    def hook(module, inputs, output):
        store['a'] = output
        output.register_hook(lambda g: store.__setitem__('g', g))

    h = layer.register_forward_hook(hook)
    arr = np.load(img_path, mmap_mode='r')
    fig, axes = plt.subplots(1, len(findings), figsize=(3 * len(findings), 3.2))
    for ax, lab in zip(np.atleast_1d(axes), findings):
        j = LABELS.index(lab)
        pos = np.where(Y[trow, j] == 1)[0]
        i = pos[np.argmax(pt[pos, j])]
        img = np.array(arr[trow[i]])
        x = _to_input(torch.from_numpy(img)[None], device).requires_grad_(True)
        model.zero_grad()
        out = model(x)
        out[0, j].backward()
        w = store['g'].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * store['a']).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=img.shape, mode='bilinear', align_corners=False)[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        ax.imshow(img, cmap='gray')
        ax.imshow(cam, cmap='jet', alpha=0.35)
        ax.set_title(f"{lab.replace('_', ' ')} (p={torch.sigmoid(out[0, j]).item():.2f})", fontsize=9)
        ax.axis('off')
    h.remove()
    fig.tight_layout()
    os.makedirs(os.path.join(out_dir, 'figures'), exist_ok=True)
    p = os.path.join(out_dir, 'figures', 'gradcam.png')
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print('Saved', p)


def package(out_dir, run_root, zip_base):
    """Zip results plus training configs and histories (no checkpoints or predictions)."""
    tmp = zip_base + '_staging'
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(out_dir, os.path.join(tmp, 'results'), ignore=shutil.ignore_patterns('*.joblib', '*.npz'))
    for v in os.listdir(run_root):
        for k in os.listdir(os.path.join(run_root, v)):
            for f in ('config.json', 'history.json'):
                src = os.path.join(run_root, v, k, f)
                if os.path.exists(src):
                    os.makedirs(os.path.join(tmp, 'runs', v, k), exist_ok=True)
                    shutil.copy(src, os.path.join(tmp, 'runs', v, k, f))
    path = shutil.make_archive(zip_base, 'zip', tmp)
    shutil.rmtree(tmp)
    print('Created', path)
    return path
