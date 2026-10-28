"""Training for the foundation-model runs, the EVA-X-recipe controls and the seed replicates.
Writes to <project>/runs_v3 and reuses the data and metric code in cxr_pipeline.py, so the split and
validation-only checkpoint selection match the main run.

The evax recipe follows hustvl/EVA-X (cxr14): plain BCE, no Mixup, layer decay 0.55, wd 0.05, drop-path 0.2,
5 warmup epochs, 30 epochs max, batch 256 with lr scaled from 1e-3 at batch 1024. Differences: our augmentation
instead of RandAugment, and checkpoints picked on validation (their script picks on the test set)."""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evax  # noqa: E402

RECIPES = {
    'shared': dict(batch_size=32, max_epochs=40, patience=5, lr=1e-4, weight_decay=1e-4, warmup_epochs=1,
                   gamma=2.0, use_pos_weight=True, augment=True, mixup_alpha=0.4, layer_decay=None, drop_path=0.1),
    'evax': dict(batch_size=256, max_epochs=30, patience=5, lr=2.5e-4, weight_decay=0.05, warmup_epochs=5,
                 gamma=0.0, use_pos_weight=False, augment=True, mixup_alpha=0.0, layer_decay=0.55, drop_path=0.2),
}


def _import_pipeline(project):
    sys.path.insert(0, project)
    import cxr_pipeline as cx  # the exact file used for the main experiment
    return cx


def train(project, cache_dir, key, recipe='shared', seed=42, run_root=None, weights_dir=None, limit=None,
          pretrained=True, overrides=None, tag=None):
    cx = _import_pipeline(project)
    run_root = run_root or os.path.join(project, 'runs_v3')
    assert os.path.abspath(run_root) != os.path.abspath(os.path.join(project, 'runs')), 'refusing to write into runs/'
    cfg = dict(RECIPES[recipe], **(overrides or {}), seed=seed, recipe=recipe, key=key, img_size=224, select_metric='val_mean_auc')
    variant = tag or (('main' if recipe == 'shared' else f'{recipe}recipe') + ('' if seed == 42 else f'_seed{seed}'))
    if overrides and not tag and recipe == 'shared':
        raise ValueError('pass tag= when overriding the shared recipe, so the run cannot be mistaken for the main one')
    out = os.path.join(run_root, variant, key)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, 'preds.npz')):
        print(f'[{variant}/{key}] already finished, skipping')
        return out
    cx.seed_everything(seed)
    device = cx._device()
    meta, Y, idx, img_path = cx._load_split(cache_dir)
    if limit:
        rng = np.random.default_rng(seed)
        idx = {s: np.sort(rng.choice(v, min(limit, len(v)), replace=False)) for s, v in idx.items()}
    lcfg = {'augment': cfg['augment'], 'batch_size': cfg['batch_size'], 'num_workers': None}
    dl_tr, dl_va, dl_te = cx._loaders(img_path, Y, idx, lcfg, device=device)

    if key.startswith('evax'):
        model = evax.build(key, weights_path=(evax.download(key, weights_dir) if pretrained else None),
                           drop_path=cfg['drop_path'])
    else:
        model = cx.create_model(key, pretrained=pretrained)
    model = model.to(device)
    if cfg['layer_decay'] and key.startswith('evax'):
        groups = evax.layer_decay_param_groups(model, cfg['lr'], cfg['weight_decay'], cfg['layer_decay'])
    else:
        groups = [{'params': model.parameters(), 'lr': cfg['lr'], 'weight_decay': cfg['weight_decay']}]
    for g in groups:
        g['base_lr'] = g['lr']
    opt = torch.optim.AdamW(groups)
    spe = len(dl_tr)
    total, warm = spe * cfg['max_epochs'], max(1, spe * cfg['warmup_epochs'])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    tr = idx['train']
    pos = Y[tr].sum(0)
    pw = torch.tensor((len(tr) - pos) / np.maximum(pos, 1), dtype=torch.float32) if cfg['use_pos_weight'] else None
    loss_fn = cx.FocalLoss(cfg['gamma'], pw).to(device)
    dtype = cx._amp_dtype(device)
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)

    start, best, bad, history = 0, -np.inf, 0, []
    last_p, best_p = os.path.join(out, 'last.pt'), os.path.join(out, 'best.pt')
    if os.path.exists(last_p):
        ck = torch.load(last_p, map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        start, best, bad, history = ck['epoch'], ck['best'], ck['bad'], ck['history']
        print(f'[{variant}/{key}] resuming after epoch {start}')
    cx._write_json({**cfg, 'variant': variant, 'pretrained': pretrained, 'limit': limit,
                    'n_train': len(idx['train']), 'n_val': len(idx['val']), 'n_test': len(idx['test'])},
                   os.path.join(out, 'config.json'))
    yv = Y[idx['val']]
    for epoch in range(start, cfg['max_epochs']):
        if bad >= cfg['patience']:
            break
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, y in dl_tr:
            x = cx._to_input(x, device)
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
        pv = cx.predict(model, dl_va, device, dtype)
        rec = {'epoch': epoch + 1, 'train_loss': tot / max(n, 1),
               'val_mean_auc': float(np.nanmean(cx.per_label_auc(yv, pv))), 'epoch_seconds': time.time() - t0}
        history.append(rec)
        if rec['val_mean_auc'] > best:
            best, bad = rec['val_mean_auc'], 0
            cx._atomic_save(model.state_dict(), best_p)
        else:
            bad += 1
        cx._atomic_save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                         'scaler': scaler.state_dict(), 'epoch': epoch + 1, 'best': best, 'bad': bad,
                         'history': history}, last_p)
        print(f"[{variant}/{key}] ep {epoch + 1:2d} loss {rec['train_loss']:.4f} | val AUC {rec['val_mean_auc']:.4f} | "
              f"{rec['epoch_seconds'] / 60:.1f} min" + ('  *best' if bad == 0 else f"  (no improvement {bad}/{cfg['patience']})"))
    model.load_state_dict(torch.load(best_p, map_location=device))
    pv = cx.predict(model, dl_va, device, dtype)
    pt = cx.predict(model, dl_te, device, dtype)
    np.savez_compressed(os.path.join(out, 'preds.npz'), val=pv, test=pt, val_rows=idx['val'], test_rows=idx['test'])
    cx._write_json(history, os.path.join(out, 'history.json'))
    os.remove(last_p)
    print(f'[{variant}/{key}] done: best val AUROC {best:.4f}')
    return out


@torch.no_grad()
def benchmark_models(project, cache_dir, out_json, roots=('runs', 'runs_v3'), batch_sizes=(1, 64), warmup=10,
                     iters=30, repeats=3):
    """Per-model inference latency/throughput and peak memory on the current GPU, FP32 and mixed precision.
    Weights are loaded from best.pt where available (timing does not depend on weights)."""
    cx = _import_pipeline(project)
    device = cx._device()
    dtype = cx._amp_dtype(device)
    res = {'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu', 'warmup': warmup, 'iters': iters,
           'repeats': repeats, 'per_model': {}}
    for root in roots:
        mdir = os.path.join(project, root, 'main')
        if not os.path.isdir(mdir):
            continue
        for key in sorted(os.listdir(mdir)):
            if key in res['per_model']:
                continue
            m = evax.build(key) if key.startswith('evax') else cx.create_model(key, pretrained=False)
            bp = os.path.join(mdir, key, 'best.pt')
            if os.path.exists(bp):
                m.load_state_dict(torch.load(bp, map_location='cpu'))
            m = m.to(device).eval()
            r = {'params_m': sum(p.numel() for p in m.parameters()) / 1e6}
            for prec, dt in [('fp32', None), ('amp', dtype)]:
                if prec == 'amp' and dt is None:
                    continue
                for bs in batch_sizes:
                    x = torch.randn(bs, 3, 224, 224, device=device)
                    times = []
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats()
                    for _ in range(repeats):
                        for _ in range(warmup):
                            with torch.autocast(device.type, dtype=dt or torch.float32, enabled=dt is not None):
                                m(x)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for _ in range(iters):
                            with torch.autocast(device.type, dtype=dt or torch.float32, enabled=dt is not None):
                                m(x)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        times.append((time.perf_counter() - t0) / iters)
                    med = float(np.median(times))
                    r[f'{prec}_bs{bs}_ms_per_batch'] = med * 1000
                    r[f'{prec}_bs{bs}_img_per_s'] = bs / med
                    if device.type == 'cuda':
                        r[f'{prec}_bs{bs}_peak_mem_mb'] = torch.cuda.max_memory_allocated() / 2 ** 20
            big = max(batch_sizes)
            pref = 'amp' if dtype is not None else 'fp32'
            r['ms_per_image'] = 1000 / r[f'{pref}_bs{big}_img_per_s']
            res['per_model'][key] = r
            print(key, {k: round(v, 2) for k, v in r.items()})
            del m
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    json.dump(res, open(out_json, 'w'), indent=2)
    return res
