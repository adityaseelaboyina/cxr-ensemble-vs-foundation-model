"""EVA-X backbones (Yao et al., npj Digit. Med. 2025). Configs and checkpoint key conversion follow
hustvl/EVA-X eva_x.py (Apache-2.0). Loading fails loudly if anything other than the head is missing.

Weights: https://huggingface.co/MapleF/eva_x (MIM on ChestX-ray14 + CheXpert + MIMIC-CXR, unlabeled).
Don't use the released *_cxr14_ft checkpoints: they were fine-tuned on images in our validation split."""
from __future__ import annotations

import torch
from timm.models.eva import Eva

HF_REPO = 'MapleF/eva_x'
WEIGHTS = {
    'evax_ti': 'eva_x_tiny_patch16_merged520k_mim.pt',
    'evax_s': 'eva_x_small_patch16_merged520k_mim.pt',
    'evax_b': 'eva_x_base_patch16_merged520k_mim.pt',
}
CONFIGS = {
    'evax_ti': dict(embed_dim=192, depth=12, num_heads=3),
    'evax_s': dict(embed_dim=384, depth=12, num_heads=6),
    'evax_b': dict(embed_dim=768, depth=12, num_heads=12, qkv_fused=False, scale_mlp=True),
}
DISPLAY = {'evax_ti': 'EVA-X-Ti', 'evax_s': 'EVA-X-S', 'evax_b': 'EVA-X-B'}


class EVAX(Eva):
    """Eva with mean pooling over patch tokens (EVA-X fine-tuning uses --use_mean_pooling)."""

    def forward_features(self, x):
        x = self.patch_embed(x)
        x, rope = self._pos_embed(x)
        for blk in self.blocks:
            x = blk(x, rope=rope)
        return self.norm(x)

    def forward_head(self, x, pre_logits: bool = False):
        x = x[:, self.num_prefix_tokens:].mean(dim=1)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def _convert(state_dict, model):
    """Same key mapping as EVA-X checkpoint_filter_fn for MIM checkpoints."""
    for k in ('model_ema', 'model', 'module', 'state_dict'):
        state_dict = state_dict.get(k, state_dict) if isinstance(state_dict, dict) else state_dict
    mim = 'mask_token' in state_dict
    out = {}
    for k, v in state_dict.items():
        if 'rope' in k or k.startswith('lm_head') or k == 'mask_token':
            continue
        if k == 'patch_embed.proj.weight' and v.ndim == 2:  # linear patchify -> conv
            _, _, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(v.shape[0], -1, H, W)
        k = k.replace('mlp.ffn_ln', 'mlp.norm').replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1').replace('mlp.w1', 'mlp.fc1_g').replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if mim and k in ('norm.weight', 'norm.bias'):
            k = k.replace('norm', 'fc_norm')  # EVA-X moves the final norm to fc_norm for fine-tuning
        out[k] = v
    return out


def build(key, num_classes=14, weights_path=None, drop_path=0.1):
    cfg = CONFIGS[key]
    model = EVAX(img_size=224, patch_size=16, mlp_ratio=4 * 2 / 3, swiglu_mlp=True, use_rot_pos_emb=True,
                 ref_feat_shape=(14, 14), num_classes=num_classes, drop_path_rate=drop_path,
                 use_fc_norm=True, **cfg)
    if weights_path:
        sd = torch.load(weights_path, map_location='cpu', weights_only=False)
        sd = _convert(sd, model)
        sd = {k: v for k, v in sd.items() if not k.startswith('head.')}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        allowed = {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias', 'norm.weight', 'norm.bias'}
        bad_missing = [k for k in missing if k not in allowed]
        if bad_missing or unexpected:
            raise RuntimeError(f'EVA-X weights did not load cleanly. missing={bad_missing[:10]} '
                               f'unexpected={list(unexpected)[:10]}')
        print(f'[{key}] loaded {len(sd)} tensors; newly initialized: {sorted(missing)}')
    return model


def download(key, cache_dir=None):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, WEIGHTS[key], cache_dir=cache_dir)


def layer_decay_param_groups(model, lr, weight_decay, layer_decay=0.55):
    """Layer-wise learning-rate decay as in the EVA-X / BEiT fine-tuning recipe."""
    n = len(model.blocks) + 1
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(('cls_token', 'pos_embed', 'patch_embed')):
            layer = 0
        elif name.startswith('blocks.'):
            layer = int(name.split('.')[1]) + 1
        else:
            layer = n
        no_wd = p.ndim == 1 or name.endswith('.bias') or name in ('cls_token', 'pos_embed')
        g = (layer, no_wd)
        if g not in groups:
            groups[g] = {'params': [], 'lr': lr * layer_decay ** (n - layer), 'weight_decay': 0.0 if no_wd else weight_decay}
        groups[g]['params'].append(p)
    return list(groups.values())
