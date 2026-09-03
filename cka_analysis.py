import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/mnt/ShaDocFormer-main')
from models.comparison_models import (
    Restormer, NAFNet, ShadowEncoder,
    ShadowGuidedRestormer_NoSGCA, ShadowGuidedRestormer_CrossAttn,
    ShadowGuidedRestormer_FiLM, ShadowGuidedRestormer_Gated,
    ShadowGuidedRestormer_Large, ShadowGuidedNAFNet_NoSGCA,
)
from models.fusion_models import (
    ShadowGuidedNAFNet_CrossAttn, ShadowGuidedNAFNet_FiLM,
    ShadowGuidedNAFNet_Gated, ShadowGuidedNAFNet_ASF,
    ShadowGuidedNAFNet_Large, ShadowGuidedRestormer_ASF,
)
from data.data_RGB import get_data

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    return (X.T @ Y).norm()**2 / ((X.T @ X).norm() * (Y.T @ Y).norm())

def flatten(feat):
    B, C, H, W = feat.shape
    return feat.permute(0, 2, 3, 1).reshape(B * H * W, C)

def run_model(model, gray, inp):
    with torch.no_grad():
        if gray is None:
            return model(inp)
        return model(gray, inp)

def collect(model, gray, inp, hook_name):
    captured = {}
    mod = dict(model.named_modules())[hook_name]
    handle = mod.register_forward_hook(
        lambda m, a, o: captured.setdefault('feat', o.detach()))
    run_model(model, gray, inp)
    handle.remove()
    return captured['feat']

def load_ckpt(model, path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()

def load_shared_shadow(src_ckpt, width):
    enc = ShadowEncoder(width=width).to(DEVICE)
    ckpt = torch.load(src_ckpt, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model', ckpt)
    sd = {k[len('shadow_encoder.'):]: v for k, v in state.items()
          if k.startswith('shadow_encoder.')}
    if sd:
        enc.load_state_dict(sd)
    return enc

RESTORMER_CFG = [
    ('Baseline',  Restormer,                     'experiment_results/restormer_sd7k/restormer_best.pth', 'decoder_level3', True),
    ('Concat',    ShadowGuidedRestormer_NoSGCA,  'experiment_results/nosgca_sd7k/shadow_guided_restormer_no_sgca_best.pth', 'decoder_level3', False),
    ('CrossAttn', ShadowGuidedRestormer_CrossAttn, 'experiment_results/sgcr_sd7k/shadow_guided_restormer_crossattn_best.pth', 'decoder_level3', False),
    ('FiLM',      ShadowGuidedRestormer_FiLM,    'experiment_results/sgfm_sd7k/shadow_guided_restormer_film_best.pth', 'decoder_level3', False),
    ('Gated',     ShadowGuidedRestormer_Gated,   'experiment_results/sggf_sd7k/shadow_guided_restormer_gated_best.pth', 'decoder_level3', False),
    ('ASF',       ShadowGuidedRestormer_ASF,     'experiment_results/restormer_asf_sd7k/shadow_guided_restormer_asf_best.pth', 'decoder_level3', False),
    ('Large',     ShadowGuidedRestormer_Large,   'experiment_results/sglarge_sd7k/shadow_guided_restormer_large_final.pth', 'decoder_level3', False),
]
NAFNET_CFG = [
    ('Baseline',  NAFNet,                       'experiment_results/nafnet_sd7k/nafnet_best.pth', 'decoders.1', True),
    ('Concat',    ShadowGuidedNAFNet_NoSGCA,    'experiment_results/nafnet_nosgca_sd7k/shadow_guided_nafnet_nosgca_best.pth', 'decoders.1', False),
    ('CrossAttn', ShadowGuidedNAFNet_CrossAttn, 'experiment_results/nafnet_crossattn_sd7k/shadow_guided_nafnet_crossattn_best.pth', 'decoders.1', False),
    ('FiLM',      ShadowGuidedNAFNet_FiLM,      'experiment_results/nafnet_film_sd7k/shadow_guided_nafnet_film_best.pth', 'decoders.1', False),
    ('Gated',     ShadowGuidedNAFNet_Gated,     'experiment_results/nafnet_gated_sd7k/shadow_guided_nafnet_gated_best.pth', 'decoders.1', False),
    ('ASF',       ShadowGuidedNAFNet_ASF,       'experiment_results/nafnet_asf_sd7k/shadow_guided_nafnet_asf_best.pth', 'decoders.1', False),
    ('Large',     ShadowGuidedNAFNet_Large,     'experiment_results/nafnet_large_sd7k/shadow_guided_nafnet_large_best.pth', 'decoders.1', False),
]

RESTORMER_HOOKS = {'D3': 'decoder_level3', 'D2': 'decoder_level2'}
NAFNET_HOOKS   = {'D3': 'decoders.1',      'D2': 'decoders.2'}

def main():
    dataset = get_data('./dataset/SD7K/test/', 'input', 'target', mode='val',
                       img_options={'h': 256, 'w': 256})
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    print('SD7K test images:', len(dataset))

    sh_rest = load_shared_shadow(RESTORMER_CFG[1][2], 48)
    sh_naf  = load_shared_shadow(NAFNET_CFG[1][2], 16)

    results = {'Restormer': {}, 'NAFNet': {}}

    for fam, cfg_list, hooks, shared_enc in [
        ('Restormer', RESTORMER_CFG, RESTORMER_HOOKS, sh_rest),
        ('NAFNet',    NAFNET_CFG,    NAFNET_HOOKS,    sh_naf),
    ]:
        print('\n===== %s =====' % fam)
        for name, cls, ckpt, hook, baseline in cfg_list:
            try:
                model = cls()
                load_ckpt(model, ckpt)
                if hasattr(model, 'shadow_encoder'):
                    model.shadow_encoder = shared_enc
                print('  %-9s %s' % (name, ckpt))

                acc_feat = {'D3': [], 'D2': []}
                acc_s    = {'D3': [], 'D2': []}
                n_used = 0
                for idx, batch in enumerate(loader):
                    inp_img, gray_img, tar_img, fname = [
                        b.to(DEVICE) if isinstance(b, torch.Tensor) else b
                        for b in batch[:4]]
                    gray_resized = F.interpolate(gray_img, size=inp_img.shape[2:],
                                                 mode='bilinear', align_corners=False)
                    s1, s2, s3 = shared_enc(gray_resized)

                    for layer in ['D3', 'D2']:
                        feat = collect(model, None if baseline else gray_img, inp_img, hooks[layer])
                        acc_feat[layer].append(feat.cpu())
                        acc_s[layer].append({'D3': s3, 'D2': s2}[layer].cpu())
                    n_used += 1
                    if n_used >= 12:
                        break

                row = {}
                for layer in ['D3', 'D2']:
                    X = torch.cat(acc_feat[layer], 0)
                    Y = torch.cat(acc_s[layer], 0)
                    if X.shape[2:] != Y.shape[2:]:
                        Y = F.interpolate(Y, size=X.shape[2:], mode='bilinear', align_corners=False)
                    cka = linear_cka(flatten(X), flatten(Y)).item()
                    row[layer] = round(cka, 4)
                results[fam][name] = row
                print('    D3-CKA=%.4f  D2-CKA=%.4f' % (row['D3'], row['D2']))
            except Exception as e:
                print('  %-9s FAILED: %s: %s' % (name, type(e).__name__, str(e)[:200]))

    with open('cka_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved cka_results.json')

if __name__ == '__main__':
    main()
