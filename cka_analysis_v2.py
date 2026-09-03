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

def load_ckpt(model, path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model

# (name, cls, ckpt, dec_hook(D3), dec_hook(D2), fusion_hook, baseline, enc_width)
RESTORMER_CFG = [
    ('Baseline',  Restormer,                     'experiment_results/restormer_sd7k/restormer_best.pth',
     'decoder_level3', 'decoder_level2', None, True, 48),
    ('Concat',    ShadowGuidedRestormer_NoSGCA,  'experiment_results/nosgca_sd7k/shadow_guided_restormer_no_sgca_best.pth',
     'decoder_level3', 'decoder_level2', 'fuse_dec3', False, 48),
    ('CrossAttn', ShadowGuidedRestormer_CrossAttn, 'experiment_results/sgcr_sd7k/shadow_guided_restormer_crossattn_best.pth',
     'decoder_level3', 'decoder_level2', 'sgcf_dec3', False, 48),
    ('FiLM',      ShadowGuidedRestormer_FiLM,    'experiment_results/sgfm_sd7k/shadow_guided_restormer_film_best.pth',
     'decoder_level3', 'decoder_level2', 'sgfm_dec3', False, 48),
    ('Gated',     ShadowGuidedRestormer_Gated,   'experiment_results/sggf_sd7k/shadow_guided_restormer_gated_best.pth',
     'decoder_level3', 'decoder_level2', 'sggf_dec3', False, 48),
    ('ASF',       ShadowGuidedRestormer_ASF,     'experiment_results/restormer_asf_sd7k/shadow_guided_restormer_asf_best.pth',
     'decoder_level3', 'decoder_level2', 'asf_dec3', False, 48),
    ('Large',     ShadowGuidedRestormer_Large,   'experiment_results/sglarge_sd7k/shadow_guided_restormer_large_final.pth',
     'decoder_level3', 'decoder_level2', 'fuse_dec3', False, 64),
]
NAFNET_CFG = [
    ('Baseline',  NAFNet,                       'experiment_results/nafnet_sd7k/nafnet_best.pth',
     'decoders.1', 'decoders.2', None, True, 16),
    ('Concat',    ShadowGuidedNAFNet_NoSGCA,    'experiment_results/nafnet_nosgca_sd7k/shadow_guided_nafnet_nosgca_best.pth',
     'decoders.1', 'decoders.2', 'fuse_dec1', False, 16),
    ('CrossAttn', ShadowGuidedNAFNet_CrossAttn, 'experiment_results/nafnet_crossattn_sd7k/shadow_guided_nafnet_crossattn_best.pth',
     'decoders.1', 'decoders.2', 'sgcf_dec1', False, 16),
    ('FiLM',      ShadowGuidedNAFNet_FiLM,      'experiment_results/nafnet_film_sd7k/shadow_guided_nafnet_film_best.pth',
     'decoders.1', 'decoders.2', 'sgfm_dec1', False, 16),
    ('Gated',     ShadowGuidedNAFNet_Gated,     'experiment_results/nafnet_gated_sd7k/shadow_guided_nafnet_gated_best.pth',
     'decoders.1', 'decoders.2', 'sggf_dec1', False, 16),
    ('ASF',       ShadowGuidedNAFNet_ASF,       'experiment_results/nafnet_asf_sd7k/shadow_guided_nafnet_asf_best.pth',
     'decoders.1', 'decoders.2', 'asf_dec1', False, 16),
    ('Large',     ShadowGuidedNAFNet_Large,     'experiment_results/nafnet_large_sd7k/shadow_guided_nafnet_large_best.pth',
     'decoders.1', 'decoders.2', 'fuse_dec1', False, 32),
]

def make_shared_enc(src_ckpt, width):
    enc = ShadowEncoder(width=width).to(DEVICE)
    ckpt = torch.load(src_ckpt, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model', ckpt)
    sd = {k[len('shadow_encoder.'):]: v for k, v in state.items()
          if k.startswith('shadow_encoder.')}
    if sd:
        enc.load_state_dict(sd)
    return enc

def analyze(model, loader, dec_hooks, fusion_hook, baseline, shared_enc, n=12):
    acc = {'D3': {'before': [], 's': []}, 'D2': {'before': [], 's': []}}
    if fusion_hook:
        acc['fused'] = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            inp_img, gray_img, tar_img, fname = [
                b.to(DEVICE) if isinstance(b, torch.Tensor) else b for b in batch[:4]]
            gray_resized = F.interpolate(gray_img, size=inp_img.shape[2:],
                                         mode='bilinear', align_corners=False)
            s1, s2, s3 = shared_enc(gray_resized)

            captured = {}
            handles = []
            for lk, hk in dec_hooks.items():
                mod = dict(model.named_modules())[hk]
                handles.append(mod.register_forward_hook(
                    lambda m, a, o, k=lk: captured.setdefault(k, o.detach())))
            if fusion_hook:
                fmod = dict(model.named_modules())[fusion_hook]
                handles.append(fmod.register_forward_hook(
                    lambda m, a, o: captured.setdefault('fused', o.detach())))

            if baseline:
                model(inp_img)
            else:
                model(gray_img, inp_img)
            for h in handles:
                h.remove()

            acc['D3']['before'].append(captured['D3'].cpu())
            acc['D3']['s'].append(s3.cpu())
            acc['D2']['before'].append(captured['D2'].cpu())
            acc['D2']['s'].append(s2.cpu())
            if 'fused' in captured:
                acc['fused'].append(captured['fused'].cpu())
            if idx + 1 >= n:
                break
    return acc

def main():
    ds = get_data('./dataset/SD7K/test/', 'input', 'target', mode='val',
                  img_options={'h': 256, 'w': 256})
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    print('SD7K test imgs:', len(ds))

    enc48 = make_shared_enc(RESTORMER_CFG[1][2], 48)
    enc64 = make_shared_enc(RESTORMER_CFG[6][2], 64)
    enc16 = make_shared_enc(NAFNET_CFG[1][2], 16)
    enc32 = make_shared_enc(NAFNET_CFG[6][2], 32)

    results = {}
    for fam, cfg_list, enc48v, enc64v in [
        ('Restormer', RESTORMER_CFG, enc48, enc64),
        ('NAFNet',    NAFNET_CFG,    enc16, enc32),
    ]:
        print('\n===== %s =====' % fam)
        fam_results = {}
        for name, cls, ckpt, hk3, hk2, fhk, baseline, width in cfg_list:
            try:
                model = load_ckpt(cls(), ckpt)
                shared = enc64v if width in (64, 32) else enc48v
                model.shadow_encoder = shared
                dec_hooks = {'D3': hk3, 'D2': hk2}
                acc = analyze(model, loader, dec_hooks, fhk, baseline, shared)

                row = {}
                for layer, sfeat in [('D3', 's'), ('D2', 's')]:
                    X = torch.cat(acc[layer]['before'], 0)
                    Y = torch.cat(acc[layer][sfeat], 0)
                    if X.shape[2:] != Y.shape[2:]:
                        Y = F.interpolate(Y, size=X.shape[2:], mode='bilinear', align_corners=False)
                    cka = linear_cka(flatten(X), flatten(Y)).item()
                    row[layer] = {'before': round(cka, 4)}
                if 'fused' in acc and acc['fused']:
                    Xf = torch.cat(acc['fused'], 0)
                    Y = torch.cat(acc['D3']['s'], 0)
                    if Xf.shape[2:] != Y.shape[2:]:
                        Xf = F.interpolate(Xf, size=Y.shape[2:], mode='bilinear', align_corners=False)
                    row['D3']['after'] = round(linear_cka(flatten(Xf), flatten(Y)).item(), 4)
                else:
                    row['D3']['after'] = None
                fam_results[name] = row
                print('  %-9s D3 before=%.4f after=%s | D2 before=%.4f'
                      % (name, row['D3']['before'], row['D3']['after'], row['D2']['before']))
            except Exception as e:
                print('  %-9s FAILED: %s: %s' % (name, type(e).__name__, str(e)[:180]))
        results[fam] = fam_results

    with open('cka_results_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved cka_results_v2.json')

if __name__ == '__main__':
    main()
