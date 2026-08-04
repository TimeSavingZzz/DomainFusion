"""OCR evaluation for remaining models (skip CrossAttn at 512, use 256)."""
import sys, os
sys.path.insert(0, '/mnt/ShaDocFormer-main')
import torch, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import easyocr

from train_compare_models import build_model
from data.dataset_RGB import DataReader

device = torch.device('cuda')
RES = 512

MODEL_CH = [
    ('FiLM',        'shadow_guided_restormer_film', 'experiment_results/sgfm_sd7k/shadow_guided_restormer_film_best.pth', 512),
    ('Large',       'shadow_guided_restormer_large', 'experiment_results/sglarge_sd7k/shadow_guided_restormer_large_best.pth', 512),
    ('Gated',       'shadow_guided_restormer_gated', 'experiment_results/sggf_sd7k/shadow_guided_restormer_gated_best.pth', 512),
    ('GatedLarge',  'shadow_guided_restormer_gated_large', 'experiment_results/sggf_large_sd7k/shadow_guided_restormer_gated_large_best.pth', 512),
    ('CrossAttn',   'shadow_guided_restormer_crossattn', 'experiment_results/sgcr_sd7k/shadow_guided_restormer_crossattn_best.pth', 256),
]

ocr = easyocr.Reader(['en'], gpu=True)

def ocr_chars(img_np):
    res = ocr.readtext(img_np, detail=0)
    return sum(len(r) for r in res)

all_r = {}
for disp_name, model_name, ck_path, res in MODEL_CH:
    print('\n=== %s (res=%d) ===' % (disp_name, res))
    try:
        model, model_type = build_model(model_name, device)
        ck = torch.load(ck_path, map_location='cuda')
        model.load_state_dict(ck)
        model.eval()

        dset = DataReader('./dataset/SD7K/test/', 'input', 'target', mode='test', ori=False,
                          img_options={'h': res, 'w': res})
        loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=0)

        ocr_gt, ocr_inp, ocr_out = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc='OCR ' + disp_name):
                inp, gray, tar, _ = batch
                inp, gray, tar = inp.to(device), gray.to(device), tar.to(device)
                out = model(gray, inp)
                inp_np = (inp[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                out_np = (out[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                tar_np = (tar[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                try:
                    ocr_gt.append(ocr_chars(tar_np))
                    ocr_inp.append(ocr_chars(inp_np))
                    ocr_out.append(ocr_chars(out_np))
                except:
                    pass

        g, i, o = np.mean(ocr_gt), np.mean(ocr_inp), np.mean(ocr_out)
        recovery = o / max(g, 1) * 100
        all_r[disp_name] = {'ocr_gt': float(g), 'ocr_inp': float(i), 'ocr_out': float(o), 'recovery': recovery}
        print('  OCR: Inp=%.1f, Out=%.1f, GT=%.1f, Recovery=%.1f%%' % (i, o, g, recovery))
        torch.cuda.empty_cache()
    except Exception as e:
        print('  SKIP %s: %s' % (disp_name, str(e)))
        torch.cuda.empty_cache()

print('\n\n=== OCR Summary ===')
print('%-14s %8s %8s %8s %10s' % ('Model', 'OCR Inp', 'OCR Out', 'OCR GT', 'Recovery'))
print('-' * 54)
for name, r in sorted(all_r.items()):
    print('%-14s %8.1f %8.1f %8.1f %8.1f%%' % (name, r['ocr_inp'], r['ocr_out'], r['ocr_gt'], r['recovery']))
print('Done.')
