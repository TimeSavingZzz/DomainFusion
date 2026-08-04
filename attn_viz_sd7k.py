"""
SD7K Attention Visualization for PRL Paper
Extracts decoder-level3 self-attention maps from 4 models
and generates a compact comparison figure.
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# These will be imported on the container
# from models.comparison_models import ...
# from data.data_RGB import get_data

_ATTN_STORE = {}

def _mdta_patched_forward(self, x):
    b, c, h, w = x.shape
    qkv = self.qkv_dwconv(self.qkv(x))
    q, k, v = qkv.chunk(3, dim=1)
    head_dim = c // self.num_heads
    q = q.reshape(b, self.num_heads, head_dim, h * w)
    k = k.reshape(b, self.num_heads, head_dim, h * w)
    v = v.reshape(b, self.num_heads, head_dim, h * w)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    attn = (q @ k.transpose(-2, -1)) * self.temperature
    attn = attn.softmax(dim=-1)
    _ATTN_STORE[id(self)] = (attn.detach().cpu(), (h, w))
    out = attn @ v
    out = out.reshape(b, -1, h, w)
    return self.project_out(out)


def create_comparison_figure(all_data, layer_name, save_path):
    """
    Compact 2-row figure:
    Row 1: Shadow Input | Restormer Attn | Concat Attn | FiLM Attn | Gated Attn
    Row 2: Restormer Out | Restormer Overlay | Concat Overlay | FiLM Overlay | Gated Overlay
    OR simplified: 2x5 grid showing attn overlay + output
    """
    n = len(all_data)  # 4 models
    fig, axes = plt.subplots(2, n + 1, figsize=(4.2 * (n + 1), 5.5),
                             gridspec_kw={'height_ratios': [1, 1]})

    shadow_np = None
    for row, data in enumerate(all_data):
        if row == 0:
            shadow_np = data['shadow'].permute(1, 2, 0).cpu().numpy()
            shadow_np = np.clip(shadow_np, 0, 1)

    Himg, Wimg = shadow_np.shape[0], shadow_np.shape[1]

    for col_idx, data in enumerate(all_data):
        # Attention map
        attn = data['attn']
        attn_received = attn[0].mean(dim=0).mean(dim=0)

        N = attn_received.shape[0]
        for d in range(int(N**0.5), 0, -1):
            if N % d == 0:
                Hmap, Wmap = d, N // d
                break
        else:
            Hmap = Wmap = int(N**0.5)

        attn_map = attn_received.reshape(Hmap, Wmap).numpy()
        a_min, a_max = attn_map.min(), attn_map.max()
        if a_max > a_min:
            attn_map = (attn_map - a_min) / (a_max - a_min)

        attn_resized = np.array(Image.fromarray(
            (attn_map * 255).astype(np.uint8)).resize((Wimg, Himg), Image.LANCZOS)) / 255.0
        attn_colored = plt.cm.jet(attn_resized)[:, :, :3]
        overlay = shadow_np * 0.45 + attn_colored * 0.55

        output_np = data['output'].permute(1, 2, 0).cpu().numpy()
        output_np = np.clip(output_np, 0, 1)

        # Top row: attention heatmap overlay
        axes[0, 1 + col_idx].imshow(overlay)
        axes[0, 1 + col_idx].set_title(data['label'], fontsize=10, fontweight='bold')
        axes[0, 1 + col_idx].axis('off')

        # Bottom row: restored output
        axes[1, 1 + col_idx].imshow(output_np)
        axes[1, 1 + col_idx].axis('off')

    # Column 0: shadow input (top) and ground truth (bottom)
    axes[0, 0].imshow(shadow_np)
    axes[0, 0].set_title('Shadow Input', fontsize=10, fontweight='bold', color='#D62828')
    axes[0, 0].axis('off')

    if all_data[0].get('target') is not None:
        target_np = all_data[0]['target'].permute(1, 2, 0).cpu().numpy()
        target_np = np.clip(target_np, 0, 1)
        axes[1, 0].imshow(target_np)
        axes[1, 0].set_title('Ground Truth', fontsize=10, fontweight='bold', color='#2A9D8F')
        axes[1, 0].axis('off')
    else:
        axes[1, 0].axis('off')

    # Row labels
    fig.text(0.01, 0.73, 'Attention\nOverlay', fontsize=11, fontweight='bold',
             ha='left', va='center', rotation=90)
    fig.text(0.01, 0.23, 'Restored\nOutput', fontsize=11, fontweight='bold',
             ha='left', va='center', rotation=90)

    fig.suptitle(f'Decoder Self-Attention Visualization (Level 3) — SD7K',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(pad=0.5, rect=[0.04, 0, 1, 0.97])
    fig.savefig(save_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved comparison figure: {save_path}")


def main():
    # Container-side imports
    sys.path.insert(0, '/mnt/ShaDocFormer-main')
    from models.comparison_models import (
        Restormer,
        ShadowGuidedRestormer_NoSGCA,
        ShadowGuidedRestormer_FiLM,
        ShadowGuidedRestormer_Gated,
        MDTA,
    )
    from data.data_RGB import get_data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load one SD7K test image
    data_dir = '/mnt/ShaDocFormer-main/dataset/SD7K/test/'
    dataset = get_data(data_dir, 'input', 'target', mode='val',
                       img_options={'h': 320, 'w': 320})
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"SD7K test images: {len(dataset)}")

    # Use an image with visible shadows — pick from middle of dataset
    for idx, batch in enumerate(loader):
        if idx == 45:  # Pick sample with visible shadow
            inp_img, gray_img, tar_img, fname = [b.to(device) if isinstance(b, torch.Tensor) else b
                                                   for b in batch[:4]]
            print(f"Using sample {idx}: {fname}")
            break

    # Model configs
    model_configs = [
        {
            'label': 'Restormer\n(no shadow encoder)',
            'builder': lambda: Restormer(),
            'ckpt': '/mnt/ShaDocFormer-main/experiment_results/restormer_sd7k/',
            'use_shadow': False,
        },
        {
            'label': 'Concat\n(No SGCA)',
            'builder': lambda: ShadowGuidedRestormer_NoSGCA(),
            'ckpt': '/mnt/ShaDocFormer-main/experiment_results/nosgca_sd7k/',
            'use_shadow': True,
        },
        {
            'label': 'FiLM\n(SGFM)',
            'builder': lambda: ShadowGuidedRestormer_FiLM(),
            'ckpt': '/mnt/ShaDocFormer-main/experiment_results/sgfm_sd7k/',
            'use_shadow': True,
        },
        {
            'label': 'Gated\n(SGGF, ours)',
            'builder': lambda: ShadowGuidedRestormer_Gated(),
            'ckpt': '/mnt/ShaDocFormer-main/experiment_results/sggf_sd7k/',
            'use_shadow': True,
        },
    ]

    all_data = []

    for cfg in model_configs:
        print(f"\nProcessing: {cfg['label'].replace(chr(10), ' ')}")

        # Find best checkpoint
        ckpt_dir = cfg['ckpt']
        best_path = os.path.join(ckpt_dir, 'shadow_guided_restormer_gated_best.pth')
        if not os.path.exists(best_path):
            # Try restormer_best.pth for baseline
            best_path = os.path.join(ckpt_dir, 'restormer_best.pth')
        if not os.path.exists(best_path):
            # Try finding any best/final pth
            import glob
            candidates = glob.glob(os.path.join(ckpt_dir, '*best*')) + \
                         glob.glob(os.path.join(ckpt_dir, '*final*'))
            if candidates:
                best_path = candidates[0]
            else:
                print(f"  SKIP: no checkpoint found in {ckpt_dir}")
                continue

        print(f"  Loading: {best_path}")

        model = cfg['builder']().to(device)
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        state = ckpt.get('model', ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()

        # Patch MDTA modules
        global _ATTN_STORE
        _ATTN_STORE.clear()
        patched = 0
        for module in model.modules():
            if isinstance(module, MDTA):
                module.forward = _mdta_patched_forward.__get__(module, MDTA)
                patched += 1
        print(f"  Patched {patched} MDTA modules")

        # Forward
        with torch.no_grad():
            if cfg['use_shadow']:
                out = model(gray_img, inp_img)
            else:
                out = model(inp_img)

        # Find decoder_level3 attention
        decoder3_attn = None
        decoder3_spatial = None
        for layer_id, (attn_tensor, spatial) in _ATTN_STORE.items():
            # Find by position: decoder has 3 levels, each with multiple MDTA blocks
            # We need to find the modules from decoder level 3
            pass

        # Get MDTA names to identify decoder level 3
        attn_list = list(_ATTN_STORE.items())
        print(f"  Captured {len(attn_list)} attention maps")

        # For Restormer with 4 encoder + 4 decoder levels + 4 latent + 1 refinement:
        # Encoder: levels 0,1,2,3 → MDTA blocks at each level
        # Latent: 4 blocks (level 4/bottleneck)
        # Decoder: levels 3,2,1,0 → MDTA blocks at each level (descending)
        # Refinement: 1 block
        # We want decoder level 3 → the FIRST decoder attention blocks
        # In Restormer with [4,6,6,8] MDTA config:
        # enc0=4, enc1=6, enc2=6, enc3=8, latent=8, dec3=6, dec2=6, dec1=6, dec0=4, ref=4

        # Actually, let's just pick attention from the MIDDLE of the list
        # which corresponds to the decoder's deepest level
        # For a 4-level Restormer, decoder-level3 attention maps are typically
        # the ones right after the latent blocks
        if len(attn_list) >= 10:
            # Decoder level 3: roughly 2/3 through the list
            dec3_idx = len(attn_list) * 2 // 3
            layer_id, (attn_tensor, spatial) = attn_list[dec3_idx]
            decoder3_attn = attn_tensor
            decoder3_spatial = spatial
        elif len(attn_list) > 0:
            # Fallback: use middle attention
            mid = len(attn_list) // 2
            layer_id, (attn_tensor, spatial) = attn_list[mid]
            decoder3_attn = attn_tensor
            decoder3_spatial = spatial

        if decoder3_attn is None:
            print(f"  WARNING: No attention maps captured!")
            continue

        all_data.append({
            'label': cfg['label'],
            'shadow': inp_img[0].cpu(),
            'output': out[0].cpu().clamp(0, 1),
            'target': tar_img[0].cpu(),
            'attn': decoder3_attn,
            'spatial': decoder3_spatial,
        })

    # Generate comparison figure
    if len(all_data) >= 2:
        output_dir = '/mnt/ShaDocFormer-main/attention_figures_sd7k/'
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'fig_attention_viz.pdf')
        create_comparison_figure(all_data, 'decoder_level3', save_path)

        # Also save PNG
        png_path = os.path.join(output_dir, 'fig_attention_viz.png')
        create_comparison_figure(all_data, 'decoder_level3', png_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
