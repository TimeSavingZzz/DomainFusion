import torch, sys
sys.path.insert(0, '/mnt/ShaDocFormer-main')
from models.comparison_models import ShadowGuidedRestormer_NoSGCA, ShadowGuidedNAFNet_NoSGCA, ShadowGuidedRestormer_CrossAttn
from models.fusion_models import ShadowGuidedNAFNet_CrossAttn

captured = {}
def make_hook(name):
    def hook(mod, args, output):
        captured[name] = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]
    return hook

for cls, name, mod_names in [
    (ShadowGuidedRestormer_NoSGCA, 'Restormer Concat', ['fuse_dec3', 'fuse_dec2']),
    (ShadowGuidedNAFNet_NoSGCA, 'NAFNet Concat', ['fuse_dec1', 'fuse_dec2']),
    (ShadowGuidedNAFNet_CrossAttn, 'NAFNet CrossAttn', ['sgcf_dec1', 'sgcf_dec2']),
    (ShadowGuidedRestormer_CrossAttn, 'Restormer CrossAttn', ['sgcf_dec3', 'sgcf_dec2']),
]:
    m = cls()
    captured.clear()
    for mn in mod_names:
        mod = dict(m.named_modules())[mn]
        mod.register_forward_hook(make_hook(mn))
    m.eval()
    inp = torch.randn(1, 3, 64, 64)
    g = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        m(g, inp)
    print(f'{name}:')
    for mn in mod_names:
        print(f'  {mn} inputs: {captured.get(mn)}')
