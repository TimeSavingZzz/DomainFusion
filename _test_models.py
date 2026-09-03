import torch
import sys
sys.path.insert(0, '/mnt/ShaDocFormer-main')

from models.comparison_models import Restormer, NAFNet, ShadowEncoder, SGCF, SGFM, SGGF
from models.fusion_models import (
    ASF, ShadowGuidedNAFNet_FiLM, ShadowGuidedNAFNet_Gated,
    ShadowGuidedNAFNet_CrossAttn, ShadowGuidedNAFNet_ASF, ShadowGuidedNAFNet_Large,
    ShadowGuidedRestormer_ASF
)

# Test ASF
asf = ASF(64, 64)
x = torch.randn(2, 64, 32, 32)
s = torch.randn(2, 64, 32, 32)
y = asf(x, s)
print(f'ASF: {x.shape} + {s.shape} -> {y.shape}  OK')

# Test NAFNet variants
for name, cls in [
    ('NAFNet_FiLM', ShadowGuidedNAFNet_FiLM),
    ('NAFNet_Gated', ShadowGuidedNAFNet_Gated),
    ('NAFNet_CrossAttn', ShadowGuidedNAFNet_CrossAttn),
    ('NAFNet_ASF', ShadowGuidedNAFNet_ASF),
    ('NAFNet_Large', ShadowGuidedNAFNet_Large),
]:
    m = cls()
    g = torch.randn(2, 1, 256, 256)
    inp = torch.randn(2, 3, 256, 256)
    out = m(g, inp)
    p = sum(p.numel() for p in m.parameters())
    print(f'{name}: {g.shape} -> {out.shape}, params={p:,}  OK')

# Test Restormer+ASF
m = ShadowGuidedRestormer_ASF()
g = torch.randn(2, 1, 256, 256)
inp = torch.randn(2, 3, 256, 256)
out = m(g, inp)
p = sum(p.numel() for p in m.parameters())
print(f'Restormer_ASF: {inp.shape} -> {out.shape}, params={p:,}  OK')

print('All models verified!')
