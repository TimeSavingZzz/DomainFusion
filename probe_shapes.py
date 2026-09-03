import torch, sys
sys.path.insert(0, '/mnt/ShaDocFormer-main')
from models.comparison_models import ShadowGuidedRestormer_NoSGCA, ShadowGuidedNAFNet_NoSGCA, Restormer, NAFNet
from models.fusion_models import ShadowGuidedNAFNet_CrossAttn, ShadowGuidedNAFNet_FiLM, ShadowGuidedNAFNet_ASF, ShadowGuidedNAFNet_Gated, ShadowGuidedNAFNet_Large
from models.comparison_models import ShadowGuidedRestormer_FiLM, ShadowGuidedRestormer_Gated, ShadowGuidedRestormer_CrossAttn

def probe(model, name, gray=False):
    model.eval()
    inp = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        if gray:
            g = torch.randn(1, 1, 64, 64)
            out = model(g, inp)
        else:
            out = model(inp)
    print(f'{name}: out={tuple(out.shape)}')

for cls, name, gray in [
    (Restormer, 'Restormer baseline', False),
    (ShadowGuidedRestormer_NoSGCA, 'Restormer Concat', True),
    (ShadowGuidedRestormer_CrossAttn, 'Restormer CrossAttn', True),
    (ShadowGuidedRestormer_FiLM, 'Restormer FiLM', True),
    (ShadowGuidedRestormer_Gated, 'Restormer Gated', True),
    (NAFNet, 'NAFNet baseline', False),
    (ShadowGuidedNAFNet_NoSGCA, 'NAFNet Concat', True),
    (ShadowGuidedNAFNet_CrossAttn, 'NAFNet CrossAttn', True),
    (ShadowGuidedNAFNet_FiLM, 'NAFNet FiLM', True),
    (ShadowGuidedNAFNet_Gated, 'NAFNet Gated', True),
    (ShadowGuidedNAFNet_ASF, 'NAFNet ASF', True),
    (ShadowGuidedNAFNet_Large, 'NAFNet Large', True),
]:
    try:
        m = cls()
        probe(m, name, gray)
    except Exception as e:
        print(f'{name}: ERROR {type(e).__name__}: {str(e)[:150]}')
