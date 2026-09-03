import sys
sys.path.insert(0, '/opt/miniconda3/lib/python3.13/site-packages')
from PIL import Image
import os

img = Image.open('attention_figures_sd7k/comparison/compare_sample00_decoder_level3_0_attn.png')
img.save('attention_figures_sd7k/fig_attention_sample00.pdf', 'PDF', resolution=300.0)
size_kb = os.path.getsize('attention_figures_sd7k/fig_attention_sample00.pdf') / 1024
print(f'PDF: {size_kb:.0f} KB')
