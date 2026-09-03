import sys
sys.path.insert(0, '/opt/miniconda3/lib/python3.13/site-packages')
from PIL import Image
import os

img = Image.open('attention_figures_sd7k/comparison/compare_sample00_decoder_level3_0_attn.png')
w, h = img.size
new_w = 2000
new_h = int(h * new_w / w)
img_resized = img.resize((new_w, new_h), Image.LANCZOS)
img_resized.save('attention_figures_sd7k/fig_attention_sample00.pdf', 'PDF', resolution=300.0)
size_kb = os.path.getsize('attention_figures_sd7k/fig_attention_sample00.pdf') / 1024
print(f'Original: {w}x{h}')
print(f'Resized: {new_w}x{new_h}')
print(f'PDF: {size_kb:.0f} KB')
