from PIL import Image; import os
img = Image.open('attention_figures_sd7k/comparison/compare_sample00_decoder_level3_0_attn.png')
img.resize((1500,1498), Image.LANCZOS).save('attention_figures_sd7k/fig_attn_1500.pdf', 'PDF', resolution=300.0)
size = os.path.getsize('attention_figures_sd7k/fig_attn_1500.pdf') / 1024
print(f'PDF 1500px: {size:.0f} KB')
