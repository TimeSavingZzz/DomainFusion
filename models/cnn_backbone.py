"""CNN U-Net backbone and Shadow-Guided variants for cross-paradigm comparison."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# ResBlock — standard residual block (replaces TransformerBlock in CNN U-Net)
# =============================================================================
class ResBlock(nn.Module):
    """2-conv residual block with optional expansion, same as TransformerBlock's position."""
    def __init__(self, dim, expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, hidden, 3, 1, 1, bias=bias)
        self.norm2 = nn.LayerNorm(hidden)
        self.conv2 = nn.Conv2d(hidden, dim, 3, 1, 1, bias=bias)
        self.act = nn.GELU()

    def forward(self, x, x_size=None):
        # x: (B, H*W, C)
        B, N, C = x.shape
        H, W = x_size if x_size else (int(math.sqrt(N)), int(math.sqrt(N)))
        shortcut = x
        x = self.norm1(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.act(self.conv1(x))
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, hidden)
        x = self.norm2(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.conv2(x)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return x + shortcut


# =============================================================================
# CNNUNet — Same U-Net structure as Restormer, but with ResBlocks (no attention)
# =============================================================================
class CNNUNet(nn.Module):
    """CNN U-Net backbone with identical structure to Restormer but using
    ResBlocks instead of TransformerBlocks. Designed for controlled comparison:
    CNN vs Transformer vs SSM U-Net — same skeleton, different block type.
    """
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],  # kept for API compatibility, unused
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(inp_channels, dim, 3, 1, 1, bias=bias),
        )

        # ——— Encoder ———
        self.encoder_level1 = nn.Sequential(*[
            ResBlock(dim, ffn_expansion_factor, bias) for _ in range(num_blocks[0])
        ])

        self.down1_2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, 1, 1, bias=bias),
            nn.PixelUnshuffle(2)
        )
        self.encoder_level2 = nn.Sequential(*[
            ResBlock(int(dim * 2), ffn_expansion_factor, bias) for _ in range(num_blocks[1])
        ])

        self.down2_3 = nn.Sequential(
            nn.Conv2d(int(dim * 2), int(dim * 4), 3, 1, 1, bias=bias),
            nn.PixelUnshuffle(2)
        )
        self.encoder_level3 = nn.Sequential(*[
            ResBlock(int(dim * 4), ffn_expansion_factor, bias) for _ in range(num_blocks[2])
        ])

        self.down3_4 = nn.Sequential(
            nn.Conv2d(int(dim * 4), int(dim * 8), 3, 1, 1, bias=bias),
            nn.PixelUnshuffle(2)
        )
        # Bottleneck
        self.latent = nn.Sequential(*[
            ResBlock(int(dim * 8), ffn_expansion_factor, bias) for _ in range(num_blocks[3])
        ])

        # ——— Decoder ———
        self.up4_3 = nn.Sequential(
            nn.Conv2d(int(dim * 8), int(dim * 4) * 4, 3, 1, 1, bias=bias),
            nn.PixelShuffle(2)
        )
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 8), int(dim * 4), 1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            ResBlock(int(dim * 4), ffn_expansion_factor, bias) for _ in range(num_blocks[2])
        ])

        self.up3_2 = nn.Sequential(
            nn.Conv2d(int(dim * 4), int(dim * 2) * 4, 3, 1, 1, bias=bias),
            nn.PixelShuffle(2)
        )
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 4), int(dim * 2), 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            ResBlock(int(dim * 2), ffn_expansion_factor, bias) for _ in range(num_blocks[1])
        ])

        self.up2_1 = nn.Sequential(
            nn.Conv2d(int(dim * 2), int(dim) * 4, 3, 1, 1, bias=bias),
            nn.PixelShuffle(2)
        )
        # Decoder level 1: operates at dim*2 (96ch) because skip concat = 48+48, no reduction
        self.decoder_level1 = nn.Sequential(*[
            ResBlock(int(dim * 2), ffn_expansion_factor, bias) for _ in range(num_blocks[0])
        ])

        # Refinement
        self.refinement = nn.Sequential(*[
            ResBlock(int(dim * 2), ffn_expansion_factor, bias) for _ in range(num_refinement_blocks)
        ])

        # Output
        self.output = nn.Conv2d(int(dim * 2), out_channels, 3, 1, 1, bias=bias)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (64 - h % 64) % 64
        mod_pad_w = (64 - w % 64) % 64
        if mod_pad_h > 0 or mod_pad_w > 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        """Returns encoder outputs + bottleneck + decoder intermediates for fusion."""
        B, C, H, W = x.shape

        # Patch embed
        x = self.patch_embed(x)  # (B, dim, H, W)
        
        # Encoder
        enc1 = self.encoder_level1(x.flatten(2).transpose(1, 2))  # (B, HW, dim)
        enc1_out = enc1.transpose(1, 2).view(B, -1, H, W)

        enc2_in = self.down1_2(enc1_out)
        _, _, H2, W2 = enc2_in.shape
        enc2 = self.encoder_level2(enc2_in.flatten(2).transpose(1, 2))
        enc2_out = enc2.transpose(1, 2).view(B, -1, H2, W2)

        enc3_in = self.down2_3(enc2_out)
        _, _, H3, W3 = enc3_in.shape
        enc3 = self.encoder_level3(enc3_in.flatten(2).transpose(1, 2))
        enc3_out = enc3.transpose(1, 2).view(B, -1, H3, W3)

        enc4_in = self.down3_4(enc3_out)
        _, _, H4, W4 = enc4_in.shape
        latent = self.latent(enc4_in.flatten(2).transpose(1, 2))
        latent_out = latent.transpose(1, 2).view(B, -1, H4, W4)

        # Decoder
        dec3_up = self.up4_3(latent_out)
        dec3_cat = torch.cat([dec3_up, enc3_out], dim=1)
        dec3_in = self.reduce_chan_level3(dec3_cat)
        dec3 = self.decoder_level3(dec3_in.flatten(2).transpose(1, 2))
        dec3_out = dec3.transpose(1, 2).view(B, -1, H3, W3)

        dec2_up = self.up3_2(dec3_out)
        dec2_cat = torch.cat([dec2_up, enc2_out], dim=1)
        dec2_in = self.reduce_chan_level2(dec2_cat)
        dec2 = self.decoder_level2(dec2_in.flatten(2).transpose(1, 2))
        dec2_out = dec2.transpose(1, 2).view(B, -1, H2, W2)

        dec1_up = self.up2_1(dec2_out)
        dec1_cat = torch.cat([dec1_up, enc1_out], 1)
        dec1 = self.decoder_level1(dec1_cat.flatten(2).transpose(1, 2))
        dec1_out = dec1.transpose(1, 2).view(B, -1, H, W)

        return dec3_out, dec2_out, dec1_out

    def forward(self, inp):
        """Standard forward: inp [B,3,H,W] -> output [B,3,H,W]."""
        inp = self.check_image_size(inp)
        _, dec2_out, dec1_out = self.forward_features(inp)
        refined = self.refinement(dec1_out.flatten(2).transpose(1, 2))
        refined = refined.transpose(1, 2).view(refined.shape[0], -1, dec1_out.shape[2], dec1_out.shape[3])
        out = self.output(refined) + inp
        return out
