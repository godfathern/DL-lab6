"""Conditional U-Net for 64x64 i-CLEVR images.

The timestep and the multi-hot label are each embedded into a vector, summed, and fed to
every residual block, where they produce a per-channel scale and shift after GroupNorm
(FiLM / adaptive GroupNorm). Self-attention runs at the 16x16 and 8x8 resolutions.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Sinusoidal embedding of integer timesteps, as in Transformers and DDPM."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def norm(channels):
    return nn.GroupNorm(32, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout):
        super().__init__()
        self.norm1 = norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm = norm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.heads = heads

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        num_classes=24,
        base_ch=128,
        ch_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16, 8),
        dropout=0.1,
        image_size=64,
    ):
        super().__init__()
        self.base_ch = base_ch
        emb_dim = base_ch * 4
        self.time_mlp = nn.Sequential(nn.Linear(base_ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        # Multi-hot labels go through an MLP; the all-zero vector is the "no condition"
        # input that classifier-free guidance uses.
        self.label_mlp = nn.Sequential(nn.Linear(num_classes, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.input_conv = nn.Conv2d(3, base_ch, 3, padding=1)
        self.down = nn.ModuleList()
        skip_chs = [base_ch]
        ch, res = base_ch, image_size
        for level, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                layers = nn.ModuleList([ResBlock(ch, out_ch, emb_dim, dropout)])
                if res in attn_resolutions:
                    layers.append(AttnBlock(out_ch))
                self.down.append(layers)
                ch = out_ch
                skip_chs.append(ch)
            if level != len(ch_mult) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                skip_chs.append(ch)
                res //= 2

        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim, dropout), AttnBlock(ch), ResBlock(ch, ch, emb_dim, dropout)])

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks + 1):
                layers = nn.ModuleList([ResBlock(ch + skip_chs.pop(), out_ch, emb_dim, dropout)])
                if res in attn_resolutions:
                    layers.append(AttnBlock(out_ch))
                self.up.append(layers)
                ch = out_ch
            if level != 0:
                self.up.append(nn.ModuleList([Upsample(ch)]))
                res *= 2

        self.out = nn.Sequential(norm(ch), nn.SiLU(), nn.Conv2d(ch, 3, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    @staticmethod
    def _run(layers, h, emb):
        for layer in layers:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        return h

    def forward(self, x, t, labels):
        emb = self.time_mlp(timestep_embedding(t, self.base_ch)) + self.label_mlp(labels)
        h = self.input_conv(x)
        skips = [h]
        for layers in self.down:
            h = self._run(layers, h, emb)
            skips.append(h)
        h = self._run(self.mid, h, emb)
        for layers in self.up:
            if isinstance(layers[0], ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
            h = self._run(layers, h, emb)
        return self.out(h)
