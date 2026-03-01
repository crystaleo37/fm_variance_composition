"""U-Net backbone for flow matching (~35M params).

Architecture: 4 resolution levels, attention at 16×16, channel multipliers [1,2,2,2].
Time conditioning via sinusoidal embedding + linear projection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for timestep t (Eq. 4, Vaswani et al.)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        emb = t[:, None].float() * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time-conditional group normalization."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention with group norm (applied at 16×16 resolution)."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # Scaled dot-product attention: q,k,v are (B, heads, head_dim, N)
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj_out(out)


class Downsample(nn.Module):
    """Spatial downsampling via strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling via nearest interpolation + convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for flow matching velocity prediction.

    Args:
        in_channels: Input image channels (3 for RGB).
        base_channels: Base channel count (scaled by channel_mult at each level).
        channel_mult: Per-level channel multipliers.
        num_res_blocks: Residual blocks per level.
        attn_resolutions: Spatial resolutions where attention is applied.
        dropout: Dropout rate in residual blocks.
        image_size: Input spatial size (used to determine attention levels).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 2, 2),
        num_res_blocks: int = 3,
        attn_resolutions: tuple = (16,),
        dropout: float = 0.1,
        image_size: int = 32,
    ):
        super().__init__()
        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        channels = [base_channels]
        ch = base_channels
        res = image_size

        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            block_list = nn.ModuleList()
            for _ in range(num_res_blocks):
                block_list.append(ResBlock(ch, out_ch, time_dim, dropout))
                if res in attn_resolutions:
                    block_list.append(AttentionBlock(out_ch))
                ch = out_ch
                channels.append(ch)
            self.down_blocks.append(block_list)
            if level < len(channel_mult) - 1:
                self.down_samples.append(Downsample(ch))
                channels.append(ch)
                res //= 2
            else:
                self.down_samples.append(nn.Identity())

        # Middle
        self.mid_block1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, time_dim, dropout)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level in reversed(range(len(channel_mult))):
            out_ch = base_channels * channel_mult[level]
            block_list = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                block_list.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout))
                if res in attn_resolutions:
                    block_list.append(AttentionBlock(out_ch))
                ch = out_ch
            self.up_blocks.append(block_list)
            if level > 0:
                self.up_samples.append(Upsample(ch))
                res *= 2
            else:
                self.up_samples.append(nn.Identity())

        # Output
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity field v_theta(x_t, t)."""
        t_emb = self.time_embed(t)
        h = self.input_conv(x)

        # Downsampling
        skips = [h]
        for level, (blocks, downsample) in enumerate(zip(self.down_blocks, self.down_samples)):
            i = 0
            while i < len(blocks):
                h = blocks[i](h, t_emb)
                i += 1
                if i < len(blocks) and isinstance(blocks[i], AttentionBlock):
                    h = blocks[i](h)
                    i += 1
                skips.append(h)
            if not isinstance(downsample, nn.Identity):
                h = downsample(h)
                skips.append(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Upsampling
        for blocks, upsample in zip(self.up_blocks, self.up_samples):
            i = 0
            while i < len(blocks):
                s = skips.pop()
                h = torch.cat([h, s], dim=1)
                h = blocks[i](h, t_emb)
                i += 1
                if i < len(blocks) and isinstance(blocks[i], AttentionBlock):
                    h = blocks[i](h)
                    i += 1
            if not isinstance(upsample, nn.Identity):
                h = upsample(h)

        return self.out_conv(F.silu(self.out_norm(h)))
