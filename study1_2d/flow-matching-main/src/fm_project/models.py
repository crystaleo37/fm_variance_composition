import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0.0, math.log(10_000.0), half, device=t.device)
        )
        phase = t[:, None] * freqs[None, :] * 2 * math.pi
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        if emb.shape[1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class VectorFieldMLP(nn.Module):
    def __init__(self, x_dim: int = 2, hidden_dim: int = 256, time_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        in_dim = x_dim + time_dim

        layers = []
        dims = [in_dim] + [hidden_dim] * n_layers + [x_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        t_emb = self.time_emb(t)
        inp = torch.cat([x, t_emb], dim=1)
        return self.net(inp)
