from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .models import VectorFieldMLP


@torch.no_grad()
def sample_batch(x1: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, x1.shape[0], (batch_size,), device=x1.device)
    x1b = x1[idx]
    x0b = torch.randn_like(x1b)
    return x0b, x1b


def train_one_variant(
    flow_matcher,
    x1: torch.Tensor,
    hidden_dim: int,
    time_dim: int,
    n_layers: int,
    n_epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    model = VectorFieldMLP(
        x_dim=x1.shape[1],
        hidden_dim=hidden_dim,
        time_dim=time_dim,
        n_layers=n_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raw_losses = []
    normalized_losses = []

    model.train()
    for _ in range(n_epochs):
        epoch_raw = 0.0
        epoch_norm = 0.0

        for _ in range(steps_per_epoch):
            x0b, x1b = sample_batch(x1, batch_size)
            t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0b, x1b)

            ut_dev = ut.to(device)
            pred = model(t.to(device), xt.to(device))
            loss = ((pred - ut_dev) ** 2).mean()
            # Per-batch target scaling for fairer cross-variant comparison.
            norm = ut_dev.var(unbiased=False).clamp_min(1e-8)
            normalized_loss = loss / norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_raw += float(loss.item())
            epoch_norm += float(normalized_loss.item())

        raw_losses.append(epoch_raw / steps_per_epoch)
        normalized_losses.append(epoch_norm / steps_per_epoch)

    return model, {"raw_mse": raw_losses, "normalized_mse": normalized_losses}


def train_all_variants(matchers: Dict[str, object], x1: torch.Tensor, **kwargs):
    models = {}
    histories = {}
    for name, fm in matchers.items():
        model, history = train_one_variant(fm, x1=x1, **kwargs)
        models[name] = model
        histories[name] = history
    return models, histories
