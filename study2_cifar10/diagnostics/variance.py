"""Per-timestep loss variance tracker.

Computes Var_x[||v_theta(x_t, t) - u_t(x_t | x1)||^2] at 20 evenly-spaced
t values in [0.05, 0.95] over a held-out batch (Section 5.2 of the study).
"""

import torch
import torch.nn as nn
import numpy as np


# Fixed evaluation timesteps: 20 values in [0.05, 0.95]
EVAL_TIMESTEPS = np.linspace(0.05, 0.95, 20)


@torch.no_grad()
def compute_per_timestep_variance(
    model: nn.Module,
    path,
    x1_batch: torch.Tensor,
    x0_batch: torch.Tensor = None,
) -> np.ndarray:
    """Compute per-timestep loss variance over a held-out batch.

    For each of 20 timesteps, computes the variance of the per-sample
    squared error ||v_theta(x_t, t) - u_t(x_t | x1)||^2 across the batch.

    Args:
        model: Velocity network v_theta(x, t).
        path: Probability path with sample_xt and conditional_vector_field.
        x1_batch: Data samples (B, C, H, W).
        x0_batch: Noise samples (B, C, H, W). If None, sampled fresh.

    Returns:
        Numpy array of shape (20,) with variance at each timestep.
    """
    B = x1_batch.shape[0]
    device = x1_batch.device

    if x0_batch is None:
        x0_batch = torch.randn_like(x1_batch)

    variances = np.zeros(20)

    for i, t_val in enumerate(EVAL_TIMESTEPS):
        t = torch.full((B,), t_val, device=device, dtype=x1_batch.dtype)
        xt = path.sample_xt(x0_batch, x1_batch, t)
        ut = path.conditional_vector_field(xt, x1_batch, t)
        vt = model(xt, t)

        # Per-sample squared error
        per_sample_loss = ((vt - ut) ** 2).flatten(1).sum(1)  # (B,)
        variances[i] = per_sample_loss.var().item()

    return variances
