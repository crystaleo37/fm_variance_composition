"""Trajectory geometry diagnostics for flow matching.

Computes curvature proxy and backtracking proxy on a fixed held-out batch
using Euler trajectories with NFE=100 (Section 5.2 of the study).
"""

import torch
import torch.nn as nn
from fm.solver import ODESolver


@torch.no_grad()
def compute_curvature_proxy(
    model: nn.Module,
    x0: torch.Tensor,
    nfe: int = 100,
) -> float:
    """Mean second-order finite difference along Euler trajectories.

    curvature = mean_batch mean_t ||x_{t+2h} - 2*x_{t+h} + x_t|| / h^2.
    Lower values indicate straighter (more OT-like) trajectories.

    Args:
        model: Velocity network v_theta(x, t).
        x0: Initial noise batch (B, C, H, W).
        nfe: Number of Euler steps.

    Returns:
        Scalar curvature proxy averaged over batch and timesteps.
    """
    solver = ODESolver(model, method="euler", nfe=nfe)
    traj = solver.trajectory(x0, steps=nfe)  # list of (nfe+1) tensors

    h = 1.0 / nfe
    curvatures = []
    for i in range(len(traj) - 2):
        diff2 = traj[i + 2] - 2 * traj[i + 1] + traj[i]
        curvatures.append(diff2.flatten(1).norm(dim=1) / (h ** 2))

    # (num_steps-2, B) -> mean over both axes
    curvatures = torch.stack(curvatures)  # (nfe-1, B)
    return curvatures.mean().item()


@torch.no_grad()
def compute_backtracking_proxy(
    model: nn.Module,
    x0: torch.Tensor,
    nfe: int = 100,
) -> float:
    """Fraction of steps where velocity points away from the trajectory endpoint.

    First runs a full Euler trajectory to get x1_final (the actual endpoint).
    Then at each step t, checks cos_angle = <v_t(x_t), x1_final - x_t>.
    Backtracking = fraction of steps where cos_angle < 0.

    Args:
        model: Velocity network v_theta(x, t).
        x0: Initial noise batch (B, C, H, W).
        nfe: Number of Euler steps.

    Returns:
        Scalar backtracking fraction in [0, 1].
    """
    solver = ODESolver(model, method="euler", nfe=nfe)
    traj = solver.trajectory(x0, steps=nfe)  # list of (nfe+1) tensors
    x1_final = traj[-1]  # actual trajectory endpoint

    total_steps = 0
    backtrack_count = 0

    for step in range(nfe):
        x_t = traj[step]
        t = step / nfe
        t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=x_t.dtype)
        v = model(x_t, t_batch)

        # Direction toward actual endpoint
        direction = x1_final - x_t

        v_flat = v.flatten(1)
        d_flat = direction.flatten(1)
        cos_angle = (v_flat * d_flat).sum(dim=1) / (
            v_flat.norm(dim=1) * d_flat.norm(dim=1) + 1e-8
        )

        backtrack_count += (cos_angle < 0).sum().item()
        total_steps += x_t.shape[0]

    return backtrack_count / max(total_steps, 1)
