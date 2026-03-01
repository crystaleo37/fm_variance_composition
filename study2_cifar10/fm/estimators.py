"""Timestep sampling estimators for CFM training.

UniformEstimator: t ~ U[0,1] (standard).
TPCEstimator: Temporal Pair Consistency with antithetic sampling and consistency penalty.
"""

import torch
import torch.nn as nn


class UniformEstimator:
    """Standard uniform timestep sampling: t ~ U[0, 1] (Eq. 20, Lipman et al.)."""

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample batch_size timesteps uniformly from [0, 1]."""
        return torch.rand(batch_size, device=device)

    def compute_loss(
        self,
        objective: nn.Module,
        model: nn.Module,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute standard CFM/StableVM loss with uniform timesteps."""
        t = self.sample(x0.shape[0], x0.device)
        loss, _ = objective(model, x0, x1, t)
        return loss


class TPCEstimator:
    """Temporal Pair Consistency estimator with antithetic sampling.

    For each step, samples antithetic pair (t1, t2 = 1-t1) and adds a
    consistency penalty ||v(x_{t1}, t1) - v(x_{t2}, t2)||^2 that couples
    velocity predictions at complementary timesteps.

    Args:
        lambda_tpc: Weight of the consistency penalty. Default 0.1.
    """

    def __init__(self, lambda_tpc: float = 0.1):
        self.lambda_tpc = lambda_tpc

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample antithetic timestep pairs: t1 ~ U[0, 0.5], t2 = 1 - t1."""
        t1 = torch.rand(batch_size, device=device) * 0.5
        t2 = 1.0 - t1
        return t1, t2

    def compute_loss(
        self,
        objective: nn.Module,
        model: nn.Module,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TPC loss: L_CFM(t1) + L_CFM(t2) + lambda * consistency_penalty."""
        B = x0.shape[0]
        t1, t2 = self.sample(B, x0.device)

        # CFM loss at t1
        loss1, xt1 = objective(model, x0, x1, t1)
        # CFM loss at t2 (same x0, x1 pair)
        loss2, xt2 = objective(model, x0, x1, t2)

        # Velocity predictions for consistency penalty
        with torch.no_grad():
            # Re-sample xt at t1 and t2 for the same (x0, x1) pair
            xt1_c = objective.path.sample_xt(x0, x1, t1)
            xt2_c = objective.path.sample_xt(x0, x1, t2)

        vt1 = model(xt1_c, t1)
        vt2 = model(xt2_c, t2)

        # Consistency penalty: ||v(x_{t1}, t1) - v(x_{t2}, t2)||^2
        consistency = ((vt1 - vt2) ** 2).flatten(1).mean()

        return loss1 + loss2 + self.lambda_tpc * consistency
