"""Coupling strategies for pairing noise x0 with data x1.

Independent coupling and BatchOT (Sinkhorn + Hungarian) coupling (Tong et al., 2023).
"""

import torch
import numpy as np
import ot  # POT library
from scipy.optimize import linear_sum_assignment


class IndependentCoupling:
    """Independent coupling: x0 ~ N(0, I) sampled independently of x1 (Eq. 19, Lipman et al.)."""

    def __call__(self, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x0, x1) with x0 ~ N(0, I) independent of x1."""
        x0 = torch.randn_like(x1)
        return x0, x1


class BatchOTCoupling:
    """Batch optimal-transport coupling via Sinkhorn + Hungarian (Tong et al., 2023, Sec. 3.2).

    Solves the entropic OT problem within each minibatch to find an assignment
    that minimizes total squared Euclidean cost, then uses Hungarian algorithm
    on the transport plan for a deterministic assignment respecting both marginals.

    Args:
        reg: Entropy regularization for Sinkhorn. Default 0.05.
        max_iter: Maximum Sinkhorn iterations. Default 100.
    """

    def __init__(self, reg: float = 0.05, max_iter: int = 100):
        self.reg = reg
        self.max_iter = max_iter

    @torch.no_grad()
    def __call__(self, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return OT-coupled (x0, x1) using Sinkhorn + Hungarian within the minibatch.

        Respects both marginals exactly via deterministic assignment.
        """
        B = x1.shape[0]
        x0 = torch.randn_like(x1)

        # Flatten for cost matrix computation
        x0_flat = x0.reshape(B, -1)
        x1_flat = x1.reshape(B, -1)

        # Squared Euclidean cost matrix (B x B)
        cost = torch.cdist(x0_flat, x1_flat, p=2).pow(2)
        # Normalize for numerical stability
        cost_norm = cost / (cost.max() + 1e-8)

        # Uniform marginals
        a = torch.ones(B, device=x1.device, dtype=x1.dtype) / B
        b = torch.ones(B, device=x1.device, dtype=x1.dtype) / B

        # Sinkhorn on GPU via POT (log-domain for numerical stability)
        transport_plan = ot.bregman.sinkhorn_log(
            a, b, cost_norm, reg=self.reg, numItermax=self.max_iter
        )

        # Deterministic assignment: Hungarian on the transport plan
        # This enforces BOTH marginals exactly (one-to-one assignment)
        # linear_sum_assignment minimizes cost, so negate T to maximize it
        if isinstance(transport_plan, torch.Tensor):
            T_np = transport_plan.cpu().numpy()
        else:
            T_np = np.array(transport_plan)
        row_ind, col_ind = linear_sum_assignment(-T_np)

        # Reorder both x0 and x1 according to the OT assignment
        row_idx = torch.from_numpy(row_ind).to(x1.device)
        col_idx = torch.from_numpy(col_ind).to(x1.device)
        return x0[row_idx], x1[col_idx]
