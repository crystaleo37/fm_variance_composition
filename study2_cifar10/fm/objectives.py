"""Training objectives for conditional flow matching.

CFMLoss: standard single-sample regression (Eq. 20, Lipman et al.).
StableVMLoss: K-sample importance-weighted marginal vector field target (variance-reduced).
"""

import torch
import torch.nn as nn
from collections import deque


class CFMLoss(nn.Module):
    """Standard CFM loss: ||v_theta(x_t, t) - u_t(x_t | x1)||^2 (Eq. 20, Lipman et al.)."""

    def __init__(self, path):
        super().__init__()
        self.path = path

    def forward(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute CFM loss and return (loss, x_t) for diagnostics.

        Returns:
            loss: scalar MSE loss averaged over batch.
            xt: the interpolated samples (for potential reuse).
        """
        xt = self.path.sample_xt(x0, x1, t)
        ut = self.path.conditional_vector_field(xt, x1, t)
        vt = model(xt, t)
        loss = ((vt - ut) ** 2).flatten(1).mean()
        return loss, xt


class StableVMLoss(nn.Module):
    """Variance-reduced CFM via K-sample importance-weighted marginal vector field.

    For each x_t, draws K reference samples from a memory bank, computes
    importance weights w_k = p_t(x_t | x1_k) / sum_j p_t(x_t | x1_j),
    and uses the weighted average target sum_k w_k * u_t(x_t | x1_k).

    Args:
        path: Probability path with sample_xt, conditional_vector_field, log_prob methods.
        K: Number of reference samples per training point. Default 16.
        bank_size: FIFO memory bank capacity. Default 4096.
    """

    def __init__(self, path, K: int = 16, bank_size: int = 4096):
        super().__init__()
        self.path = path
        self.K = K
        self.bank_size = bank_size
        self.bank = deque(maxlen=bank_size)
        self._bank_tensor = None
        self._bank_dirty = True
        self._step_count = 0

    @property
    def is_warmed_up(self) -> bool:
        """Bank is considered ready when it contains at least K*2 samples."""
        return len(self.bank) >= self.K * 2

    def _update_bank(self, x1: torch.Tensor) -> None:
        """Update FIFO memory bank with new data samples."""
        for i in range(x1.shape[0]):
            self.bank.append(x1[i].detach())
        self._bank_dirty = True
        self._step_count += 1

    def _get_bank_tensor(self, device: torch.device) -> torch.Tensor:
        """Return memory bank as a stacked tensor, caching when possible."""
        if self._bank_dirty or self._bank_tensor is None:
            self._bank_tensor = torch.stack(list(self.bank)).to(device)
            self._bank_dirty = False
        return self._bank_tensor

    def forward(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute StableVM loss with importance-weighted marginal target.

        Falls back to standard CFM if bank is too small.
        Returns a tag string via self._warmup_tag for CSV logging.
        """
        self._update_bank(x1)

        xt = self.path.sample_xt(x0, x1, t)

        # Fall back to standard CFM if bank is not yet full enough
        if not self.is_warmed_up:
            self._warmup_tag = "[STABLEVM_WARMUP]"
            ut = self.path.conditional_vector_field(xt, x1, t)
            vt = model(xt, t)
            loss = ((vt - ut) ** 2).flatten(1).mean()
            return loss, xt

        self._warmup_tag = ""

        bank = self._get_bank_tensor(x1.device)
        B = x1.shape[0]

        # Sample K reference indices from the bank (without replacement)
        K = min(self.K, len(self.bank))
        indices = torch.randint(0, len(self.bank), (B, K), device=x1.device)
        x1_refs = bank[indices]  # (B, K, C, H, W)

        # Compute log importance weights: log p_t(x_t | x1_k) for each k
        # Expand xt for broadcasting: (B, 1, C, H, W)
        xt_exp = xt.unsqueeze(1).expand(-1, K, *xt.shape[1:])
        t_exp = t.unsqueeze(1).expand(-1, K).reshape(B * K)

        # Reshape for batch computation
        xt_flat = xt_exp.reshape(B * K, *xt.shape[1:])
        x1_refs_flat = x1_refs.reshape(B * K, *x1.shape[1:])

        log_weights = self.path.log_prob(xt_flat, x1_refs_flat, t_exp)  # (B*K,)
        log_weights = log_weights.reshape(B, K)

        # Self-normalized importance weights (softmax for numerical stability)
        weights = torch.softmax(log_weights, dim=1)  # (B, K)

        # Compute conditional vector fields for all references
        ut_refs = self.path.conditional_vector_field(xt_flat, x1_refs_flat, t_exp)
        ut_refs = ut_refs.reshape(B, K, *xt.shape[1:])  # (B, K, C, H, W)

        # Weighted average target: sum_k w_k * u_t(x_t | x1_k)
        weights_exp = weights.view(B, K, *([1] * (xt.dim() - 1)))
        ut_weighted = (weights_exp * ut_refs).sum(dim=1)  # (B, C, H, W)

        vt = model(xt, t)
        loss = ((vt - ut_weighted) ** 2).flatten(1).mean()
        return loss, xt
