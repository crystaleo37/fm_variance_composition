"""Probability paths for conditional flow matching.

Implements VP-diffusion and OT (linear) paths from Lipman et al. (2022), Section 4.
"""

import torch
import math


class OTProbabilityPath:
    """Optimal-transport conditional probability path (Eq. 22, Lipman et al.).

    Linear interpolation: mu_t = t*x1, sigma_t = 1 - (1 - sigma_min)*t.
    """

    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def mu_t(self, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Mean of p_t(x | x1) = t * x1 (Eq. 22)."""
        return t * x1

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Std of p_t(x | x1) = 1 - (1 - sigma_min) * t (Eq. 22)."""
        return 1.0 - (1.0 - self.sigma_min) * t

    def sample_xt(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x_t ~ p_t(x | x1) = N(mu_t(x1), sigma_t^2 I) (Eq. 22)."""
        t_ = t.view(-1, *([1] * (x1.dim() - 1)))
        mu = self.mu_t(x1, t_)
        sigma = self.sigma_t(t_).view_as(t_)
        return mu + sigma * x0

    def conditional_vector_field(
        self, x: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Conditional vector field u_t(x | x1) (Theorem 3, Lipman et al.).

        u_t(x | x1) = (x1 - (1 - sigma_min) * x) / (1 - (1 - sigma_min) * t).
        """
        t_ = t.view(-1, *([1] * (x1.dim() - 1)))
        return (x1 - (1.0 - self.sigma_min) * x) / (1.0 - (1.0 - self.sigma_min) * t_)

    def log_prob(self, x: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Log p_t(x | x1) — Gaussian density for importance weighting."""
        t_ = t.view(-1, *([1] * (x1.dim() - 1)))
        mu = self.mu_t(x1, t_)
        sigma = self.sigma_t(t_).view_as(t_)
        d = x.shape[1:].numel()
        return -0.5 * d * math.log(2 * math.pi) - d * sigma.log().view(-1) - 0.5 * (
            ((x - mu) / sigma) ** 2
        ).flatten(1).sum(1)


class VPDiffusionPath:
    """VP-diffusion conditional probability path (Eq. 23, Lipman et al.).

    Uses the VP schedule: beta(s) = beta_min + s*(beta_max - beta_min),
    alpha_s = exp(-0.5 * integral_0^s beta(r) dr).
    mu_t(x1) = alpha_{1-t} * x1, sigma_t = sqrt(1 - alpha_{1-t}^2).
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def _alpha(self, s: torch.Tensor) -> torch.Tensor:
        """Compute alpha_s = exp(-0.5 * integral_0^s beta(r) dr) (VP schedule)."""
        log_alpha = -0.5 * (self.beta_min * s + 0.5 * (self.beta_max - self.beta_min) * s ** 2)
        return torch.exp(log_alpha)

    def mu_t(self, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Mean of p_t(x | x1) = alpha_{1-t} * x1 (Eq. 23)."""
        alpha = self._alpha(1.0 - t)
        return alpha * x1

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Std of p_t(x | x1) = sqrt(1 - alpha_{1-t}^2) (Eq. 23)."""
        alpha = self._alpha(1.0 - t)
        return torch.sqrt(1.0 - alpha ** 2)

    def sample_xt(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x_t ~ p_t(x | x1) = N(mu_t(x1), sigma_t^2 I) (Eq. 23)."""
        t_ = t.view(-1, *([1] * (x1.dim() - 1)))
        mu = self.mu_t(x1, t_)
        sigma = self.sigma_t(t_).view_as(t_)
        return mu + sigma * x0

    def conditional_vector_field(
        self, x: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Conditional vector field u_t(x | x1) (Theorem 3 applied to VP path).

        u_t(x | x1) = d/dt[mu_t]*inv_sigma_t*(x - mu_t)/sigma_t + d/dt[sigma_t]*...
        Simplified: (d_alpha * x1 * sigma_t^2 - d_sigma * (x - alpha*x1) * sigma_t) / sigma_t^2
        where we use the time derivative form from the paper.
        """
        t_ = t.view(-1, *([1] * (x1.dim() - 1)))
        s = 1.0 - t_  # VP time variable
        alpha = self._alpha(s.squeeze())
        alpha = alpha.view_as(t_)
        sigma = torch.sqrt(1.0 - alpha ** 2)

        # d(alpha_{1-t})/dt = -d(alpha_s)/ds = alpha_s * 0.5 * beta(s)
        beta_s = self.beta_min + s * (self.beta_max - self.beta_min)
        d_alpha_dt = alpha * 0.5 * beta_s  # positive because d/dt of alpha_{1-t}

        # d(sigma_{1-t})/dt = -alpha * (-0.5*beta(s)) * alpha / sigma = alpha^2 * 0.5*beta(s) / sigma
        d_sigma_dt = -alpha ** 2 * 0.5 * beta_s / sigma

        # u_t(x|x1) = d_mu/dt + d_sigma/dt / sigma * (x - mu)
        ut = d_alpha_dt * x1 + d_sigma_dt / sigma * (x - alpha * x1)
        return ut

    def log_prob(self, x: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Log p_t(x | x1) — Gaussian density for importance weighting."""
        t_ = t.view(-1, *([1] * (x1.dim() - 1)))
        mu = self.mu_t(x1, t_)
        sigma = self.sigma_t(t_).view_as(t_)
        d = x.shape[1:].numel()
        return -0.5 * d * math.log(2 * math.pi) - d * sigma.log().view(-1) - 0.5 * (
            ((x - mu) / sigma) ** 2
        ).flatten(1).sum(1)
