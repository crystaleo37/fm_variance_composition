"""ODE solvers for sampling from learned flow matching models.

Wrappers for Euler, midpoint, RK4 (fixed-step), and dopri5 (adaptive-step).
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODESolver:
    """Unified ODE solver interface for flow matching sampling.

    Integrates dx/dt = v_theta(x, t) from t=0 to t=1.

    Args:
        model: Velocity network v_theta(x, t).
        method: One of 'euler', 'midpoint', 'rk4', 'dopri5'.
        nfe: Number of function evaluations (fixed-step methods). Ignored for dopri5.
        atol: Absolute tolerance (dopri5 only). Default 1e-5.
        rtol: Relative tolerance (dopri5 only). Default 1e-5.
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = "euler",
        nfe: int = 100,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        self.model = model
        self.method = method
        self.nfe = nfe
        self.atol = atol
        self.rtol = rtol

    @torch.no_grad()
    def sample(self, x0: torch.Tensor) -> torch.Tensor:
        """Integrate from noise x0 at t=0 to data at t=1."""
        if self.method == "dopri5":
            return self._solve_adaptive(x0)
        elif self.method == "euler":
            return self._solve_euler(x0)
        elif self.method == "midpoint":
            return self._solve_midpoint(x0)
        elif self.method == "rk4":
            return self._solve_rk4(x0)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _solve_euler(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Euler integration with fixed step size (Eq. 1)."""
        dt = 1.0 / self.nfe
        t = 0.0
        for _ in range(self.nfe):
            t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
            v = self.model(x, t_batch)
            x = x + dt * v
            t += dt
        return x

    def _solve_midpoint(self, x: torch.Tensor) -> torch.Tensor:
        """Midpoint method (2nd-order Runge-Kutta)."""
        dt = 1.0 / (self.nfe // 2)  # 2 evaluations per step
        t = 0.0
        for _ in range(self.nfe // 2):
            t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
            t_mid = torch.full((x.shape[0],), t + 0.5 * dt, device=x.device, dtype=x.dtype)
            k1 = self.model(x, t_batch)
            k2 = self.model(x + 0.5 * dt * k1, t_mid)
            x = x + dt * k2
            t += dt
        return x

    def _solve_rk4(self, x: torch.Tensor) -> torch.Tensor:
        """Classic 4th-order Runge-Kutta (4 evaluations per step)."""
        dt = 1.0 / (self.nfe // 4)
        t = 0.0
        for _ in range(self.nfe // 4):
            t0 = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
            t_half = torch.full((x.shape[0],), t + 0.5 * dt, device=x.device, dtype=x.dtype)
            t1 = torch.full((x.shape[0],), t + dt, device=x.device, dtype=x.dtype)
            k1 = self.model(x, t0)
            k2 = self.model(x + 0.5 * dt * k1, t_half)
            k3 = self.model(x + 0.5 * dt * k2, t_half)
            k4 = self.model(x + dt * k3, t1)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
        return x

    def _solve_adaptive(self, x0: torch.Tensor) -> torch.Tensor:
        """Adaptive dopri5 integration via torchdiffeq."""

        def odefunc(t, x):
            t_batch = torch.full((x.shape[0],), t.item(), device=x.device, dtype=x.dtype)
            return self.model(x, t_batch)

        t_span = torch.tensor([0.0, 1.0], device=x0.device, dtype=x0.dtype)
        solution = odeint(odefunc, x0, t_span, method="dopri5", atol=self.atol, rtol=self.rtol)
        return solution[-1]

    @torch.no_grad()
    def trajectory(self, x0: torch.Tensor, steps: int = 100) -> list[torch.Tensor]:
        """Return full Euler trajectory as list of (steps+1) states for diagnostics."""
        dt = 1.0 / steps
        t = 0.0
        x = x0.clone()
        traj = [x.clone()]
        for _ in range(steps):
            t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
            v = self.model(x, t_batch)
            x = x + dt * v
            t += dt
            traj.append(x.clone())
        return traj


class NLLComputer:
    """Compute bits-per-dimension via instantaneous change-of-variables (Grathwohl et al.).

    Uses Hutchinson's trace estimator with dopri5 for the augmented ODE:
      d[x, log_p]/dt = [v(x,t), -Tr(dv/dx)]
    Integrates backward from t=1 (data) to t=0 (prior).

    Args:
        model: Velocity network v_theta(x, t).
        atol: Absolute tolerance for dopri5. Default 1e-5.
        rtol: Relative tolerance for dopri5. Default 1e-5.
        n_hutchinson: Number of Hutchinson samples. Default 1.
    """

    def __init__(self, model: nn.Module, atol: float = 1e-5, rtol: float = 1e-5, n_hutchinson: int = 1):
        self.model = model
        self.atol = atol
        self.rtol = rtol
        self.n_hutchinson = n_hutchinson

    def compute_bpd(self, x1: torch.Tensor) -> torch.Tensor:
        """Compute bits-per-dimension for a batch of data samples x1."""
        import math

        B = x1.shape[0]
        d = x1.shape[1:].numel()

        def augmented_dynamics(t, state):
            x = state[:B]
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                t_batch = torch.full((B,), t.item(), device=x.device, dtype=x.dtype)
                v = self.model(x, t_batch)

                # Hutchinson trace estimator
                trace = torch.zeros(B, device=x.device, dtype=x.dtype)
                for _ in range(self.n_hutchinson):
                    eps = torch.randn_like(x)
                    vjp = torch.autograd.grad(v, x, eps, create_graph=False)[0]
                    trace += (vjp * eps).flatten(1).sum(1)
                trace /= self.n_hutchinson

            return torch.cat([v.reshape(B, -1), -trace.unsqueeze(1)], dim=1).reshape_as(state)

        # Augmented state: [x (B, d), log_det (B, 1)]
        x_flat = x1.reshape(B, -1)
        init_state = torch.cat([x_flat, torch.zeros(B, 1, device=x1.device, dtype=x1.dtype)], dim=1)

        # Integrate backward: t=1 -> t=0
        t_span = torch.tensor([1.0, 0.0], device=x1.device, dtype=x1.dtype)
        result = odeint(
            augmented_dynamics,
            init_state.reshape(B, -1),
            t_span,
            method="dopri5",
            atol=self.atol,
            rtol=self.rtol,
        )

        final = result[-1]
        x0_pred = final[:, :-1]
        log_det = final[:, -1]

        # Prior log-probability: N(0, I)
        log_prior = -0.5 * d * math.log(2 * math.pi) - 0.5 * (x0_pred ** 2).sum(1)

        # Data log-likelihood = log p_prior(x0) + log |det df/dx|
        log_px = log_prior + log_det

        # Convert to bits-per-dimension
        bpd = -log_px / (d * math.log(2))
        return bpd
