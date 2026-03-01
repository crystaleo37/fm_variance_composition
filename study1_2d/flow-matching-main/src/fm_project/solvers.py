from typing import Dict

import torch


def evals_per_step(method: str) -> int:
    if method == "euler":
        return 1
    if method in {"heun", "midpoint"}:
        return 2
    if method == "rk4":
        return 4
    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def ode_solve(model, x0: torch.Tensor, n_steps: int, method: str = "euler") -> Dict[str, torch.Tensor]:
    dt = 1.0 / n_steps
    xt = x0.clone()
    traj = [xt.clone()]

    for i in range(n_steps):
        t = torch.full((x0.shape[0],), i / n_steps, device=x0.device)
        if method == "euler":
            vt = model(t, xt)
            xt = xt + dt * vt
        elif method == "midpoint":
            vt1 = model(t, xt)
            x_mid = xt + 0.5 * dt * vt1
            t_mid = torch.full((x0.shape[0],), (i + 0.5) / n_steps, device=x0.device)
            vt_mid = model(t_mid, x_mid)
            xt = xt + dt * vt_mid
        elif method == "heun":
            vt1 = model(t, xt)
            x_mid = xt + dt * vt1
            t2 = torch.full((x0.shape[0],), (i + 1) / n_steps, device=x0.device)
            vt2 = model(t2, x_mid)
            xt = xt + 0.5 * dt * (vt1 + vt2)
        elif method == "rk4":
            k1 = model(t, xt)
            t_half = torch.full((x0.shape[0],), (i + 0.5) / n_steps, device=x0.device)
            k2 = model(t_half, xt + 0.5 * dt * k1)
            k3 = model(t_half, xt + 0.5 * dt * k2)
            t_full = torch.full((x0.shape[0],), (i + 1) / n_steps, device=x0.device)
            k4 = model(t_full, xt + dt * k3)
            xt = xt + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"Unknown method: {method}")
        traj.append(xt.clone())

    traj = torch.stack(traj, dim=0)
    return {"x_final": xt, "trajectory": traj}


@torch.no_grad()
def ode_solve_budget(model, x0: torch.Tensor, nfe_budget: int, method: str = "euler") -> Dict[str, torch.Tensor]:
    n_eval = evals_per_step(method)
    n_steps = max(1, nfe_budget // n_eval)
    out = ode_solve(model, x0, n_steps=n_steps, method=method)
    out["effective_nfe"] = n_steps * n_eval
    out["n_steps"] = n_steps
    return out
