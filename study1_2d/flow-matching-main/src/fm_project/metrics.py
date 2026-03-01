import math
from typing import Dict

import numpy as np
import torch


def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, n_proj: int = 128) -> float:
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    d = x_np.shape[1]

    vals = []
    for _ in range(n_proj):
        p = np.random.randn(d)
        p /= np.linalg.norm(p) + 1e-12
        xp = np.sort(x_np @ p)
        yp = np.sort(y_np @ p)
        vals.append(np.mean(np.abs(xp - yp)))
    return float(np.mean(vals))


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, gamma: float = 2.0) -> float:
    x = x.detach()
    y = y.detach()

    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2

    kxx = torch.exp(-gamma * xx).mean()
    kyy = torch.exp(-gamma * yy).mean()
    kxy = torch.exp(-gamma * xy).mean()

    return float(kxx + kyy - 2 * kxy)


def trajectory_curvature_ratio(traj: torch.Tensor) -> torch.Tensor:
    diffs = traj[1:] - traj[:-1]
    seg_len = torch.linalg.norm(diffs, dim=-1)
    path_len = seg_len.sum(dim=0)
    chord = torch.linalg.norm(traj[-1] - traj[0], dim=-1)
    return path_len / (chord + 1e-8)


def trajectory_mean_speed(traj: torch.Tensor) -> torch.Tensor:
    diffs = traj[1:] - traj[:-1]
    seg_len = torch.linalg.norm(diffs, dim=-1)
    return seg_len.mean(dim=0)


def trajectory_alignment(traj: torch.Tensor) -> torch.Tensor:
    direction = traj[-1] - traj[0]
    direction = direction / (torch.linalg.norm(direction, dim=-1, keepdim=True) + 1e-8)

    diffs = traj[1:] - traj[:-1]
    tangent = diffs / (torch.linalg.norm(diffs, dim=-1, keepdim=True) + 1e-8)
    cos = (tangent * direction.unsqueeze(0)).sum(dim=-1)
    return cos.mean(dim=0)


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt((x**2).sum() * (y**2).sum()) + 1e-12
    return float((x * y).sum() / den)


def auc_trapezoid(y: np.ndarray) -> float:
    if y.ndim != 1:
        raise ValueError("auc_trapezoid expects a 1D array")
    x = np.linspace(0.0, 1.0, len(y))
    return float(np.trapz(y, x))


def summarize_trajectory_metrics(traj: torch.Tensor) -> Dict[str, float]:
    curvature = trajectory_curvature_ratio(traj)
    speed = trajectory_mean_speed(traj)
    align = trajectory_alignment(traj)
    return {
        "curvature_mean": float(curvature.mean().item()),
        "curvature_std": float(curvature.std().item()),
        "speed_mean": float(speed.mean().item()),
        "alignment_mean": float(align.mean().item()),
    }
