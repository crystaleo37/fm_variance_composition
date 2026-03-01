import numpy as np
import torch


def make_checkerboard(n: int, n_tiles: int = 4, noise: float = 0.02) -> np.ndarray:
    x = np.random.rand(n, 2)
    x = 2.0 * x - 1.0
    ix = np.floor((x[:, 0] + 1.0) * n_tiles / 2.0).astype(int)
    iy = np.floor((x[:, 1] + 1.0) * n_tiles / 2.0).astype(int)
    mask = (ix + iy) % 2 == 0
    x = x[mask]
    while x.shape[0] < n:
        extra = make_checkerboard(n - x.shape[0], n_tiles=n_tiles, noise=noise)
        x = np.concatenate([x, extra], axis=0)
    x = x[:n]
    x += noise * np.random.randn(n, 2)
    return x.astype(np.float32)


def make_two_moons(n: int, noise: float = 0.05) -> np.ndarray:
    n1 = n // 2
    n2 = n - n1

    t1 = np.random.rand(n1) * np.pi
    t2 = np.random.rand(n2) * np.pi

    moon1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    moon2 = np.stack([1.0 - np.cos(t2), -np.sin(t2) - 0.5], axis=1)

    x = np.concatenate([moon1, moon2], axis=0)
    x += noise * np.random.randn(n, 2)
    x -= x.mean(axis=0, keepdims=True)
    x /= x.std(axis=0, keepdims=True) + 1e-8
    x *= 0.8
    return x.astype(np.float32)


def make_gaussian_mixture(n: int, n_modes: int = 8, radius: float = 1.0, noise: float = 0.08) -> np.ndarray:
    ids = np.random.randint(0, n_modes, size=n)
    theta = 2 * np.pi * ids / n_modes
    centers = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    x = centers + noise * np.random.randn(n, 2)
    return x.astype(np.float32)


def make_source_noise(n: int, scale: float = 1.0) -> np.ndarray:
    return (scale * np.random.randn(n, 2)).astype(np.float32)


def get_dataset(name: str, n: int) -> torch.Tensor:
    if name == "checkerboard":
        x = make_checkerboard(n)
    elif name == "two_moons":
        x = make_two_moons(n)
    elif name == "gaussian_mixture":
        x = make_gaussian_mixture(n)
    else:
        raise ValueError(f"Unknown dataset '{name}'")
    return torch.from_numpy(x)
