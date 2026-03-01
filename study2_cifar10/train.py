"""Main training loop for conditional flow matching variance-reduction study.

Supports all 8 ablation cells via YAML config. Logs training loss, geometry
diagnostics, and per-timestep variance to CSV. Saves checkpoints with EMA.

Usage:
    python train.py --config configs/cell_000.yaml --seed 0
    python train.py --config configs/cell_000.yaml --seed 0 --debug   # fast CPU smoke-test
"""

import argparse
import copy
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from data.datasets import get_dataloader
from diagnostics.geometry import compute_curvature_proxy, compute_backtracking_proxy
from diagnostics.variance import compute_per_timestep_variance, EVAL_TIMESTEPS
from fm.coupling import IndependentCoupling, BatchOTCoupling
from fm.estimators import UniformEstimator, TPCEstimator
from fm.objectives import CFMLoss, StableVMLoss
from fm.paths import OTProbabilityPath, VPDiffusionPath
from models.unet import UNet

# ── Debug-mode overrides (small model, few steps, tiny batches) ──────────────
DEBUG_OVERRIDES = {
    "batch_size": 4,
    "base_channels": 32,
    "channel_mult": [1, 1, 2],
    "num_res_blocks": 1,
    "attn_resolutions": [8],
    "total_steps": 50,
    "log_every": 10,
    "diag_every": 25,
    "ckpt_every": 50,
    "stable_vm_K": 4,
    "stable_vm_bank": 32,
}

# ── Fast-mode overrides (T4, ~1.5h per cell — 20.5M images seen) ──
FAST_OVERRIDES = {
    "base_channels": 64,
    "num_res_blocks": 2,
    "batch_size": 256,
    "total_steps": 80_000,
    "log_every": 100,
    "diag_every": 5000,
    "ckpt_every": 80_000,
    "diag_nfe": 50,
    "diag_batch_size": 64,
}

# Convergence detection window
CONVERGENCE_WINDOW = 2000
CONVERGENCE_THRESHOLD = 1e-4


def build_components(cfg: dict):
    """Instantiate path, coupling, objective, and estimator from config dict."""
    # Path
    path_name = cfg.get("path", "ot")
    if path_name == "ot":
        path = OTProbabilityPath(sigma_min=cfg.get("sigma_min", 1e-4))
    elif path_name == "vp":
        path = VPDiffusionPath(
            beta_min=cfg.get("beta_min", 0.1), beta_max=cfg.get("beta_max", 20.0)
        )
    else:
        raise ValueError(f"Unknown path: {path_name}")

    # Coupling
    coupling_name = cfg.get("coupling", "independent")
    if coupling_name == "independent":
        coupling = IndependentCoupling()
    elif coupling_name == "batch_ot":
        coupling = BatchOTCoupling(
            reg=cfg.get("ot_reg", 0.05), max_iter=cfg.get("ot_max_iter", 100)
        )
    else:
        raise ValueError(f"Unknown coupling: {coupling_name}")

    # Objective
    objective_name = cfg.get("objective", "cfm")
    if objective_name == "cfm":
        objective = CFMLoss(path)
    elif objective_name == "stable_vm":
        objective = StableVMLoss(
            path, K=cfg.get("stable_vm_K", 16), bank_size=cfg.get("stable_vm_bank", 4096)
        )
    else:
        raise ValueError(f"Unknown objective: {objective_name}")

    # Estimator
    estimator_name = cfg.get("estimator", "uniform")
    if estimator_name == "uniform":
        estimator = UniformEstimator()
    elif estimator_name == "tpc":
        estimator = TPCEstimator(lambda_tpc=cfg.get("lambda_tpc", 0.1))
    else:
        raise ValueError(f"Unknown estimator: {estimator_name}")

    return path, coupling, objective, estimator


def ema_update(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999):
    """Exponential moving average update of model parameters."""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p, alpha=1.0 - decay)


def train(cfg: dict, seed: int):
    """Run one training cell."""
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output directory
    cell_name = cfg.get("cell_name", "unnamed")
    out_dir = os.path.join(cfg.get("output_dir", "outputs"), cell_name, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Data
    dataset = cfg.get("dataset", "cifar10")
    batch_size = cfg.get("batch_size", 128)
    loader = get_dataloader(dataset, batch_size, train=True, data_root=cfg.get("data_root", "./data"))

    # Model
    image_size = 32
    model = UNet(
        in_channels=3,
        base_channels=cfg.get("base_channels", 128),
        channel_mult=tuple(cfg.get("channel_mult", [1, 2, 2, 2])),
        num_res_blocks=cfg.get("num_res_blocks", 3),
        attn_resolutions=tuple(cfg.get("attn_resolutions", [16])),
        dropout=cfg.get("dropout", 0.1),
        image_size=image_size,
    ).to(device)

    ema_model = copy.deepcopy(model)
    ema_model.eval()

    # Components
    path, coupling, objective, estimator = build_components(cfg)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.999)),
        weight_decay=0.0,
    )

    # Mixed precision (only on CUDA — no-op on CPU)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training params
    total_steps = cfg.get("total_steps", 200_000)
    log_every = cfg.get("log_every", 100)
    diag_every = cfg.get("diag_every", 2000)
    ckpt_every = cfg.get("ckpt_every", 20_000)
    ema_decay = cfg.get("ema_decay", 0.9999)
    diag_nfe = cfg.get("diag_nfe", 100)

    # CSV loggers
    train_csv = os.path.join(out_dir, "train_loss.csv")
    diag_csv = os.path.join(out_dir, "diagnostics.csv")
    var_csv = os.path.join(out_dir, "variance.csv")

    with open(train_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "wall_time"])

    with open(diag_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "curvature", "backtracking"])

    with open(var_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + [f"t_{t:.3f}" for t in EVAL_TIMESTEPS])

    # Fixed held-out batch for diagnostics
    diag_batch_size = cfg.get("diag_batch_size", 256)
    diag_loader = get_dataloader(dataset, diag_batch_size, train=False, data_root=cfg.get("data_root", "./data"))
    diag_x1 = next(iter(diag_loader))[0].to(device)
    diag_x0 = torch.randn_like(diag_x1)

    # Training loop
    data_iter = iter(loader)
    step = 0
    start_time = time.time()
    recent_losses = []  # rolling window for convergence detection

    print(f"Training cell {cell_name}, seed {seed}, {total_steps} steps on {device}")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count / 1e6:.1f}M")

    while step < total_steps:
        # Get batch (infinite iterator)
        try:
            x1, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x1, _ = next(data_iter)

        x1 = x1.to(device, non_blocking=True)

        # Apply coupling
        x0, x1 = coupling(x1)

        # Forward pass with mixed precision
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = estimator.compute_loss(objective, model, x0, x1)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # EMA update
        ema_update(ema_model, model, decay=ema_decay)

        step += 1

        # Logging
        if step % log_every == 0:
            wall_time = time.time() - start_time
            loss_val = loss.item()

            # Early convergence detection
            recent_losses.append(loss_val)
            if len(recent_losses) > CONVERGENCE_WINDOW // log_every:
                recent_losses.pop(0)
            if len(recent_losses) >= CONVERGENCE_WINDOW // log_every:
                loss_variation = max(recent_losses) - min(recent_losses)
                if loss_variation < CONVERGENCE_THRESHOLD:
                    print(f"  [CONVERGED_EARLY] Loss variation {loss_variation:.6f} < {CONVERGENCE_THRESHOLD} over last {CONVERGENCE_WINDOW} steps")

            # Check for StableVM warmup tag
            warmup_tag = ""
            if hasattr(objective, '_warmup_tag'):
                warmup_tag = objective._warmup_tag

            with open(train_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, f"{loss_val:.6f}", f"{wall_time:.1f}"])
            tag_str = f" {warmup_tag}" if warmup_tag else ""
            print(f"Step {step}/{total_steps} | Loss: {loss_val:.4f} | Time: {wall_time:.0f}s{tag_str}")

        # Diagnostics
        if step % diag_every == 0:
            ema_model.eval()

            curvature = compute_curvature_proxy(ema_model, diag_x0, nfe=diag_nfe)
            backtracking = compute_backtracking_proxy(ema_model, diag_x0, nfe=diag_nfe)
            variances = compute_per_timestep_variance(ema_model, path, diag_x1, diag_x0)

            with open(diag_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, f"{curvature:.6f}", f"{backtracking:.6f}"])

            with open(var_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step] + [f"{v:.6f}" for v in variances])

            print(f"  Diagnostics @ {step}: curvature={curvature:.4f}, backtracking={backtracking:.4f}")

        # Checkpoint
        if step % ckpt_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "seed": seed,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    ckpt_path = os.path.join(ckpt_dir, "final.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "seed": seed,
    }, ckpt_path)
    print(f"Training complete. Final checkpoint: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a CFM model for variance study")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug/smoke-test mode: tiny model, 50 steps, batch_size=4 (runs in minutes on CPU)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: 70k steps, batch 256, lighter diagnostics (~3h/cell on T4)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        print("=== DEBUG MODE: overriding config for fast CPU smoke-test ===")
        cfg.update(DEBUG_OVERRIDES)
        cfg["diag_batch_size"] = 8
        cfg["diag_nfe"] = 10
    elif args.fast:
        print("=== FAST MODE: 70k steps, batch 256, lighter diagnostics ===")
        cfg.update(FAST_OVERRIDES)

    train(cfg, args.seed)


if __name__ == "__main__":
    main()
