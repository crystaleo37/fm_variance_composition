"""Evaluation script: FID-NFE curves and NLL computation.

Generates 10k samples for FID using torch-fidelity, computes BPD via dopri5.
Uses EMA weights for all evaluations.

Usage:
    python evaluate.py --checkpoint outputs/cell_000/seed_0/checkpoints/final.pt \
                       --dataset cifar10 --output_dir results/cell_000/seed_0
"""

import argparse
import json
import os
import tempfile

import numpy as np
import torch
import yaml
from tqdm import tqdm

from data.datasets import get_dataloader
from fm.paths import OTProbabilityPath, VPDiffusionPath
from fm.solver import ODESolver, NLLComputer
from models.unet import UNet


def load_ema_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load EMA model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = UNet(
        in_channels=3,
        base_channels=cfg.get("base_channels", 128),
        channel_mult=tuple(cfg.get("channel_mult", [1, 2, 2, 2])),
        num_res_blocks=cfg.get("num_res_blocks", 3),
        attn_resolutions=tuple(cfg.get("attn_resolutions", [16])),
        dropout=0.0,  # No dropout at eval
        image_size=32,
    ).to(device)

    model.load_state_dict(ckpt["ema_state_dict"])
    model.eval()
    return model, cfg


def generate_samples(
    model: torch.nn.Module,
    method: str,
    nfe: int,
    n_samples: int,
    batch_size: int = 256,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Generate n_samples using the given solver method and NFE budget."""
    solver = ODESolver(model, method=method, nfe=nfe)
    samples = []
    remaining = n_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        x0 = torch.randn(bs, 3, 32, 32, device=device)
        x1 = solver.sample(x0)
        samples.append(x1.cpu())
        remaining -= bs
    return torch.cat(samples, dim=0)[:n_samples]


def save_samples_as_images(samples: torch.Tensor, output_dir: str):
    """Save samples as individual PNG files for torch-fidelity."""
    from torchvision.utils import save_image
    os.makedirs(output_dir, exist_ok=True)
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0, 1)
    for i in range(samples.shape[0]):
        save_image(samples[i], os.path.join(output_dir, f"{i:05d}.png"))


def compute_fid(samples_dir: str, dataset: str = "cifar10") -> float:
    """Compute FID using torch-fidelity against the reference dataset."""
    import torch_fidelity

    if dataset == "cifar10":
        input2 = "cifar10-train"
    else:
        input2 = dataset

    metrics = torch_fidelity.calculate_metrics(
        input1=samples_dir,
        input2=input2,
        cuda=torch.cuda.is_available(),
        fid=True,
        verbose=False,
    )
    return metrics["frechet_inception_distance"]


def evaluate_fid_nfe(
    model: torch.nn.Module,
    dataset: str,
    output_dir: str,
    n_samples: int = 10000,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """Compute FID for all solver/NFE combinations."""
    methods = ["euler", "midpoint", "rk4"]
    nfes = [5, 10, 20, 40, 100]
    results = {}

    for method in methods:
        for nfe in nfes:
            key = f"{method}_{nfe}"
            print(f"Generating {n_samples} samples: {key}...")
            samples = generate_samples(model, method, nfe, n_samples, device=device)

            with tempfile.TemporaryDirectory() as tmpdir:
                save_samples_as_images(samples, tmpdir)
                fid = compute_fid(tmpdir, dataset)

            results[key] = fid
            print(f"  FID ({key}): {fid:.2f}")

    # Save results
    fid_path = os.path.join(output_dir, "fid_nfe.json")
    with open(fid_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"FID results saved to {fid_path}")

    return results


def evaluate_nll(
    model: torch.nn.Module,
    path,
    dataset: str,
    output_dir: str,
    device: torch.device = torch.device("cuda"),
    data_root: str = "./data",
) -> float:
    """Compute bits-per-dimension on test set using dopri5."""
    test_loader = get_dataloader(dataset, batch_size=64, train=False, data_root=data_root)
    nll_computer = NLLComputer(model, atol=1e-5, rtol=1e-5, n_hutchinson=1)

    all_bpd = []
    for batch_idx, (x1, _) in enumerate(tqdm(test_loader, desc="Computing NLL")):
        x1 = x1.to(device)
        bpd = nll_computer.compute_bpd(x1)
        all_bpd.append(bpd.cpu())
        if batch_idx >= 50:  # Cap for time
            break

    all_bpd = torch.cat(all_bpd)
    mean_bpd = all_bpd.mean().item()
    std_bpd = all_bpd.std().item()

    nll_path = os.path.join(output_dir, "nll.json")
    with open(nll_path, "w") as f:
        json.dump({"mean_bpd": mean_bpd, "std_bpd": std_bpd, "n_samples": len(all_bpd)}, f, indent=2)
    print(f"NLL: {mean_bpd:.3f} ± {std_bpd:.3f} BPD (saved to {nll_path})")

    return mean_bpd


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CFM model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet32"])
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for evaluation results")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples for FID")
    parser.add_argument("--data_root", type=str, default="./data", help="Data root directory")
    parser.add_argument("--skip_fid", action="store_true", help="Skip FID computation")
    parser.add_argument("--nll", action="store_true",
                        help="Compute NLL (optional, ~20min per cell). Off by default.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_ema_model(args.checkpoint, device)

    if not args.skip_fid:
        evaluate_fid_nfe(model, args.dataset, args.output_dir, args.n_samples, device)

    if args.nll:
        # Build path for NLL
        path_name = cfg.get("path", "ot")
        if path_name == "ot":
            path = OTProbabilityPath(sigma_min=cfg.get("sigma_min", 1e-4))
        else:
            path = VPDiffusionPath(beta_min=cfg.get("beta_min", 0.1), beta_max=cfg.get("beta_max", 20.0))
        evaluate_nll(model, path, args.dataset, args.output_dir, device, args.data_root)
    else:
        print("NLL skipped (use --nll to enable)")


if __name__ == "__main__":
    main()
