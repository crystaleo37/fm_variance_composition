"""Launch the full 2^3 = 8-cell ablation table.

Generates YAML configs for all combinations of:
  - Coupling: Independent vs. BatchOT
  - Objective: CFM vs. StableVM
  - Estimator: Uniform vs. TPC

Also generates 2 sanity-check configs: SM-Diffusion and FM-Diffusion.

Usage:
    # Run a specific cell and seed:
    python run_ablation.py --cell 0 --seed 0

    # Generate all configs without training:
    python run_ablation.py --generate_configs

    # Run all cells (sequentially):
    python run_ablation.py --all --seed 0
"""

import argparse
import os
import subprocess
import sys

import yaml


# 8-cell ablation table
CELLS = {
    0: {"coupling": "independent", "objective": "cfm", "estimator": "uniform", "cell_name": "cell_000"},
    1: {"coupling": "independent", "objective": "cfm", "estimator": "tpc", "cell_name": "cell_001"},
    2: {"coupling": "independent", "objective": "stable_vm", "estimator": "uniform", "cell_name": "cell_010"},
    3: {"coupling": "independent", "objective": "stable_vm", "estimator": "tpc", "cell_name": "cell_011"},
    4: {"coupling": "batch_ot", "objective": "cfm", "estimator": "uniform", "cell_name": "cell_100"},
    5: {"coupling": "batch_ot", "objective": "cfm", "estimator": "tpc", "cell_name": "cell_101"},
    6: {"coupling": "batch_ot", "objective": "stable_vm", "estimator": "uniform", "cell_name": "cell_110"},
    7: {"coupling": "batch_ot", "objective": "stable_vm", "estimator": "tpc", "cell_name": "cell_111"},
}

# TPC lambda ablation on cell 001 (Independent + CFM + TPC)
# These run in fast mode (30k steps) — enough to compare lambda effect
LAMBDA_ABLATION = {
    "001_lam001": {"coupling": "independent", "objective": "cfm", "estimator": "tpc",
                   "lambda_tpc": 0.01, "cell_name": "cell_001_lam001"},
    "001_lam01":  {"coupling": "independent", "objective": "cfm", "estimator": "tpc",
                   "lambda_tpc": 0.1, "cell_name": "cell_001_lam01"},
    "001_lam10":  {"coupling": "independent", "objective": "cfm", "estimator": "tpc",
                   "lambda_tpc": 1.0, "cell_name": "cell_001_lam10"},
}

# Sanity check configs from Lipman et al.
SANITY_CHECKS = {
    "sm_diffusion": {
        "coupling": "independent",
        "objective": "cfm",
        "estimator": "uniform",
        "path": "vp",
        "cell_name": "sm_diffusion",
    },
    "fm_diffusion": {
        "coupling": "independent",
        "objective": "cfm",
        "estimator": "uniform",
        "path": "vp",
        "cell_name": "fm_diffusion",
    },
}


def make_base_config(dataset: str = "cifar10") -> dict:
    """Base config shared by all cells."""
    return {
        # Data
        "dataset": dataset,
        "data_root": "./data",
        "batch_size": 512 if dataset == "cifar10" else 512,
        # Model
        "base_channels": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 3,
        "attn_resolutions": [16],
        "dropout": 0.1,
        # Path (default OT, overridden for VP cells)
        "path": "ot",
        "sigma_min": 1e-4,
        # Training
        "lr": 1e-4,
        "beta1": 0.9,
        "beta2": 0.999,
        "total_steps": 150_000 if dataset == "cifar10" else 300_000,
        "ema_decay": 0.9999,
        "log_every": 100,
        "diag_every": 2000,
        "ckpt_every": 20_000,
        "output_dir": "outputs",
        # StableVM defaults
        "stable_vm_K": 16,
        "stable_vm_bank": 4096,
        # TPC defaults
        "lambda_tpc": 0.1,
        # OT coupling defaults
        "ot_reg": 0.05,
        "ot_max_iter": 100,
    }


def generate_config(cell_spec: dict, dataset: str = "cifar10") -> dict:
    """Merge cell specification into base config."""
    cfg = make_base_config(dataset)
    cfg.update(cell_spec)
    return cfg


def generate_all_configs(config_dir: str = "configs", dataset: str = "cifar10"):
    """Write all YAML configs to disk."""
    os.makedirs(config_dir, exist_ok=True)

    # 8 ablation cells
    for cell_id, cell_spec in CELLS.items():
        cfg = generate_config(cell_spec, dataset)
        path = os.path.join(config_dir, f"cell_{cell_id:03b}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Written: {path}")

    # Lambda ablation configs
    for name, cell_spec in LAMBDA_ABLATION.items():
        cfg = generate_config(cell_spec, dataset)
        path = os.path.join(config_dir, f"cell_{name}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Written: {path}")

    # Sanity checks
    for name, cell_spec in SANITY_CHECKS.items():
        cfg = generate_config(cell_spec, dataset)
        path = os.path.join(config_dir, f"{name}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Written: {path}")


# Multi-seed strategy: 3 seeds for extreme cells, 1 seed for intermediate
MULTI_SEED_CELLS = ["000", "111"]
ALL_SEEDS = [0, 1, 2]
SINGLE_SEED_CELLS = ["001", "010", "011", "100", "101", "110"]


def run_cell(cell_id: int, seed: int, config_dir: str = "configs", extra_flags: list = None):
    """Launch training for a single cell."""
    config_path = os.path.join(config_dir, f"cell_{cell_id:03b}.yaml")

    if not os.path.exists(config_path):
        print(f"Config not found, generating: {config_path}")
        generate_all_configs(config_dir)

    cmd = [sys.executable, "train.py", "--config", config_path, "--seed", str(seed)]
    if extra_flags:
        cmd.extend(extra_flags)
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_lambda_ablation(config_dir: str = "configs", extra_flags: list = None):
    """Run TPC lambda ablation (3 configs, fast mode, seed 0 only)."""
    for name in LAMBDA_ABLATION:
        config_path = os.path.join(config_dir, f"cell_{name}.yaml")
        if not os.path.exists(config_path):
            generate_all_configs(config_dir)
        cmd = [sys.executable, "train.py", "--config", config_path, "--seed", "0", "--fast"]
        if extra_flags:
            cmd.extend(extra_flags)
        print(f"Running lambda ablation: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--cell", type=int, default=None, help="Cell index (0-7)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet32"])
    parser.add_argument("--generate_configs", action="store_true", help="Only generate YAML configs")
    parser.add_argument("--all", action="store_true", help="Run all 8 cells sequentially")
    parser.add_argument("--all_seeds", action="store_true",
                        help="Run targeted multi-seed: 3 seeds for cell_000/111, 1 seed for rest")
    parser.add_argument("--lambda_ablation", action="store_true", help="Run TPC lambda ablation")
    parser.add_argument("--config_dir", type=str, default="configs", help="Config directory")
    parser.add_argument("--fast", action="store_true", help="Pass --fast flag to train.py")
    args = parser.parse_args()

    extra_flags = ["--fast"] if args.fast else []

    if args.generate_configs:
        generate_all_configs(args.config_dir, args.dataset)
        return

    if args.lambda_ablation:
        run_lambda_ablation(args.config_dir, extra_flags)
        return

    if args.all_seeds:
        # Targeted multi-seed strategy
        for cell_id in range(8):
            cell_bin = f"{cell_id:03b}"
            if cell_bin in MULTI_SEED_CELLS:
                for seed in ALL_SEEDS:
                    run_cell(cell_id, seed, args.config_dir, extra_flags)
            else:
                run_cell(cell_id, 0, args.config_dir, extra_flags)
        return

    if args.all:
        for cell_id in range(8):
            run_cell(cell_id, args.seed, args.config_dir, extra_flags)
        return

    if args.cell is not None:
        run_cell(args.cell, args.seed, args.config_dir, extra_flags)
    else:
        print("Specify --cell <0-7>, --all, --all_seeds, --lambda_ablation, or --generate_configs")
        parser.print_help()


if __name__ == "__main__":
    main()
