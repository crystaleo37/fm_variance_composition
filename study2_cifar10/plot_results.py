"""Generate all figures for the variance-reduction composition study.

Produces 6 publication-quality figures from training logs and evaluation results.

Usage:
    python plot_results.py --results_dir outputs --output_dir figures
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

# Cell labels for the 8-cell ablation
CELL_NAMES = [
    "cell_000", "cell_001", "cell_010", "cell_011",
    "cell_100", "cell_101", "cell_110", "cell_111",
]
CELL_LABELS = [
    "Ind+CFM+Uni", "Ind+CFM+TPC", "Ind+SVM+Uni", "Ind+SVM+TPC",
    "OT+CFM+Uni", "OT+CFM+TPC", "OT+SVM+Uni", "OT+SVM+TPC",
]
SEEDS = [0, 1, 2]

# Plot style
COLORS = list(sns.color_palette("Set2", 8))
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def load_csv(path: str) -> dict:
    """Load a CSV file as a dict of columns (lists of floats)."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                data.setdefault(key, []).append(float(val))
    return data


def load_fid_results(results_dir: str) -> dict:
    """Load FID-NFE JSON results for all cells and seeds."""
    results = {}
    for cell in CELL_NAMES:
        cell_results = {}
        for seed in SEEDS:
            path = os.path.join(results_dir, cell, f"seed_{seed}", "fid_nfe.json")
            if os.path.exists(path):
                with open(path) as f:
                    cell_results[seed] = json.load(f)
        if cell_results:
            results[cell] = cell_results
    return results


def load_diagnostics(results_dir: str) -> dict:
    """Load diagnostics CSVs for all cells and seeds."""
    diag = {}
    for cell in CELL_NAMES:
        cell_diag = {}
        for seed in SEEDS:
            path = os.path.join(results_dir, cell, f"seed_{seed}", "diagnostics.csv")
            if os.path.exists(path):
                cell_diag[seed] = load_csv(path)
        if cell_diag:
            diag[cell] = cell_diag
    return diag


def load_variance(results_dir: str) -> dict:
    """Load variance CSVs for all cells and seeds."""
    var = {}
    for cell in CELL_NAMES:
        cell_var = {}
        for seed in SEEDS:
            path = os.path.join(results_dir, cell, f"seed_{seed}", "variance.csv")
            if os.path.exists(path):
                cell_var[seed] = load_csv(path)
        if cell_var:
            var[cell] = cell_var
    return var


def aggregate_seeds(data: dict, key: str) -> tuple:
    """Compute mean ± std across seeds for a given metric key."""
    values = []
    for seed in SEEDS:
        if seed in data:
            values.append(np.array(data[seed].get(key, [])))
    if not values:
        return None, None
    min_len = min(len(v) for v in values)
    values = [v[:min_len] for v in values]
    arr = np.stack(values)
    return arr.mean(axis=0), arr.std(axis=0)


# ─── Figure 1: FID–NFE Curves ───────────────────────────────────────────────

def plot_fig1_fid_nfe(fid_results: dict, output_dir: str):
    """Fig 1: FID–NFE curves for all 8 cells (3 subplots: Euler / midpoint / RK4)."""
    methods = ["euler", "midpoint", "rk4"]
    nfes = [5, 10, 20, 40, 100]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, method in zip(axes, methods):
        for i, (cell, label) in enumerate(zip(CELL_NAMES, CELL_LABELS)):
            if cell not in fid_results:
                continue

            means, stds = [], []
            for nfe in nfes:
                key = f"{method}_{nfe}"
                vals = [
                    fid_results[cell][s][key]
                    for s in SEEDS
                    if s in fid_results[cell] and key in fid_results[cell][s]
                ]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))

            if means:
                ax.errorbar(
                    nfes[:len(means)], means, yerr=stds,
                    label=label, color=COLORS[i], marker="o", markersize=4, linewidth=1.5,
                )

        ax.set_xlabel("NFE")
        ax.set_title(method.capitalize())
        ax.set_xscale("log")
        ax.set_xticks(nfes)
        ax.set_xticklabels(nfes)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("FID ↓")
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.suptitle("FID vs. NFE by Solver", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_fid_nfe.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved fig1_fid_nfe.png")


# ─── Figure 2: Per-timestep Loss Variance Heatmap ───────────────────────────

def plot_fig2_variance_heatmap(var_data: dict, output_dir: str):
    """Fig 2: Rows = 8 cells, columns = 20 t-values, color = variance magnitude."""
    from diagnostics.variance import EVAL_TIMESTEPS

    # Use the last logged step for each cell (average across seeds)
    matrix = np.zeros((8, 20))
    labels_used = []

    for i, (cell, label) in enumerate(zip(CELL_NAMES, CELL_LABELS)):
        if cell not in var_data:
            labels_used.append(label)
            continue

        t_cols = [k for k in sorted(var_data[cell].get(SEEDS[0], {}).keys()) if k.startswith("t_")]
        if not t_cols:
            labels_used.append(label)
            continue

        row_vals = []
        for seed in SEEDS:
            if seed in var_data[cell]:
                seed_vals = []
                for tc in t_cols:
                    vals = var_data[cell][seed].get(tc, [])
                    if vals:
                        seed_vals.append(vals[-1])  # last step
                if seed_vals:
                    row_vals.append(seed_vals)

        if row_vals:
            min_len = min(len(v) for v in row_vals)
            row_vals = [v[:min_len] for v in row_vals]
            matrix[i, :min_len] = np.mean(row_vals, axis=0)

        labels_used.append(label)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        matrix, ax=ax, cmap="YlOrRd",
        xticklabels=[f"{t:.2f}" for t in EVAL_TIMESTEPS],
        yticklabels=labels_used,
        cbar_kws={"label": "Loss Variance"},
    )
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Method")
    ax.set_title("Per-Timestep Loss Variance (final checkpoint)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_variance_heatmap.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved fig2_variance_heatmap.png")


# ─── Figure 3: Curvature Proxy vs. Training Step ────────────────────────────

def plot_fig3_curvature(diag_data: dict, output_dir: str):
    """Fig 3: Curvature proxy vs. training step for all 8 cells."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (cell, label) in enumerate(zip(CELL_NAMES, CELL_LABELS)):
        if cell not in diag_data:
            continue
        mean, std = aggregate_seeds(diag_data[cell], "curvature")
        if mean is None:
            continue
        steps_mean, _ = aggregate_seeds(diag_data[cell], "step")
        if steps_mean is None:
            continue
        ax.plot(steps_mean, mean, label=label, color=COLORS[i], linewidth=1.5)
        ax.fill_between(steps_mean, mean - std, mean + std, alpha=0.15, color=COLORS[i])

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Curvature Proxy ↓")
    ax.set_title("Trajectory Curvature vs. Training", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_curvature.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved fig3_curvature.png")


# ─── Figure 4: Backtracking Fraction vs. Training Step ──────────────────────

def plot_fig4_backtracking(diag_data: dict, output_dir: str):
    """Fig 4: Backtracking fraction vs. training step for all 8 cells."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (cell, label) in enumerate(zip(CELL_NAMES, CELL_LABELS)):
        if cell not in diag_data:
            continue
        mean, std = aggregate_seeds(diag_data[cell], "backtracking")
        if mean is None:
            continue
        steps_mean, _ = aggregate_seeds(diag_data[cell], "step")
        if steps_mean is None:
            continue
        ax.plot(steps_mean, mean, label=label, color=COLORS[i], linewidth=1.5)
        ax.fill_between(steps_mean, mean - std, mean + std, alpha=0.15, color=COLORS[i])

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Backtracking Fraction ↓")
    ax.set_title("Backtracking Events vs. Training", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_backtracking.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved fig4_backtracking.png")


# ─── Figure 5: Summary Bar Chart ────────────────────────────────────────────

def plot_fig5_summary_bars(fid_results: dict, output_dir: str):
    """Fig 5: FID@NFE=10 and FID@NFE=100 for all 8 cells, error bars from seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, nfe in zip(axes, [10, 100]):
        means, stds = [], []
        for cell in CELL_NAMES:
            if cell in fid_results:
                vals = [
                    fid_results[cell][s].get(f"euler_{nfe}", float("nan"))
                    for s in SEEDS
                    if s in fid_results[cell]
                ]
                means.append(np.nanmean(vals))
                stds.append(np.nanstd(vals))
            else:
                means.append(0)
                stds.append(0)

        x = np.arange(8)
        bars = ax.bar(x, means, yerr=stds, color=COLORS, capsize=4, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(CELL_LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("FID ↓")
        ax.set_title(f"FID @ NFE={nfe} (Euler)", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Summary: FID by Method", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig5_summary_bars.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved fig5_summary_bars.png")


# ─── Figure 6: Interaction Plot ─────────────────────────────────────────────

def plot_fig6_interaction(fid_results: dict, output_dir: str):
    """Fig 6: 2×2×2 interaction plot — main effects and pairwise interactions on FID@NFE=10."""
    # Axis labels for the 3 binary factors
    factors = {
        "Coupling": ["Independent", "BatchOT"],
        "Objective": ["CFM", "StableVM"],
        "Estimator": ["Uniform", "TPC"],
    }

    # Collect FID@NFE=10 for each cell
    fid_values = {}
    for idx, cell in enumerate(CELL_NAMES):
        if cell in fid_results:
            vals = [
                fid_results[cell][s].get("euler_10", float("nan"))
                for s in SEEDS
                if s in fid_results[cell]
            ]
            fid_values[idx] = (np.nanmean(vals), np.nanstd(vals))

    # Binary encoding: cell_idx -> (coupling, objective, estimator) each 0/1
    def decode(idx):
        return (idx >> 2) & 1, (idx >> 1) & 1, idx & 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Pairwise interaction plots
    pairs = [("Coupling", "Objective"), ("Coupling", "Estimator"), ("Objective", "Estimator")]
    factor_keys = list(factors.keys())

    for ax, (f1_name, f2_name) in zip(axes, pairs):
        f1_idx = factor_keys.index(f1_name)
        f2_idx = factor_keys.index(f2_name)
        f3_idx = 3 - f1_idx - f2_idx  # the remaining factor

        for f2_val in [0, 1]:
            means_line = []
            stds_line = []
            for f1_val in [0, 1]:
                # Average over the third factor
                cell_vals = []
                for f3_val in [0, 1]:
                    bits = [0, 0, 0]
                    bits[f1_idx] = f1_val
                    bits[f2_idx] = f2_val
                    bits[f3_idx] = f3_val
                    cell_idx = bits[0] * 4 + bits[1] * 2 + bits[2]
                    if cell_idx in fid_values:
                        cell_vals.append(fid_values[cell_idx][0])
                if cell_vals:
                    means_line.append(np.mean(cell_vals))
                    stds_line.append(np.std(cell_vals))
                else:
                    means_line.append(0)
                    stds_line.append(0)

            f2_label = factors[f2_name][f2_val]
            ax.errorbar(
                [0, 1], means_line, yerr=stds_line,
                label=f"{f2_name}={f2_label}",
                marker="o", linewidth=2, markersize=6, capsize=4,
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(factors[f1_name])
        ax.set_xlabel(f1_name)
        ax.set_ylabel("FID@NFE=10 ↓")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Factor Interaction Effects on FID@NFE=10", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig6_interaction.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved fig6_interaction.png")


# ─── Figure 7: TPC Lambda Ablation ────────────────────────────────────────────

def plot_fig7_lambda_ablation(results_dir: str, output_dir: str):
    """Fig 7: FID@NFE=10 vs lambda for TPC (cell 001 variants)."""
    lambda_cells = [
        ("cell_001_lam001", 0.01),
        ("cell_001_lam01", 0.1),
        ("cell_001_lam10", 1.0),
    ]

    lambdas, fids, fid_stds = [], [], []
    for cell_name, lam in lambda_cells:
        fid_path = os.path.join(results_dir, cell_name, "seed_0", "fid_nfe.json")
        if not os.path.exists(fid_path):
            continue
        with open(fid_path) as f:
            data = json.load(f)
        fid_val = data.get("euler_10", None)
        if fid_val is not None:
            lambdas.append(lam)
            fids.append(fid_val)
            fid_stds.append(0)  # single seed

    if not lambdas:
        print("Skipping fig7_lambda_ablation.png (no data)")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(lambdas, fids, yerr=fid_stds, marker="o", linewidth=2, markersize=8, capsize=5, color="tab:blue")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda_{\mathrm{TPC}}$")
    ax.set_ylabel("FID@NFE=10 (Euler) ↓")
    ax.set_title(r"TPC $\lambda$ Ablation (cell 001: Ind+CFM+TPC)", fontweight="bold")
    ax.set_xticks(lambdas)
    ax.set_xticklabels([str(l) for l in lambdas])
    ax.grid(True, alpha=0.3)
    # Highlight the chosen value
    if 0.1 in lambdas:
        idx = lambdas.index(0.1)
        ax.annotate("chosen", (lambdas[idx], fids[idx]), textcoords="offset points",
                     xytext=(15, 10), fontsize=10, color="tab:red",
                     arrowprops=dict(arrowstyle="->", color="tab:red"))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig7_lambda_ablation.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved fig7_lambda_ablation.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate all figures for the study")
    parser.add_argument("--results_dir", type=str, default="outputs", help="Directory with training outputs")
    parser.add_argument("--output_dir", type=str, default="figures", help="Directory for saved figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all data
    fid_results = load_fid_results(args.results_dir)
    diag_data = load_diagnostics(args.results_dir)
    var_data = load_variance(args.results_dir)

    # Generate figures (each function handles missing data gracefully)
    plot_fig1_fid_nfe(fid_results, args.output_dir)
    plot_fig2_variance_heatmap(var_data, args.output_dir)
    plot_fig3_curvature(diag_data, args.output_dir)
    plot_fig4_backtracking(diag_data, args.output_dir)
    plot_fig5_summary_bars(fid_results, args.output_dir)
    plot_fig6_interaction(fid_results, args.output_dir)
    plot_fig7_lambda_ablation(args.results_dir, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
