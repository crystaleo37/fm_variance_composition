#!/usr/bin/env python3
"""Generate combined figures for the variance-composition report.

Reads CSVs/JSONs from study2_cifar10/ and produces PDF figures in figures/combined/.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

STUDY2 = Path(__file__).resolve().parent.parent / "study2_cifar10"
OUTDIR = Path(__file__).resolve().parent / "figures" / "combined"
OUTDIR.mkdir(parents=True, exist_ok=True)

CELLS = ["000", "001", "010", "011", "100", "101", "110", "111"]
CELL_LABELS = {
    "000": "Indep+CFM+Uni",
    "001": "Indep+CFM+TPC",
    "010": "Indep+SVM+Uni",
    "011": "Indep+SVM+TPC",
    "100": "OT+CFM+Uni",
    "101": "OT+CFM+TPC",
    "110": "OT+SVM+Uni",
    "111": "OT+SVM+TPC",
}

CELL_STATUS = {
    "000": "NaN",
    "001": "NaN",
    "010": "unstable",
    "011": "unstable",
    "100": "converged",
    "101": "NaN",
    "110": "unstable",
    "111": "unstable",
}


def load_train_loss(cell):
    """Load training loss CSV, return (steps, losses) arrays."""
    path = STUDY2 / "outputs" / f"cell_{cell}" / "seed_0" / "train_loss.csv"
    steps, losses = [], []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            s, l = int(parts[0]), parts[1]
            steps.append(s)
            losses.append(float(l) if l != "nan" else np.nan)
    return np.array(steps), np.array(losses)


def load_fid(cell):
    """Load FID-NFE JSON."""
    path = STUDY2 / "results" / f"cell_{cell}" / "seed_0" / "fid_nfe.json"
    with open(path) as f:
        return json.load(f)


# ---------- Figure 1: Combined Training Curves (2x4 grid) ----------
def fig_combined_training_curves():
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    axes = axes.flatten()

    for i, cell in enumerate(CELLS):
        ax = axes[i]
        steps, losses = load_train_loss(cell)

        # Find first NaN
        nan_mask = np.isnan(losses)
        first_nan = np.where(nan_mask)[0]

        # Plot valid losses
        valid = ~nan_mask
        ax.plot(steps[valid], losses[valid], linewidth=0.5, color="C0", alpha=0.7)

        # Mark NaN onset
        if len(first_nan) > 0:
            nan_step = steps[first_nan[0]]
            ax.axvline(nan_step, color="red", linestyle="--", linewidth=1.2,
                       label=f"NaN @ {nan_step}")
            ax.legend(fontsize=7, loc="upper right")

        status = CELL_STATUS[cell]
        color = {"converged": "green", "NaN": "red", "unstable": "orange"}[status]
        ax.set_title(f"cell_{cell}\n{CELL_LABELS[cell]}", fontsize=9, color=color,
                     fontweight="bold")
        ax.set_ylabel("Loss" if i % 4 == 0 else "")
        ax.set_xlabel("Step" if i >= 4 else "")
        ax.tick_params(labelsize=7)

        # Limit y-axis for unstable cells
        if status == "unstable":
            valid_losses = losses[valid]
            if len(valid_losses) > 0:
                p95 = np.percentile(valid_losses, 95)
                ax.set_ylim(0, min(p95 * 1.5, valid_losses.max()))

    fig.suptitle("Training Loss Curves — All 8 Ablation Cells", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR / "combined_training_curves.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  -> combined_training_curves.pdf")


# ---------- Figure 2: cell_100 FID detail ----------
def fig_cell_100_fid_detail():
    fid = load_fid("100")
    nfes = [5, 10, 20, 40, 100]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for solver, marker, color in [("euler", "o", "C0"), ("midpoint", "s", "C1"),
                                   ("rk4", "^", "C2")]:
        vals = [fid[f"{solver}_{n}"] for n in nfes]
        ax.plot(nfes, vals, marker=marker, color=color, linewidth=2,
                markersize=7, label=solver.capitalize())

    ax.set_xlabel("Number of Function Evaluations (NFE)", fontsize=11)
    ax.set_ylabel("FID (↓)", fontsize=11)
    ax.set_title("cell_100 (BatchOT + CFM + Uniform) — FID vs NFE", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks(nfes)
    ax.set_xticklabels(nfes)
    ax.grid(True, alpha=0.3)

    # Annotate best
    best_fid = fid["rk4_100"]
    ax.annotate(f"Best: {best_fid:.1f}", xy=(100, best_fid),
                xytext=(60, best_fid + 8), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="C2"), color="C2")

    fig.tight_layout()
    fig.savefig(OUTDIR / "cell_100_fid_detail.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  -> cell_100_fid_detail.pdf")


# ---------- Figure 3: Failure Taxonomy Table ----------
def fig_failure_taxonomy():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    # Table data
    headers = ["Cell", "Coupling", "Objective", "Estimator", "Status",
               "Best FID", "Checkpoint"]
    cell_data = [
        ["000", "Independent", "CFM", "Uniform", "NaN divergence", "679.2", "step 10k"],
        ["001", "Independent", "CFM", "TPC", "NaN divergence", "420.1", "step 10k"],
        ["010", "Independent", "StableVM", "Uniform", "Unstable", "256.8", "final"],
        ["011", "Independent", "StableVM", "TPC", "Unstable", "197.9", "final"],
        ["100", "BatchOT", "CFM", "Uniform", "Converged", "18.4", "final"],
        ["101", "BatchOT", "CFM", "TPC", "NaN divergence", "679.2", "step 10k"],
        ["110", "BatchOT", "StableVM", "Uniform", "Unstable", "211.4", "final"],
        ["111", "BatchOT", "StableVM", "TPC", "Unstable", "273.6", "final"],
    ]

    colors = []
    for row in cell_data:
        status = row[4]
        if "Converged" in status:
            colors.append(["#d4edda"] * len(headers))
        elif "NaN" in status:
            colors.append(["#f8d7da"] * len(headers))
        else:
            colors.append(["#fff3cd"] * len(headers))

    table = ax.table(cellText=cell_data, colLabels=headers, cellColours=colors,
                     colColours=["#e9ecef"] * len(headers),
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    ax.set_title("Ablation Outcome Summary — 8-Cell Factorial Design",
                 fontsize=12, pad=20, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUTDIR / "failure_taxonomy.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  -> failure_taxonomy.pdf")


# ---------- Figure 4: FID comparison bar chart ----------
def fig_fid_comparison():
    fig, ax = plt.subplots(figsize=(10, 5))

    best_fids = {}
    for cell in CELLS:
        fid = load_fid(cell)
        best_fids[cell] = min(fid.values())

    x = np.arange(len(CELLS))
    colors_map = {"converged": "#28a745", "NaN": "#dc3545", "unstable": "#ffc107"}
    colors = [colors_map[CELL_STATUS[c]] for c in CELLS]

    bars = ax.bar(x, [best_fids[c] for c in CELLS], color=colors, edgecolor="black",
                  linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"cell_{c}\n{CELL_LABELS[c]}" for c in CELLS],
                       fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Best FID (↓)", fontsize=11)
    ax.set_title("Best FID Across All 8 Cells", fontsize=12)

    # Add value labels
    for bar, cell in zip(bars, CELLS):
        val = best_fids[cell]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{val:.1f}", ha="center", fontsize=8)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#28a745", label="Converged"),
        mpatches.Patch(color="#dc3545", label="NaN divergence"),
        mpatches.Patch(color="#ffc107", label="Unstable"),
    ]
    ax.legend(handles=legend_patches, fontsize=9)
    ax.set_ylim(0, 750)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fid_comparison_bars.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  -> fid_comparison_bars.pdf")


if __name__ == "__main__":
    print("Generating combined figures...")
    fig_combined_training_curves()
    fig_cell_100_fid_detail()
    fig_failure_taxonomy()
    fig_fid_comparison()
    print("Done.")
