import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable

# Set writable cache paths.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplcache_fm_project"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "cache_fm_project"))
import matplotlib
try:
    from IPython import get_ipython

    _IN_NOTEBOOK = get_ipython() is not None
except Exception:
    _IN_NOTEBOOK = False

# Use a non-interactive backend only outside notebooks.
if not _IN_NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _variant_colors() -> Dict[str, str]:
    return {
        "ot": "#1f77b4",
        "vp": "#ff7f0e",
        "target": "#2ca02c",
        "schrodinger": "#d62728",
    }


def _color_for(name: str) -> str:
    colors = _variant_colors()
    return colors.get(name, "#444444")


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#cccccc",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.alpha": 0.5,
            "grid.linestyle": "-",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#cccccc",
        }
    )


def plot_training_losses(losses: Dict[str, Iterable[float]], out_path: Path, ylabel: str) -> None:
    _apply_style()
    plt.figure(figsize=(8, 4))
    for name, values in losses.items():
        y = np.asarray(list(values), dtype=float)
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, label=name, color=_color_for(name), linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.yscale("log")
    plt.title("Training Curves by Variant")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_samples_grid(
    real: torch.Tensor,
    generated: Dict[str, torch.Tensor],
    out_path: Path,
) -> None:
    _apply_style()
    n = len(generated) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharex=True, sharey=True)

    r = real.detach().cpu().numpy()
    lim = np.max(np.abs(r)) * 1.1 + 1e-8
    axes[0].scatter(r[:, 0], r[:, 1], s=3, alpha=0.35, color="#222222")
    axes[0].set_title("Target data")

    for i, (name, x) in enumerate(generated.items(), start=1):
        x_np = x.detach().cpu().numpy()
        lim = max(lim, np.max(np.abs(x_np)) * 1.1 + 1e-8)
        axes[i].scatter(x_np[:, 0], x_np[:, 1], s=3, alpha=0.35, color=_color_for(name))
        axes[i].set_title(name)

    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_nfe_tradeoff(nfe_to_metric: Dict[str, Dict[int, float]], metric_name: str, out_path: Path) -> None:
    _apply_style()
    plt.figure(figsize=(7, 4))
    for name, values in nfe_to_metric.items():
        xs = sorted(values)
        ys = [values[k] for k in xs]
        plt.plot(
            xs,
            ys,
            marker="o",
            label=name,
            color=_color_for(name),
            linewidth=2.0,
            markersize=5,
        )
    plt.xlabel("NFE")
    plt.ylabel(metric_name)
    plt.xscale("log", base=2)
    plt.title(f"NFE Tradeoff ({metric_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_curvature_error_scatter(
    per_variant_points: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    _apply_style()
    n = len(per_variant_points)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    axes = axes[0]
    for ax, (name, points) in zip(axes, per_variant_points.items()):
        x = points["curvature"]
        y = points["error"]
        ax.scatter(x, y, s=6, alpha=0.25, color=_color_for(name))
        if len(x) > 1:
            coeff = np.polyfit(x, y, deg=1)
            xx = np.linspace(np.min(x), np.max(x), 100)
            yy = coeff[0] * xx + coeff[1]
            ax.plot(xx, yy, color="black", linewidth=1.5)
        ax.set_title(name)
        ax.set_xlabel("Curvature ratio")
        ax.set_ylabel("Low-vs-high NFE error")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_progress_curves(progress: Dict[str, Dict[str, np.ndarray]], out_path: Path) -> None:
    _apply_style()
    plt.figure(figsize=(8, 4))
    for name, values in progress.items():
        y = values["swd"]
        x = np.linspace(0.0, 1.0, len(y))
        plt.plot(x, y, label=name, color=_color_for(name), linewidth=2.0)
    plt.xlabel("Normalized time t")
    plt.ylabel("SWD(x_t, target)")
    plt.title("Time-to-Structure Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_solver_tradeoff(
    solver_nfe_swd: Dict[str, Dict[str, Dict[int, float]]],
    out_path: Path,
) -> None:
    _apply_style()
    methods = list(solver_nfe_swd.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), squeeze=False)
    axes = axes[0]

    for ax, method in zip(axes, methods):
        per_variant = solver_nfe_swd[method]
        for variant, values in per_variant.items():
            xs = sorted(values)
            ys = [values[k] for k in xs]
            ax.plot(
                xs,
                ys,
                marker="o",
                label=variant,
                color=_color_for(variant),
                linewidth=1.8,
                markersize=4,
            )
        ax.set_title(method)
        ax.set_xlabel("NFE budget")
        ax.set_ylabel("SWD")
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_main_metrics_bars(main_metrics: Dict[str, Dict[str, float]], out_path: Path) -> None:
    _apply_style()
    variants = list(main_metrics.keys())
    swd = [main_metrics[v]["SWD@64"] for v in variants]
    mmd = [main_metrics[v]["MMD@64"] for v in variants]
    x = np.arange(len(variants))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    bars1 = ax1.bar(
        x - width / 2,
        swd,
        width=width,
        color=[_color_for(v) for v in variants],
        alpha=0.85,
        label="SWD@64",
    )
    ax1.set_ylabel("SWD@64")
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        mmd,
        width=width,
        color="#666666",
        alpha=0.45,
        label="MMD@64",
    )
    ax2.set_ylabel("MMD@64")
    ax1.set_title("Main Metrics by Variant (lower is better)")

    for b in list(bars1) + list(bars2):
        h = b.get_height()
        ax = ax1 if b in bars1 else ax2
        ax.text(
            b.get_x() + b.get_width() / 2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scorecard_heatmap(results: Dict[str, Dict[str, float]], out_path: Path) -> None:
    _apply_style()
    variants = list(results.keys())
    metrics = list(next(iter(results.values())).keys())
    mat = np.array([[results[v][m] for m in metrics] for v in variants], dtype=float)

    minv = np.min(mat, axis=0, keepdims=True)
    maxv = np.max(mat, axis=0, keepdims=True)
    norm = (mat - minv) / (maxv - minv + 1e-12)

    fig, ax = plt.subplots(figsize=(1.8 * len(metrics) + 2, 0.9 * len(variants) + 2))
    im = ax.imshow(norm, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels(variants)
    ax.set_title("Variant Scorecard (column-wise normalized, lower raw is better)")

    for i in range(len(variants)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_multi_dataset_wins(wins: Dict[str, Dict[str, int]], out_path: Path) -> None:
    _apply_style()
    variants = list(wins.keys())
    metrics = list(next(iter(wins.values())).keys())
    data = np.array([[wins[v][m] for m in metrics] for v in variants], dtype=float)

    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(variants))
    fig, ax = plt.subplots(figsize=(2.2 * len(metrics) + 2, 4.5))
    for i, v in enumerate(variants):
        ax.bar(
            x - 0.4 + (i + 0.5) * width,
            data[i],
            width=width,
            label=v,
            color=_color_for(v),
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylabel("Number of datasets won")
    ax.set_title("Multi-dataset wins by metric (lower is better)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_multi_dataset_heatmap(
    scorecards: Dict[str, Dict[str, Dict[str, float]]],
    out_path: Path,
) -> None:
    _apply_style()
    datasets = list(scorecards.keys())
    variants = list(next(iter(scorecards.values())).keys())
    metrics = list(next(iter(next(iter(scorecards.values())).values())).keys())

    rows = []
    row_labels = []
    for ds in datasets:
        for v in variants:
            row_labels.append(f"{ds}:{v}")
            rows.append([scorecards[ds][v][m] for m in metrics])
    mat = np.array(rows, dtype=float)

    # Normalize by dataset+metric (to compare within dataset fairly).
    norm = mat.copy()
    for dsi in range(len(datasets)):
        sl = slice(dsi * len(variants), (dsi + 1) * len(variants))
        block = norm[sl]
        mn = block.min(axis=0, keepdims=True)
        mx = block.max(axis=0, keepdims=True)
        norm[sl] = (block - mn) / (mx - mn + 1e-12)

    fig, ax = plt.subplots(figsize=(2.0 * len(metrics) + 2, 0.35 * len(row_labels) + 2.5))
    im = ax.imshow(norm, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("Multi-dataset scorecard (normalized within each dataset)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Normalized score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
