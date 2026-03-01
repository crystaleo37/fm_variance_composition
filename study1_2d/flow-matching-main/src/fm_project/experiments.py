from pathlib import Path
from typing import Dict

import torch

from .config import ExperimentConfig
from .data import get_dataset
from .flow_matchers import build_matchers
from .metrics import (
    auc_trapezoid,
    mmd_rbf,
    pearson_corr,
    sliced_wasserstein,
    summarize_trajectory_metrics,
    trajectory_curvature_ratio,
)
from .plots import (
    plot_curvature_error_scatter,
    plot_main_metrics_bars,
    plot_nfe_tradeoff,
    plot_progress_curves,
    plot_samples_grid,
    plot_scorecard_heatmap,
    plot_solver_tradeoff,
    plot_training_losses,
)
from .solvers import ode_solve, ode_solve_budget
from .training import train_all_variants
from .utils import ensure_dir, resolve_device, save_json, set_seed


def run_full_experiment(config: ExperimentConfig) -> Dict:
    set_seed(config.seed)
    device = resolve_device(config.device)

    root = ensure_dir(config.output_dir)
    fig_dir = ensure_dir(root / "figures")

    x_train = get_dataset(config.dataset_name, config.n_train).to(device)
    x_eval = get_dataset(config.dataset_name, config.n_eval).to(device)

    matchers = build_matchers(config.variants, sigma=config.sigma)

    models, histories = train_all_variants(
        matchers,
        x1=x_train,
        hidden_dim=config.hidden_dim,
        time_dim=config.time_dim,
        n_layers=config.n_layers,
        n_epochs=config.n_epochs,
        steps_per_epoch=config.steps_per_epoch,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        device=device,
    )

    raw_losses = {k: v["raw_mse"] for k, v in histories.items()}
    normalized_losses = {k: v["normalized_mse"] for k, v in histories.items()}
    plot_training_losses(raw_losses, fig_dir / "training_losses_raw.png", ylabel="Raw MSE loss")
    plot_training_losses(
        normalized_losses,
        fig_dir / "training_losses_normalized.png",
        ylabel="Normalized MSE loss",
    )

    x0_eval = torch.randn_like(x_eval)

    main_metrics = {}
    nfe_metrics_sw = {name: {} for name in models}
    nfe_metrics_mmd = {name: {} for name in models}
    generated_for_plot = {}
    traj_metric_summary = {}
    curvature_error_corr = {}
    curvature_error_points = {}
    progress_curves = {}
    progress_auc = {}
    solver_nfe_swd = {
        method: {name: {} for name in models} for method in config.solver_methods
    }

    for name, model in models.items():
        out64 = ode_solve(model, x0_eval, n_steps=64, method="heun")
        x_gen = out64["x_final"]
        traj = out64["trajectory"]
        generated_for_plot[name] = x_gen

        sw = sliced_wasserstein(x_gen, x_eval)
        mmd = mmd_rbf(x_gen, x_eval)
        main_metrics[name] = {"SWD@64": sw, "MMD@64": mmd}
        traj_metric_summary[name] = summarize_trajectory_metrics(traj)

        for nfe in config.nfe_list:
            out = ode_solve_budget(model, x0_eval, nfe_budget=nfe, method="heun")
            xn = out["x_final"]
            nfe_metrics_sw[name][int(nfe)] = sliced_wasserstein(xn, x_eval)
            nfe_metrics_mmd[name][int(nfe)] = mmd_rbf(xn, x_eval)

        for method in config.solver_methods:
            for nfe in config.nfe_list:
                out = ode_solve_budget(model, x0_eval, nfe_budget=nfe, method=method)
                solver_nfe_swd[method][name][int(out["effective_nfe"])] = sliced_wasserstein(
                    out["x_final"], x_eval
                )

        low = ode_solve(model, x0_eval, n_steps=min(config.nfe_list), method="euler")
        high = ode_solve(model, x0_eval, n_steps=max(config.nfe_list), method="heun")
        curv = trajectory_curvature_ratio(high["trajectory"])
        err = torch.linalg.norm(low["x_final"] - high["x_final"], dim=-1)
        curvature_error_corr[name] = pearson_corr(curv, err)
        curvature_error_points[name] = {
            "curvature": curv.detach().cpu().numpy(),
            "error": err.detach().cpu().numpy(),
        }

        progress_swd = []
        for xt in high["trajectory"]:
            progress_swd.append(sliced_wasserstein(xt, x_eval, n_proj=32))
        progress_swd_arr = torch.tensor(progress_swd).cpu().numpy()
        progress_curves[name] = {"swd": progress_swd_arr}
        progress_auc[name] = auc_trapezoid(progress_swd_arr)

    plot_samples_grid(x_eval, generated_for_plot, fig_dir / "samples_grid.png")
    plot_nfe_tradeoff(nfe_metrics_sw, "Sliced Wasserstein", fig_dir / "nfe_vs_swd.png")
    plot_nfe_tradeoff(nfe_metrics_mmd, "MMD", fig_dir / "nfe_vs_mmd.png")
    plot_main_metrics_bars(main_metrics, fig_dir / "main_metrics_bars.png")
    plot_curvature_error_scatter(curvature_error_points, fig_dir / "curvature_vs_error.png")
    plot_progress_curves(progress_curves, fig_dir / "time_to_structure.png")
    plot_solver_tradeoff(solver_nfe_swd, fig_dir / "solver_tradeoff_swd.png")

    nfe_summary = {}
    for name in models:
        low_nfe = min(config.nfe_list)
        high_nfe = max(config.nfe_list)
        nfe_summary[name] = {
            "SWD_low_nfe": nfe_metrics_sw[name][low_nfe],
            "SWD_high_nfe": nfe_metrics_sw[name][high_nfe],
            "MMD_low_nfe": nfe_metrics_mmd[name][low_nfe],
            "MMD_high_nfe": nfe_metrics_mmd[name][high_nfe],
        }
    scorecard = {}
    for name in models:
        scorecard[name] = {
            "SWD@64": main_metrics[name]["SWD@64"],
            "MMD@64": main_metrics[name]["MMD@64"],
            "SWD@lowNFE": nfe_summary[name]["SWD_low_nfe"],
            "curvature_mean": traj_metric_summary[name]["curvature_mean"],
            "progress_auc_swd": progress_auc[name],
        }
    plot_scorecard_heatmap(scorecard, fig_dir / "scorecard_heatmap.png")

    results = {
        "config": config.__dict__,
        "training_history": histories,
        "main_metrics": main_metrics,
        "nfe_summary": nfe_summary,
        "scorecard": scorecard,
        "progress_auc_swd": progress_auc,
        "solver_nfe_swd": solver_nfe_swd,
        "trajectory_metrics": traj_metric_summary,
        "nfe_swd": nfe_metrics_sw,
        "nfe_mmd": nfe_metrics_mmd,
        "curvature_error_corr": curvature_error_corr,
        "device": str(device),
    }

    save_json(results, root / "results.json")
    return results
