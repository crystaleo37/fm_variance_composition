from copy import deepcopy
from typing import Dict, List

from .config import ExperimentConfig
from .experiments import run_full_experiment
from .utils import ensure_dir, save_json


def _extract_dataset_summary(results: Dict) -> Dict[str, Dict[str, float]]:
    main = results["main_metrics"]
    low = results["nfe_summary"]
    geom = results["trajectory_metrics"]
    auc = results.get("progress_auc_swd", {})
    out = {}
    for variant in main:
        out[variant] = {
            "SWD@64": float(main[variant]["SWD@64"]),
            "MMD@64": float(main[variant]["MMD@64"]),
            "SWD@lowNFE": float(low[variant]["SWD_low_nfe"]),
            "curvature_mean": float(geom[variant]["curvature_mean"]),
            "progress_auc_swd": float(auc.get(variant, float("nan"))),
        }
    return out


def run_multi_dataset_suite(
    base_config: ExperimentConfig,
    datasets: List[str],
    output_dir: str,
) -> Dict:
    # Lazy import with reload so notebook kernels pick up latest plotting functions.
    from importlib import reload

    from . import plots as _plots

    _plots = reload(_plots)
    plot_multi_dataset_wins = _plots.plot_multi_dataset_wins
    plot_multi_dataset_heatmap = _plots.plot_multi_dataset_heatmap

    root = ensure_dir(output_dir)
    per_dataset = {}
    scorecards = {}

    for ds in datasets:
        cfg = deepcopy(base_config)
        cfg.dataset_name = ds
        cfg.output_dir = str(root / ds)
        res = run_full_experiment(cfg)
        per_dataset[ds] = {
            "config": res["config"],
            "main_metrics": res["main_metrics"],
            "nfe_summary": res["nfe_summary"],
            "scorecard": res["scorecard"],
            "progress_auc_swd": res.get("progress_auc_swd", {}),
        }
        scorecards[ds] = _extract_dataset_summary(res)

    # Win counts across datasets for each metric (lower is better).
    metrics = ["SWD@64", "MMD@64", "SWD@lowNFE", "curvature_mean", "progress_auc_swd"]
    variants = list(next(iter(scorecards.values())).keys())
    wins = {v: {m: 0 for m in metrics} for v in variants}
    for ds in datasets:
        for m in metrics:
            best = min(variants, key=lambda v: scorecards[ds][v][m])
            wins[best][m] += 1

    fig_dir = ensure_dir(root / "figures")
    plot_multi_dataset_wins(wins, fig_dir / "multi_dataset_wins.png")
    plot_multi_dataset_heatmap(scorecards, fig_dir / "multi_dataset_heatmap.png")

    final = {
        "datasets": datasets,
        "per_dataset": per_dataset,
        "aggregate_scorecards": scorecards,
        "wins": wins,
    }
    save_json(final, root / "results_multi_dataset.json")
    return final
