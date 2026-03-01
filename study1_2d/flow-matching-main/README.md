# Flow Matching Project (Reproduction + Extensions)

Clean codebase to compare flow-matching variants and study trajectory geometry beyond baseline results.

## Structure

- `src/fm_project/`: reusable modules
- `scripts/run_experiments.py`: CLI launcher
- `notebooks/flow_matching_project.ipynb`: visualization notebook
- `reports/project_report.md`: report draft (max 5 pages target)
- `reports/presentation_outline.md`: 5-minute presentation script
- `results/`: generated metrics/figures

## Install

```bash
cd flow_matching_project
pip install -r requirements.txt
```

## Run experiments

```bash
python scripts/run_experiments.py --dataset checkerboard --epochs 200 --steps-per-epoch 50 --device cpu
```

This generates, in `results/figures/`:
- `training_losses_raw.png`
- `training_losses_normalized.png`
- `samples_grid.png`
- `main_metrics_bars.png`
- `scorecard_heatmap.png`
- `nfe_vs_swd.png`
- `nfe_vs_mmd.png`
- `curvature_vs_error.png`
- `time_to_structure.png`
- `solver_tradeoff_swd.png`

## Run multi-dataset suite

```bash
python scripts/run_multi_dataset.py --datasets checkerboard two_moons gaussian_mixture --epochs 200 --steps-per-epoch 50 --device cpu
```

Outputs are saved in `results_multi_dataset/`:
- one subfolder per dataset with full single-dataset outputs
- `results_multi_dataset/results_multi_dataset.json`
- `results_multi_dataset/figures/multi_dataset_wins.png`
- `results_multi_dataset/figures/multi_dataset_heatmap.png`

## Open notebook

```bash
jupyter lab notebooks/flow_matching_project.ipynb
```
