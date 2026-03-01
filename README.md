# Variance-Reduction Composition in Conditional Flow Matching

**Master 2 Deep Learning — March 2026**
**Authors:** Sacha Leiderr, Amine Soukane

## Overview

This repository contains the code, data, and report for a study on whether three variance-reduction techniques for Conditional Flow Matching (CFM) compose additively:

1. **Coupling**: Independent vs. Minibatch Optimal Transport (BatchOT)
2. **Objective**: Standard CFM vs. StableVM (importance-weighted marginal VF)
3. **Timestep Estimator**: Uniform vs. TPC (Temporal Pair Consistency)

## Structure

```
fm_variance_composition/
├── study1_2d/             # 2D synthetic experiments (Sacha Leiderr)
│   └── flow-matching-main/
│       ├── src/           # FM variants: OT, VP, Target, SB
│       ├── results/       # Single-dataset (checkerboard)
│       └── results_multi_dataset/  # 3 datasets
├── study2_cifar10/        # CIFAR-10 ablation (Amine Soukane)
│   ├── fm/                # Core: paths, coupling, objectives, estimators
│   ├── models/            # U-Net backbone
│   ├── configs/           # 8 YAML configs (cell_000 to cell_111)
│   ├── results/           # FID-NFE JSONs (all 8 cells)
│   ├── outputs/           # Training CSVs and per-cell figures
│   └── figures/           # Aggregate analysis figures
└── report/
    ├── main.tex           # LaTeX report entry point
    ├── sections/          # One .tex per section
    ├── figures/           # study1/, study2/, combined/
    ├── bibliography.bib
    └── make_figures.py    # Generate combined figures
```

## Key Results

- **Study 1 (2D):** SB-CFM wins endpoint quality; OT-CFM produces the straightest trajectories and best low-NFE robustness. All variants work well in 2D.
- **Study 2 (CIFAR-10):** Only 1/8 cells converged — **cell_100** (BatchOT + CFM + Uniform) with FID 18.4. StableVM causes unstable oscillation; TPC causes NaN divergence. The three axes are NOT additive.

## Building the Report

```bash
cd report
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Generating Combined Figures

```bash
cd report
pip install matplotlib numpy
python make_figures.py
```

## Reproducing Study 2

See `study2_cifar10/slurm/` for SLURM job scripts targeting AMD MI210 GPUs:

```bash
cd study2_cifar10
bash slurm/setup_env.sh    # Create venv + install deps
bash slurm/launch_all.sh   # Submit all 8 cells to SLURM
```

## Reference

Based on: Lipman et al. (2023), "Flow Matching for Generative Modeling" (ICLR 2023).
