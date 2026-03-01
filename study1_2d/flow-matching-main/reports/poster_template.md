# Poster Template - Flow Matching Project

## Title
**Flow Matching for Generative Modeling: Reproduction and Trajectory Geometry Analysis**

## Authors
- [Your Name]
- [Teammate Name]
- Course: [Course Name], [School], Spring 2026

## 1. Motivation
- Compare diffusion-like and OT-like probability paths in flow matching.
- Understand not only *which* variant works better, but *why* (trajectory geometry).

## 2. Method
- Same network and training setup for all variants.
- Variants: OT-CFM, VP-CFM, Target-CFM, SB-CFM.
- Datasets: checkerboard, two moons, Gaussian mixture.
- Metrics: SWD, MMD, NFE vs quality tradeoff.

## 3. Main Results
Insert figures:
- `training_losses.png`
- `samples_grid.png`
- `nfe_vs_swd.png`
- `nfe_vs_mmd.png`

Add table:

| Variant | SWD@64 | MMD@64 |
|---|---:|---:|
| OT-CFM | [fill] | [fill] |
| VP-CFM | [fill] | [fill] |
| Target-CFM | [fill] | [fill] |
| SB-CFM | [fill] | [fill] |

## 4. Beyond Baseline: Trajectory Geometry
- Curvature ratio, alignment, and mean speed.
- Correlation between trajectory curvature and low-NFE error.
- Key finding: straighter trajectories are generally more robust under low solver budget.

## 5. Conclusion
- Reproduced baseline trend (OT-style paths better at low NFE).
- Added geometry-based explanation and diagnostics.
- Proposed path toward larger-scale benchmarks (images).

## 6. Reproducibility
- Code: `flow_matching_project/src/fm_project`
- Notebook: `flow_matching_project/notebooks/flow_matching_project.ipynb`
- Run command:

```bash
python scripts/run_experiments.py --dataset checkerboard --epochs 200
```
