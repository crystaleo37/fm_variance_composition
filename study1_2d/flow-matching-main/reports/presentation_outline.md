# 5-Minute Presentation Outline

## Slide 1 - Problem (30s)
- Goal: compare flow-matching paths for generation quality at fixed compute.
- Key question: why do some paths sample better with fewer ODE steps?

## Slide 2 - Setup (60s)
- Same model and optimizer across variants.
- Variants: OT-CFM, VP-CFM, Target-CFM, SB-CFM.
- Datasets: checkerboard / moons / Gaussian mixture.
- Metrics: SWD, MMD, NFE tradeoff.

## Slide 3 - Reproduction Results (60s)
- Show sample grids and SWD/MMD table.
- Highlight baseline claim reproduced: OT-like paths are more efficient than diffusion-like paths at low NFE.

## Slide 4 - My Extension (90s)
- Introduce trajectory geometry metrics.
- Show curvature and alignment differences between variants.
- Show correlation plot: curvature vs low-NFE error.
- Main message: straighter trajectories are more solver-friendly.

## Slide 5 - Additional Novel Experiments (60s)
- True NFE-budget comparison across solvers (Euler/Heun/RK4).
- Time-to-structure AUC to quantify how early each path forms target geometry.
- Practical takeaway: path quality should be evaluated jointly with solver robustness.

## Slide 6 - Conclusion (30s)
- Reproduction + new diagnostic insight.
- Clean codebase for future extensions.
- Next step: evaluate same framework on image datasets.
