# Flow Matching for Generative Modeling: Reproduction and Extensions

## 1. Introduction
The objective of this project is to study whether the probability path used in flow matching has a measurable impact on sample quality and computational efficiency. The first part of the work reproduces the baseline comparison between diffusion-like and optimal-transport-like paths. The second part extends the analysis with trajectory-geometry diagnostics and solver-aware evaluation, with the goal of explaining *why* some variants are more robust than others under limited numerical budget.

To keep the study reproducible, all experiments are implemented in a modular codebase (`src/fm_project`) and executed from a report-style notebook. The same network architecture, optimizer, and training protocol are shared across variants; only the flow-matcher definition changes.

## 2. Experimental Setup
The main experiments are run on 2D synthetic distributions because they provide direct visual access to trajectories and generated samples. I first analyzed `checkerboard`, then extended the study to `two_moons` and `gaussian_mixture` for cross-dataset robustness.

The compared variants are OT-CFM, VP-CFM, Target-CFM, and Schrödinger-Bridge CFM (SB-CFM). The vector field is a time-conditioned MLP trained by velocity regression. Sampling is performed by solving the learned ODE, and quality is measured with SWD and MMD. Beyond standard endpoint quality, I evaluate: (i) low-vs-high NFE behavior, (ii) trajectory geometry (curvature, alignment), (iii) solver robustness under equal NFE budget, and (iv) time-to-structure via SWD-over-time AUC.

## 3. Single-Dataset Results (Checkerboard)
On checkerboard, SB-CFM gives the best endpoint quality at NFE=64, with OT-CFM very close. The measured values are:

| Variant | SWD@64 | MMD@64 |
|---|---:|---:|
| OT-CFM | 0.02049 | 0.000503 |
| VP-CFM | 0.02655 | 0.000792 |
| Target-CFM | 0.03004 | 0.001068 |
| SB-CFM | **0.02016** | **0.000449** |

Compared to VP-CFM, SB-CFM improves SWD@64 by about 24% and MMD@64 by about 43%, which confirms that path choice materially changes practical generation quality.

The low-budget regime is even more discriminative. At low NFE, OT and SB remain around 0.022 SWD, whereas VP and Target are much higher (around 0.042 and 0.039). This means OT/SB are roughly 47% better than VP in this regime. As NFE increases, VP improves strongly, which indicates higher discretization sensitivity; OT/SB improve less because they are already strong at low budget.

Trajectory metrics support this behavior. OT and SB are the straightest (curvature means close to 1.0) and the most aligned with source-to-target displacement. Target is much more curved in this run. However, the per-sample curvature/error correlation is weak and slightly negative here, so the relation is not a single monotonic law in this dataset/seed setting.

Finally, time-to-structure reveals an interesting nuance: Target-CFM obtains the best AUC on checkerboard (faster early structural progress), but does not win endpoint SWD/MMD. This suggests that “early geometric progress” and “final sample fidelity” are related but not equivalent objectives.

## 4. Multi-Dataset Results
To move beyond a single dataset, I ran the full suite on checkerboard, two_moons, and gaussian_mixture. The aggregate results show a clear pattern.

On endpoint quality (SWD@64 and MMD@64), SB-CFM wins all three datasets. On low-NFE SWD, OT-CFM wins two datasets and SB-CFM wins one, indicating that both are robust under strict compute budgets. On curvature, OT-CFM is consistently the straightest across all datasets. On time-to-structure AUC, SB-CFM wins two datasets while Target wins one.

This is summarized by the multi-dataset win counts:

| Variant | SWD@64 wins | MMD@64 wins | SWD@lowNFE wins | Curvature wins | AUC wins |
|---|---:|---:|---:|---:|---:|
| OT-CFM | 0 | 0 | **2** | **3** | 0 |
| VP-CFM | 0 | 0 | 0 | 0 | 0 |
| Target-CFM | 0 | 0 | 0 | 0 | 1 |
| SB-CFM | **3** | **3** | 1 | 0 | **2** |

These multi-dataset results are stronger than the single-dataset claim: SB-CFM is the most reliable endpoint performer, while OT-CFM provides the most consistently straight trajectories and strongest low-NFE robustness on most datasets.

## 5. What Is New Beyond Reproduction
This project is not a notebook-only baseline rerun. It contributes a reusable experimental framework and three concrete extensions: budget-fair solver comparisons, trajectory-geometry diagnostics, and time-to-structure analysis. The extensions are directly tied to the practical question of sampling efficiency, and the multi-dataset sweep provides stronger evidence than a single benchmark.

## 6. Limitations
The study remains in 2D synthetic settings, so conclusions should be interpreted as mechanistic and comparative rather than definitive for high-dimensional image generation. In addition, some secondary relations (for example curvature/error correlation sign) are seed- and dataset-dependent, which motivates multi-seed aggregation as the next step.

## 7. Conclusion
The reproduction objective is successful: path design clearly affects generation quality and low-budget behavior. The extension objective is also successful: trajectory geometry and solver-aware diagnostics reveal complementary information that average endpoint metrics miss. Across three datasets, SB-CFM is the most consistent winner on final quality, while OT-CFM is strongest on geometric straightness and often on low-NFE robustness. Together, these results support a practical recommendation: evaluate flow-matching variants jointly on endpoint quality, low-budget robustness, and trajectory geometry, rather than relying on a single metric.

## 8. Contribution Statement
I designed and implemented the modular experimental pipeline, reproduced the baseline variant comparison, and developed the extension analyses (geometry, solver-budget fairness, and time-to-structure), including multi-dataset aggregation and report-ready visualizations.
