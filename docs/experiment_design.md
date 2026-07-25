# Experiment Design

This project treats the simulator as the sole source of quantitative evidence.
It does not compare against measurements from a physical 6G deployment.

## Claim–evidence mapping

| Claim tested | Evidence | Split / perturbation | Metrics |
| --- | --- | --- | --- |
| AI assistance can correct structured RSS bias | Four-method main comparison | spatially held-out in-domain samples | mean, RMSE, median, P90, maximum error |
| The correction remains useful in blocked paths | LoS/NLoS stratification | wall-intersection labels from the simulator | LoS/NLoS mean and median, NLoS P90 |
| Robustness is condition dependent | noise, blockage, dropout and domain-shift sweeps | fixed root model under independently seeded measurement realizations | mean and P90 error |
| Residual features contribute distinct information | feature-group ablation | unchanged validation/test protocol | mean, RMSE and P90 error |
| The system is practical for a classroom demonstration | warmed-up one-row latency, batch throughput and model-size measurements | CPU-only local execution | one-row ms, batch/amortized time, training time, MB |

## Split policy

Training and validation positions are sampled outside a reserved spatial region.
The spatial-holdout set is sampled inside that region, preventing dense
neighbouring fingerprints from leaking across the boundary. Trajectories are
kept separate from static training positions. Domain-shift and anchor-failure
sets are regenerated with changed propagation or availability parameters.
The reported Quick headline repeats the complete sampling, training, validation,
and spatial-holdout evaluation independently for seeds 42, 123, and 2026.
Robustness uses the seed-42 model across three seeded measurement realizations;
the exported schema records both model and measurement seed scope.

## Baseline policy

The geometric least-squares estimator is the interpretable traditional
baseline. KNN fingerprinting is a non-parametric site-specific baseline. A
compact MLP is the direct learning baseline, and Extra Trees predicts a
correction to the geometric estimate. Kalman filtering is reported only as
trajectory post-processing.

Feature ablations preserve the full Residual AI tree count/depth. The spatial
bias training ablation reuses positions and paired propagation randomness while
only disabling the spatial field. Kalman rows are evaluated on trajectories and
are exported in a separate filtering table, never ranked against static
point-localization feature variants.

## Evidence boundaries

Every exported table and figure is generated from a saved CSV. Negative
improvements are retained. No statistical significance, real-world accuracy or
centimetre-level 6G deployment claim is inferred from the simulator.
Dashboard data-preparation timing excludes Streamlit/Plotly rendering and is
labelled as a proxy rather than end-to-end browser latency.
