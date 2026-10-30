# Changelog

## 0.2.0 — unreleased

### Joint controlled processes

- Extended composed, Merton, GARCH and Markov scenarios to multiple dependent targets.
- Added correlated shocks, shared jump events, shared hidden regimes, and state-specific correlation matrices.
- Added CCC-GARCH with exact joint simulation and explicitly labelled numerical cross-covariance.
- Added source notebook 15 and application notebook 27, including dependence-fault and repair examples.
- Preserved scalar-case outputs, seed behaviour, and replay records.


TimeGym now provides a reproducible verification and diagnosis workflow for
forecasting pipelines, with a new v2 manuscript and bounded controlled studies.

- Explicit contracts for point means, marginal quantiles, and joint samples.
- Controlled scalar and joint compositions and parameter changes, Merton jump diffusion,
  GARCH volatility, and hidden Markov regimes with optional jumps.
- Non-Gaussian conditional simulation references with explicit quantile and
  filtering approximation metadata.
- Conditional Gaussian references and reproducible multivariate scenarios.
- Parameter-aware plot labels distinguish fixed-mean white noise, linear trend
  plus noise, and random walks with or without drift, including mixed-target drift.
- Fresh-fitting pipeline adapters and training-only recursive sklearn forecasts.
- Structured diagnostics, validated report loading, and allowlisted model replay.
- A command-line alignment-fault demo, report inspection, and replay.
- Retained forecast matrices with observation dots and unconnected forecast
  crosses; plots reuse fitted results.
- Notebook-first source with deterministic nbdev export and drift checks.
- Univariate and multivariate uncertainty demos, fault/repair examples, and local
  real-data applications. The 11 original notebooks remain archived unchanged.
- Strict documentation and distribution checks, citation metadata, and CI jobs
  that exercise an installed wheel outside the checkout.

The 2021 API and dependency assumptions changed substantially. Follow the
[migration guide](docs/migration.md). Direct/multioutput forecasting strategies,
automatic repair discovery, and universal fault detection are not claimed.
