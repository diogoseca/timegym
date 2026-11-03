# Changelog

## 1.1.0 — 2026-11-03

- Added reversible target transforms to `timegym.target`: `LogTransform`,
  `BoxCoxTransform`, `DifferenceTransform` and `ComposedTransform`, with
  `transform_config` and `transform_from_config` for configuration and replay.
- `Pipeline(..., target_transform=None)` fits a fresh copy of the transform on each
  context and inverts the model's forecast before evaluation.
- `SklearnForecaster` adds the `direct` and `multioutput` reductions beside
  `recursive`; `timegym.models.REDUCTIONS` lists them and `Pipeline(reduction=...)`
  selects one.
- Pipeline configuration schema 3 records `target_transform`; schema 2
  configurations remain readable.
- Added tutorials 29 (target transforms) and 30 (reductions).
- Removed the command-line interface: the `timegym` console script,
  `python -m timegym`, the `timegym.cli` module and source notebook 13.
  TimeGym is used as a Python package.
- `examples/reproduce_report.py` is a standalone script: it replays one case with
  `TestReport.read(...).replay_case(...)`, prints the comparison and exits 1 when
  the case is not reproduced. The installed-wheel CI job replays the report saved
  by `tools/smoke_installed.py --report`.
- Removed `Pipeline.plot_error()`, the 1.0 alias of `Pipeline.plot_errors()`.
- Pipeline configuration schema 1 (`strategy` instead of `reduction`) is no longer
  read; `pipeline_from_config` accepts schemas 2 and 3.
- Removed `timegym.joint_processes.resolve_joint_process()`; call
  `timegym.processes.resolve_process()` with the same arguments.

## 1.0.0 — 2026-10-30

Prepared release; publication is performed only by the version-tag workflow.

- Unified package, citation and manuscript version metadata; added checked
  citation rendering and a tag-driven PyPI/GitHub release workflow.
- Added scheduled Python compatibility updates following pandas support,
  bounded runtime dependencies, and a pinned development Docker image.
- Documented module imports, built-in models, fault fixtures, and release setup.
- Excluded historical notebooks from the source archive and consolidated CI checks.
- `get_stock` now defaults to `auto_adjust=True` (adjusted OHLC); pass `False` for the
  provider's unadjusted fields.

- Renamed Pipeline’s `strategy` parameter and attribute to `reduction`. Pipeline
  configuration schema 2 uses `reduction`; saved schema-1 configurations remain
  replayable. `NaiveForecaster(strategy=...)` continues to select its baseline.

- Renamed `Pipeline.plot_error()` to `Pipeline.plot_errors()` for consistency with
  retained runs. The singular spelling remains a forwarding alias for the 1.0
  release; migrate callers before the next release.

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
migration guide. Direct/multioutput forecasting strategies,
automatic repair discovery, and universal fault detection are not claimed.
