# Changelog

## Unreleased

- Added the scorecard: `Pipeline(model).scorecard()` runs a forecaster on ten
  kinds of series at low and high noise and returns a `Scorecard`. Its image
  gives a 0-100 score, what the model is good at and fails at, the mistakes it
  makes (for example *Values one step late* or *Wrong dates*), and for each
  kind of series what can be predicted and what cannot by design. `print(card)`
  gives the same as Markdown; `card.save(path)` writes PNG, SVG or PDF; the
  JSON report stays available as `card.report`. Source notebook 13 builds it.
- Added the scenario grid behind it: `PATTERNS`, `scenario_grid()` and
  `GridCell` from `timegym`, with `noise_levels()` in `timegym.grid`. Two noise
  levels by default; more levels fill the range between the same two extremes.
- Added the `ar2` scenario kind (stationary AR(2), irregular cycles) and an
  optional `noise` parameter for `ar1`, `ar2` and `random_walk`: measurement
  noise on the hidden process, with an exact Kalman-filter reference. Scenarios
  without `noise` keep their references and manifests.
- One seed drives a whole run: `Pipeline.test`, `run`, `plot_tests` and
  `plot_errors` take `seed=0` and `instances`, using seeds `seed, seed + 1, ...`.
  The default is unchanged (`seeds=[0, 1, 2]`). `seed=None` draws a fresh seed
  and records it in the report metadata. An explicit `seeds` list still works.
- A model's `predict` may return a pandas DataFrame indexed by the requested
  dates, or a Series for one series, as well as a `Forecast`. A plain class
  with `fit` and `predict` can be passed as an instance; each case fits a deep
  copy and the original is never fitted.
- `report.checks` counts each output check in plain words (for example
  *Forecasts the requested dates*) with the cases that passed and failed.
  `assert_passed()` names the failed checks first. The scorecard reports how
  many steps wrong dates are shifted.
- `pip install timegym` now includes Matplotlib and scikit-learn. The `plots`
  and `sklearn` extras remain as empty aliases so older install commands work.

- The API reference pages, except Market data, end with signatures and
  docstrings generated from the code by mkdocstrings; the `docs` extra adds
  `mkdocstrings` and `mkdocstrings-python`. Every public class, method and
  property now has a docstring.
- Added a capability map to the Concepts pages. It crosses the behaviors a
  forecaster needs with the four kinds of evidence, and links the scenario
  family, diagnostic and tutorial for each pair that a tutorial covers.

## 1.1.0 — 2026-11-05

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
- Renamed the supplement's paired loss fields `repair_minus_faulty_mse` and
  `repair_minus_faulty_reference_mse` to `clean_minus_faulty_mse` and
  `clean_minus_faulty_reference_mse` in `experiments/results-v2`, as a recorded
  offline migration; no value changed.

## 1.0.0 — 2026-10-30

Not published to PyPI; 1.1.0 is the first PyPI release.

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

TimeGym now provides a reproducible verification and diagnosis workflow for
forecasting pipelines, with a new manuscript and bounded controlled studies.

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

The 2021 API and dependency assumptions changed substantially.
Direct/multioutput forecasting strategies, automatic repair discovery, and
universal fault detection are not claimed.

### Joint controlled processes

- Extended composed, Merton, GARCH and Markov scenarios to multiple dependent targets.
- Added correlated shocks, shared jump events, shared hidden regimes, and state-specific correlation matrices.
- Added CCC-GARCH with exact joint simulation and explicitly labeled numerical cross-covariance.
- Added source notebook 15 and application notebook 27, including dependence-fault and repair examples.
- Preserved scalar-case outputs, seed behavior, and replay records.
