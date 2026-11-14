# Changelog

## Unreleased

- Added an interval to every scorecard score. `card.table` has new columns
  `score_low` and `score_high`, a 95% percentile bootstrap interval over the
  cell's series, and `settled`, True when both ends fall in the score's band;
  `card.unsettled` lists the rows that are not settled. With the default ten
  series the interval is rough and tends to be too narrow; raise `instances`
  to tighten it. Scores and verdicts are unchanged.
- The printed scorecard marks a verdict that is not settled with a question
  mark (`56 Struggles?`) and ends with how many verdicts are not settled and
  how many instances to try. The image draws each interval as a short line
  beside its score and gives an unsettled verdict a question mark and a paler
  badge.

## 1.4.1 — 2026-11-14

- Deprecated these names. Each still works as in 1.4.0, warns with a
  `DeprecationWarning` that says what to use instead, and leaves in TimeGym
  2.0; the documentation no longer lists them.
  - `timegym.smoothing`: `simple_moving_average`, `exponential_moving_average`;
    `timegym.features`: `add_lagged_target`, `cyclical_encoding`,
    `add_time_features`, `add_tsfel_features`; `timegym.target`:
    `multioutput_target`, `onestep_target`, `step_ahead_target`,
    `fractional_weights`, `fractional_differencing`, and noninteger orders of
    `differencing` with its `fractional_coefficients` argument. The package has
    no replacement: source notebooks 03, 04 and 05 keep the recipes as worked
    examples, and `differencing` with whole orders stays.
  - `timegym.data.real.get_stock`, with its API page. The package has no
    replacement; source notebook 02 keeps it as a worked example. The
    `features` and `market-data` extras stay until 2.0.
  - `Pipeline.plot_tests(...)`: use `pipeline.run(...).plot_forecasts()`.
    `Pipeline.plot_errors(...)`: use `pipeline.run(...).plot_errors()`, with
    `instances=20` for the same 20 draws.
  - `timegym.processes.generate_process` and
    `timegym.joint_processes.generate_joint_process`: use
    `Scenario(kind, context_length, horizon, parameters, n_targets, seed).generate()`.
    `timegym.processes.process_label`: use `timegym.scenarios.scenario_label`.
  - `timegym.metrics.empirical_crps`: use `crps`.
  - `timegym.scorecard.scorecard(pipeline, ...)`: use `pipeline.scorecard(...)`.
  - `timegym.pipeline.replay_case(report, i)`: use `report.replay_case(i)`.
- `default_scenarios` is now defined in `timegym.scenarios`. The old import,
  `from timegym.data.synthetic import default_scenarios`, keeps working until
  2.0.
- `Scenario` checks `correlation` with the same check as the composed and
  financial processes. Error messages for an invalid correlation are worded
  differently, the tolerances for symmetry, unit diagonal and positive
  semidefiniteness are slightly stricter, and a matrix that is symmetric only
  to rounding error is made exactly symmetric before drawing. Scalar
  correlations draw the same series as before.
- Run and report metadata no longer repeat the NumPy and pandas versions as
  top-level `numpy` and `pandas` keys; read them from
  `metadata["dependencies"]`, as before.
- The API reference no longer documents nine helpers that pipelines and
  reports use internally: `pipeline_config`, `pipeline_from_config`,
  `runtime_provenance`, `diagnostic_settings`, `transform_config`,
  `transform_from_config`, `fresh_model`, `is_sklearn_regressor` and
  `exception_evidence`. They still import without a warning.
- Removed the empty `plots` and `sklearn` extras. They have installed nothing
  since 1.2.0, when Matplotlib and scikit-learn became core dependencies. An
  install command that still names one, such as `pip install "timegym[plots]"`,
  installs the same packages as before and pip warns that the extra is unknown;
  drop the extra from the command.

## 1.4.0 — 2026-11-12

- Added `LastValueForecaster`, the last-value baseline under its own name: the
  same forecasts as `NaiveForecaster("last")` with nothing to set, so a
  scorecard reads "LastValueForecaster: 51/100". It replays from a report like
  the other built-in models, and the README, the quick start and the example
  scripts now score it. The hand-written last-value example in the README and
  the custom-model tutorial is now called `RepeatLast`.
- The tutorials moved out of `notebooks/` into `tutorials/`, numbered 01–12 in
  reading order, with "Write and test a custom model" second. The 15 source
  notebooks that generate the package are in `source/`, with the 2021 archive
  in `source/archive/`, and the README and site images in `docs/images/`. The
  old tutorial addresses on the documentation site redirect to the new ones,
  and the README's images now show on PyPI too.

## 1.3.0 — 2026-11-11

- A custom model's `predict` may return a plain list or NumPy array: one value
  per requested step for one series, or one row per step and one column per
  series. TimeGym labels the rows with `future_index` and the columns with the
  context's series. `Forecast(future_index, values)` takes one value per step
  for a single series, and `target_names` now defaults to `None`, meaning the
  task's series in order. The last-value forecaster needs no NumPy and no
  target names: `self.last = context.iloc[-1]` in `fit`, then
  `Forecast(future_index, [self.last] * len(future_index))` in `predict`.
- `Pipeline.predict(context, future_index)` also takes a list or NumPy array as
  the past (labelled 0, 1, 2, ...) and any sequence of labels as
  `future_index`.
- Forecast and error plots use the scorecard's look: gray dots for
  observations, a black answer key, blue forecast crosses and one key above the
  panels in plain words. Each forecast panel shows the case's score as a colored
  badge, and only the last four horizons of history unless
  `plot_forecasts(history="all")` or a number of steps is given. `plot_errors`
  marks each row's median.
- The scorecard says "1 kind of series" for a one-row card, and the compact
  image puts each noise level's sd on its own line when three or more levels
  would run their headers together.

## 1.2.0 — 2026-11-10

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
- Forecast plots title each panel with the case's 0-100 score, its error
  against the answer key and its error against what happened, instead of
  "observed MSE" and "mean-reference MSE"; `plot_errors` labels its axis
  "Error against the answer key". `ForecastRun.titles` names rows in plain
  words, and a scorecard's run uses its pattern and noise titles.
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

## 1.0.0 — 2026-10-30

Not published to PyPI; 1.1.0 is the first PyPI release. It replaced the 2021 prototype: forecast contracts for point means, marginal quantiles and joint samples; known conditional references for Gaussian, composed, jump, GARCH and hidden-regime scenarios, for one or several dependent targets; fresh-fitting pipelines, structured diagnostics, validated report loading and allowlisted replay; notebook-first source; the tutorials, the manuscript and two controlled studies; tag-driven PyPI releases and citation metadata. Pipeline's `strategy` parameter became `reduction`, and `get_stock` defaults to `auto_adjust=True`. The 2021 API changed substantially.
