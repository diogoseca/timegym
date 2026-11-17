# Changelog

## 1.10.1 — 2026-11-17

- Added ways to score your own data. `timegym.cases_from_series(series,
  origins, horizon, *, name="series", manifest=None)` cuts a series, or a
  table of series, into cases that end their history at different times: an
  origin is a number of rows, a share of the rows, or an index label. Each
  case keeps the rows after its origin as what happened, has no answer key,
  and records in its manifest where its history ends and a fingerprint of the
  whole series. `Pipeline.test` and `Pipeline.run` take these, or any list of
  `Case` objects, as `cases=[...]`, with a fresh model for every case.
  `ForecastRun.from_forecasts(cases, forecasts)` checks and scores forecasts
  made earlier, without a model, and gives a run with the usual report and
  plots.
- Changed `evaluate_forecast` to accept a case whose manifest has no `kind` or
  `seed`, as your own cases may: it is labelled `"case"` with no seed. A
  manifest written by a TimeGym scenario must still have both.
- Changed replay to say why it cannot replay a case that was not drawn from a
  TimeGym scenario, or a report with no pipeline configuration, such as one
  from `ForecastRun.from_forecasts`.
- Changed the forecast plots to name a case with no seed by where its history
  ends, and the error messages of `plot_errors` to say `seed 2` rather than
  `seed=2`.
- Added a section on your own data to the scope page, a recipe on the data
  preparation page for putting readings taken at irregular times on a regular
  grid without looking ahead, and a page, Extend TimeGym, that lists every
  place to plug in your own code and data. Tutorial 07 now builds its cases
  with `cases_from_series` and scores them with `pipeline.run(cases=...)`.
- Added scores for forecasts of yes/no outcomes, such as rain or no rain:
  `timegym.metrics.brier(observed, predicted)`, the mean squared gap between
  the forecast chance and what happened (0 or 1), and `log_loss(observed,
  predicted, *, eps=None)`, which punishes a confident wrong forecast hard.
  A forecast of exactly 0 or 1 that turns out wrong makes the log loss
  infinite, so `log_loss` raises `UndefinedMetricError` naming its position
  unless you pass `eps`, such as `eps=1e-6`, to clip forecasts first. Both
  refuse observations other than 0 and 1 and chances outside [0, 1].
- Added four scores to `timegym.metrics`: `rmse`, the root mean squared error;
  `rmsse(observed, predicted, context, seasonality=1)`, the root mean squared
  error divided by that of repeating the value one season back in the history;
  `interval_score(observed, quantiles, levels)`, the width of the widest
  symmetric interval plus a penalty for each value outside it; and
  `weighted_interval_score(observed, quantiles, levels)`, the mean of interval
  scores over every symmetric pair of levels and the median, which needs 0.5
  among the levels. MAPE and sMAPE are left out on purpose: they blow up near
  zero and treat over- and under-forecasts unevenly.
- Added `scores=` to `evaluate_forecast`, `Pipeline.test`, `Pipeline.run` and
  `ForecastRun.from_forecasts`, to record scores beyond the usual ones. Give a
  list of names from `timegym.metrics` and functions of your own, or a
  dictionary that names each, for example to pass settings with
  `functools.partial`. A function says what it needs by its argument names:
  `observed`, `predicted`, `quantiles`, `levels`, `samples` and `context`.
  Each value lands in `case.metrics` under its name. A request that cannot
  work, such as an unknown name, a name every report records already, or an
  argument TimeGym cannot give, is refused before anything runs. For each
  case, a score whose input the forecast lacks gives a `metric_<name>`
  diagnostic marked `UNSUPPORTED`, and one that raises gives `ERROR` with the
  exception named. With no `scores=`, reports are unchanged.
- Changed replay to compute again the requested scores from `timegym.metrics`,
  with their settings, which the run records under `metadata["scores"]`. Your
  own functions cannot be rebuilt from a saved report, so replay leaves them
  out of the comparison and lists them under `"not_recomputed"`.
- Added the new scores and `scores=` to the scores page and the Extend TimeGym
  page, and a section to the scores notebook that scores two forecasts of rain
  or no rain with the Brier score and the log loss.

- Added a confidence interval to every scorecard score. `card.table` has new
  columns `score_low` and `score_high`, and `settled`, True when both ends fall
  in the score's band; `card.unsettled` lists the rows that are not settled.
  With other series of the same kind a score would come out a little different,
  and the interval shows by how much. `interval_level`, a new argument of
  `Pipeline.scorecard`, sets its confidence level: 0.95 by default, and for
  example 0.99 for a wider, more cautious interval. The card keeps it in the
  new field `Scorecard.interval_level`. With the default ten series the
  interval is wide; raise `instances` to tighten it. It reaches further below
  the score than above it, because a few series usually carry most of the
  error and a small card more often flatters a model than it wrongs it. It
  only knows the series the card drew, not kinds of series the card did not
  draw. For the statistics: each series is left out in turn and the rest are
  scored, and the spread of those scores on the log-odds scale, times a
  Student t quantile, gives how far the interval reaches above the score;
  below it reaches 1 + 1/√n times as far, with n the number of series. Nothing
  is drawn at random. Scores and verdicts are unchanged.
- The printed scorecard marks a verdict that is not settled with a question
  mark (`56 Struggles?`) and ends with how many verdicts are not settled and
  how many instances to try. The image draws each interval as a short line
  beside its score and gives an unsettled verdict a question mark and a paler
  badge.
- Added what a fitted model could score to every scorecard row. `card.table`
  has a new column `attainable`, after `settled`: the score that the pattern's
  own model family, fitted to each history, reaches on 200 fresh series of the
  row. 100 minus it is what the row's own model family loses to estimating its
  parameters from the history; the distance from a model's score up to it is
  what that model loses beyond that. It is itself an estimate, good to about a
  point, and a model can land above it on a given card. The image marks it
  with a small triangle above each score's line, also when a cell has no
  interval. Scores, intervals and verdicts are unchanged.
- Each `Pattern` has a new field `fitted`, a factory for an estimator of the
  generator's own model family (a sample mean, a least-squares line, a
  sinusoid of known period, a mean after one estimated break, or AR(1),
  AR(2) and random-walk models with measurement noise fitted by maximum
  likelihood). It defaults to `None`, so existing six-argument `Pattern(...)`
  calls still work.
- The default grid's attainable scores are stored. Off the default grid
  (another noise level, history length or horizon) `attainable` is NaN unless
  you pass `scorecard(attainable=True)`, which simulates the missing values
  once per session: about 3 to 5 seconds for each autoregressive or
  random-walk cell and well under a second for the others.
- Added `Scorecard.save_baseline(path)` and `Scorecard.compare(baseline)`,
  to compare a new version of a pipeline with the version you trusted on the
  same series. `save_baseline` writes a small JSON file with each series'
  avoidable error (about 17 kB for the default grid at 30 instances), and
  `compare` accepts that file or another `Scorecard` and returns a
  `ScorecardComparison`. Each cell is "unchanged", "changed, no worse" or
  "changed and worse": worse when, on the same series, the new version is
  further from the answer key by more than chance would explain (a paired
  sign-flip test with Holm's correction across the changed cells), or when a
  series scored in the baseline now crashes or fails an output check.
  `assert_no_regression()` raises on the worse cells. A different
  grid, seed or number of instances raises `ValueError`, and fewer than 20
  instances warn; use 30. The test does not say that the new version is
  better, and a regression too small for the number of series passes.
- Added `timegym.diagnostics.rank_alarm(observed, calibration, *, alpha=0.05)`,
  the one function that now computes the p-value of TimeGym's statistical
  checks: it asks whether a number is unusually large next to numbers you know
  are fine. Its `RankAlarm` result holds the p-value
  `(1 + number of calibration values >= observed) / (n + 1)`, the status, and
  the threshold the observed value must exceed to fail.
  `covariance_dependence_check` now calls it. Its statuses, messages and
  p-values are unchanged. Its `critical_value` now always agrees with the
  p-value: for some uncommon pairs of `n_calibration` and `alpha`, where
  `(n_calibration + 1) * (1 - alpha)` rounds to just above a whole number
  (999 calibration ensembles at `alpha=0.059`, or 99 at `alpha=0.41`), it
  used to be the next calibration value up. At `alpha` 0.001, 0.005, 0.01,
  0.025, 0.05, 0.1 and 0.2 it is unchanged for every `n_calibration` up to
  200,000.
- Added a documentation page,
  [What TimeGym tests, and what it does not](docs/concepts/scope.md).

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
