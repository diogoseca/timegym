# Changelog

## Unreleased

- Made `Pipeline.run` the one call for testing a model. Its result now
  carries the scorecard: `result = pipeline.run()`, then
  `result.scorecard.plot()` or `print(result.scorecard)`, with
  `result.assert_passed()`, `result.to_json(path)` and `result.checks` read
  straight from its report. With no arguments `run()` draws the scorecard's
  ten kinds of series at low and high noise, 10 times each (200 series), and
  it takes the scorecard's settings: `noise_levels`, `history_length`,
  `horizon`, `title`, `attainable` and `interval_level`. Scenarios you built
  give one row each. A run of your own data (`cases=`) has no scorecard,
  since nobody knows its answer key, and reading `result.scorecard` says so
  and points to `result.checks` and the errors in `result.report`.
- `run()` now drops the series and forecasts once each one is scored, so a
  large run stays small in memory. The report, the scorecard with its
  picture, `save_baseline`, `compare` and `plot_errors()` work as before;
  `keep_data=True` keeps every series and forecast for `plot_forecasts()`,
  `calibration` and your own reading, and `result.kept_data` says which.
  Code that read `result.cases[i].case` or `.forecast` needs `keep_data=True`.
  With nothing to forecast named, `run()` used to draw the five kinds of
  `default_scenarios()` 3 times each; it now draws the scorecard's series, and
  with scenarios named it draws each 10 times instead of 3.
- `Pipeline.test(...)` and `Pipeline.scorecard(...)` still work as before and
  now warn: use `pipeline.run(...)` and `pipeline.run(...).scorecard`. They
  will be removed in a later release. `timegym.scorecard.scorecard` warns the
  same way, and the scorecard's closing line now says
  `Run pipeline.run(instances=30) to narrow them.`
- Moved the docs, tutorials, source notebooks, examples, the paper's sentence
  on the scorecard call and the installed-package check to the new call.

- One word for the past a forecaster sees: history. Write `fit(self, history)`
  in your models, `Scenario(kind, history_length=120, horizon=12)`,
  `task.history`, `pipeline.predict(history, future_index)`,
  `mase(observed, predicted, history)`, `rmsse(observed, predicted, history)`,
  `check_shift_equivariance(pipeline, history, future_index)` and
  `history_length=` in `run()`, `scenario_grid` and `default_scenarios`. A
  score or a check of your own asks for the past by naming its argument
  `history`. The built-in models and target transforms name their argument
  `history` too, `NaiveForecaster` keeps what it fitted as `history_`, and
  error messages say history where they said context.
- The old names still work and warn, and will be removed in a later release:
  `context_length=` wherever it was taken (`Scenario`, `Pipeline.scorecard`,
  `scenario_grid`, `default_scenarios`, `generate_process`,
  `generate_joint_process`, `resolve_process` and
  `timegym.scorecard.scorecard`), `scenario.context_length`,
  `ForecastTask(context=...)`, `task.context`, `context=` in
  `Pipeline.predict`, `mase`, `rmsse` and `check_shift_equivariance`,
  `NaiveForecaster.context_`, a score or a check whose argument is named
  `context` (it warns once per run), and a `build` of your own whose scenario
  has `context_length` instead of `history_length`. A model of your own
  written `fit(self, context)` needs no change, since TimeGym passes the
  history by position.
- New cases, reports and scorecard baselines record `history_length` where
  they recorded `context_length`. Reports and baselines saved by 1.13.1 and
  earlier still load, compare with new ones and replay. Code that reads
  `case.manifest["context_length"]` from a new case needs
  `case.manifest["history_length"]`: that is the one change that does not
  warn. Scorecard baselines are now format version 2, so TimeGym 1.13.1
  refuses a new baseline and asks to be upgraded instead of failing on the
  missing key; the report format is unchanged.
- Moved the docs, the glossary (History, where it said Context), the README,
  tutorials, examples, source notebooks, the installed-package check and the
  paper to the word history wherever they meant the past a forecaster sees.

- One class for each built-in baseline, named for the forecast it makes:
  `LastValueForecaster()`, `SeasonalLastValueForecaster(period)` (seasonal
  naive: each step gets the last value seen at the same point of the cycle),
  `MeanForecaster()`, `SeasonalMeanForecaster(period)` and `DriftForecaster()`,
  all from `timegym`. Only the two seasonal ones take a `period`. Reports and
  scorecards name each by its class, and each replays from a report.
- `NaiveForecaster(strategy, period)` still works, makes the same forecasts as
  before and now warns, naming the class that makes the same forecasts for the
  strategy given, such as `SeasonalLastValueForecaster(period=12)`; it will be
  removed in a later release. A pipeline warns once, not once per case. Reports
  and scorecard baselines that name `NaiveForecaster` still load, compare and
  replay, and replaying one does not warn. `LastValueForecaster` is still a
  `NaiveForecaster`, as before.
- Moved the docs, tutorials, examples, source notebooks and the
  installed-package check to the new classes.

- `Pipeline` takes the model to test first, in one of four forms: a model
  (anything with `fit` and `predict`, copied fresh for every case, even if it
  can also be called); a list of scikit-learn steps ending in a regressor,
  such as `Pipeline([StandardScaler(), Ridge()], window_length=24)`, which
  becomes one scikit-learn pipeline (a scikit-learn pipeline you already have
  is taken as it is); a class; or a function that returns a fresh model, now
  written `Pipeline(make_model)` and called for every case, as
  `Pattern(attainable_by=...)` calls it. A list that does not end in a
  regressor is refused when the pipeline is made.
- Reports, comparisons and the scorecard title name a scikit-learn pipeline by
  its steps, such as "StandardScaler + Ridge". It is described in the report
  and not replayed from it, like other models of your own.
- `.add()`, `model_factory=` and a pipeline with no model still work as before
  and now warn: use the list of steps, which makes the same forecasts,
  `Pipeline(make_model)`, and `Pipeline(LastValueForecaster())`, which makes
  the forecasts a pipeline with no model makes. `.add()` warns once per
  pipeline and a pipeline with no model warns when it is first used. They will
  be removed in a later release.
- Moved the docs, tutorials and source notebooks to the list of steps and to
  a model in every pipeline. The scripts behind the paper's scorecard figure
  and the follow-up study keep their `.add()` chain, which the figure's title
  and the study's registered protocol name.

- Added `timegym.diagnostics.calibration(run, *, bins=10, cluster=None,
  level=0.95, seed=0)`, which asks whether forecasts of a yes/no outcome come
  true as often as they say: of the days given a 70% chance of rain, did it
  rain on about 70%? It reads a run's point forecasts, or the mean of the
  sample paths, or arrays with `calibration(observed, predicted)`, and returns
  a `Calibration`: a `table` of bins of equal width with the number of
  forecasts, their mean and the share that happened; the calibration `slope`
  (1 is calibrated, below 1 too confident, above 1 too timid) with a bootstrap
  `slope_interval` that resamples whole cases, or whole clusters with
  `cluster=`; the calibration `intercept` (0 is right on average); and
  `plot()`, a reliability diagram drawn by the new
  `timegym.plotting.plot_calibration`. Forecasts of exactly 0 or 1 are counted
  in `certain`, since the slope reads them as 0.000001 and 0.999999. Nothing
  else changes.
- Added a section on calibration to the scores page, a sentence on the scope
  page and in the README, and a section to the scores notebook with twenty
  weather stations, where a share learned from few cycles says 0% or 100% too
  often and adding one rainy and one dry step to each count fixes it.
- Added a sentence to the Pipeline page saying that `scorecard(patterns=...)`
  takes your own `Pattern` objects beside the keys of `PATTERNS`.

## 1.13.1 — 2026-11-24

- Added your own kinds of series to the scorecard. `Pattern` is now exported
  from `timegym`, and `Pipeline.scorecard(patterns=...)` and `scenario_grid`
  take your own `Pattern` objects beside the keys of `PATTERNS`: a title, the
  sentences saying what is predictable and what is not, and
  `simulate(rng, history, horizon, noise)`, a function that draws one series
  and returns its `values` with the true `mean` and `variance` of each step to
  forecast (and optionally a `covariance`). TimeGym cuts the history from the
  future and builds the case and its answer key. A kind of series that needs
  more can give `build(noise, context_length, horizon)` instead, returning an
  object that generates whole cases. Before anything runs, the grid draws one
  case per cell and says what is missing or has the wrong shape, or that a key
  belongs to a built-in pattern. With `attainable_by=` and `attainable=True`,
  the row's attainable score is simulated as for the built-in rows; otherwise
  it is NaN. The table, `save_baseline` and `compare` know the row by its key.
  `GridCell` has a new field `definition`, the `Pattern` the row comes from.
  Built-in rows are unchanged.
- Renamed `Pattern.fitted` to `Pattern.attainable_by`, which takes the row's
  own model itself (copied fresh for each series) as well as a class or
  function that returns one. The built-in rows hold their models, so
  `Pipeline(PATTERNS["trend"].attainable_by)` scores one. `fitted=` and
  `Pattern.fitted()` still work and warn that the name changed.
- Added `NaiveForecaster("seasonal_mean", period=...)`: each step forecast with
  the mean of all past values at the same point of the cycle, which for yes/no
  values is the share that happened there.
- Added a section on your own kinds of series to the scorecard page, with a
  yes/no row (rain or no rain, with a chance that follows a cycle) where the
  score becomes the share of the expected Brier score that no forecaster could
  avoid, a row on the Extend TimeGym page, and a section to the scorecard
  notebook.

## 1.12.1 — 2026-11-23

- Added an Examples section to the documentation, placed right after the home
  page: forecasting jobs on real public data, from the first model to a
  decision, with TimeGym in the loop. The first twenty-two are weekly shop
  sales turned into stock orders, spare parts where the lowest average error
  goes to a forecast of zero, a Christmas order that five Decembers cannot
  settle and generated shops can, a call center's forecast made again at ten
  o'clock, train station riders forecast with the real weather to see what a
  weather feed could add, daily births forecast with different amounts of
  history, a day-ahead electricity forecast improved by asking which days went
  wrong, a range for a solar farm's whole day that half-hour ranges cannot
  give, a 90% range for a river's flow, rain tomorrow scored against the
  weather service's own forecast, Sydney's temperature forecast with other
  cities' help, London's air scored only where it was measured, a flu season
  forecast from the calendar of earlier winters, which works until the peak
  comes late, the sunspot cycle split by years ahead, an El Niño forecast whose
  reach depends on the month it is made, a jump in a page's traffic where the
  ordinary backtest picks the wrong forecaster, a site's capacity booked four
  weeks ahead and tested at that lead time, a Bitcoin backtest whose edge was
  one input reaching into the future, a range for the S&P 500's next week that
  holds in crashes, a random forest that cannot follow growing airline traffic,
  the winner of a 50-setting backtest that comes tenth the next year, and a
  pretrained model whose bad backtest year comes down to five products. The
  notebooks are in `examples/`, with a small sample of each dataset committed
  in `examples/data` beside its source, license and credit, so they run
  offline. The pretrained model's forecasts were made once and are saved in
  `examples/forecasts`. `examples/example_data.py` loads the samples and, with
  `load(name, full=True)`, downloads and checks the whole dataset. `pip install
  -e '.[examples]'` from a clone installs what the notebooks need.
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
- Added `timegym.compare_reports(candidate, baseline, *, metric, by=None,
  cluster=None, alpha=0.05, higher_is_better=False, seed=0)`, to say whether
  a forecaster is better or worse than another on the same cases, or shows no
  clear difference. It pairs the two reports case by case, refuses reports
  whose case manifests differ, and returns a `ReportComparison` with `table`,
  `better`, `worse`, `too_few`, `assert_no_regression()` and a printed
  summary. `by` names a manifest key and gives one row per group, with Holm's
  correction across groups. `cluster` names a manifest key whose cases move
  together, such as the store a case was cut from; those cases count as one
  piece of evidence. A row where too few cases or clusters differ for any
  difference to reach `alpha` says "too few to tell" and fails
  `assert_no_regression()`. A case the baseline scored that crashes or fails
  an output check in the candidate makes its row worse. `candidate` may be a
  list or a dictionary of reports: each candidate's p-value is then adjusted
  for every candidate and group tried, so the best of many does not beat the
  baseline by luck. The test is a two-sided paired sign-flip test.
- Changed where the sign flips and Holm's correction behind
  `Scorecard.compare` live: `timegym.diagnostics`, shared with
  `compare_reports`. `Scorecard.compare` keeps its one-sided question, and its
  results are unchanged.
- Added a section on comparing two forecasters to the scores page, a sentence
  pointing to it on the scope page, and a section to the scores notebook that
  compares two forecasters of store sales by region and measures how often
  cases that move together, and the best of many variants, give a false
  verdict.
- Added `checks=` to `evaluate_forecast`, `Pipeline.test`, `Pipeline.run` and
  `ForecastRun.from_forecasts`, for output checks of your own, and
  `timegym.checks.within(lower=None, upper=None)`, a ready-made range check:
  `within(0, None)` for rainfall or sales, `within(0, 1)` for a chance. It
  reads the point forecast, the quantiles and the samples; `None` leaves a
  side open, and a dictionary from series name to bound gives each series its
  own range. Your own check is a function that asks for its inputs by
  argument name, as a score does, plus `target_names`, and returns `True` or a
  message saying what is wrong; its title is the first line of its docstring.
  A failed check is a `FAIL` named `check_<name>`, is counted in
  `report.checks` after the built-in checks and makes `assert_passed` raise.
  Each finding keeps the check's title, so a saved report loads without your
  code; the run records the checks under `metadata["checks"]`. Replay runs
  `within` again and lists your own functions under `"checks_not_rerun"`.
  Without `checks=`, reports are unchanged.
- Added a section on your own output checks to the scores page and the Extend
  TimeGym page, a catalogue entry for `check_<name>`, a sentence in the README,
  and a section to the scores notebook that checks forecasts of store sales
  for values below zero and for totals that don't add up.

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
