# Pipeline and reports

Running a scenario suite and reading, saving and replaying its report:
`Pipeline`, `ForecastRun`, `TestReport` and the result types from `timegym`,
with plots in `timegym.plotting` and schema constants in `timegym.provenance` and `timegym.reporting`.

`Pipeline(model, window_length=12, reduction="recursive", *,
target_transform=None, name=None)` takes the model to test first, as one of:

- a model: anything with `fit(history)` and `predict(future_index)`, copied
  fresh before every case (through `clone_unfitted()` when it has one), even if
  it can also be called;
- a list of scikit-learn steps ending in a regressor, such as
  `Pipeline([StandardScaler(), Ridge()], window_length=24)`, which becomes one
  scikit-learn pipeline; a scikit-learn pipeline you already have is taken as
  it is;
- a class, or a function that returns a fresh model, called for every case.

A scikit-learn regressor forecasts from windows of the last `window_length`
steps; `reduction` selects one of `REDUCTIONS`, with the case horizon as the
reduction horizon. Neither changes native models. Reports and the scorecard
name a scikit-learn pipeline by its steps, such as "StandardScaler + Ridge".
`target_transform` is an instance implementing `clone_unfitted()`,
`fit_transform(history)` and `inverse_transform(forecast)`. A fresh copy is
fitted on each history before the model and inverts the forecast the model
returns, so evaluation sees the original scale.

`name` names the model once for everything a run shows:
`Pipeline(RandomForestRegressor(), window_length=12, name="Forest")` titles
the scorecard "Forest", heads `plot_forecasts()` and `plot_errors()`, labels
the run in a comparison and is `result.metadata["name"]`, while
`result.metadata["model"]` still names the class. The name is saved with the
pipeline's configuration, so a replay keeps it. Without a name, the scorecard
is titled by the model and the plots have no heading.

The older ways still work, with their old behaviour, and warn:
`.add(estimator, **parameters)` adds one step at a time and makes the same
forecasts as the list; `model_factory=make_model` is `Pipeline(make_model)`;
and a pipeline with no model tests the last value, as
`Pipeline(LastValueForecaster())` does. They will be removed in a later
release.

`.predict(history, future_index)` returns the model's Forecast for inspection.
`history` may be a DataFrame, a Series, or a list or NumPy array (labelled 0, 1,
2, ...), and `future_index` any sequence of the labels to forecast, such as
`range(120, 132)` after 120 values.
A model's `predict(future_index)` returns one value per requested step: a list
or NumPy array (one column per series when there are several), a `Forecast`, a
pandas `DataFrame` indexed by the requested dates with one column per series,
or a `Series` for a single series. A list or array has no dates, so its rows are
taken as the requested steps in order. A DataFrame keeps its own index and
column names, so wrong dates or a wrong series order still fail their checks.

`.run(scenarios=None, seeds=None, *, seed=0, instances=None, cases=None,
noise_levels=None, history_length=None, horizon=None, keep_data=False,
simulate_attainable=False, interval_level=None, metrics=None, checks=None)` forecasts
every case with a fresh model, checks each forecast and measures its errors,
and returns a `ForecastRun`. It is the one call for testing a model:

```python
from timegym import LastValueForecaster, Pipeline

result = Pipeline(LastValueForecaster()).run()
result.scorecard.plot()     # the scorecard picture
print(result.scorecard)     # the same as a table
result.assert_passed()      # raises if a forecast failed an output check
result.save("report.json")  # the report, to read back with TestReport.load
```

What to forecast goes first. With nothing, `run()` draws the scorecard's ten
kinds of series at low and high noise, 120 steps of history and 24 to
forecast, 10 times each: 200 series. Keys of `PATTERNS` such as
`["trend", "seasonal"]`, and `Pattern` objects of your own beside them, give
those rows of the scorecard; `noise_levels`, `history_length` and `horizon`
set its columns and lengths. `Scenario` objects, or anything with
`generate(seed)` and `kind`, are drawn as they were built, one row each.
Kinds of series and scenarios you built cannot be mixed in one run.

One `seed` drives the whole run: each row is drawn `instances` times (10 by
default) with seeds `seed, seed + 1, ...`, so the default equals
`seeds=list(range(10))`. `seed=None` draws a fresh seed and records it, with
the instance count, in `result.metadata`. An explicit `seeds` list is used as
given and cannot be combined with `seed` or `instances`. Run seeds override a
Scenario's default seed. Model/evaluation exceptions are recorded with
traceback evidence. Invalid scenario configuration is a caller error and is
raised rather than counted as a model fault.

The result holds the report and the scorecard. `result.report` is the
`TestReport`; `result.checks`, `result.assert_passed()`, `result.save(path)`
and `result.to_json()` read it directly. `result.scorecard` is a
`Scorecard` ([Scorecards and the scenario grid](scorecard.md)), titled by
the pipeline's name or, without one, by its model; `simulate_attainable` and
`interval_level` (0.95 by default) are its settings. A run of your own data
has no scorecard, because nobody knows the answer key of real data: reading
`result.scorecard` then raises an error that points to `result.checks` and the
errors against what happened in `result.metrics`.

Each entry of `result.cases` is an `EvaluatedCase` that answers for its case
directly:

```python
for entry in result.cases:
    entry.history          # the past the model saw, one column per series
    entry.future_index     # the dates it forecast
    entry.observed         # what happened on them
    entry.reference        # the answer key (None for your own data)
    entry.forecast         # what the model returned (None if it crashed)
    entry.metrics["mae"]   # each metric by name
    entry.diagnostics      # every finding
    entry.manifest         # the case's description: its kind, seed, ...
```

`metrics`, `diagnostics` and `manifest` are the case's entry in the report,
`result.report.cases`. `entry.case` is the whole `Case`, to hand to another
call, and `scenario_index` and `seed_index` place it in the run.
`entry.result`, the old way to the report's entry, still works and warns.

`result.metrics` puts every case's metrics in one table, a pandas DataFrame
with one row per case, in order. The case's labels come first and tell the
scenarios apart. On the scorecard's kinds of series they are the scorecard's
row and column, `pattern` and `noise` (such as `"Low noise"`), then the
`seed`, so grouping by `pattern` and `noise` gives one value per cell of the
scorecard. Scenarios you built are labelled by `scenario`, the name of their
row on the scorecard: the kind, such as `"ar1"`, then `"ar1 (2)"` for a
second scenario of that kind. Cases cut from your own data are labelled by
`kind` (the name given to `cases_from_series`) and `origin`, where the history
ends. One column per metric follows, in alphabetical order. A case that
crashed or failed an output check is not scored, so its metrics are NaN. The
table reads only the metrics, so it works whether or not the run kept its
data. The run records its scenarios' labels in
`result.metadata["scenario_labels"]`, so `report.metrics` gives the same table
on a report, also one read back from its file; a report saved before this
release has `kind` and `seed` there instead.

```python
errors = result.metrics["avoidable_mse"]                     # one value per case
result.metrics.groupby(["pattern", "noise"])["mae"].mean()   # one value per cell of the scorecard
```

By default the run drops the series, answer keys and forecasts once every
case is measured, so a large run stays small in memory: `history`,
`future_index`, `observed`, `reference`, `forecast` and `case` are then
`None`, while `metrics`, `diagnostics` and `manifest` stay, and
`result.kept_data` is `False`. The report, the scorecard (with its picture)
and `result.plot_errors()` work either way. `keep_data=True` keeps them for
`result.plot_forecasts()`, `result.calibration()` and your own
reading. This in-memory object keeps arrays separate from the portable report
schema.

`.test(...)` and `.scorecard(...)` still work, with their old behaviour, and
warn. `pipeline.test(...)` is `pipeline.run(...).report`, except that with no
scenarios it draws the five kinds of `default_scenarios()`, 3 times each.
`pipeline.scorecard(patterns, ...)` is
`pipeline.run(patterns, ..., keep_data=True).scorecard`. Both will be removed
in a later release.

`metrics=` records more metrics in every case and `checks=` adds output checks
of your own ([Metrics and diagnostics](diagnostics.md#your-own-output-checks)).
`checks=[dependence()]`, with `dependence` from `timegym.checks`, adds the check
that the sample paths of each case move together as its answer key says; the
report records its settings under `metadata["diagnostic_settings"]`. The old
arguments that asked for it, `check_dependence`, `n_calibration`, `alpha` and
`dependence_seed_offset`, still work and warn.

`cases=[...]` tests ready-made `Case` objects instead of drawing them, for
example cases cut from your own data with
[`cases_from_series`](tasks-and-forecasts.md#your-own-data). Each case gets a
fresh model, fitted on its history only, and the same output checks and
metrics as a drawn case. Cases that share a manifest `kind` share a row of the
plots, in the order given. `cases` cannot be combined with `scenarios`,
`seeds`, `seed`, `instances` or the scorecard's settings, and the report's
metadata has no seed. Pass a list, even of one case.

`ForecastRun.from_forecasts(cases, forecasts, *, name=None, metadata=None,
metrics=None, checks=None)` checks forecasts made earlier and measures their errors,
with no model: `forecasts[i]` belongs to `cases[i]` and can take any form a
model's `predict` returns. `None` marks a case with no recorded forecast, which
fails "Runs without errors". The result is a `ForecastRun`, so its report and
plots work as usual. `name` says what made the forecasts, as
`Pipeline(..., name=...)` does: it is `run.metadata["name"]`, heads the plots
and labels the run in a comparison. `metadata` is added to the run's
metadata, for example `{"made_on": "2024-05-01"}`; naming the forecasts there
with `{"model": ...}` still works and warns. With the rain `cases` from
[Your own data](tasks-and-forecasts.md#your-own-data):

```python
import numpy as np
from timegym import ForecastRun

recorded = [np.full(7, case.task.history.iloc[-1, 0]) for case in cases]  # made elsewhere
run = ForecastRun.from_forecasts(cases, recorded, name="repeat last")
run.report.assert_passed()
```

`run.plot_forecasts(*, show_quantiles=False, figsize=None, history=None)` needs
a run made with `keep_data=True`. It draws
scenario/target rows and seed columns in the scorecard's look: gray dots are
observations, the black line is the answer key, blue crosses are the forecast,
and one key above the panels names them. `history` is how many past steps each
panel shows: four times the horizon by default, `"all"` for the whole history.
Optional shading shows marginal intervals. Above each panel, the case's 0-100
score sits in its colored band (as on the scorecard), next to its error against
the answer key and its error against what happened.
`run.plot_errors(metric="avoidable_mse", *, bins=None, missing="raise",
figsize=None)` shows one value of the metric per whole case, with a black line at each row's
median; `missing="annotate"` explicitly permits unavailable values and labels
their counts and reasons. `metric="mean_reference_mse"`, the old name of the
default, still works and warns. `run.titles` optionally names each scenario position in plain
words for both plots; a run of kinds of series sets them to the kind and noise
level, so `pipeline.run(["trend"], keep_data=True).plot_forecasts()` zooms
into those rows of the scorecard. Neither method refits models. Matplotlib is installed with
TimeGym and imported only when a plot is drawn.

`.to_config()` describes the configured model and `.from_config(config)` can
restore allowlisted built-in models with a bundled target transform or none.
Factories, custom models, sklearn steps and other target transforms are
explicitly marked non-replayable. A JSON report records model configuration,
source fingerprint, dependency versions and diagnostic settings. Use
`report.replay_case(case_index=0)` or `python examples/reproduce_report.py report.json`
to reproduce a supported built-in case. Replay draws the case again from its
manifest, so it refuses, saying why, a case that was not drawn from a TimeGym
scenario, such as one cut from your own data, and a report with no pipeline
configuration, such as one made by `ForecastRun.from_forecasts`. Keep your data
and the code that made the forecasts to reproduce those. The script prints the comparison and exits
with status 1 when the case is not reproduced.
Replay accepts a `TestReport`; load dictionaries through `TestReport.from_dict`
first. The `metadata["replay"]` result contains `reproduced` (metrics, diagnostics,
and manifest agree) and `same_environment` (source, dependencies, and Python
version agree). Diagnostic comparison excludes traceback fields, including those
in nested exception evidence; `traceback_match` records their exact agreement
separately. A changed exception type, message, or other evidence still prevents
reproduction. No arbitrary module import is performed
from a report. Reproduction on different numerical-library versions may require
tolerance rather than exact floating-point equality.

`DiagnosticStatus` defines `PASS`, `FAIL`, `INCONCLUSIVE`, `UNSUPPORTED`, and
`ERROR`. Only `FAIL` and `ERROR` count as failures (`DiagnosticResult.failed`).
`DiagnosticCategory` defines `correctness` (forecast contracts), `evaluation`
(evaluator inputs), `capability` (descriptive metrics), `uncertainty` (intervals),
`distribution` (joint-law checks), `metamorphic` (declared transformations), and
`execution` (model execution). A check keeps its category across outcomes;
`reference_covariance` is always `distribution`, `shift_equivariance` always
`metamorphic`, and `metric_<name>` always `capability`.
Both enums and `DiagnosticResult` and `CaseResult` are importable from `timegym`.
Result construction validates these fields, nonempty names, dictionary evidence,
nonnegative integer or absent seeds, and finite numeric metrics. Serialization
canonicalizes NumPy scalars and builtin subclasses and rejects nonfinite values
in evidence and metadata as well as metrics; it also revalidates mutable results.
`report.save(path)` writes diagnostics, measurements and manifests to a JSON
file, and `report.to_json()` returns the same JSON as text. `to_json(path)`,
the old way to write the file, still does and warns.
`TestReport.load(path)`, `TestReport.from_json(text)`, and
`TestReport.from_dict(payload)` load reports without executing saved model
descriptions; `TestReport.read(path)`, the old name of `load`, still works and
warns.
They reject unknown schemas/statuses/categories, nonfinite values, duplicate JSON keys,
malformed records, and stored summary counts that disagree with the diagnostics.
`from_json` always expects JSON text; use `load` for a file. Evidence and metadata
are copied so later edits to the input dictionary cannot change the loaded report.
`report.checks` counts, per output check in plain words, the cases that passed
and failed: one `{"check", "title", "passed", "failed"}` dictionary per check
that ran (`runs`, `dates`, `series`, `shape`, `finite`, `quantiles`, and
`dependence` when `checks=[dependence()]` asked for it), then one per check
asked for with `checks=`, such as `within(0, None)`
([your own output checks](diagnostics.md#your-own-output-checks)).
Metrics are not checks: they have no pass mark.
`report.metrics` is every case's metrics in one table, one row per case with
its labels first, as `result.metrics` above.
`report.assert_passed()` raises for detected FAIL/ERROR outcomes. Its message
names each failed check first, for example `Forecasts the requested dates:
failed in 4 of 4 cases`, then lists the underlying diagnostics. Callers must
inspect unsupported/inconclusive checks to decide whether their CI coverage is
adequate. A pass is always scoped to a check.

## Report schema

`timegym.reporting.REPORT_SCHEMA_VERSION = "2.0"` identifies the report envelope.
`timegym.provenance.PIPELINE_CONFIG_SCHEMA_VERSION = 3` identifies current
pipeline configuration writes, which add `target_transform` (`null` or a
description of the transform). Schema 2 configurations, written without that
field, remain readable; schema 1 configurations, which used `strategy`, are
rejected. A named pipeline's configuration also holds its `name`; one without
a name writes none. Scenario manifests use integer version 1,
named by `timegym.scenarios.MANIFEST_SCHEMA_VERSION`. These are independent
schemas: the report version is a string, while configuration and
manifest versions are integers.

## Signatures

::: timegym.Pipeline

::: timegym.ForecastRun

::: timegym.EvaluatedCase

::: timegym.TestReport

::: timegym.CaseResult

::: timegym.DiagnosticResult

::: timegym.DiagnosticStatus

::: timegym.DiagnosticCategory

::: timegym.plotting.plot_forecasts

::: timegym.plotting.plot_errors
