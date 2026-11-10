# Pipeline and reports

Running a scenario suite and reading, saving and replaying its report:
`Pipeline`, `ForecastRun`, `TestReport` and the result types from `timegym`,
with `replay_case` in `timegym.pipeline`, plots in `timegym.plotting`, and
schema constants in `timegym.provenance` and `timegym.reporting`.

`Pipeline(model=None, window_length=12, reduction="recursive", *, model_factory=None,
target_transform=None)` uses a last-value baseline by default. Native model
classes/factories or the `clone_unfitted()` protocol ensure fresh fits. Optional
sklearn steps can be added with `.add(estimator, **parameters)`; `reduction`
selects one of `REDUCTIONS` for them, with the case horizon as the reduction
horizon, and does not change native models. `target_transform` is an instance
implementing `clone_unfitted()`, `fit_transform(context)` and
`inverse_transform(forecast)`. A fresh copy is fitted on each context before the
model and inverts the forecast the model returns, so evaluation sees the
original scale.

`.predict(context, future_index)` returns the model's Forecast for inspection.
A model's `predict(future_index)` may return a `Forecast`, a pandas `DataFrame`
indexed by the requested dates with one column per series, or a `Series` for a
single series. A DataFrame keeps its own index and column names, so wrong dates
or a wrong series order still fail their checks.

`.test(scenarios=None, seeds=None, *, seed=0, instances=None,
check_dependence=False, n_calibration=99, alpha=0.05,
dependence_seed_offset=100_000)` evaluates cases and returns a TestReport. One
`seed` drives the whole test: each scenario is run `instances` times (3 by
default) with seeds `seed, seed + 1, ...`, so the default equals
`seeds=[0, 1, 2]`. `seed=None` draws a fresh seed and records it, with the
instance count, in `report.metadata`. An explicit `seeds` list is used as given
and cannot be combined with `seed` or `instances`. Test seeds override a
Scenario's default seed. Model/evaluation exceptions are recorded with
traceback evidence. Invalid scenario configuration is a caller error and is
raised rather than counted as a model fault.

`.scorecard(patterns=None, *, noise_levels=2, instances=10, seed=0,
context_length=120, horizon=24, title=None)` runs the scenario grid and returns
a `Scorecard`; see [Scorecards and the scenario grid](scorecard.md).

`.run(scenarios=None, seeds=None, *, seed=0, instances=None, ...)` accepts the
same seeding and diagnostic settings and returns a `ForecastRun`. Its `.report` is the usual `TestReport`; `.cases`
retains each `EvaluatedCase`, including the generated case, forecast (or `None`
on model failure), result, and scenario/seed positions. This in-memory inspection
object keeps arrays separate from the portable report schema.

`run.plot_forecasts(*, show_quantiles=False, figsize=None)` draws scenario/target
rows and seed columns with observation dots and unconnected forecast crosses.
Optional shading shows marginal intervals. `run.plot_errors(metric="mean_reference_mse", *,
bins=None, missing="raise", figsize=None)` shows one score per whole case;
`missing="annotate"` explicitly permits unavailable scores and labels their
counts and reasons. Each forecast panel's title gives the case's 0-100 score
(as on the scorecard), its error against the answer key and its error against
what happened. `run.titles` optionally names each scenario position in plain
words for both plots; a scorecard's run sets them to the pattern and noise
level, so `card.run.plot_forecasts()` zooms into its rows. Neither method refits models. Matplotlib is installed with
TimeGym and imported only when a plot is drawn.

`pipeline.plot_tests(...)` and `pipeline.plot_errors(...)` are convenient one-call
experiments; both take `seed` and `instances` (20 by default for `plot_errors`). Each creates a fresh run. Use `run()` when multiple views must share
exactly the same predictions, especially for stochastic models.

`.to_config()` describes the configured model and `.from_config(config)` can
restore allowlisted built-in models with a bundled target transform or none.
Factories, custom models, sklearn steps and other target transforms are
explicitly marked non-replayable. A JSON report records model configuration,
source fingerprint, dependency versions and diagnostic settings. Use
`report.replay_case(case_index=0)` or `python examples/reproduce_report.py report.json`
to reproduce a supported built-in case. The script prints the comparison and exits
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
(evaluator inputs), `capability` (descriptive scores), `uncertainty` (intervals),
`distribution` (joint-law checks), `metamorphic` (declared transformations), and
`execution` (model execution). A check keeps its category across outcomes;
`reference_covariance` is always `distribution`, `shift_equivariance` always
`metamorphic`, and `metric_<name>` always `capability`.
Both enums and `DiagnosticResult` and `CaseResult` are importable from `timegym`.
Result construction validates these fields, nonempty names, dictionary evidence,
nonnegative integer or absent seeds, and finite numeric metrics. Serialization
canonicalizes NumPy scalars and builtin subclasses and rejects nonfinite values
in evidence and metadata as well as metrics; it also revalidates mutable results.
`report.to_json(path=None)` serializes diagnostics, measurements and manifests.
`TestReport.from_dict(payload)`, `TestReport.from_json(text)`, and
`TestReport.read(path)` load reports without executing saved model descriptions.
They reject unknown schemas/statuses/categories, nonfinite values, duplicate JSON keys,
malformed records, and stored summary counts that disagree with the diagnostics.
`from_json` always expects JSON text; use `read` for a file. Evidence and metadata
are copied so later edits to the input dictionary cannot change the loaded report.
`report.checks` counts, per output check in plain words, the cases that passed
and failed: one `{"check", "title", "passed", "failed"}` dictionary per check
that ran (`runs`, `dates`, `series`, `shape`, `finite`, `quantiles`,
`dependence`). Scores are not checks: they have no pass mark.
`report.assert_passed()` raises for detected FAIL/ERROR outcomes. Its message
names each failed check first, for example `Forecasts the requested dates:
failed in 4 of 4 cases`, then lists the underlying diagnostics. Callers must
inspect unsupported/inconclusive checks to decide whether their CI coverage is
adequate. A pass is always scoped to a check.

## Report schema

`timegym.reporting.REPORT_SCHEMA_VERSION = "2.0"` identifies the report envelope.
`timegym.provenance.PIPELINE_CONFIG_SCHEMA_VERSION = 3` identifies current
pipeline configuration writes, which add `target_transform` (`null` or a
`transform_config` description). Schema 2 configurations, written without that
field, remain readable; schema 1 configurations, which used `strategy`, are
rejected. Scenario manifests use integer version 1,
named by `timegym.scenarios.MANIFEST_SCHEMA_VERSION`. These are independent
schemas: the report version is a string, while configuration and
manifest versions are integers.

## Signatures

Generated from the docstrings in the code.

::: timegym.Pipeline

::: timegym.ForecastRun

::: timegym.EvaluatedCase

::: timegym.TestReport

::: timegym.CaseResult

::: timegym.DiagnosticResult

::: timegym.DiagnosticStatus

::: timegym.DiagnosticCategory

::: timegym.pipeline.replay_case

::: timegym.provenance.pipeline_config

::: timegym.provenance.pipeline_from_config

::: timegym.provenance.runtime_provenance

::: timegym.provenance.diagnostic_settings

::: timegym.plotting.plot_forecasts

::: timegym.plotting.plot_errors
