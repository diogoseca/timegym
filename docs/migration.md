# Migrating from the 2021 prototype

TimeGym 0.2 introduces a new diagnostic core and a new v2 manuscript. It does
not promise execution compatibility with the exploratory 2021 notebooks.
The historical implementation remains available in Git history. The ten former
`dev/` notebooks and original Quick Start now live together in
[the notebook archive](../notebooks/archive/README.md), with their contents and
outputs preserved. Their APIs and environments reflect the earlier prototype.

## The supported entry point

V2 is an unreleased development artifact. Install this checkout from its root
with `pip install -e .`; the published PyPI package does not provide this v2
API. Core imports require NumPy, pandas, and SciPy. Install optional extras
from the checkout with commands such as `pip install -e '.[sklearn]'`.
Other extras are `features`, `plots`, `market-data`, and `notebooks`. The
[maintained notebook series](../notebooks/README.md) demonstrates the v2 API,
including univariate and multivariate uncertainty, integration, and diagnosis.

```python
from timegym import NaiveForecaster, Pipeline, Scenario

scenario = Scenario("random_walk", context_length=120, horizon=12, seed=7)
case = scenario.generate()
pipeline = Pipeline(NaiveForecaster(strategy="last"))
forecast = pipeline.predict(case.task.context, case.task.future_index)
assert forecast.validate(case.task) == []

report = pipeline.test([scenario], seeds=[0, 1, 2])
print(report.summary)
report.to_json("timegym-report.json")
```

A native model implements `fit(context: pandas.DataFrame)` and
`predict(future_index: pandas.Index) -> Forecast`. Only observed context and
the requested index are passed to the model. `Pipeline.add(...)` supports
scikit-learn transformers followed by a regressor when the `sklearn` extra is
installed. The focused v2 supports recursive reduction only; other reduction
strategies are not implemented. For a native model, supply
`Pipeline(model_factory=lambda: MyForecaster(...))`, a zero-argument model
class, or an instance implementing `clone_unfitted()` that returns a fresh,
unfitted instance. Scikit-learn estimators are cloned. A deep copy is not
assumed to clear fitted state: the supplied factory or clone method must
honor the fresh-instance contract so one case cannot train the next.

## Notebook development and visual inspection

The active collection lives directly under `notebooks/`: source notebooks
01–15 build all 25 package modules, while Quick Start 00 and applications 20–27
use them. This continues the original notebook-first development approach.
Selected nbdev cells replace `%%runwrite` append operations; save source edits,
then run `python tools/export_notebooks.py`. A normal notebook run does not
write modules. `python tools/export_notebooks.py --check` verifies that generated
files match their notebooks. See [contributing](contributing.md).

`Pipeline.test(...)` still returns the serializable `TestReport`. To inspect
the actual outputs and plot them without fitting the model again, use:

```python
run = pipeline.run([scenario], seeds=[0, 1, 2])
run.plot_forecasts()
run.plot_errors(metric="mean_reference_mse")
report = run.report
```

`run.cases` holds the generated cases, forecasts, and diagnostic results.
The matrix plot uses discrete blue crosses for means, gray observation dots,
thin conditional-reference curves, and the forecast origin. The convenience
methods `pipeline.plot_tests(...)` and `pipeline.plot_errors(...)` each perform
their own run; retain a `ForecastRun` when several views should share the same
fitted outputs. Some original themes, such as semi-synthetic generation and
automatic optimization, remain research directions rather than implemented APIs.

## Forecasts and references are explicit

`Forecast.point` has shape `(horizon, targets)` and denotes a predictive mean.
`Forecast.quantiles` has shape `(horizon, targets, levels)`, with explicit
strictly increasing probability levels. `Forecast.samples` has shape
`(samples, horizon, targets)` and represents joint draws. Index and target
order must match the task. Malformed forecasts can be constructed so their
errors can be diagnosed with `validate()`; they are not silently reshaped.

`Scenario.generate()` returns a `Case` containing `task`, `observed`,
`reference`, and a replayable `manifest`. The evaluator's `Reference` holds
the Gaussian conditional mean, variance, full covariance, marginal quantiles,
and joint sampling method. AR(1) and random-walk references condition on the
last observed context value. All references additionally know the generator
parameters; estimated models do not receive those parameters. These are
controlled diagnostic references with an information advantage, not attainable
accuracy guarantees on arbitrary real-world data.

The old `SyntheticSeries`, `GluonSyntheticSeries`, GluonTS recipes,
module-level generated datasets, and `get_data(include_oracle=True)` tuples
are removed. No compatibility wrapper calls an arbitrary latent component an
oracle. The supported families are `white_noise`, `trend`, `seasonal`, `ar1`,
and `random_walk`. The additional `composite`, `merton_jump_diffusion`,
`garch`, and `markov_switching` families add explicitly specified stress cases;
see [the stress-case guide](stress-cases.md). Other historical families have not
been ported automatically. Import
`default_scenarios` from `timegym.data.synthetic` for fresh specifications of
these five families; generation occurs only when explicitly requested.

Reports distinguish `PASS`, `FAIL`, `INCONCLUSIVE`, `UNSUPPORTED`, and `ERROR`.
A model without predictive samples cannot establish distributional or joint
dependence correctness. `assert_passed()` rejects detected failures and errors;
unsupported and inconclusive checks remain visible and are not evidence of
comprehensive success.

## Helper behavior corrections

- `cyclical_encoding` requires a known `period` or explicit inclusive discrete
  bounds. Use `period=7` for weekdays and `period=24` for hours. Zero bounds
  are honored. The period and phase origin are never inferred from the sample.
- `add_time_features` always emits the same calendar columns, including
  microseconds. Calendar periods are fixed. Its `time` column is elapsed Unix
  epoch days, rather than a value divided by the current sample's standard
  deviation. This prevents feature values from changing between data slices.
- `differencing(order=2)` now means two repeated first differences. Use
  `differencing(order=1, lag=12)` for a seasonal difference. Integer-valued
  floats are accepted, and order zero returns an unchanged copy. Fractional
  differencing is a truncated binomial approximation; its warmup is exactly
  `(fractional_coefficients - 1) * lag`, and internal missing values propagate.
- `add_tsfel_features` loads TSFEL only when called. At row `t`, its window
  ends at `t-lag`; the default `lag=1` excludes the current observation.
  Insufficient history raises a clear error.
- `exponential_moving_average` requires an explicit alpha and supports only
  `statistic="mean"`. An exponentially weighted median is not implemented.

Existing fitted models using the old feature values must be retrained after
these corrections. They cannot safely reuse the old preprocessing parameters.

Replay orchestration now lives with Pipeline. Import the standalone helper as
`from timegym.pipeline import replay_case`; `report.replay_case()` is unchanged.
`provenance.pipeline_from_config()` validates and returns constructor arguments;
use `Pipeline.from_config()` to reconstruct a pipeline.

Use `Pipeline(reduction="recursive")` for the sklearn reduction. The old Pipeline
`strategy` keyword and attribute are now named `reduction`; `NaiveForecaster`
continues to use `strategy` for its baseline method. New pipeline configurations
use schema version 2 and a `reduction` field. `Pipeline.from_config()` and report
replay also accept schema version 1's `strategy` field and normalize it on export.
The enclosing report schema remains `"2.0"`.
