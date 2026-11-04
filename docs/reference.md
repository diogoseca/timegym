# API contracts

## Public surface

The top-level package keeps the main workflow small. Scores, checks, preprocessing
and other helpers are public through their modules; for example,
`from timegym.metrics import mase` and
`from timegym.checks import check_shift_equivariance`.
The import column names the module to import each listed name from.
“Public” means supported in this release; consult the changelog when upgrading.
Schema constants describe stored formats independently of the package version.
Underscore-prefixed names and modules, including `_internal` and
`_process_common`, are private and have no compatibility guarantee.

| Name | Import path | Stability |
| --- | --- | --- |
| `ForecastTask`, `Forecast`, `Reference`, `SimulationReference`, `Case`, `Scenario` | `timegym` | Public |
| `Pipeline`, `NaiveForecaster`, `GaussianAR1Forecaster` | `timegym` | Public |
| `ForecastRun`, `EvaluatedCase`, `TestReport`, `DiagnosticResult`, `CaseResult` | `timegym` | Public |
| `DiagnosticStatus`, `DiagnosticCategory`, `__version__` | `timegym` | Public |
| `DEFAULT_LEVELS` | `timegym.contracts` | Public |
| `mse`, `mae`, `mase`, `mean_reference_mse`, `pinball`, `crps`, `empirical_crps`, `energy_score`, `variogram_score`, `UndefinedMetricError` | `timegym.metrics` | Public; `empirical_crps` aliases `crps` |
| `evaluate_forecast`, `covariance_dependence_check`, `exception_evidence` | `timegym.diagnostics` | Public |
| `check_shift_equivariance` | `timegym.checks` | Public; opt-in model property |
| `SklearnForecaster`, `fresh_model`, `is_sklearn_regressor` | `timegym.models` | Public; sklearn loaded when needed |
| `REDUCTIONS` | `timegym.models` | Public |
| `MUTATIONS`, `mutate_forecast` | `timegym.mutations` | Public controlled fixtures |
| `plot_forecasts`, `plot_errors` | `timegym.plotting` | Public; requires plots extra |
| `wilson_interval`, `paired_t_interval` | `timegym.statistics` | Public; independent cases |
| `default_scenarios` | `timegym.data.synthetic` | Public |
| `get_stock` | `timegym.data.real` | Public; provider data may change |
| `add_lagged_target`, `cyclical_encoding`, `add_time_features`, `add_tsfel_features` | `timegym.features` | Public; TSFEL requires features extra |
| `multioutput_target`, `onestep_target`, `step_ahead_target`, `fractional_weights`, `fractional_differencing`, `differencing` | `timegym.target` | Public |
| `LogTransform`, `BoxCoxTransform`, `DifferenceTransform`, `ComposedTransform`, `transform_config`, `transform_from_config` | `timegym.target` | Public; replay rebuilds bundled transforms only |
| `simple_moving_average`, `exponential_moving_average` | `timegym.smoothing` | Public |
| `CORE_KINDS`, `scenario_label`, `MANIFEST_SCHEMA_VERSION` | `timegym.scenarios` | Public; manifest schema versioned separately |
| `PROCESS_KINDS`, `resolve_process`, `process_label`, `generate_process` | `timegym.processes` | Public low-level generators |
| `generate_joint_process` | `timegym.joint_processes` | Public; takes `resolve_process` output |
| `replay_case` | `timegym.pipeline` | Public; allowlisted models only |
| `PIPELINE_CONFIG_SCHEMA_VERSION`, `pipeline_config`, `pipeline_from_config`, `runtime_provenance`, `diagnostic_settings` | `timegym.provenance` | Public; config schema versioned separately |
| `REPORT_SCHEMA_VERSION` | `timegym.reporting` | Public; report schema versioned separately |

## Inputs and outputs

`ForecastTask(context, future_index)` validates a finite pandas DataFrame, unique
ordered target names, increasing unique time labels, and a future index after the
context. Context is all that ordinary models may use to fit. Future targets and
privileged reference parameters remain with the evaluator.

`Forecast(index, point=None, quantiles=None, levels=(), samples=None,
target_names=("y",))` accepts unvalidated outputs so failures can be reported.
Call `forecast.validate(task)` to obtain stable code/message dictionaries.

| Output | Shape | Meaning |
| --- | --- | --- |
| point | H × D | Predictive mean |
| quantiles | H × D × Q | Marginal quantiles at strictly increasing levels in (0,1) |
| samples | S × H × D | Complete joint trajectories, preserving horizon/target pairing |

At least one output is required. Missing output types receive `UNSUPPORTED` when
an evaluation needs them. Shapes, finite values, index alignment, target order and
quantile monotonicity are checked before any score is calculated.

## Scenarios and references

`Scenario(kind, context_length=120, horizon=12, parameters=<factory>, n_targets=1,
seed=0).generate(seed=None)` returns a `Case(task, observed, reference, manifest)`.
`parameters` uses `dataclasses.field(default_factory=dict)`, so each instance
gets its own dictionary. The constructor accepts positional fields;
`.generate(seed=None)` also accepts a positional seed override.
Core kinds: `white_noise`, `trend`, `seasonal`, `ar1`, `random_walk`. Additional
joint-capable kinds: `composite`, `merton_jump_diffusion`, `garch`,
`markov_switching`. Parameters are
validated and serialized in their resolved form, including target-wise values
and the innovation correlation matrix. The four additional controlled families
resolve their parameters once at construction; create a new `Scenario` to change
them. Their scalar manifests retain scalar values, accepting a one-target
`correlation` without adding dependence fields. All random draws use local generators.

`Case` is defined in `timegym.contracts` and remains importable from `timegym` and
`timegym.scenarios`. Direct `generate_process(...)` calls resolve raw parameters;
`generate_joint_process(...)` accepts parameters already returned by
`resolve_process(...)`.

`scenario.display_name` provides the same parameter-aware name used in plots.
The identifiers above remain unchanged in configurations and reports.

| Identifier | Display name | What a shock does |
| --- | --- | --- |
| `white_noise` | Gaussian white noise (fixed mean) | A fresh deviation from a fixed mean; no carryover |
| `trend` | Linear trend + white noise | A fresh deviation from a deterministic line; no carryover |
| `seasonal` | Seasonality + white noise | A fresh deviation from a deterministic cycle; no carryover |
| `ar1` | Stationary AR(1) | Its effect decays, possibly alternating in sign |
| `random_walk`, `drift=0` | Random walk without drift | Its effect persists in the level |
| `random_walk`, nonzero `drift` | Random walk with drift | Its effect persists, alongside an expected change per step |

A random walk is **not white noise in levels**. It accumulates white-noise
innovations: `Y[t] = Y[t-1] + drift + epsilon[t]`. Its increments, after
subtracting the drift, are white noise. By contrast, the `trend` scenario is
`Y[t] = intercept + slope*t + epsilon[t]`: only its deterministic mean trends;
the noise does not accumulate. With known parameters, a scalar random walk's
conditional variance at horizon `h` is `h*sigma**2`, whereas white noise and
linear trend plus white noise, and fixed seasonality plus white noise have variance `sigma**2` at each horizon.
A walk with zero drift can still appear to trend in one realization.

`drift` defaults to zero and can differ across targets. Plot facets label each
target separately; a case combining zero and nonzero drifts is labeled
“Random walk (mixed drift across targets)” in aggregate views. The helper
`timegym.scenarios.scenario_label(kind, parameters=None, *, target_index=None)` also
formats recorded manifest parameters and leaves unknown custom kind names
unchanged.

`Reference.mean` and `variance` are H × D; `covariance` is (H·D) × (H·D) in
horizon-major C order. `sample(n_samples, seed)` returns S × H × D and
`quantile(levels)` returns H × D × Q. `forecast(n_samples=256, seed=0, levels=DEFAULT_LEVELS)` constructs
an explicitly privileged reference forecast for diagnostic calibration.
`timegym.contracts.DEFAULT_LEVELS` is `(0.1, 0.5, 0.9)`; it also supplies the
default marginal levels for the Gaussian AR(1) model.

The five core families have exact known-parameter Gaussian reference laws. For AR(1), |phi| < 1 and the
context starts in its stationary distribution. A random walk starts at its
configured intercept at index zero. Future innovations are integrated out.

## Composed and financial processes

The additional four kinds use the same `Scenario` and `Pipeline` entry points
and support both scalar and joint targets. These are controlled **data generators**; forecasting
models are supplied separately. See the [stress-case guide](stress-cases.md) and
[executable application](../notebooks/26_composed_and_financial_stresses.ipynb).

| Kind | Important parameters | Units and reference |
| --- | --- | --- |
| `composite` | `intercept`, `slope`, `amplitude`, `period`, `phase`, `sigma`, `residual`; optional `jump_rate`, `jump_mean`, `jump_sigma`, `change_point`, `changes` | Additive levels; analytic Gaussian or compound-Poisson reference |
| `merton_jump_diffusion` | `mu`, `sigma`, `jump_rate`, `jump_mean`, `jump_sigma`, `dt`, `initial_price`, `output` | Physical expected-price-growth rate; exact conditional samples and moments |
| `garch` | `mu`, `omega`, `alpha`, `beta`, `initial_variance`, `initial_price`, `output` | Gaussian GARCH(1,1) log returns; each future sample evolves its own variance |
| `markov_switching` | `transition`, `initial_probs`, `regime_mu`, `regime_sigma`, `regime_jump_rate`, `regime_jump_mean`, `regime_jump_sigma`, `dt`, `initial_price`, `output` | Hidden return regimes; filter observed history, then propagate state uncertainty |

`composite` adds a deterministic line and sinusoid to one residual process:
`white_noise` or `random_walk`. Poisson jumps require `random_walk` residuals.
The `changes` dictionary may change `intercept`, `slope`, `amplitude`, `period`,
`phase`, or `sigma` from `change_point` onward. Mean parameters apply to absolute
time, so a slope change can also create a level jump. The schedule is known to
the evaluator; the model receives only its ordinary history and future index.
Fixed seasonality, changing conditional volatility, and switching hidden states
are distinct from changing the process parameters.

For Merton and Markov families, `dt` is the observation-step length in a
chosen base time unit and defaults to `1/252`. Growth and jump intensity
(`mu`, `jump_rate`, and their regime variants) are per base time unit;
diffusion scale `sigma` is per square root of that unit. For example, with
years as the base unit, `dt=1/252` represents one trading day. Jump sizes
are **log-price increments**.
Outputs are `price`, `log_price`, or `log_return`. GARCH's `mu` and variance
parameters apply per observation and it supports only `log_return` or
`log_price`. Price moments are not inferred by exponentiating log-price means.
Composite jump rates are per observation, not annual rates.

```python
suite = [
    Scenario("composite", parameters={
        "slope": .03, "amplitude": 2., "period": 24.,
        "residual": "random_walk", "sigma": .2,
        "jump_rate": .04, "jump_sigma": .9,
        "change_point": 80, "changes": {"amplitude": 4.},
    }),
    Scenario("merton_jump_diffusion", parameters={"jump_rate": 6., "output": "price"}),
    Scenario("garch"),
    Scenario("markov_switching", parameters={
        "regime_jump_rate": [0., 12.], "regime_jump_mean": [0., -.06],
        "regime_jump_sigma": [0., .10], "output": "log_return",
    }),
]
run = Pipeline().run(suite, seeds=[11, 22, 33])
run.plot_forecasts()
```

`SimulationReference` accepts aligned finite mean and covariance arrays as in
`Reference`, plus `sampler(n_samples, seed)`, `sampler_kind`, `quantile_samples`
(default 8192), and `quantile_seed` (default 0). It validates trajectory shape and
finiteness, and estimates marginal quantiles empirically from a reproducible
ensemble. It never infers a Gaussian distribution from moments. Generator
reference samplers use random streams separated from observed paths.

The `capabilities` dictionary and recorded `conditioning` metadata identify
analytic or explicitly estimated covariance, Monte Carlo quantiles, and exact versus approximate
conditional sampling. Regimes with jumps use a truncated Poisson-mixture
observation likelihood. Its omitted probability mass is recorded; it is not a
posterior-error bound. The covariance rank diagnostic is `UNSUPPORTED` for
approximate conditional samplers. Ordinary forecast contracts and descriptive
scores remain available. Parameter validation rejects unsupported cases rather
than silently substituting a different law. Mixed atomic/continuous Markov
emissions are rejected. In the univariate implementation, entirely discrete log-return emissions use exact atom
matching; informative discrete price/log-price histories are rejected unless
all regimes are identically deterministic, because numerical log/difference
conversion cannot preserve distinct atoms reliably. Initial state probabilities
refer to the state before the first return; each return follows one transition.

## Built-in models

`NaiveForecaster(strategy="last", period=1)` returns explicit point forecasts.
`last` repeats the final observation and `mean` repeats the context mean; both
need at least one observation. `seasonal` repeats the final `period` observations
and requires at least that many rows. `drift` extrapolates the slope from the
first to the last observation and needs at least two rows. `period` is a positive
integer. With two or more context rows, built-in step models require a regular
index and immediately following consecutive future labels. With one row, the
naive models can only require increasing future labels after the context.

`GaussianAR1Forecaster(n_samples=256, seed=0, levels=DEFAULT_LEVELS)` requires at
least five context observations. It fits an intercept and lag coefficient by
OLS per target and pools residual cross-target covariance with `n - 3` degrees
of freedom for `n` context rows. Its `Forecast` carries the iterated fitted
conditional mean, paired Gaussian trajectories of shape `(n_samples, H, D)`, and
empirical marginal quantiles at the requested levels (by default 0.1, 0.5, 0.9).
At least two samples are required. A nonnegative integer seed repeats the draws;
`seed=None` uses fresh randomness. Parameter-estimation uncertainty is omitted;
this fitted model is not the known-parameter reference.

`timegym.models.SklearnForecaster(steps, window_length=12, reduction="recursive",
horizon=None)` reduces a cloned sklearn pipeline to multi-step forecasts. Each
input flattens `window_length` consecutive rows across all targets, oldest row
first; transforms fit only those training windows. `timegym.models.REDUCTIONS`
is `("recursive", "direct", "multioutput")`. `recursive` fits one one-step
regressor per target and appends each predicted target vector to the window.
`direct` fits one regressor per horizon step and target, each predicting from
the last observed window. `multioutput` fits one regressor per target for the
whole horizon vector, wrapping the final step in `MultiOutputRegressor` when it
lacks native multioutput support. `direct` and `multioutput` need a positive
integer `horizon`, forecast at most that many steps, and need at least
`window_length + horizon` context rows; `recursive` needs `window_length + 1`.
All three return point outputs only. `steps` must end in an sklearn regressor. `fresh_model(model)` uses
`clone_unfitted()` for native instances or sklearn's configuration clone;
it rejects native instances without that protocol and clones returning the
original object. `is_sklearn_regressor(model)` reads sklearn tags lazily after
checking for `get_params`. Install `timegym[sklearn]` when using sklearn objects.

## Output-fault fixtures

`timegym.mutations.mutate_forecast(forecast, name, *, seed=0, enabled=True,
strength=1.0)` deep-copies outputs and coerces their arrays before checking a
fixture's preconditions. `timegym.mutations.MUTATIONS` lists these eight names.
The validator codes below assume an otherwise valid forecast for its task;
truncation can also produce output-shape violations.

| Name | Transformation | Precondition | Validator code |
| --- | --- | --- | --- |
| `index_shift` | Add one to index labels | Integer index | `index_mismatch` |
| `target_swap` | Reverse target names, leave values in place | At least two targets | `target_mismatch` |
| `horizon_truncation` | Drop the last horizon in every output and index | No additional condition | `index_mismatch` |
| `nonfinite_point` | Set the first point to NaN | Nonempty explicit points | `nonfinite_point` |
| `quantile_crossing` | Reverse the quantile axis, retain levels | At least two distinct ordered quantiles | `quantile_crossing` |
| `sample_axis_swap` | Transpose sample and horizon axes | Samples with unequal sample/horizon sizes | `sample_shape` |
| `variance_shrink` | Multiply deviations from points by `1 - 0.75 * strength` | Point, quantile and sample outputs | None; distribution checks may detect it |
| `dependence_shuffle` | Independently permute selected samples per coordinate | Sample outputs | None; marginals are preserved |

`strength` lies in `[0, 1]` and controls only the last two transformations;
`seed` controls only the shuffle and must be a nonnegative integer. The other
fixtures ignore these controls after validation. `enabled=False` returns the
prepared copy without applying the fault, but still validates the name, seed,
strength and preconditions. This bypass is a specified repair, not repair discovery.

## Pipeline and reports

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
`.test(scenarios=None, seeds=(0, 1, 2), *, check_dependence=False,
n_calibration=99, alpha=0.05, dependence_seed_offset=100_000)` evaluates cases and
returns a TestReport. Explicit test seeds override a Scenario's default seed.
Model/evaluation exceptions are recorded with traceback evidence. Invalid scenario
configuration is a caller error and is raised rather than counted as a model fault.

`.run(scenarios=None, seeds=(0,1,2), ...)` accepts the same diagnostic settings
and returns a `ForecastRun`. Its `.report` is the usual `TestReport`; `.cases`
retains each `EvaluatedCase`, including the generated case, forecast (or `None`
on model failure), result, and scenario/seed positions. This in-memory inspection
object keeps arrays separate from the portable report schema.

`run.plot_forecasts(*, show_quantiles=False, figsize=None)` draws scenario/target
rows and seed columns with observation dots and unconnected forecast crosses.
Optional shading shows marginal intervals. `run.plot_errors(metric="mean_reference_mse", *,
bins=None, missing="raise", figsize=None)` shows one score per whole case;
`missing="annotate"` explicitly permits unavailable scores and labels their
counts and reasons. Neither method refits models. Install `.[plots]` for plotting.

`pipeline.plot_tests(...)` and `pipeline.plot_errors(...)` are convenient one-call
experiments. Each creates a fresh run. Use `run()` when multiple views must share
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
`report.assert_passed()` raises for detected FAIL/ERROR outcomes; callers must
inspect unsupported/inconclusive checks to decide whether their CI coverage is
adequate. A pass is always scoped to a check.

### Report schema

`timegym.reporting.REPORT_SCHEMA_VERSION = "2.0"` identifies the report envelope.
`timegym.provenance.PIPELINE_CONFIG_SCHEMA_VERSION = 3` identifies current
pipeline configuration writes, which add `target_transform` (`null` or a
`transform_config` description). Schema 2 configurations, written without that
field, remain readable; schema 1 configurations, which used `strategy`, are no
longer read. Scenario manifests use integer version 1,
named by `timegym.scenarios.MANIFEST_SCHEMA_VERSION`. These are independent
schemas: the historical report version is a string, while configuration and
manifest versions are integers.

## Statistical and metamorphic diagnostics

`timegym.diagnostics.covariance_dependence_check(reference, samples, *, seed=0,
n_calibration=99, alpha=0.05)` tests independent sampling from the specified conditional reference
law. It compares normalized covariance discrepancy with independent ensembles of
the same sample size using a conservative finite Monte Carlo rank p-value. Its
rejection can reflect variance or dependence errors. The threshold is per check;
no multiple-comparison adjustment is silently applied. Too few samples or
insufficient p-value resolution yield `INCONCLUSIVE`. Fewer than two coordinates
or a deterministic reference yield `UNSUPPORTED`. Both this check and
`evaluate_forecast` require explicit nonnegative integer dependence seeds.
Invalid seeds, calibration counts, alpha, or dependence flags are caller errors
and raise before evaluation, including when dependence checking is disabled.
The evaluator's `mase` scales by the lag-1 in-context naive MAE regardless of
scenario period; `point_accuracy` evidence records `mase_seasonality: 1`.
Point, quantile, and sample accuracy diagnostics include their available scores
in evidence, with reference capabilities when a simulation reference supplies them.

`timegym.checks.check_shift_equivariance(pipeline, context, future_index, *,
shift=10.0, atol=1e-8, rtol=1e-7)` tests
f(y+c)=f(y)+c. Use it only when this property is part of the model's intended
contract; many legitimate models are not location equivariant.

The evaluator records accuracy, interval coverage, and proper scores as
measurements. It does not turn arbitrary tolerances into diagnoses of software
bugs. Failure provenance identifies the violated check; causal root-cause
localization still requires inspection or a controlled intervention.

## Joint composed and financial processes

The four additional kinds accept `n_targets > 1`. Their scalar implementations
retain existing seeds and output values. Joint cases use separate random-stream
namespaces for observations, conditional reference draws, and numerical moments.

| Parameter | Multivariate meaning |
| --- | --- |
| `correlation` | Scalar equicorrelation or `(D,D)` Gaussian innovation correlation matrix; default identity |
| `jump_correlation` | Scalar equicorrelation or `(D,D)` correlation of Gaussian jump marks; default identity |
| Composite mean parameters, `sigma`, `jump_mean`, `jump_sigma` | Scalars or length-`D` vectors; scheduled changes use the same convention |
| Merton `mu`, `sigma`, `initial_price`, `jump_mean`, `jump_sigma` | Scalars or length-`D` vectors |
| Composite/Merton `jump_rate` | One scalar intensity for the **shared** Poisson arrival clock |
| GARCH `mu`, `omega`, `alpha`, `beta`, `initial_variance`, `initial_price` | Scalars or length-`D` vectors; `alpha + beta < 1` for each target |
| GARCH `moment_samples`, `moment_seed` | Independent covariance-integration ensemble size (default 16384, at least 2) and seed (default 0) |
| Markov `regime_mu`, `regime_sigma`, `regime_jump_mean`, `regime_jump_sigma` | `(K,)` values broadcast over targets, or explicit `(K,D)` arrays |
| Markov `regime_jump_rate` | `(K,)` intensities for one state-dependent shared jump clock |
| Markov `regime_correlation` | Optional `(K,D,D)` matrices; otherwise every state uses `correlation` |

`dt` remains scalar. `transition` and `initial_probs` describe a **shared** hidden
chain. Initial prices accept target vectors. Markov references use all observed
targets to infer the terminal-state distribution; they never use the realized
hidden state. Multivariate filtering requires strictly positive diffusion
scales and positive-definite state correlation matrices. The other joint
families accept positive-semidefinite, including singular, correlation matrices.
Independent per-target jump clocks, separate hidden chains, DCC-GARCH, and
cross-lag spillover terms are not part of these declared laws.

```python
case = Scenario("merton_jump_diffusion", n_targets=3, horizon=8, parameters={
    "mu": [.03, .05, .02], "sigma": [.15, .2, .25],
    "correlation": .5, "jump_rate": 3., "jump_correlation": .8,
    "jump_mean": [-.03, -.06, -.10], "output": "log_return",
}).generate(seed=11)
assert case.reference.sample(256, seed=91).shape == (256, 8, 3)
assert case.reference.covariance.shape == (24, 24)
```

The covariance flattens `(horizon,target)` in C order. Joint `composite` and
Merton moments are analytic. Joint Markov moments are analytic conditional on
the filtered state distribution; a jump-mixture posterior is approximate, so
its exact-reference covariance diagnostic stays `UNSUPPORTED`.

CCC-GARCH has exact conditional sampling, analytical means and marginal
variances, and zero cross-time return covariance. Its future contemporaneous
cross-covariance contains `R[i,j] * E[sqrt(v[h,i] * v[h,j])]`, which generally
requires integration. The implementation averages conditional covariance
matrices using a separate Monte Carlo ensemble and rescales their diagonals
to analytical variances, preserving positive semidefiniteness. Log-level
covariances accumulate these return blocks. First-step, zero-correlation, and
fully deterministic-variance cases use analytical covariance. The resolved joint
GARCH parameters retain `moment_samples` and `moment_seed` for replay even in
these analytical cases, where neither setting is used.

Both cases retain `kind="analytic_moments_simulation"`. For the numerical case,
`SimulationReference` records
`moments_kind="analytic_mean_monte_carlo_covariance"`, `covariance_kind="monte_carlo"`,
`moment_samples`, and `moment_seed`. Changing the moment seed changes only the
numerical reference covariance, not observations, analytical means, or sampled
future paths. Exact-sampler rank calibration remains available: the covariance
target is fixed independently of the forecast/null ensembles, and both receive
the identical statistic. This is a conditional-law diagnostic, not an assertion
that the estimated covariance is exact. A covariance check also cannot detect
all higher-order dependence.

The five core families are listed by `timegym.scenarios.CORE_KINDS`, derived from
the scenario defaults. `timegym.data.synthetic.default_scenarios(context_length=120,
horizon=12, *, n_targets=1, seed=0)` creates fresh specifications for those families.
`Pipeline.run()` uses that helper; `Pipeline.plot_tests()` requests the same families
with a context length of 240 and horizon of 80.

## Case-level uncertainty intervals

`timegym.statistics.wilson_interval(values)` accepts a nonempty vector of binary
observations and returns `count`, `n`, `rate`, and the Wilson `ci95` bounds.
`timegym.statistics.paired_t_interval(values)` accepts at least two finite
case-level differences and returns `mean`, `n`, and Student t `ci95` bounds.
Cases must be independent; horizons and targets within a case are not replications.
The t interval is for continuous score differences, not differences in alarm rates.

## Optional market data

`timegym.data.real.get_stock(ticker, columns="Close", *, auto_adjust=True,
cache_path=None, refresh=False)` is a public opt-in network helper. Install
`timegym[market-data]` for a fetch. It requests one ticker's maximum daily history
through [yfinance's history API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html).
It always returns a new pandas DataFrame, including for a single column, in the
requested column order with the provider's timezone and dates (nanosecond index
resolution). Missing observations are preserved; empty history, unavailable
columns, and duplicate or unordered dates raise `ValueError`.

`auto_adjust=True`, the default, requests adjusted OHLC including `Close`, so
splits and dividends do not appear as jumps; `False` keeps the provider's
unadjusted fields. Choose and record this setting before defining the
forecasting target. Selecting `"Close"` used to
return a Series; callers that need one can now select `result["Close"]`.

`cache_path` names a JSON snapshot file containing full history, ticker and
adjustment metadata. A matching existing snapshot is read without importing
yfinance or making a network request. It can serve different column selections.
Malformed snapshots or different ticker/adjustment settings raise; they never
trigger an implicit refetch. `refresh=True` fetches again and atomically replaces
the snapshot only after validation. Fetch errors propagate, and an unsuccessful
refresh leaves the previous snapshot intact. No cache is written when the path
is omitted. JSON preserves numeric values to 15 decimal places; comparison of
floating-point values across a cache round trip should allow rounding tolerance.

```python
from timegym.data.real import get_stock

prices = get_stock("MSFT", auto_adjust=True, cache_path="data/msft-adjusted.json")
```

The first call above uses the network unless the snapshot already exists.
Archive the snapshot and adjustment setting with an experiment: provider history
can be revised, and today's adjusted prices are not point-in-time data. This
helper supplies observations, not a known conditional reference. The maintained
notebooks instead use the bundled AirPassengers CSV and run offline on CPU.

## Data preparation

The functions below are public through `timegym.features`, `timegym.target`, and
`timegym.smoothing`; import them from those modules. They are standalone pandas
transformations, not automatic Pipeline stages. They never mutate the supplied
object. Row offsets are positional, so a lag of one means the previous row even
when timestamps are irregular. Apply transformations within each training split
and keep future labels out of predictor matrices. `SklearnForecaster` constructs
its own lag windows; these helpers support explicit notebook experiments and
custom models. The reversible target transforms below are the exception: a
`Pipeline` fits and inverts them on each case.

### Smoothing

`simple_moving_average(df, window_length=30, min_periods=1)` and
`exponential_moving_average(df, exponential_alpha=None, statistic="mean", *,
min_periods=1)` both return the same Series/DataFrame type, index, column order
and labels as the input, containing only smoothed values for **every column**.
SMA previously appended one renamed column using only the first input column;
callers should now plot or concatenate the original data explicitly if needed.

Both means use the current and earlier rows. Shift the result by one row to use
it for predicting the current observation. SMA uses a trailing `window_length`
(positive integer); `min_periods` is a nonnegative integer no larger than that
window. EMA requires finite `exponential_alpha` in `(0, 1]` and supports only
`statistic="mean"`. It normalizes geometric weights (`adjust=True`) and uses
absolute row positions through gaps (`ignore_na=False`). Both skip missing
observations when averaging and return NaN until each column meets
`min_periods`. These are historical transformations, not future forecasts.

### Features

| Function | Contract |
| --- | --- |
| `add_lagged_target(df, lags=10, dropna=True)` | Append `{column}_t-1` through `{column}_t-{lags}` by source column; zero lags adds no columns. Retain originals; exclude a contemporaneous target from predictors. |
| `cyclical_encoding(values, scaled=False, min_value=None, max_value=None, *, period=None)` | Return cosine/sine features of finite real values using a fixed period and phase. Prefer `period=7, min_value=0` for weekdays. Explicit bounds are inclusive discrete labels; `scaled=True` accepts cycle fractions and forbids period/bounds. No sample-dependent scaling. |
| `add_time_features(df)` | Require a DatetimeIndex without NaT. Append a fixed calendar schema: elapsed UTC days since the Unix epoch, local year and cosine/sine pairs for quarter, month, day, weekday, hour, minute, second, microsecond and month-relative day. |
| `add_tsfel_features(df, use_cols=None, window_size=100, domain=None, frequency=100, dropna=True, show_progress=False, lag=1)` | Optional `timegym[features]` helper. Append TSFEL features to selected original columns. At row `t`, the window ends at `t-lag`; default `lag=1` excludes the current row. `frequency` is the sampling frequency in Hz. |

Frame helpers require unique column labels and reject generated-name collisions
instead of overwriting observations. `use_cols` is a name or sequence of names;
omitting it selects every column. For lagged and TSFEL features, `dropna=True`
removes rows missing any returned value, including original values. `False`
retains the original index and missing warmup features. TSFEL needs at least
`window_size + lag` rows to produce one feature row, with positive window size
and nonnegative lag. Optional extraction and progress imports happen only when
requested; core imports do not require TSFEL or tqdm.

### Targets and differences

| Function | Contract |
| --- | --- |
| `multioutput_target(df, horizon=10, dropna=True)` | Append `{column}_t+1` through `{column}_t+{horizon}`, by source column. Positive integer horizon. |
| `onestep_target(df, dropna=True)` | The same target construction with horizon one. |
| `step_ahead_target(df, step=5, dropna=True)` | Append only `{column}_t+{step}` for each original column. Positive integer step. |
| `fractional_weights(d, lags)` | Return a `(lags, 1)` array of the first binomial coefficients of `(1-B)^d`, newest first. Finite real `d`; positive integer coefficient count. |
| `fractional_differencing(series, order, fractional_coefficients, *, lag=1)` | Apply a finite truncated `(1-B^lag)^order` filter independently to each column. Trim exactly `(fractional_coefficients-1)*lag` warmup rows; order zero returns an unchanged copy. |
| `differencing(series, order=1, fractional_coefficients=10, *, lag=1)` | Integer order means repeated differences, not lag distance: order two is `y[t]-2*y[t-lag]+y[t-2*lag]`. Trim `order*lag` rows. Noninteger order delegates to the finite fractional filter. |

Target constructors retain originals, reject duplicate or colliding columns, and
return a new DataFrame. `dropna=True` removes any row with a missing original or
future value; `False` retains the index and incomplete tail. Future columns are
labels, never inputs available at the forecast origin. These constructors are
not model adapters; `SklearnForecaster` implements the direct and multioutput
reductions.

Differencing accepts a Series or DataFrame and preserves its type and labels on
the retained rows. Orders must be finite and nonnegative, and lag must be a
positive integer. Missing observations inside the series propagate; they are
not filled with zero. `fractional_coefficients` is used only for noninteger
orders in `differencing`. A finite fractional filter is an approximation, not
an exact infinite-memory transform. These functions apply no inverse; use
`DifferenceTransform` below for differences that a pipeline undoes.

### Reversible target transforms

`LogTransform()`, `BoxCoxTransform()` and `DifferenceTransform(order=1, lag=1)`
implement `fit_transform(context)`, `inverse_transform(forecast)` and
`clone_unfitted()`. `ComposedTransform(transforms)` applies bundled members in
order and inverts them in reverse order. Each transform is fitted on the context
only. Log and Box-Cox require strictly positive finite targets; Box-Cox fits one
lambda per target by maximum likelihood, bounded to `[-1, 2]`, and rejects a
constant target and transformed forecasts without a preimage. Inversion
requires a valid forecast with the fitted target names in the fitted order.

- Samples and quantiles invert elementwise. Monotone transforms preserve
  quantile order.
- After a nonlinear inverse, the point is the mean of the inverted samples,
  else the inverted 0.5 quantile, else the inverted point read as a median.
- `DifferenceTransform` integrates points and samples exactly. It recomputes
  quantiles from the integrated samples, or omits them when there are none,
  and assumes the forecast rows immediately follow the context.

`transform_config(transform)` describes exact bundled classes as `builtin` and
any other object, including a subclass, as `external`.
`transform_from_config(config)` rebuilds an unfitted bundled transform; replay
rebuilds bundled transforms only.
