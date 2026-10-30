# API contracts

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

`Scenario(kind, context_length=120, horizon=12, parameters={}, n_targets=1,
seed=0).generate(seed=None)` returns a `Case(task, observed, reference, manifest)`.
Core kinds: `white_noise`, `trend`, `seasonal`, `ar1`, `random_walk`. Additional
joint-capable kinds: `composite`, `merton_jump_diffusion`, `garch`,
`markov_switching`. Parameters are
validated and serialized in their resolved form, including target-wise values
and the innovation correlation matrix. All random draws use local generators.

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
`timegym.scenarios.scenario_label(kind, parameters, target_index=...)` also
formats recorded manifest parameters and leaves unknown custom kind names
unchanged.

`Reference.mean` and `variance` are H × D; `covariance` is (H·D) × (H·D) in
horizon-major C order. `sample(n_samples, seed)` returns S × H × D and
`quantile(levels)` returns H × D × Q. `forecast(n_samples=256, seed=0)` constructs
an explicitly privileged reference forecast for diagnostic calibration.

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

## Pipeline and reports

`Pipeline(model=None, window_length=12, strategy="recursive", model_factory=None)`
uses a last-value baseline by default. Native model classes/factories or the
`clone_unfitted()` protocol ensure fresh fits. Optional sklearn steps can be added
with `.add(estimator, **parameters)`. Only recursive reduction is implemented.

`.predict(context, future_index)` returns the model's Forecast for inspection.
`.test(scenarios=None, seeds=(0,1,2), check_dependence=False)` evaluates cases and
returns a TestReport. Explicit test seeds override a Scenario's default seed.
Model/evaluation exceptions are recorded with traceback evidence. Invalid scenario
configuration is a caller error and is raised rather than counted as a model fault.

`.run(scenarios=None, seeds=(0,1,2), ...)` accepts the same diagnostic settings
and returns a `ForecastRun`. Its `.report` is the usual `TestReport`; `.cases`
retains each `EvaluatedCase`, including the generated case, forecast (or `None`
on model failure), result, and scenario/seed positions. This in-memory inspection
object keeps arrays separate from the portable report schema.

`run.plot_forecasts(show_quantiles=False, figsize=None)` draws scenario/target
rows and seed columns with observation dots and unconnected forecast crosses.
Optional shading shows marginal intervals. `run.plot_errors(metric="mean_reference_mse",
bins=None, missing="raise", figsize=None)` shows one score per whole case;
`missing="annotate"` explicitly permits unavailable scores and labels their
counts and reasons. Neither method refits models. Install `.[plots]` for plotting.

`pipeline.plot_tests(...)` and `pipeline.plot_error(...)` are convenient one-call
experiments. Each creates a fresh run. Use `run()` when multiple views must share
exactly the same predictions, especially for stochastic models.

`.to_config()` describes the configured model and `.from_config(config)` can
restore allowlisted built-in models. Factories, custom models and sklearn steps
are explicitly marked non-replayable. A JSON report records model configuration,
source fingerprint, dependency versions and diagnostic settings. Use
`report.replay_case(case_index=0)` or `python examples/reproduce_report.py report.json`
to reproduce a supported built-in case. No arbitrary module import is performed
from a report. Reproduction on different numerical-library versions may require
tolerance rather than exact floating-point equality.

Outcomes: `PASS`, `FAIL`, `INCONCLUSIVE`, `UNSUPPORTED`, `ERROR`.
`report.to_json(path=None)` serializes diagnostics, measurements and manifests.
`TestReport.from_dict(payload)`, `TestReport.from_json(text)`, and
`TestReport.read(path)` load reports without executing saved model descriptions.
They reject unknown schemas/statuses, nonfinite metrics, duplicate JSON keys,
malformed records, and stored summary counts that disagree with the diagnostics.
`from_json` always expects JSON text; use `read` for a file. Evidence and metadata
are copied so later edits to the input dictionary cannot change the loaded report.
`report.assert_passed()` raises for detected FAIL/ERROR outcomes; callers must
inspect unsupported/inconclusive checks to decide whether their CI coverage is
adequate. A pass is always scoped to a check.

## Statistical and metamorphic diagnostics

`covariance_dependence_check(reference, samples, seed=0, n_calibration=99,
alpha=.05)` tests independent sampling from the specified conditional reference
law. It compares normalized covariance discrepancy with independent ensembles of
the same sample size using a conservative finite Monte Carlo rank p-value. Its
rejection can reflect variance or dependence errors. The threshold is per check;
no multiple-comparison adjustment is silently applied. Too few samples or
insufficient p-value resolution yield `INCONCLUSIVE`.

`check_shift_equivariance(pipeline, context, future_index, shift=10)` tests
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
fully deterministic-variance cases use analytical covariance.

For the numerical case, `SimulationReference` records
`moments_kind="analytic_mean_monte_carlo_covariance"`, `covariance_kind="monte_carlo"`,
`moment_samples`, and `moment_seed`. Changing the moment seed changes only the
numerical reference covariance, not observations, analytical means, or sampled
future paths. Exact-sampler rank calibration remains available: the covariance
target is fixed independently of the forecast/null ensembles, and both receive
the identical statistic. This is a conditional-law diagnostic, not an assertion
that the estimated covariance is exact. A covariance check also cannot detect
all higher-order dependence.
