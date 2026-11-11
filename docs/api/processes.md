# Composed and joint processes

The four additional scenario kinds, for one target and for several: `Scenario`
and `SimulationReference` from `timegym`, with low-level generators in
`timegym.processes` and `timegym.joint_processes`.

## Composed and financial processes

The additional four kinds use the same `Scenario` and `Pipeline` entry points
and support both scalar and joint targets. These are controlled **data generators**; forecasting
models are supplied separately. See the [stress-case guide](../stress-cases.md) and
[executable application](../../tutorials/11_composed_and_financial_stresses.ipynb).

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

## Joint composed and financial processes

The four additional kinds accept `n_targets > 1`. Joint cases use separate random-stream
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
the filtered state distribution.

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

## Signatures

Generated from the docstrings in the code.

::: timegym.processes.resolve_process

::: timegym.processes.generate_process

::: timegym.processes.process_label

::: timegym.joint_processes.generate_joint_process
