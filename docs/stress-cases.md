# Synthetic forecasting stress cases

TimeGym separates the **process generating a test case** from the **forecasting
model being tested**. A generator supplies controlled observations and conditional
references; the same `Pipeline` can test a statistical, econometric, or machine
learning forecaster against them. The additional generators below do not install
or train new forecasting models.

## Seasonality, drift, and changing mechanisms

Seasonality repeats: for example, a cycle with fixed amplitude, period, and phase.
Process drift is an expected increment, as in a random walk with drift. Concept
drift concerns a changing predictive relationship; distribution shift can also
change the inputs. Changes may affect trend slope, seasonal amplitude or period,
noise variance or tails, jump intensity, or regime probabilities.
[Gama et al. (2014)](https://doi.org/10.1145/2523813) distinguish these forms of
distributional change.

A stable seasonal cycle is not automatically a changing mechanism when calendar
phase is part of the conditioning information. Likewise, changing volatility in
a fixed GARCH process, or switching states under a fixed transition matrix, need
not mean that the generating law changed. Test **state changes** and **parameter
changes** separately.

## Available mechanisms

Use these `Scenario` kinds through the existing `Pipeline.run` and
`Pipeline.test` entry points. All four extensions support `n_targets > 1`, alongside the original five
Gaussian families. Joint draws retain `(sample, horizon, target)` order.

| Kind | Controlled mechanism | Output and interpretation |
| --- | --- | --- |
| `composite` | Deterministic trend and sinusoid; one residual process, either white noise or random walk; optional Poisson increments and a change schedule | General time-series levels; component changes are explicit |
| `merton_jump_diffusion` | Diffusion with compound-Poisson jumps | `price`, `log_price`, or `log_return`; physical expected-return drift includes jump compensation |
| `garch` | Gaussian GARCH(1,1) innovations with recursive conditional variance | `log_return` or `log_price`; no price output |
| `markov_switching` | Hidden regimes with state-dependent return distributions and optional regime-dependent jumps | `price`, `log_price`, or `log_return`; the reference filters the hidden regime |

[Merton (1976)](https://www.cmat.edu.uy/~mordecki/hk2010/merton76.pdf) motivates
discontinuous price dynamics; [Bollerslev (1986)](https://public.econ.duke.edu/~boller/Published_Papers/joe_86.pdf)
motivates conditional volatility; [Hamilton (1989)](https://doi.org/10.2307/1912559)
motivates hidden regime switching. They are established generating mechanisms,
not a claim that these models win every stock-forecasting task.

The composite change schedule is known to the evaluator. It does not demonstrate
that a fitted model could anticipate an unannounced future break. Adding a cycle
to log returns also differs from adding it to log-price levels: specify the
forecast target before composing mechanisms.

## Build an interpretable test ladder

1. **Isolate:** run each mechanism alone, including simple no-change controls.
2. **Pair:** vary one property, such as jump intensity, seasonal amplitude, or
   variance persistence. Keep other parameters fixed and match innovations where
   the construction permits it.
3. **Compose:** add interactions, such as trend plus seasonality plus persistent
   noise, or hidden regimes with regime-dependent jumps. Use the supported
   combinations above; arbitrary addition of generator objects is not an API.
4. **Reserve evidence:** hold out seeds, parameter regions, and combinations from
   development. Compare correct, faulty, and repaired implementations on matched
   cases. A capability gap alone does not diagnose a software fault.
5. **Transfer:** freeze the selected pipeline and assess it on untouched real
   data with rolling forecast origins and task-appropriate metrics.

A synthetic suite can itself be overfit. Passing it establishes behavior under
its declared assumptions; it does not establish real-market accuracy or profits.
[FinStressTS (2026)](https://arxiv.org/abs/2606.03184) provides closely related
mechanism-based financial forecasting benchmarks. TimeGym additionally emphasizes
implementation contracts, controlled faults, and repair verification.

## Match the model and diagnostic to the target

| Forecast target | Useful model classes to test |
| --- | --- |
| Levels or conditional return means | Last-value controls, linear regressions, autoregressions |
| Return variance and predictive tails | ARCH/GARCH and probabilistic forecasters |
| Latent levels, trends, or regimes | State-space and regime-switching models |
| Nonlinear dependence on observed predictors | Trees and neural networks |
| General forecasting transfer | Time-series foundation models with explicit input/output contracts |

These are candidate classes, not rankings. Financial evaluation depends on the
target and loss; [FinTSB (2026)](https://arxiv.org/abs/2502.18834v3) emphasizes
task-specific evaluation and realistic market constraints. Price, log price, returns, and
volatility are different prediction problems.

## Reference limits

References integrate future innovations and hidden-state uncertainty rather than
using the realized future path. Analytic means and conditional samplers are
provided where supported; multivariate GARCH labels its numerical cross-covariance; non-Gaussian marginal quantiles are empirical and
identified as such. Multi-step GARCH distributions need not be Gaussian.

Regime switching with jumps uses a numerical posterior approximation. Its
covariance rank diagnostic is disabled rather than presented as an exact test.
Inspect reference metadata and `UNSUPPORTED` outcomes alongside scores. See the
[API contracts](reference.md) for output shapes and diagnostic semantics.

## Extend a mechanism to dependent targets

Use `n_targets=2` (or more) and a scalar equicorrelation or explicit correlation
matrix. Target-specific parameters accept vectors. The joint mechanisms are:

- Composed processes have correlated Gaussian shocks and optional shared jump
  arrivals, with vector jump sizes and target-specific parameter changes.
- Merton processes have correlated diffusion and a shared Poisson jump clock.
  `jump_correlation` controls jump-size dependence. A shared clock can induce
  dependence even when Gaussian shocks and jump marks have zero correlation.
- CCC-GARCH has one variance recursion per target and constant conditional
  innovation correlation. This follows [Bollerslev (1990)](https://public.econ.duke.edu/~boller/Published_Papers/restat_90.pdf).
  It is not a dynamic-conditional-correlation or cross-lag spillover model.
- Markov targets share one hidden chain. The evaluator filters it using all
  observed targets; optional `regime_correlation` matrices change dependence
  between states. Jump arrivals can also depend on the shared state.

CCC-GARCH's exact conditional simulator recursively updates every target on
each future path. General future cross-series covariance is numerically
integrated using an independent Monte Carlo stream, with count and seed recorded.
Means and marginal variances remain analytic. Rank calibration holds the same
estimated covariance matrix fixed for the forecast and every null ensemble.
This does not make that matrix an exact population moment.

The [joint uncertainty notebook](../notebooks/27_joint_financial_uncertainty.ipynb)
compares paired futures with deliberately mismatched target trajectories. The
mutation preserves every target's marginal distribution and temporal path;
joint diagnostics and aggregate uncertainty reveal what marginal checks miss.
See the [API reference](reference.md) for shapes, units and filtering boundaries.
