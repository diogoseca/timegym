# Synthetic scenarios

A scenario is a recipe for a kind of series. TimeGym draws a past and a future
from it, gives your model the past, and keeps the answer key: the expected value
and the spread of every future step, given that past. Because the recipe is
known, so is the limit of what any forecaster can achieve on it. The useful
question for each scenario is what a forecast *can* get right, and what nobody
can.

## The core scenarios

| Scenario | What the series does | What a good forecast gets right | What nobody can predict |
| --- | --- | --- | --- |
| Noise around a fixed level (`white_noise`) | Scatters randomly around a constant | The level | Each individual wiggle |
| Noise around a trend (`trend`) | A straight line plus scatter | The line, extended | The scatter |
| Noise around a season (`seasonal`) | A repeating wave plus scatter | The wave, repeated | The scatter |
| Mean reversion (`ar1`, also called AR(1)) | Wanders, but is pulled back to its mean | The next few steps, sliding back to the mean | Anything beyond that: far ahead, the best forecast is the mean |
| Random walk (`random_walk`) | Each step adds a fresh shock to the last value, and shocks never fade | Only that it stays near where it is now | Which way it goes; the uncertainty grows with every step ahead |
| Random walk with drift (`random_walk` with `drift`) | A random walk with a steady push | The push | The wandering around it |

In the first three, the noise is fresh at every step and leaves no trace: the
best forecast is the underlying level, line or wave, and the error that remains
is the noise itself. In the last three, each shock carries into the future, so
where the series is now matters.

**Mean reversion and random walks are the same equation.** Write the series as
its distance from a mean `m`. Each step keeps a fraction `phi` of the last
step's distance and adds a fresh shock `e[t]`:

```text
y[t] - m = phi * (y[t-1] - m) + e[t]
```

- With `phi` between 0 and 1 (TimeGym's default is 0.8) the series is
  *mean-reverting*, also called a stationary *autoregressive* process of order
  one, AR(1): "autoregressive" because each value is regressed on the one before
  it. A shock fades by a factor `phi` per step, so after `h` steps only
  `phi**h` of today's distance from the mean remains (0.8 to the 12th power is
  about 0.07). The best forecast slides from today's value back to the mean.
- With `phi = 1` the pull disappears: `y[t] = y[t-1] + e[t]` is a *random walk*.
  Shocks never fade, the best forecast is today's value, and its uncertainty
  keeps growing: `h` steps ahead the variance is `h` times the variance of one
  shock. A drift adds a constant to every step.
- With `phi` between -1 and 0, the series alternates around the mean. A
  positive `phi` does not make the series cycle; a regular cycle is what the
  seasonal scenario provides.

A random walk without drift can still look like a trend in one draw. That is
why every scenario is drawn many times: one draw can mislead, the spread of
errors across draws does not.

## Composed and financial scenarios

Four more recipes combine or extend the core ones. Each works for one series or
for several correlated series.

| Scenario | What it adds | What it tests |
| --- | --- | --- |
| `composite` | A trend, a season and noise together, with optional scheduled changes and jumps | Whether a model separates the parts, and how it copes when one changes |
| `merton_jump_diffusion` | A price that drifts and diffuses, with occasional sudden jumps | Whether forecasts and their spread account for jumps |
| `garch` | Returns whose volatility clusters: calm periods and turbulent periods | Whether the forecast spread widens and narrows with recent volatility |
| `markov_switching` | Hidden regimes, such as calm and stressed, with different behavior in each | Whether a model infers the regime it cannot see |

The rest of this page explains these four in more depth: what they can and
cannot show, and how to build a test out of them.

## Changing mechanisms

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

Use these `Scenario` kinds through the `Pipeline.run` and
`Pipeline.test` entry points. All four kinds, like the five core Gaussian
families, support `n_targets > 1`. Joint draws retain `(sample, horizon, target)` order.

| Kind | Controlled mechanism | Output and interpretation |
| --- | --- | --- |
| `composite` | Deterministic trend and sinusoid; one residual process, either white noise or random walk; optional Poisson increments and a change schedule | General time-series levels; component changes are explicit |
| `merton_jump_diffusion` | Diffusion with compound-Poisson jumps | `price`, `log_price`, or `log_return`; physical expected-return drift includes jump compensation |
| `garch` | Gaussian GARCH(1,1) innovations with recursive conditional variance | `log_return` or `log_price`; no price output |
| `markov_switching` | Hidden regimes with state-dependent return distributions and optional regime-dependent jumps | `price`, `log_price`, or `log_return`; the reference filters the hidden regime |

[Merton (1976)](https://doi.org/10.1016/0304-405X(76)90022-2) motivates
discontinuous price dynamics; [Bollerslev (1986)](https://doi.org/10.1016/0304-4076(86)90063-1)
motivates conditional volatility; [Hamilton (1989)](https://doi.org/10.2307/1912559)
motivates hidden regime switching.

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
[FinStressTS (2026)](https://doi.org/10.1145/3770855.3817578) provides closely related
mechanism-based financial forecasting benchmarks. TimeGym additionally emphasizes
implementation contracts, controlled faults, and repair verification.

## Match the model class to the target

| Forecast target | Useful model classes to test |
| --- | --- |
| Levels or conditional return means | Last-value controls, linear regressions, autoregressions |
| Return variance and predictive tails | ARCH/GARCH and probabilistic forecasters |
| Latent levels, trends, or regimes | State-space and regime-switching models |
| Nonlinear dependence on observed predictors | Trees and neural networks |
| General forecasting transfer | Time-series foundation models with explicit input/output contracts |

These are candidate classes, not rankings. Financial evaluation depends on the
target and loss; [FinTSB (2026)](https://doi.org/10.1007/s11704-026-51064-5) emphasizes
task-specific evaluation and realistic market constraints. Price, log price, returns, and
volatility are different prediction problems.

## Reference limits

References integrate future innovations and hidden-state uncertainty rather than
using the realized future path. Analytic means and conditional samplers are
provided where supported; multivariate GARCH labels its numerical cross-covariance; non-Gaussian marginal quantiles are empirical and
identified as such. Multi-step GARCH distributions need not be Gaussian.

Regime switching with jumps uses a numerical posterior approximation. Its
covariance rank diagnostic is disabled rather than presented as an exact test.
Inspect reference metadata and `UNSUPPORTED` outcomes alongside scores. See
[Tasks and forecasts](api/tasks-and-forecasts.md) for output shapes and
[Scores and diagnostics](api/diagnostics.md) for diagnostic semantics.

## Extend a mechanism to dependent targets

Use `n_targets=2` (or more) and a scalar equicorrelation or explicit correlation
matrix. Target-specific parameters accept vectors. The joint mechanisms are:

- Composed processes have correlated Gaussian shocks and optional shared jump
  arrivals, with vector jump sizes and target-specific parameter changes.
- Merton processes have correlated diffusion and a shared Poisson jump clock.
  `jump_correlation` controls jump-size dependence. A shared clock can induce
  dependence even when Gaussian shocks and jump marks have zero correlation.
- CCC-GARCH has one variance recursion per target and constant conditional
  innovation correlation. This follows [Bollerslev (1990)](https://doi.org/10.2307/2109358).
  It is not a dynamic-conditional-correlation or cross-lag spillover model.
- Markov targets share one hidden chain. The evaluator filters it using all
  observed targets; optional `regime_correlation` matrices change dependence
  between states. Jump arrivals can also depend on the shared state.

The [joint uncertainty notebook](../notebooks/27_joint_financial_uncertainty.ipynb)
compares paired futures with deliberately mismatched target trajectories. The
mutation preserves every target's marginal distribution and temporal path;
joint diagnostics and aggregate uncertainty reveal what marginal checks miss.
See [Composed and joint processes](api/processes.md) for shapes, units and filtering boundaries.
