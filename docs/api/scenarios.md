# Scenarios and references

Synthetic cases and their known conditional references: `Scenario`, `Case` and
`Reference` from `timegym`, `DEFAULT_LEVELS` from `timegym.contracts`, and
`scenario_label` from `timegym.scenarios`.

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

The five core families are listed by `timegym.scenarios.CORE_KINDS`, derived from
the scenario defaults. `timegym.data.synthetic.default_scenarios(context_length=120,
horizon=12, *, n_targets=1, seed=0)` creates fresh specifications for those families.
`Pipeline.run()` uses that helper; `Pipeline.plot_tests()` requests the same families
with a context length of 240 and horizon of 80.
