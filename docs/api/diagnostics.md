# Scores and diagnostics

**In short.** Scores measure how far a forecast is, both from what actually
happened (`mse`, `mae`, `mase`) and from the answer key (`mean_reference_mse`);
lower is better. A diagnostic is one check's result: a verdict (`PASS`, `FAIL`,
`INCONCLUSIVE`, `UNSUPPORTED` or `ERROR`), a message, and the numbers it rests
on, called its evidence. Accuracy diagnostics are always `INCONCLUSIVE`,
because TimeGym sets no pass mark for accuracy; that threshold is yours.
[How TimeGym decides](../concepts/how-timegym-decides.md) explains the verdicts.

## Scores

All scores are losses, so lower is better. Each takes finite arrays and raises
`ValueError` for empty, nonfinite or misshapen input: `predicted` matches the
shape of `observed`, quantiles add a trailing level axis, and samples add a
leading sample axis. `UndefinedMetricError`, a `ValueError` subclass, marks a
score that is undefined for otherwise valid data.

### Point scores

A point score is minimized in expectation by one functional of the predictive
distribution, so it has to match what the point denotes
([Gneiting 2011](https://doi.org/10.1016/j.ijforecast.2009.12.015)).
`Forecast.point` denotes a predictive mean.

| Function | What it computes | What it scores | Evaluator records it |
| --- | --- | --- | --- |
| `mse(observed, predicted)` | Mean squared error over all coordinates | The mean: squared error is minimized in expectation by the predictive mean | With a point output |
| `mae(observed, predicted)` | Mean absolute error over all coordinates | The median: absolute error is minimized in expectation by the predictive median | With a point output |
| `mase(observed, predicted, context, seasonality=1)` | Absolute error divided, per target, by the mean absolute `seasonality`-step change of that target in the context (time is axis zero), then averaged | The median, as for `mae`, because each target's scale is fixed by the context | With a point output, always with `seasonality=1` |
| `mean_reference_mse(reference_mean, predicted)` | Mean squared distance from a supplied conditional mean | The conditional mean itself: with a valid conditional mean it is the conditional excess squared-error risk, not realized loss against a future observation | With a point output and a case reference |

`mase` raises `UndefinedMetricError` when a target's naive-error scale is zero,
as for a constant target, or when the context is no longer than `seasonality`;
it adds no epsilon to the denominator and does not drop constant targets.
`evaluate_forecast` always calls `mase` with `seasonality=1`, so the evaluator
scales each target by its lag-1 naive in-context MAE whatever the scenario's
seasonal period; `point_accuracy` evidence records `mase_seasonality: 1`.
`mean_reference_mse` is not a reference for other losses.

### Uncertainty scores

Calibration and sharpness are the two properties these scores reward.
Calibration is the statistical consistency between the predictive distribution
and the observations; sharpness is the concentration of the predictive
distribution, a property of the forecast alone. The aim is to maximize sharpness
subject to calibration
([Gneiting, Balabdaoui and Raftery 2007](https://doi.org/10.1111/j.1467-9868.2007.00587.x)).
A proper scoring rule rewards both at once: its expected value is lowest when the
forecast is the distribution the observation is drawn from
([Gneiting and Katzfuss 2014](https://doi.org/10.1146/annurev-statistics-062713-085831)).

| Function | What it computes | What it scores | Evaluator records it |
| --- | --- | --- | --- |
| `pinball(observed, quantiles, levels)` | Mean quantile (pinball) loss over coordinates and levels, without the optional factor of two; levels must increase strictly in (0, 1) and quantiles must not cross | The quantile at each level: pinball loss at level α is minimized in expectation by the α-quantile | With a quantile output |
| `crps(observed, samples)` | Mean marginal CRPS of the finite empirical ensemble: the expected absolute error of a draw minus half the expected absolute difference between two draws, diagonal pairs included | Each coordinate's whole marginal distribution, as a proper scoring rule; it is unchanged when samples are permuted independently per coordinate, so it cannot see dependence | With a sample output |
| `empirical_crps` | The same function as `crps` | As `crps` | Recorded as `crps` |
| `energy_score(observed, samples, chunk_size=128)` | Energy score of the whole flattened horizon-target vector with the Euclidean norm, not divided by the number of coordinates; diagonal pairs included | The whole joint distribution, as a proper scoring rule for multivariate forecasts; with one coordinate it equals CRPS | With a sample output |
| `variogram_score(observed, samples, exponent=0.5)` | Mean squared difference between the realized and the expected `abs(x_i - x_j) ** exponent` over all distinct coordinate pairs, equally weighted; `exponent` lies in (0, 2) | Pairwise dependence between coordinates; it ignores a shift shared by all coordinates and is not strictly proper over unrestricted joint distributions | With a sample output; undefined for one coordinate |

`crps` and `energy_score` score the empirical distribution of the finite
ensemble; neither is the fair-ensemble estimator. `chunk_size` only bounds the
memory of the pairwise distances.

With at least two quantile levels the evaluator also records
`interval_coverage`, the share of coordinates between the lowest and the highest
quantile; `interval_nominal_coverage`, the highest level minus the lowest; and
`interval_width`, the mean distance between those two quantiles. Coverage bears
on calibration and width on sharpness. One path's coverage across possibly
dependent coordinates does not establish calibration, so the `interval_coverage`
diagnostic stays `INCONCLUSIVE`.

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
Point, quantile, and sample accuracy diagnostics include their available scores
in evidence, with reference capabilities when a simulation reference supplies them.

`timegym.checks.check_shift_equivariance(pipeline, context, future_index, *,
shift=10.0, atol=1e-8, rtol=1e-7)` tests
f(y+c)=f(y)+c. Use it only when this property is part of the model's intended
contract; many legitimate models are not location equivariant.

## Diagnostics catalogue

Each row is one diagnostic `name` in a report, with the statuses it can return,
the tolerance that separates its outcomes and its oracle type. The entries
below the table add its evidence keys, the function that emits it and what a
failure usually means; `<name>` and `<code>` stand for parts filled in at run
time. A row without a threshold is descriptive and returns `INCONCLUSIVE`.

| Diagnostic | Category | Statuses it can return | Tolerance | Oracle type |
| --- | --- | --- | --- | --- |
| `forecast_contract` | `correctness` | `PASS`, `ERROR` | Exact rules | Partial oracle |
| `contract_<code>` | `correctness` | `FAIL` | Exact rules | Partial oracle |
| `evaluation_data` | `evaluation` | `ERROR` | Exact rules | None: checks the evaluator's own inputs |
| `evaluation_reference` | `evaluation` | `ERROR` | Exact rules | None: checks the evaluator's own inputs |
| `point_accuracy` | `capability` | `INCONCLUSIVE`, `UNSUPPORTED` | None (descriptive) | Simplified data with known results |
| `quantile_accuracy` | `capability` | `INCONCLUSIVE`, `UNSUPPORTED` | None (descriptive) | Simplified data with known results |
| `sample_scores` | `capability` | `INCONCLUSIVE`, `UNSUPPORTED` | None (descriptive) | Simplified data with known results |
| `metric_<name>` | `capability` | `INCONCLUSIVE`, `ERROR` | None (descriptive) | Simplified data with known results |
| `interval_coverage` | `uncertainty` | `INCONCLUSIVE`, `UNSUPPORTED` | None (descriptive) | Simplified data with known results |
| `reference_covariance` | `distribution` | `PASS`, `FAIL`, `INCONCLUSIVE`, `UNSUPPORTED`, `ERROR` | `alpha`, default 0.05 | Statistical metamorphic test |
| `shift_equivariance` | `metamorphic` | `PASS`, `FAIL`, `UNSUPPORTED`, `ERROR` | `atol` and `rtol` | Known properties |
| `model_execution` | `execution` | `ERROR` | Any exception | Execution record |

Exception evidence is `exception_type`, `exception_message` and `traceback`, as
returned by `exception_evidence`. `reference` evidence appears only with a
`SimulationReference` and holds its `kind` and capability entries. The `<code>`
in `contract_<code>` is one of `index_type`, `index_missing`, `index_nonfinite`,
`index_empty`, `index_duplicate`, `index_order`, `index_mismatch`,
`target_names`, `target_mismatch`, `invalid_levels`, `duplicate_levels`,
`nonmonotonic_levels`, `missing_quantiles`, `missing_prediction`, `point_shape`,
`quantile_shape`, `sample_shape`, `quantile_crossing`, or `nonreal_` or
`nonfinite_` followed by `point`, `quantiles` or `samples`.
[Fault fixtures](fault-fixtures.md) lists the code each output fault fixture
triggers. The study's pseudo-oracle is an experimental comparator, not a package
diagnostic, so it has no row.

### `forecast_contract`

- **Emitted by:** `evaluate_forecast`, through `Forecast.validate(task)`
- **Evidence keys:** `PASS`: none. `ERROR`: exception evidence
- **Threshold or tolerance:** None: exact rules
- **What a FAIL or ERROR usually means:** `ERROR`: validation itself raised, usually on an output object it cannot inspect. A violation is reported as `contract_<code>` instead.

### `contract_<code>`

- **Emitted by:** `evaluate_forecast`, one row per `Forecast.validate(task)` finding
- **Evidence keys:** `code`
- **Threshold or tolerance:** None: exact rules
- **What a FAIL or ERROR usually means:** The output breaks the named rule, such as a shifted index, swapped targets, a wrong shape, nonfinite values or crossing quantiles. The case is not scored.

### `evaluation_data`

- **Emitted by:** `evaluate_forecast`, after the contract holds
- **Evidence keys:** Exception evidence
- **Threshold or tolerance:** None: exact rules
- **What a FAIL or ERROR usually means:** The held-out observations do not match the task's shape, index or target order, or are not finite: a case-construction problem, not a model fault.

### `evaluation_reference`

- **Emitted by:** `evaluate_forecast`, when the case has a reference
- **Evidence keys:** Exception evidence
- **Threshold or tolerance:** None: exact rules
- **What a FAIL or ERROR usually means:** The reference fails its own validation or does not match the task's index and target order: usually a custom case or reference, not a model fault.

### `point_accuracy`

- **Emitted by:** `evaluate_forecast`
- **Evidence keys:** `INCONCLUSIVE`: the computed `mse`, `mae`, `mase` and `mean_reference_mse`, `mase_seasonality`, `reference`. `UNSUPPORTED`: none
- **Threshold or tolerance:** None: descriptive
- **What a FAIL or ERROR usually means:** Never `FAIL` or `ERROR`; a score that cannot be recorded appears as `metric_<name>`. `UNSUPPORTED`: no point output.

### `quantile_accuracy`

- **Emitted by:** `evaluate_forecast`
- **Evidence keys:** `INCONCLUSIVE`: the computed `pinball`, `reference`. `UNSUPPORTED`: none
- **Threshold or tolerance:** None: descriptive
- **What a FAIL or ERROR usually means:** Never `FAIL` or `ERROR`. `UNSUPPORTED`: no quantile output.

### `sample_scores`

- **Emitted by:** `evaluate_forecast`
- **Evidence keys:** `INCONCLUSIVE`: the computed `crps`, `energy_score` and `variogram_score`, `reference`. `UNSUPPORTED`: none
- **Threshold or tolerance:** None: descriptive
- **What a FAIL or ERROR usually means:** Never `FAIL` or `ERROR`. `UNSUPPORTED`: no sample output.

### `metric_<name>`

- **Emitted by:** `evaluate_forecast`, only when a score cannot be recorded; `<name>` is `mse`, `mae`, `mase`, `mean_reference_mse`, `pinball`, `interval_width`, `crps`, `energy_score` or `variogram_score`
- **Evidence keys:** `INCONCLUSIVE`: `reason` (`undefined_metric`). `ERROR`: exception evidence
- **Threshold or tolerance:** None: descriptive
- **What a FAIL or ERROR usually means:** `ERROR`: the calculation raised or returned a nonfinite value, such as an overflowing interval width; the score is missing from `metrics`. `INCONCLUSIVE`: the score is undefined for valid data, such as `mase` on a constant target or `variogram_score` with one coordinate.

### `interval_coverage`

- **Emitted by:** `evaluate_forecast`, with a quantile output
- **Evidence keys:** `INCONCLUSIVE`: `observed_coverage`, `nominal_coverage`, `lower_level`, `upper_level`, and `mean_width` when `interval_width` was recorded. `UNSUPPORTED`: none
- **Threshold or tolerance:** None: descriptive; one path's coverage does not establish calibration
- **What a FAIL or ERROR usually means:** Never `FAIL` or `ERROR`. `UNSUPPORTED`: fewer than two quantile levels.

### `reference_covariance`

- **Emitted by:** `covariance_dependence_check`; `evaluate_forecast`, `Pipeline.test` and `Pipeline.run` with `check_dependence=True`, where an exception becomes the `ERROR`
- **Evidence keys:** `statistic`, `statistic_name`, `p_value`, `alpha`, `n_calibration`, `critical_value`, `minimum_p_value`, `n_samples`, `n_coordinates`, `seed`, `null`, `multiplicity`; `reference`, and `covariance_target` for a Monte Carlo covariance target. Early `UNSUPPORTED` and `INCONCLUSIVE` results record fewer keys; `ERROR` records exception evidence
- **Threshold or tolerance:** `FAIL` when the Monte Carlo rank p-value is at most `alpha` (defaults `alpha=0.05`, `n_calibration=99`). `INCONCLUSIVE` below 20 samples or when `1 / (n_calibration + 1)` exceeds `alpha`
- **What a FAIL or ERROR usually means:** `FAIL`: the ensemble covariance differs from the reference law, through a variance or dependence error or samples that are not independent draws. `ERROR`: the comparison raised, for example on invalid reference calibration draws.

### `shift_equivariance`

- **Emitted by:** `check_shift_equivariance`, called directly; `Pipeline.test` does not run it
- **Evidence keys:** `PASS` and `FAIL`: `shift`, `max_absolute_error`, `atol`, `rtol`. `ERROR`: `violations` or exception evidence. `UNSUPPORTED`: none
- **Threshold or tolerance:** `FAIL` unless `np.allclose(shifted, original + shift, atol=atol, rtol=rtol)`; defaults `shift=10.0`, `atol=1e-8`, `rtol=1e-7`
- **What a FAIL or ERROR usually means:** `FAIL`: the shifted point forecast is not the original plus the shift, which is a fault only if the model declares location equivariance. `ERROR`: either forecast broke its contract, or prediction raised.

### `model_execution`

- **Emitted by:** `Pipeline.run` and `Pipeline.test`
- **Evidence keys:** Exception evidence
- **Threshold or tolerance:** None: any exception
- **What a FAIL or ERROR usually means:** The model or its target transform raised while fitting or predicting, or `predict` returned something other than a `Forecast`; nothing was scored for that case.

## Case-level uncertainty intervals

`timegym.statistics.wilson_interval(values)` accepts a nonempty vector of binary
observations and returns `count`, `n`, `rate`, and the Wilson `ci95` bounds.
`timegym.statistics.paired_t_interval(values)` accepts at least two finite
case-level differences and returns `mean`, `n`, and Student t `ci95` bounds.
Cases must be independent; horizons and targets within a case are not replications.
The t interval is for continuous score differences, not differences in alarm rates.

## Signatures

::: timegym.metrics.mse

::: timegym.metrics.mae

::: timegym.metrics.mase

::: timegym.metrics.mean_reference_mse

::: timegym.metrics.pinball

::: timegym.metrics.crps

::: timegym.metrics.energy_score

::: timegym.metrics.variogram_score

::: timegym.metrics.UndefinedMetricError

::: timegym.diagnostics.evaluate_forecast

::: timegym.diagnostics.covariance_dependence_check

::: timegym.diagnostics.exception_evidence

::: timegym.checks.check_shift_equivariance

::: timegym.statistics.wilson_interval

::: timegym.statistics.paired_t_interval
