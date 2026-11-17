# Scores and diagnostics

**In short.** Scores measure how far a forecast is, both from what actually
happened (`mse`, `mae`, `mase`) and from the answer key (`mean_reference_mse`);
lower is better. Every report records these; more scores, such as `brier` and
`log_loss` for yes/no outcomes, or functions of your own, are recorded when you
ask for them with `scores=`. A diagnostic is one check's result: a verdict (`PASS`, `FAIL`,
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
| `rmse(observed, predicted)` | Square root of `mse`, in the data's units | The mean, as for `mse` | When asked for with `scores=` |
| `rmsse(observed, predicted, context, seasonality=1)` | Each target's RMSE divided by the root mean squared `seasonality`-step change of that target in the context, then averaged over targets: the scaled error of the M5 competition ([Makridakis, Spiliotis and Assimakopoulos 2022](https://doi.org/10.1016/j.ijforecast.2021.11.013)) | The mean, as for `mse` | When asked for with `scores=` |
| `mean_reference_mse(reference_mean, predicted)` | Mean squared distance from a supplied conditional mean | The conditional mean itself: with a valid conditional mean it is the conditional excess squared-error risk, not realized loss against a future observation | With a point output and a case reference |

`mase` and `rmsse` raise `UndefinedMetricError` when a target's naive-error
scale is zero, as for a constant target, or when the context is no longer than
`seasonality`; they add no epsilon to the denominator and do not drop constant
targets. MAPE and sMAPE are left out on purpose: they are undefined when an
actual value is zero and favour forecasts that are too low
([Hyndman and Koehler 2006](https://doi.org/10.1016/j.ijforecast.2006.03.001));
`mase` and `rmsse` meet the same need.
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
| `interval_score(observed, quantiles, levels)` | For the band from the lowest to the highest level, which must be symmetric (such as 0.1 and 0.9, a central band of 1 − α = 80%): its width plus 2/α times the distance by which the outcome falls outside it, averaged over coordinates ([Winkler 1972](https://doi.org/10.1080/01621459.1972.10481224); [Gneiting and Raftery 2007](https://doi.org/10.1198/016214506000001437)) | The band's two quantiles, as a proper score for a central interval: narrow bands that still hold the outcome score best | When asked for with `scores=` |
| `weighted_interval_score(observed, quantiles, levels)` | The median's absolute error times 1/2 plus each symmetric pair's interval score times α/2, divided by the number of pairs plus 1/2 ([Bracher et al. 2021](https://doi.org/10.1371/journal.pcbi.1008618)); levels must be 0.5 plus pairs symmetric around it. It equals twice `pinball` over the same levels | Every quantile at once, as a proper score that approximates CRPS | When asked for with `scores=` |
| `crps(observed, samples)` | Mean marginal CRPS of the finite empirical ensemble: the expected absolute error of a draw minus half the expected absolute difference between two draws, diagonal pairs included | Each coordinate's whole marginal distribution, as a proper scoring rule; it is unchanged when samples are permuted independently per coordinate, so it cannot see dependence | With a sample output |
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

### Yes/no outcomes

A forecast of a yes/no outcome, such as rain tomorrow, a customer leaving this
month or a machine failing within a day, is a probability, and the outcome is
1 or 0. Put the probability in `Forecast.point`, the predictive mean.

| Function | What it computes | What it scores | Evaluator records it |
| --- | --- | --- | --- |
| `brier(observed, predicted)` | Mean squared distance between the probability and the 0/1 outcome ([Brier 1950](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2)): `mse`, after checking that outcomes are 0 or 1 and forecasts lie in [0, 1] | The probability, as a proper score | When asked for with `scores=` |
| `log_loss(observed, predicted, *, eps=None)` | Mean of minus the natural log of the probability given to what happened ([Good 1952](https://doi.org/10.1111/j.2517-6161.1952.tb00104.x)) | The probability, as a strictly proper score that punishes confident mistakes hard | When asked for with `scores=` |

Both raise `ValueError` naming the first outcome that is not 0 or 1, or the
first forecast outside [0, 1]. A forecast of exactly 0 or 1 that turned out
wrong gives an infinite log loss, so `log_loss` raises `UndefinedMetricError`,
which a report shows as `INCONCLUSIVE` with the reason. Pass `eps` (between 0
and 0.5, such as `1e-6`) to clip every forecast to [eps, 1 − eps] first; nothing
is clipped unless you ask, and the size of `eps` sets how much a confident
mistake costs.

### Your own scores

`evaluate_forecast`, `Pipeline.test`, `Pipeline.run` and
`ForecastRun.from_forecasts` take `scores=`: a list of names of functions in
`timegym.metrics` and functions of your own, or a dictionary from the name to
record to a function. Each value is stored in `case.metrics` under its name,
beside the scores every report records, so `plot_errors(name)`, the saved JSON
and comparisons can use it.

```python
from functools import partial

import numpy as np
from timegym.metrics import log_loss


def within_10_percent(observed, predicted):
    """Share of forecasts within 10% of what happened; higher is better."""
    return float(np.mean(np.abs(predicted - observed) <= 0.1 * np.abs(observed)))


scores = ["rmse", within_10_percent]                                  # by name, or your own
scores = {"log_loss": partial(log_loss, eps=1e-6), "close": within_10_percent}  # with settings
```

A score says what it needs by its argument names: `observed` (what happened),
`predicted` (the point forecast), `quantiles`, `levels`, `samples` and `context`
(the history), each a NumPy array shaped as above, time on the first axis of
`observed` and `context`. An argument with a default is passed only when the
forecast has it; any other argument must have a default, or the request is
refused before anything runs. A request is also refused when its name is
already recorded in every report, such as `"mse"`, or asked for twice: record a
score again with other settings under a new name, such as
`{"mase_12": partial(mase, seasonality=12)}`. A function needs a name of its
own, so give a `lambda` or a `partial` one with a dictionary.

For each case, a score whose required input the forecast lacks is
`UNSUPPORTED`, one that raises `UndefinedMetricError` is `INCONCLUSIVE`, and one
that raises anything else, or returns something other than one finite number,
is an `ERROR` whose message names the exception. Each appears as
`metric_<name>`. The report's metadata lists what was asked for under
`"scores"`: each score's `name`, `function`, keyword `settings` and whether it is
`replayable`. Replay computes again the scores that come from
`timegym.metrics`, with plain settings, and leaves the others out of the
comparison, listing them in `metadata["replay"]["not_recomputed"]`. Nothing is
recorded unless you ask, so reports saved earlier replay exactly as before.

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

`timegym.diagnostics.rank_alarm(observed, calibration, *, alpha=0.05)` asks
whether a number is unusually large next to numbers you know are fine. This
check uses it, and so does
[comparing a scorecard with a saved baseline](scorecard.md#compare-with-a-saved-baseline).
It ranks the observed value among `n` calibration values from something
known to be correct, which gives a Monte Carlo p-value, and returns a frozen
`RankAlarm` with `observed`, `p_value`, `status`, `threshold`, `rank`, `n` and
`alpha`:

```
p_value = (1 + number of calibration values >= observed) / (n + 1)
```

Ties count against the alarm. `status` is `FAIL` when `p_value <= alpha`,
`PASS` when it is larger, and `INCONCLUSIVE` when `1 / (n + 1) > alpha`,
because then no observed value could fail. `threshold` is the calibration
value at position `rank = ceil((n + 1) * (1 - alpha))` in increasing order,
without interpolation, so `observed > threshold` exactly when
`p_value <= alpha`; the rank is computed so that rounding cannot break that
equivalence. When `rank` exceeds `n` there is no such value: `threshold` is
`math.inf` and the status is `INCONCLUSIVE`. If the observed value and the
calibration values are exchangeable, for example independent draws of the
same statistic from the same correct source, `FAIL` happens with probability
at most `alpha`. Large values are the alarming ones. A calibration array that
is empty, not one-dimensional or not finite, a nonfinite observed value, or an
alpha outside (0, 1) raises `ValueError`.

```python
from timegym.diagnostics import rank_alarm

result = rank_alarm(observed, calibration, alpha=0.05)
result.p_value, result.status, result.threshold
```

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
| `metric_<name>` | `capability` | `INCONCLUSIVE`, `UNSUPPORTED`, `ERROR` | None (descriptive) | Simplified data with known results |
| `interval_coverage` | `uncertainty` | `INCONCLUSIVE`, `UNSUPPORTED` | None (descriptive) | Simplified data with known results |
| `reference_covariance` | `distribution` | `PASS`, `FAIL`, `INCONCLUSIVE`, `UNSUPPORTED`, `ERROR` | `alpha`, default 0.05 | Statistical metamorphic test |
| `shift_equivariance` | `metamorphic` | `PASS`, `FAIL`, `UNSUPPORTED`, `ERROR` | `atol` and `rtol` | Known properties |
| `model_execution` | `execution` | `ERROR` | Any exception | Execution record |

Exception evidence is `exception_type`, `exception_message` and `traceback`.
`reference` evidence appears only with a
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

- **Emitted by:** `evaluate_forecast`, only when a score cannot be recorded; `<name>` is `mse`, `mae`, `mase`, `mean_reference_mse`, `pinball`, `interval_width`, `crps`, `energy_score`, `variogram_score`, or the name of a score asked for with `scores=`
- **Evidence keys:** `INCONCLUSIVE`: `reason` (`undefined_metric`). `ERROR`: exception evidence. `UNSUPPORTED`: `needs`, the inputs the requested score needs
- **Threshold or tolerance:** None: descriptive
- **What a FAIL or ERROR usually means:** `ERROR`: the calculation raised or returned a nonfinite value, such as an overflowing interval width, or a requested score raised, such as `brier` on outcomes that are not 0 or 1; the score is missing from `metrics`. `INCONCLUSIVE`: the score is undefined for valid data, such as `mase` on a constant target, `variogram_score` with one coordinate or `log_loss` on a confident mistake without `eps`. `UNSUPPORTED`: a requested score needs an output the forecast lacks, such as samples.

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

::: timegym.metrics.rmse

::: timegym.metrics.rmsse

::: timegym.metrics.mean_reference_mse

::: timegym.metrics.pinball

::: timegym.metrics.interval_score

::: timegym.metrics.weighted_interval_score

::: timegym.metrics.crps

::: timegym.metrics.energy_score

::: timegym.metrics.variogram_score

::: timegym.metrics.brier

::: timegym.metrics.log_loss

::: timegym.metrics.UndefinedMetricError

::: timegym.diagnostics.evaluate_forecast

::: timegym.diagnostics.covariance_dependence_check

::: timegym.diagnostics.rank_alarm

::: timegym.diagnostics.RankAlarm

::: timegym.checks.check_shift_equivariance

::: timegym.statistics.wilson_interval

::: timegym.statistics.paired_t_interval
