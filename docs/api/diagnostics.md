# Scores and diagnostics

**In short.** Scores measure how far a forecast is, both from what actually
happened (`mse`, `mae`, `mase`) and from the answer key (`mean_reference_mse`);
lower is better. Every report records these; more scores, such as `brier` and
`log_loss` for yes/no outcomes, or functions of your own, are recorded when you
ask for them with `scores=`. Output checks of your own, such as a range the
values must stay in, are added with `checks=`. For forecasts of a yes/no outcome,
`calibration` says whether a forecast of 70% came true about 70% of the time. A diagnostic is one check's result: a verdict (`PASS`, `FAIL`,
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
| `mase(observed, predicted, history, seasonality=1)` | Absolute error divided, per target, by the mean absolute `seasonality`-step change of that target in the history (time is axis zero), then averaged | The median, as for `mae`, because each target's scale is fixed by the history | With a point output, always with `seasonality=1` |
| `rmse(observed, predicted)` | Square root of `mse`, in the data's units | The mean, as for `mse` | When asked for with `scores=` |
| `rmsse(observed, predicted, history, seasonality=1)` | Each target's RMSE divided by the root mean squared `seasonality`-step change of that target in the history, then averaged over targets: the scaled error of the M5 competition ([Makridakis, Spiliotis and Assimakopoulos 2022](https://doi.org/10.1016/j.ijforecast.2021.11.013)) | The mean, as for `mse` | When asked for with `scores=` |
| `mean_reference_mse(reference_mean, predicted)` | Mean squared distance from a supplied conditional mean | The conditional mean itself: with a valid conditional mean it is the conditional excess squared-error risk, not realized loss against a future observation | With a point output and a case reference |

`mase` and `rmsse` raise `UndefinedMetricError` when a target's naive-error
scale is zero, as for a constant target, or when the history is no longer than
`seasonality`; they add no epsilon to the denominator and do not drop constant
targets. MAPE and sMAPE are left out on purpose: they are undefined when an
actual value is zero and favour forecasts that are too low
([Hyndman and Koehler 2006](https://doi.org/10.1016/j.ijforecast.2006.03.001));
`mase` and `rmsse` meet the same need.
`evaluate_forecast` always calls `mase` with `seasonality=1`, so the evaluator
scales each target by its lag-1 naive MAE over the history whatever the scenario's
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

Whether a forecaster's chances are honest, so that its forecasts of 70% come
true about 70% of the time, is a question for many forecasts at once:
[`calibration`](#are-the-chances-honest).

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
`predicted` (the point forecast), `quantiles`, `levels`, `samples` and `history`
(the past the model saw), each a NumPy array shaped as above, time on the first
axis of `observed` and `history`. An argument with a default is passed only when
the forecast has it; any other argument must have a default, or the request is
refused before anything runs. An argument named `context`, the old name of
`history`, still gets the history and warns once per run. A request is also
refused when its name is already recorded in every report, such as `"mse"`, or
asked for twice: record a score again with other settings under a new name,
such as `{"mase_12": partial(mase, seasonality=12)}`. A function needs a name
of its own, so give a `lambda` or a `partial` one with a dictionary.

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

## Your own output checks

The output checks every forecast goes through (dates, series, shape, finite
values, quantile order) know nothing about your data: a rainfall forecast of
−3 mm or a chance of rain of 1.2 passes them. `evaluate_forecast`,
`Pipeline.run` and `ForecastRun.from_forecasts` take
`checks=`, a list of functions or a dictionary from name to function, and add
each one to `result.checks` and `assert_passed` beside the built-in checks.

`timegym.checks.within(lower=None, upper=None)` makes the most common one, a
range:

```python
from timegym.checks import within

result = pipeline.run(cases=cases, checks=[within(0, 1)])
result.checks            # now also "Values between 0 and 1"
result.assert_passed()   # raises when a value falls outside 0 to 1
```

It reads the point forecast, the quantiles and the samples, whichever the
forecast has. A value equal to a bound passes, and `None` leaves a side open:
`within(0, None)` for rainfall or sales. A dictionary from series name to
bound gives each series its own range, and a series it leaves out is open on
that side:

```python
within(lower={"rain_mm": 0, "humidity": 0}, upper={"humidity": 100})
```

A failed range check names each output and series out of range, how many
values and the most extreme one, such as `quantiles: 7 values below 0 (lowest
-3.088)`. A dictionary that names a series the forecast doesn't have is an
`ERROR`, so a typo can't switch the check off. Its name in a report is
`within`; to use two, name them with a dictionary.

A check of your own is a function that asks for its inputs by argument name,
as a score does (`observed`, `predicted`, `quantiles`, `levels`, `samples`,
`history`), and may also ask for `target_names`, the names of the series in
order. It returns `True` when the forecast passes, or a message saying what is
wrong. Its title, shown in `report.checks` and in the error of
`assert_passed`, is the first line of its docstring without a final full stop,
or its name when it has no docstring:

```python
import numpy as np


def totals_add_up(predicted, target_names):
    """Store totals add up."""
    north, south, total = (predicted[:, target_names.index(name)] for name in ("north", "south", "total"))
    gap = float(np.max(np.abs(north + south - total)))
    return gap <= 1e-6 or f"the total is off by up to {gap:.3g}"


result = pipeline.run(cases=cases, checks=[within(0, None), totals_add_up])
```

A check runs only on a forecast that passed the built-in checks. Its finding
is `check_<name>`: `PASS`; `FAIL` with the message, or when the function
returns `False`; `ERROR` when it raises or returns anything else, such as
`None`; or `UNSUPPORTED` when it needs an output the forecast lacks, and then
it isn't counted for that case. A request is refused before anything runs when
a function has no name of its own (give a `lambda` one with a dictionary),
takes the name of a built-in check such as `runs`, is asked for twice, or needs
an argument TimeGym cannot give. Each finding keeps the check's title in its
evidence, so a saved report lists and counts the checks without your code. The
report's metadata lists what was asked for under `"checks"`, as for scores.
Replay runs `within` checks again and leaves your own functions out of the
comparison, listing them in `metadata["replay"]["checks_not_rerun"]`. Without
`checks=`, reports are exactly as before.

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

`timegym.checks.check_shift_equivariance(pipeline, history, future_index, *,
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
| `check_<name>` | `correctness` | `PASS`, `FAIL`, `UNSUPPORTED`, `ERROR` | Set by the check, such as the bounds of `within` | Partial oracle |
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

### `check_<name>`

- **Emitted by:** `evaluate_forecast`, for each check asked for with `checks=`; `<name>` is its name, such as `within`
- **Evidence keys:** `title`, the check's title. `UNSUPPORTED`: also `needs`, the inputs the check needs. `ERROR`: also exception evidence when the check raised
- **Threshold or tolerance:** Set by the check, such as the bounds of `within`
- **What a FAIL or ERROR usually means:** `FAIL`: the forecast breaks your rule, such as a value outside the range of `within`; the message says where. `ERROR`: the check raised or returned something other than True or a message, or `within` names a series the forecast doesn't have. `UNSUPPORTED`: the check needs an output the forecast lacks.

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

## Compare two forecasters on the same cases

`timegym.compare_reports(candidate, baseline, *, metric, by=None,
cluster=None, alpha=0.05, higher_is_better=False, seed=0)` says whether a
candidate forecaster is better or worse than a baseline on the same cases, or
shows no clear difference. Both reports must come from the same cases, in the
same order: it pairs them by position and refuses two reports whose case
manifests differ, naming the first difference. Name the model in the report's
metadata, not in the case manifest. `metric` is any score in `case.metrics`,
your own included; lower is better unless `higher_is_better=True`.

```python
import numpy as np
import pandas as pd
from timegym import LastValueForecaster, NaiveForecaster, Pipeline, cases_from_series, compare_reports

rng = np.random.default_rng(0)
days = pd.date_range("2024-01-01", periods=140, freq="D")
cases = []
for store in range(16):
    weekly = 10 * np.sin(2 * np.pi * np.arange(140) / 7 + store)
    sales = pd.Series(40 + weekly + rng.normal(0, 3, 140), index=days, name="sales")
    cases += cases_from_series(sales, origins=[84, 98, 112, 126], horizon=7, name="sales",
                               manifest={"store": f"store {store}", "region": ["north", "south"][store % 2]})

baseline = Pipeline(LastValueForecaster()).run(cases=cases)
candidate = Pipeline(NaiveForecaster("seasonal", period=7)).run(cases=cases)
diff = compare_reports(candidate, baseline, metric="mae", by="region", cluster="store")
print(diff)
diff.assert_no_regression()
```

```text
mae against the baseline (lower is better; alpha 0.05; cases in one store move together)
  region north  better               3.52 against 9.15 (-62%) over 32 cases in 8 clusters, p = 0.017
  region south  better               3.44 against 9.08 (-62%) over 32 cases in 8 clusters, p = 0.017
```

`diff.table` has one row per candidate and group: `candidate`, `group` (the
`by` value, or `"all cases"`), `cases` scored in both reports and `clusters`
among them, `baseline_mean`, `candidate_mean`, `mean_difference` (candidate
minus baseline), `p_value`, `verdict`, and the cases that are `newly_failing`
or `newly_passing`. `diff.better`, `diff.worse` and `diff.too_few` hold those
rows. `assert_no_regression()` raises when a row is worse or too few to tell.

- **Groups.** `by` names a manifest key, such as `"region"`; each value gets a
  row, and Holm's correction runs across the groups.
- **Cases that move together.** `cluster` names a manifest key, such as
  `"store"`, whose cases share their ups and downs, as cases cut from one
  series do. All cases of a cluster count as one piece of evidence. Leaving it
  out when cases do move together makes chance differences look real far more
  often than `alpha`.
- **Too few to tell.** With `k` clusters (or cases, without `cluster`) that
  differ, the smallest possible p-value is `2 / 2**k`, so with five or fewer
  nothing can reach 0.05. Such a row says `"too few to tell"` and fails
  `assert_no_regression()`, rather than pass a test it could never fail. With
  several groups, Holm's correction needs more clusters per group: with two
  regions of six stores each, the example above would show no clear
  difference even though every store improved.
- **Crashes.** A case the baseline scored that crashes or fails an output
  check in the candidate makes its row `"worse"`. A case the candidate scores
  and the baseline did not is counted in `newly_passing` and left out of the
  test.
- **Roundoff.** Two scores of a case that differ by no more than `1e-9`, or
  `1e-8` times the largest score in the row, count as the same, as in a
  scorecard's comparison. Two reports with the same scores show no clear
  difference, with a p-value of 1.

### Many candidates against one baseline

Pass a list of reports, labelled by the model each report's metadata names, or
a dictionary of named reports. Each candidate's p-value is adjusted for how
many were tried, so the best of many does not beat the baseline by luck.

```python
candidates = {
    "same weekday last week": candidate,
    "mean so far": Pipeline(NaiveForecaster("mean")).run(cases=cases),
    "straight line": Pipeline(NaiveForecaster("drift")).run(cases=cases),
}
print(compare_reports(candidates, baseline, metric="mae", cluster="store"))
```

```text
mae against the baseline (lower is better; alpha 0.05; cases in one store move together)
  same weekday last week  better               3.48 against 9.11 (-62%) over 64 cases in 16 clusters, p = 0.0001
  mean so far             better               6.54 against 9.11 (-28%) over 64 cases in 16 clusters, p = 0.0004
  straight line           no clear difference  9.2 against 9.11 (+0.91%) over 64 cases in 16 clusters, p = 0.074
```

### How the test works

For each pair of scores, the difference is candidate minus baseline. If the
two forecasters were alike, each difference, or each cluster's sum of
differences, would be as likely to be positive as negative. The test flips
their signs at random 9,999 times, seeded from `seed` and the number of
clusters (or cases), and the p-value is the share of sign patterns, the observed one
included, whose mean difference is at least as far from zero as the observed
one. It is two-sided, so the verdict can go either way. With `cluster`, all
cases of a cluster flip together ([Canay, Romano and Shaikh
2017](https://doi.org/10.3982/ECTA13081)). With several candidates, one set
of sign flips is applied to every candidate and group. Each row's statistic is
divided by the square root of its sum of squared differences, which no flip
changes, so a group with larger errors does not dominate. Each row's p-value
is then the share of sign patterns whose largest statistic, over all rows, is
at least that row's observed one ([White
2000](https://doi.org/10.1111/1468-0262.00152); [Romano and Wolf
2005](https://doi.org/10.1111/j.1468-0262.2005.00615.x)). Rows where too few
clusters differ are left out of both corrections.

The test assumes that different clusters are independent and that, when the
two forecasters are alike, a cluster's difference is as likely to be positive
as negative. It says how these two forecasters did on these cases; whether
the difference holds on other cases depends on whether these are a fair
sample of them. A scorecard's comparison with a saved baseline uses the same
sign flips and Holm's correction, one-sided, because it asks only whether the
new version is worse.

## Are the chances honest?

When a forecast of a yes/no outcome says 70%, it should happen about 70% of
the time. `timegym.diagnostics.calibration(run, *, bins=10, cluster=None,
level=0.95, seed=0)` checks that on a run of such forecasts (made with `keep_data=True`, since it
reads every forecast and what happened), and
`calibration(observed, predicted, ...)` on arrays of outcomes (0 or 1) and
chances. It returns a `Calibration`:

- `table`: one row per bin of equal width, from `low` to `high`, with the
  number of `forecasts` in it, their `mean_forecast` and the share that
  `happened`. A calibrated forecaster's last two columns agree.
- `slope`, from a logistic regression of what happened on the log-odds of the
  forecast ([Cox 1958](https://doi.org/10.1093/biomet/45.3-4.562)): 1 is
  calibrated; below 1 the forecasts swing further from the average than what
  happened supports (too confident), above 1 not far enough (too timid).
  `slope_interval` is its bootstrap interval at `level`.
- `intercept`, from the same regression with the slope held at 1: 0 is right
  on average, below 0 the forecasts are too high, above 0 too low
  ([Van Calster et al. 2019](https://doi.org/10.1186/s12916-019-1466-7)).
- `plot()`: the table as a reliability diagram
  ([Murphy and Winkler 1977](https://doi.org/10.2307/2346866)), with the
  number of forecasts in each bin below it.

Here twenty weather stations record rain (1) or none (0), with a chance that
rises and falls over a 12-step cycle, and five cases are cut from each. The
forecaster gives the share of rainy steps so far at the same point of the
cycle, after adding one rainy and one dry step to each count:

```python
import numpy as np
import pandas as pd
from timegym import Pipeline, cases_from_series
from timegym.diagnostics import calibration

rng = np.random.default_rng(5)
chance = 0.5 + 0.37 * np.sin(2 * np.pi * np.arange(240) / 12)  # rises and falls over 12 steps
cases = []
for station in range(20):
    rain = pd.Series((rng.random(240) < chance).astype(float), name="rain")
    cases += cases_from_series(rain, origins=[120, 144, 168, 192, 216], horizon=24, name="rain",
                               manifest={"station": station})


class ShareAtThisPointOfTheCycle:
    """The share of rainy steps so far at this point of the cycle, after adding one rainy and one dry step."""

    def fit(self, history):
        values = history.to_numpy()[:, 0]
        self.length_ = len(values)
        self.share_ = np.array([(values[phase::12].sum() + 1) / (len(values[phase::12]) + 2)
                                for phase in range(12)])
        return self

    def predict(self, future_index):
        return self.share_[np.arange(self.length_, self.length_ + len(future_index)) % 12]


cal = calibration(Pipeline(ShareAtThisPointOfTheCycle()).run(cases=cases, keep_data=True), cluster="station")
print(cal)
```

```text
Calibration of 2,400 forecasts from 100 cases in 20 clusters by station
  slope 0.92 (95% interval 0.86 to 1.01): 1 is calibrated, below 1 too confident, above 1 too timid
  intercept +0.00: 0 is right on average, below 0 too high, above 0 too low
  forecast      mean  happened  forecasts
  0.00 to 0.10  0.07      0.20         60
  0.10 to 0.20  0.15      0.16        260
  0.20 to 0.30  0.25      0.25        274
  0.30 to 0.40  0.34      0.32        272
  0.40 to 0.50  0.43      0.41        182
  0.50 to 0.60  0.53      0.53        370
  0.60 to 0.70  0.65      0.68        294
  0.70 to 0.80  0.74      0.75        314
  0.80 to 0.90  0.83      0.79        288
  0.90 to 1.00  0.93      0.93         86
```

The slope is 0.92 and its interval reaches 1.01, so this forecaster is close
to calibrated, perhaps a little too sure of itself. The five cases of a
station share its history, so `cluster="station"` resamples them together.

**Which forecasts it reads.** From a run, the point forecast is the chance, or
the mean of the sample paths when there is no point forecast. Cases that
crashed, failed a built-in check, or have neither are left out and counted in
`skipped`. Each case is resampled whole for the interval, since its steps
share one history; `cluster=` names a manifest key whose cases are resampled
together. With arrays, each value is resampled on its own, or `cluster` gives
one label per value.

**Forecasts of exactly 0 or 1.** Their log-odds are infinite, so the slope and
the intercept read them as 0.000001 and 0.999999, and `certain` counts them. A
sure forecast that turns out wrong then pulls the slope far toward 0. Without
the added steps, the forecaster above says 0 or 1 exactly 130 times and its
slope drops to 0.46. That is the right direction, since a sure forecast that
fails was too sure, but the size of the drop depends on how 0 and 1 are read:
read the table beside the slope.

**When there is no number.** With every outcome the same there is no slope or
intercept; with every forecast the same, no slope; when the forecasts separate
the outcomes perfectly, the slope has no finite value. `note` says which, and
the value is NaN. A slope near 0 next to forecasts that stay near the average
means they carry no information, not that they are too sure: the share of rainy
steps so far, `NaiveForecaster("mean")`, forecasts between 40% and 60% on these
cases and has a slope of −0.14.

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

::: timegym.compare_reports

::: timegym.diagnostics.ReportComparison

::: timegym.diagnostics.calibration

::: timegym.diagnostics.Calibration

::: timegym.plotting.plot_calibration

::: timegym.diagnostics.covariance_dependence_check

::: timegym.diagnostics.rank_alarm

::: timegym.diagnostics.RankAlarm

::: timegym.checks.check_shift_equivariance

::: timegym.checks.within

::: timegym.statistics.wilson_interval

::: timegym.statistics.paired_t_interval
