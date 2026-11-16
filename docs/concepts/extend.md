# Extend TimeGym

When TimeGym lacks the exact tool you need, you can usually plug in your own.
This page lists every place that takes your own code or data, each with a short
example and a link to its full description. The examples run in order, as one
script.

| To bring your own | Write or call | Full description |
| --- | --- | --- |
| Model | an object with `fit(context)` and `predict(future_index)` | [Pipeline and reports](../api/pipeline.md) |
| scikit-learn regressor | `Pipeline(Ridge(), window_length=12)` | [Models and reductions](../api/models.md) |
| Target transform | an object with `clone_unfitted()`, `fit_transform(context)` and `inverse_transform(forecast)` | [Data preparation](../api/data-preparation.md#reversible-target-transforms) |
| Data | `cases_from_series(series, origins, horizon)` or `Case(...)`, then `pipeline.test(cases=...)` | [Tasks and forecasts](../api/tasks-and-forecasts.md#your-own-data) |
| Forecasts made elsewhere | `ForecastRun.from_forecasts(cases, forecasts)` | [Pipeline and reports](../api/pipeline.md) |
| Answer key from a curve you know | `Reference.from_function(curve, dates, covariance=...)` | [Scenarios and references](../api/scenarios.md) |
| Kind of series, with its answer key | an object with `kind` and `generate(seed)` that returns a `Case` | this page |

## Your model

A model is any object with two methods. `fit(context)` gets the history, a
DataFrame with one column per series. `predict(future_index)` gets the dates to
forecast and returns one value per date: a list, a NumPy array, a pandas Series
or DataFrame, or a `Forecast`. TimeGym fits a fresh copy for every case, so
nothing learned on one case leaks into the next.

```python
import numpy as np
import pandas as pd
from timegym import Pipeline


class RepeatLast:
    def fit(self, context):
        self.last = context.iloc[-1].to_numpy()   # the last row seen
        return self

    def predict(self, future_index):
        return np.tile(self.last, (len(future_index), 1))


report = Pipeline(RepeatLast()).test()
```

[Write and test a custom model](../../tutorials/02_custom_model.ipynb) adds
quantiles, several series and a model with settings.

## A scikit-learn regressor

A regressor learns to predict each value from the `window_length` values before
it. `.add()` puts preprocessing steps in front of it; each is fitted on the
history of each case only.

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

report = Pipeline(window_length=12).add(StandardScaler()).add(Ridge()).test()
```

## A target transform

A transform changes the scale the model learns on, then turns the forecast
back. TimeGym fits a fresh copy on each history. This one divides counts by 100
before fitting and multiplies every part of the forecast back:

```python
from dataclasses import replace


class PerHundred:
    def clone_unfitted(self):
        return PerHundred()

    def fit_transform(self, context):
        return context / 100

    def inverse_transform(self, forecast):
        def back(values):
            return None if values is None else np.asarray(values) * 100
        return replace(forecast, point=back(forecast.point),
                       quantiles=back(forecast.quantiles), samples=back(forecast.samples))


report = Pipeline(RepeatLast(), target_transform=PerHundred()).test()
```

## Your own data

Cut a series into cases that end their history at different times, then test
any model on them. There is no answer key for your own data, so each forecast is
scored against what happened.

```python
from timegym import LastValueForecaster, cases_from_series

days = pd.date_range("2024-01-01", periods=120, freq="D")
rain = pd.Series(np.random.default_rng(0).gamma(2.0, 2.0, 120), index=days, name="rain_mm")
cases = cases_from_series(rain, origins=[60, 80, 100], horizon=7, name="rain")
report = Pipeline(LastValueForecaster()).test(cases=cases)
```

To prepare readings taken at irregular times, see
[Irregular readings](../api/data-preparation.md#irregular-readings).

## Forecasts made elsewhere

Forecasts from a model TimeGym cannot run, made earlier or by another program,
get the same checks and scores. Pass one per case, in order; `None` marks a case
with no forecast.

```python
from timegym import ForecastRun

made_elsewhere = [np.full(7, case.task.context.iloc[-1, 0]) for case in cases]
run = ForecastRun.from_forecasts(cases, made_elsewhere, metadata={"model": "weather service"})
run.report.assert_passed()
```

## An answer key from a curve you know

When you know the truth from outside the data, for example a curve you chose
plus noise you add yourself, give the case an answer key. The covariance says
how far the series may stray from the curve.

```python
from timegym import Reference

dates = pd.RangeIndex(48, 60)
key = Reference.from_function(lambda index: (2.0 + 0.1 * index.to_numpy())[:, None], dates,
                              covariance=0.25 * np.eye(len(dates)))
```

## A kind of series, with its answer key

`pipeline.test` accepts any object with a `kind` and a `generate(seed)` method
that returns a `Case`, next to TimeGym's own scenarios. Put `kind` and `seed` in
the manifest, so the report can name each case.

```python
from timegym import Case, ForecastTask


class NoisyLine:
    """A straight line, 2 + 0.1 per step, plus noise with standard deviation 0.5."""

    kind = "noisy_line"

    def generate(self, seed=None):
        steps = pd.RangeIndex(0, 60)
        line = 2.0 + 0.1 * steps.to_numpy()
        noise = np.random.default_rng(seed).normal(0.0, 0.5, len(steps))
        values = pd.DataFrame({"y": line + noise}, index=steps)
        history, future = values.iloc[:48], values.iloc[48:]
        key = Reference.from_function(lambda index: (2.0 + 0.1 * index.to_numpy())[:, None],
                                      future.index, covariance=0.25 * np.eye(len(future)))
        return Case(ForecastTask(history, future.index), future, key,
                    {"kind": self.kind, "seed": seed})


report = Pipeline(RepeatLast()).test([NoisyLine()], seeds=[0, 1, 2])
```

With an answer key, the report has the error against it
(`mean_reference_mse`), which separates a weak model from an unlucky one.
Replay rebuilds only TimeGym's own scenarios, so keep this code with the
report.
