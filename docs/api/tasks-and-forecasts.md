# Tasks and forecasts

**In short.** Your model's `fit` receives the past as a pandas DataFrame, one row
per time step and one column per series. Its `predict` receives the time labels
to forecast, which continue where the past ends (after steps 0 to 119, the next
12 are 120 to 131), and returns one value per label: a list, a NumPy array, a
pandas Series or DataFrame, or a `Forecast`. A list or array has no labels of
its own, so TimeGym reads its rows as the requested steps in order and its
columns as the history's series. Before scoring, TimeGym runs the output checks
(the *forecast contract*): right dates, right series in the right order, the
right shape, no NaN or infinity, and quantiles that do not cross. A forecast
that fails one fails the test with a named reason and is not scored.

`ForecastTask(history, future_index)` validates a finite pandas DataFrame, unique
ordered target names, increasing unique time labels, and a future index after the
history. The history is all that ordinary models may use to fit. Future targets and
privileged reference parameters remain with the evaluator.

`Forecast(index, point=None, quantiles=None, levels=(), samples=None,
target_names=None)` accepts unvalidated outputs so failures can be reported.
`target_names=None` means the task's series, in order, so a model only names
them when it wants that order checked. Call `forecast.validate(task)` to obtain
stable code/message dictionaries.

```python
from timegym import Forecast

class RepeatLast:
    def fit(self, history):
        self.last = history.iloc[-1]   # the last row seen
        return self

    def predict(self, future_index):
        return Forecast(future_index, [self.last] * len(future_index))
```

When a model's output reaches the pipeline, a one-dimensional `point` for a
single series is read as one value per step. Outside the pipeline,
`validate` still reports it as `point_shape`.

| Output | Shape | Meaning |
| --- | --- | --- |
| point | H × D | Predictive mean |
| quantiles | H × D × Q | Marginal quantiles at strictly increasing levels in (0,1) |
| samples | S × H × D | Complete joint trajectories, preserving horizon/target pairing |

At least one output is required. Missing output types receive `UNSUPPORTED` when
an evaluation needs them. Shapes, finite values, index alignment, target order and
quantile monotonicity are checked before any score is calculated.

## Your own data

`cases_from_series(series, origins, horizon, *, name="series", manifest=None)`
cuts one series, or a table of series, into cases that end their history at
different times. `series` is a pandas Series or a DataFrame with one column per
series and an increasing, unique index; an unnamed Series is named `name`.
Each entry of `origins` says where one history ends:

- a whole number is the number of rows of history: `60` keeps the first 60 rows;
- a number between 0 and 1 is a share of the rows, rounded: `0.75` keeps three quarters;
- anything else is an index label, and the history holds every row at or before
  it: `"2024-04-10"` on a daily index ends the history on 10 April.

`horizon` is the number of rows to forecast after each origin, or `"rest"` for
every row that follows. Each case's history stops at its origin, and the rows
after it are kept as what actually happened (`case.observed`), so no forecast
sees the future it is scored on. The cases have no answer key
(`reference=None`), so they are scored against what happened, and the checks
that need a key say `UNSUPPORTED`.

```python
import numpy as np
import pandas as pd
from timegym import LastValueForecaster, Pipeline, cases_from_series

days = pd.date_range("2024-01-01", periods=120, freq="D")
rain = pd.Series(np.random.default_rng(0).gamma(2.0, 2.0, 120), index=days, name="rain_mm")
cases = cases_from_series(rain, origins=[60, 80, 100], horizon=7, name="rain")
result = Pipeline(LastValueForecaster()).run(cases=cases)
```

Each case's manifest records `kind` (`name`), `seed` (`None`), `origin` (the
last history label, as text), `history_length`, `horizon`, `target_names` and
`series_sha256`, a SHA-256 fingerprint of the whole series: its column names,
its index labels as text and its values as 64-bit floats. Two cases cut from
the same data have the same fingerprint. The keys of `manifest` are added to
every case, for example `{"station": "north"}`, and must not reuse those
names. Missing or infinite values in a case's history or future are refused,
with the origin named; [Data preparation](data-preparation.md#irregular-readings)
shows how to put irregular readings on a regular grid first.

You can also build a `Case(task, observed, reference, manifest)` yourself. For
your own data the manifest can be any JSON-safe dictionary: without `kind` the
case is labelled `"case"`, and without `seed` it has none. Only a manifest
written by a TimeGym scenario, the one with `schema_version`, must keep both,
because replay rebuilds the case from it.

## Signatures

::: timegym.ForecastTask

::: timegym.Forecast

::: timegym.cases_from_series
