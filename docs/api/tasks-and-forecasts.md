# Tasks and forecasts

**In short.** Your model's `fit` receives the past as a pandas DataFrame, one row
per time step and one column per series. Its `predict` receives the time labels
to forecast, which continue where the past ends (after steps 0 to 119, the next
12 are 120 to 131), and returns one value per label: a list, a NumPy array, a
pandas Series or DataFrame, or a `Forecast`. A list or array has no labels of
its own, so TimeGym reads its rows as the requested steps in order and its
columns as the context's series. Before scoring, TimeGym runs the output checks
(the *forecast contract*): right dates, right series in the right order, the
right shape, no NaN or infinity, and quantiles that do not cross. A forecast
that fails one fails the test with a named reason and is not scored.

`ForecastTask(context, future_index)` validates a finite pandas DataFrame, unique
ordered target names, increasing unique time labels, and a future index after the
context. Context is all that ordinary models may use to fit. Future targets and
privileged reference parameters remain with the evaluator.

`Forecast(index, point=None, quantiles=None, levels=(), samples=None,
target_names=None)` accepts unvalidated outputs so failures can be reported.
`target_names=None` means the task's series, in order, so a model only names
them when it wants that order checked. Call `forecast.validate(task)` to obtain
stable code/message dictionaries.

```python
from timegym import Forecast

class RepeatLast:
    def fit(self, context):
        self.last = context.iloc[-1]   # the last row seen
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

## Signatures

::: timegym.ForecastTask

::: timegym.Forecast
