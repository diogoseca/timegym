# Tasks and forecasts

What a model receives and what it must return: `ForecastTask` and `Forecast`,
imported from `timegym`.

`ForecastTask(context, future_index)` validates a finite pandas DataFrame, unique
ordered target names, increasing unique time labels, and a future index after the
context. Context is all that ordinary models may use to fit. Future targets and
privileged reference parameters remain with the evaluator.

`Forecast(index, point=None, quantiles=None, levels=(), samples=None,
target_names=("y",))` accepts unvalidated outputs so failures can be reported.
Call `forecast.validate(task)` to obtain stable code/message dictionaries.

| Output | Shape | Meaning |
| --- | --- | --- |
| point | H × D | Predictive mean |
| quantiles | H × D × Q | Marginal quantiles at strictly increasing levels in (0,1) |
| samples | S × H × D | Complete joint trajectories, preserving horizon/target pairing |

At least one output is required. Missing output types receive `UNSUPPORTED` when
an evaluation needs them. Shapes, finite values, index alignment, target order and
quantile monotonicity are checked before any score is calculated.
