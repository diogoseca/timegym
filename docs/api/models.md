# Models and reductions

Built-in forecasters and the sklearn reduction: `NaiveForecaster` and
`GaussianAR1Forecaster` from `timegym`; `SklearnForecaster`, `REDUCTIONS`,
`fresh_model` and `is_sklearn_regressor` from `timegym.models`.

`NaiveForecaster(strategy="last", period=1)` returns explicit point forecasts.
`last` repeats the final observation and `mean` repeats the context mean; both
need at least one observation. `seasonal` repeats the final `period` observations
and requires at least that many rows. `drift` extrapolates the slope from the
first to the last observation and needs at least two rows. `period` is a positive
integer. With two or more context rows, built-in step models require a regular
index and immediately following consecutive future labels. With one row, the
naive models can only require increasing future labels after the context.

`GaussianAR1Forecaster(n_samples=256, seed=0, levels=DEFAULT_LEVELS)` requires at
least five context observations. It fits an intercept and lag coefficient by
OLS per target and pools residual cross-target covariance with `n - 3` degrees
of freedom for `n` context rows. Its `Forecast` carries the iterated fitted
conditional mean, paired Gaussian trajectories of shape `(n_samples, H, D)`, and
empirical marginal quantiles at the requested levels (by default 0.1, 0.5, 0.9).
At least two samples are required. A nonnegative integer seed repeats the draws;
`seed=None` uses fresh randomness. Parameter-estimation uncertainty is omitted;
this fitted model is not the known-parameter reference.

`timegym.models.SklearnForecaster(steps, window_length=12, reduction="recursive",
horizon=None)` reduces a cloned sklearn pipeline to multi-step forecasts. Each
input flattens `window_length` consecutive rows across all targets, oldest row
first; transforms fit only those training windows. `timegym.models.REDUCTIONS`
is `("recursive", "direct", "multioutput")`. `recursive` fits one one-step
regressor per target and appends each predicted target vector to the window.
`direct` fits one regressor per horizon step and target, each predicting from
the last observed window. `multioutput` fits one regressor per target for the
whole horizon vector, wrapping the final step in `MultiOutputRegressor` when it
lacks native multioutput support. `direct` and `multioutput` need a positive
integer `horizon`, forecast at most that many steps, and need at least
`window_length + horizon` context rows; `recursive` needs `window_length + 1`.
All three return point outputs only. `steps` must end in an sklearn regressor. `fresh_model(model)` uses
`clone_unfitted()` for native instances or sklearn's configuration clone;
it rejects native instances without that protocol and clones returning the
original object. `is_sklearn_regressor(model)` reads sklearn tags lazily after
checking for `get_params`. Install `timegym[sklearn]` when using sklearn objects.
