# Models and reductions

Built-in forecasters and the sklearn reduction: `LastValueForecaster`,
`SeasonalLastValueForecaster`, `MeanForecaster`, `SeasonalMeanForecaster`,
`DriftForecaster` and `GaussianAR1Forecaster` from `timegym`; `SklearnForecaster`
and `REDUCTIONS` from `timegym.models`.

Five one-line baselines return point forecasts, one class each, named for the
forecast it makes:

- `LastValueForecaster()` repeats each series' last value at every step (the
  naive forecast). It is the baseline the README and the quick start score.
- `SeasonalLastValueForecaster(period)` repeats the last `period` values, so
  each step gets the last value seen at the same point of the cycle (seasonal
  naive). It needs at least `period` rows.
- `MeanForecaster()` repeats the mean of the whole history (the mean method).
- `SeasonalMeanForecaster(period)` forecasts each step with the mean of all past
  values at the same point of the cycle (for yes/no values, the share that
  happened there). It needs at least `period` rows.
- `DriftForecaster()` extends the straight line from the first to the last
  observation (the drift method) and needs at least two rows.

Only the two seasonal ones take a `period`, a positive integer. The others have
nothing to set, so reports and scorecards show them by name
(*LastValueForecaster: 51/100*). Every one replays from a report. With two or
more rows of history, built-in step models require a regular index and
immediately following consecutive future labels. With one row, the baselines
can only require increasing future labels after the history.

`NaiveForecaster(strategy="last", period=1)` is the older spelling of the five,
chosen by `strategy` (`"last"`, `"mean"`, `"seasonal"`, `"seasonal_mean"` or
`"drift"`). It makes the same forecasts as the class for its strategy and still
works, with a warning that names that class; it will be removed in a later
release. Reports that name it still replay.

`GaussianAR1Forecaster(n_samples=256, seed=0, levels=DEFAULT_LEVELS)` requires at
least five observations of history. It fits an intercept and lag coefficient by
OLS per target and pools residual cross-target covariance with `n - 3` degrees
of freedom for `n` rows of history. Its `Forecast` carries the iterated fitted
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
`window_length + horizon` rows of history; `recursive` needs `window_length + 1`.
All three return point outputs only. `steps` must end in an sklearn regressor. Before each case a pipeline copies
the model with `clone_unfitted()` for native instances, sklearn's configuration clone for
estimators, and otherwise a deep copy, so a plain class with `fit` and
`predict` can be passed as an instance; the original is never fitted. A
`clone_unfitted()` that returns the original object is rejected. Whether a model is an sklearn regressor
is read from its sklearn tags after checking for `get_params`. scikit-learn is installed
with TimeGym and imported only when an sklearn object is used.

## Signatures

::: timegym.LastValueForecaster

::: timegym.SeasonalLastValueForecaster

::: timegym.MeanForecaster

::: timegym.SeasonalMeanForecaster

::: timegym.DriftForecaster

::: timegym.GaussianAR1Forecaster

::: timegym.models.SklearnForecaster
