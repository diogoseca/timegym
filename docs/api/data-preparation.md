# Data preparation

Smoothing, feature and target helpers, and the reversible target transforms that
a `Pipeline` can fit and invert.

The functions below are public through `timegym.features`, `timegym.target`, and
`timegym.smoothing`; import them from those modules. They are standalone pandas
transformations, not automatic Pipeline stages. They never mutate the supplied
object. Row offsets are positional, so a lag of one means the previous row even
when timestamps are irregular. Apply transformations within each training split
and keep future labels out of predictor matrices. `SklearnForecaster` constructs
its own lag windows; these helpers support explicit notebook experiments and
custom models. The reversible target transforms below are the exception: a
`Pipeline` fits and inverts them on each case.

## Smoothing

`simple_moving_average(df, window_length=30, min_periods=1)` and
`exponential_moving_average(df, exponential_alpha=None, statistic="mean", *,
min_periods=1)` both return the same Series/DataFrame type, index, column order
and labels as the input, containing only smoothed values for **every column**.

Both means use the current and earlier rows. Shift the result by one row to use
it for predicting the current observation. SMA uses a trailing `window_length`
(positive integer); `min_periods` is a nonnegative integer no larger than that
window. EMA requires finite `exponential_alpha` in `(0, 1]` and supports only
`statistic="mean"`. It normalizes geometric weights (`adjust=True`) and uses
absolute row positions through gaps (`ignore_na=False`). Both skip missing
observations when averaging and return NaN until each column meets
`min_periods`. These are historical transformations, not future forecasts.

## Features

| Function | Contract |
| --- | --- |
| `add_lagged_target(df, lags=10, dropna=True)` | Append `{column}_t-1` through `{column}_t-{lags}` by source column; zero lags adds no columns. Retain originals; exclude a contemporaneous target from predictors. |
| `cyclical_encoding(values, scaled=False, min_value=None, max_value=None, *, period=None)` | Return cosine/sine features of finite real values using a fixed period and phase. Prefer `period=7, min_value=0` for weekdays. Explicit bounds are inclusive discrete labels; `scaled=True` accepts cycle fractions and forbids period/bounds. No sample-dependent scaling. |
| `add_time_features(df)` | Require a DatetimeIndex without NaT. Append a fixed calendar schema: elapsed UTC days since the Unix epoch, local year and cosine/sine pairs for quarter, month, day, weekday, hour, minute, second, microsecond and month-relative day. |
| `add_tsfel_features(df, use_cols=None, window_size=100, domain=None, frequency=100, dropna=True, show_progress=False, lag=1)` | Optional `timegym[features]` helper. Append TSFEL features to selected original columns. At row `t`, the window ends at `t-lag`; default `lag=1` excludes the current row. `frequency` is the sampling frequency in Hz. |

Frame helpers require unique column labels and reject generated-name collisions
instead of overwriting observations. `use_cols` is a name or sequence of names;
omitting it selects every column. For lagged and TSFEL features, `dropna=True`
removes rows missing any returned value, including original values. `False`
retains the original index and missing warmup features. TSFEL needs at least
`window_size + lag` rows to produce one feature row, with positive window size
and nonnegative lag. Optional extraction and progress imports happen only when
requested; core imports do not require TSFEL or tqdm.

## Targets and differences

| Function | Contract |
| --- | --- |
| `multioutput_target(df, horizon=10, dropna=True)` | Append `{column}_t+1` through `{column}_t+{horizon}`, by source column. Positive integer horizon. |
| `onestep_target(df, dropna=True)` | The same target construction with horizon one. |
| `step_ahead_target(df, step=5, dropna=True)` | Append only `{column}_t+{step}` for each original column. Positive integer step. |
| `fractional_weights(d, lags)` | Return a `(lags, 1)` array of the first binomial coefficients of `(1-B)^d`, newest first. Finite real `d`; positive integer coefficient count. |
| `fractional_differencing(series, order, fractional_coefficients, *, lag=1)` | Apply a finite truncated `(1-B^lag)^order` filter independently to each column. Trim exactly `(fractional_coefficients-1)*lag` warmup rows; order zero returns an unchanged copy. |
| `differencing(series, order=1, fractional_coefficients=10, *, lag=1)` | Integer order means repeated differences, not lag distance: order two is `y[t]-2*y[t-lag]+y[t-2*lag]`. Trim `order*lag` rows. Noninteger order delegates to the finite fractional filter. |

Target constructors retain originals, reject duplicate or colliding columns, and
return a new DataFrame. `dropna=True` removes any row with a missing original or
future value; `False` retains the index and incomplete tail. Future columns are
labels, never inputs available at the forecast origin. These constructors are
not model adapters; `SklearnForecaster` implements the direct and multioutput
reductions.

Differencing accepts a Series or DataFrame and preserves its type and labels on
the retained rows. Orders must be finite and nonnegative, and lag must be a
positive integer. Missing observations inside the series propagate; they are
not filled with zero. `fractional_coefficients` is used only for noninteger
orders in `differencing`. A finite fractional filter is an approximation, not
an exact infinite-memory transform. These functions apply no inverse; use
`DifferenceTransform` below for differences that a pipeline undoes.

## Reversible target transforms

`LogTransform()`, `BoxCoxTransform()` and `DifferenceTransform(order=1, lag=1)`
implement `fit_transform(context)`, `inverse_transform(forecast)` and
`clone_unfitted()`. `ComposedTransform(transforms)` applies bundled members in
order and inverts them in reverse order. Each transform is fitted on the context
only. Log and Box-Cox require strictly positive finite targets; Box-Cox fits one
lambda per target by maximum likelihood, bounded to `[-1, 2]`, and rejects a
constant target and transformed forecasts without a preimage. Inversion
requires a valid forecast with the fitted target names in the fitted order.

- Samples and quantiles invert elementwise. Monotone transforms preserve
  quantile order.
- After a nonlinear inverse, the point is the mean of the inverted samples,
  else the inverted 0.5 quantile, else the inverted point read as a median.
- `DifferenceTransform` integrates points and samples exactly. It recomputes
  quantiles from the integrated samples, or omits them when there are none,
  and assumes the forecast rows immediately follow the context.

`transform_config(transform)` describes exact bundled classes as `builtin` and
any other object, including a subclass, as `external`.
`transform_from_config(config)` rebuilds an unfitted bundled transform; replay
rebuilds bundled transforms only.

## Signatures

Generated from the docstrings in the code.

::: timegym.smoothing.simple_moving_average

::: timegym.smoothing.exponential_moving_average

::: timegym.features.add_lagged_target

::: timegym.features.cyclical_encoding

::: timegym.features.add_time_features

::: timegym.features.add_tsfel_features

::: timegym.target.multioutput_target

::: timegym.target.onestep_target

::: timegym.target.step_ahead_target

::: timegym.target.fractional_weights

::: timegym.target.fractional_differencing

::: timegym.target.differencing

::: timegym.target.LogTransform

::: timegym.target.BoxCoxTransform

::: timegym.target.DifferenceTransform

::: timegym.target.ComposedTransform

::: timegym.target.transform_config

::: timegym.target.transform_from_config
