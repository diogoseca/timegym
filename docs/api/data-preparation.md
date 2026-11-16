# Data preparation

Differences of a target series, the reversible target transforms that a
`Pipeline` can fit and invert, and how to put readings taken at irregular
times on a regular grid without letting the future in.

`differencing` is public through `timegym.target`; import it from that module. It
is a standalone pandas transformation, not an automatic Pipeline stage, and it
never mutates the supplied object. Row offsets are positional, so a lag of one
means the previous row even when timestamps are irregular. The reversible target
transforms below are the exception: a `Pipeline` fits and inverts them on each
case. The smoothing, feature, label and fractional-difference helpers of TimeGym
1.4.0 are deprecated, leave the package in TimeGym 2.0 and are no longer
documented here; the changelog names what to use instead.

## Differences

`differencing` takes a whole-number `order` (default 1) and a positive `lag`
(default 1). `order` counts repeated differences and is not the lag distance: order two is
`y[t]-2*y[t-lag]+y[t-2*lag]`. It trims `order*lag` rows. Its
`fractional_coefficients` argument serves only the deprecated fractional orders.

Differencing accepts a Series or DataFrame and preserves its type and labels on
the retained rows. Orders must be finite and nonnegative, and lag must be a
positive integer. Missing observations inside the series propagate; they are
not filled with zero. It applies no inverse; use
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

A pipeline's saved configuration describes exact bundled classes as `builtin`
and any other object, including a subclass, as `external`; replay rebuilds
bundled transforms only.

## Irregular readings

Sensors, logs and sales arrive at irregular times, and a TimeGym history needs
one row per step. Put the readings on a regular grid by taking, at each step,
the last reading at or before it, never one after it. pandas' default
resampling does not do this: `readings.resample("1h").last()` labels each hour
by its start, so a reading taken at 10:50 becomes the value at 10:00, and a
forecast made at 10:00 sees 50 minutes of the future.
`pd.merge_asof` with `direction="backward"` looks only backwards:

```python
import pandas as pd

readings = pd.Series([1.0, 2.0, 3.0, 4.0], name="level", index=pd.to_datetime(
    ["2024-05-01 09:10", "2024-05-01 10:50", "2024-05-01 12:05", "2024-05-01 19:30"]))
grid = pd.date_range("2024-05-01 10:00", "2024-05-01 20:00", freq="1h")
hourly = pd.merge_asof(pd.DataFrame(index=grid), readings.to_frame(),
                       left_index=True, right_index=True,
                       direction="backward",           # the last reading at or before each hour
                       tolerance=pd.Timedelta("6h"))   # older than 6 hours: missing
```

At 11:00 `hourly` holds the 10:50 reading, and at 19:00 it is missing, because
the last reading, at 12:05, is more than six hours old. Choose the tolerance
on purpose: it says how old a reading may be and still count as current.

TimeGym does not fill gaps for you: `ForecastTask` and `cases_from_series`
refuse a history with missing values. Cut the series at a gap, or fill it in
a way you can defend, before you
[cut it into cases](tasks-and-forecasts.md#your-own-data).

## Signatures

::: timegym.target.differencing

::: timegym.target.LogTransform

::: timegym.target.BoxCoxTransform

::: timegym.target.DifferenceTransform

::: timegym.target.ComposedTransform
